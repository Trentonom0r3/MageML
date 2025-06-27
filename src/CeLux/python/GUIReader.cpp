// Python/GUIReader.cpp

#include "Python/GUIReader.hpp"
#include <torch/torch.h> // Ensure you have included the necessary Torch headers
#include <TensorBuilder.hpp>

#define CHECK_TENSOR(tensor)                                                  \
	if (!tensor.defined() || tensor.numel() == 0)                              \
	{                                                                         \
		throw std::runtime_error("Invalid tensor: undefined or empty");        \
	}

GUIReader::GUIReader(const std::string& filePath, int numThreads,
                         std::vector<std::shared_ptr<FilterBase>> filters, std::string tensorShape)
    : decoder(nullptr), currentIndex(0), start_frame(0), end_frame(-1),
      start_time(-1.0), end_time(-1.0), filters_(filters)
{
    //set ffmpeg log level
    CELUX_INFO("GUIReader constructor called with filePath: {}", filePath);

    if (numThreads > std::thread::hardware_concurrency())
    {
        throw std::invalid_argument("Number of threads cannot exceed hardware concurrency");
	}

    try
    {

        torch::Device torchDevice = torch::Device(torch::kCPU);
        CELUX_INFO("Creating GUIReader instance");
       
        decoder =
            celux::Factory::createDecoder(torchDevice, filePath, numThreads, filters_);
        CELUX_INFO("Decoder created successfully");

        audio = std::make_shared<Audio>(decoder); // Create audio object

        torch::Dtype torchDataType;

        torchDataType = torch::kUInt8;

        // Retrieve video properties
        properties = decoder->getVideoProperties();
        
        for (auto& filter : filters_)
        { // Iterate through filters_
            // Use dynamic_cast to check if the filter is of type Scale
            if (Scale* scaleFilter = dynamic_cast<Scale*>(filter.get()))
            {
                properties.width = std::stoi(scaleFilter->getWidth());
                properties.height = std::stoi(scaleFilter->getHeight());
            }
        }
    
        CELUX_INFO("Video properties retrieved: width={}, height={}, fps={}, "
                   "duration={}, totalFrames={}, pixelFormat={}, hasAudio={}",
                   properties.width, properties.height, properties.fps,
                   properties.duration, properties.totalFrames,
                   av_get_pix_fmt_name(properties.pixelFormat), properties.hasAudio);

        TensorBuilder builder(tensorShape);
        builder.createTensor(properties.height, properties.width, torchDataType,
                             torchDevice);
        // Initialize tensor
        tensor = builder.getTensor().contiguous().clone();
        CHECK_TENSOR(tensor);
        
  //  list_ffmpeg_filters("ffmpeg_filters.json");
    }
    catch (const std::exception& ex)
    {
        CELUX_ERROR("Exception in GUIReader constructor: {}", ex.what());
        throw; // Re-throw exception after logging
    }
}

std::shared_ptr<GUIReader::Audio> GUIReader::getAudio()
{
    return audio;
}

// -------------------------
// Audio Class Implementation
// -------------------------

GUIReader::Audio::Audio(std::shared_ptr<celux::Decoder> decoder)
    : decoder(std::move(decoder))
{
    if (!this->decoder)
    {
        throw std::runtime_error("Audio: Invalid decoder instance provided.");
    }
}

torch::Tensor GUIReader::Audio::getAudioTensor()
{
    return decoder->getAudioTensor();
}

bool GUIReader::Audio::extractToFile(const std::string& outputFilePath)
{
    return decoder->extractAudioToFile(outputFilePath);
}

celux::Decoder::VideoProperties GUIReader::Audio::getProperties() const
{
    return decoder->getVideoProperties();
}

GUIReader::~GUIReader()
{
    CELUX_INFO("GUIReader destructor called");
    close();
}

void GUIReader::setRange(std::variant<int, double> start,
                           std::variant<int, double> end)
{
    // Check if both start and end are of the same type
    if (start.index() != end.index())
    {

        throw std::invalid_argument("Start and end must be of the same type.");
    }

    // Set the range based on the type of start and end
    if (std::holds_alternative<int>(start) && std::holds_alternative<int>(end))
    {
        int startFrame = std::get<int>(start);
        int endFrame = std::get<int>(end);
        setRangeByFrames(startFrame, endFrame);
    }
    else if (std::holds_alternative<double>(start) &&
             std::holds_alternative<double>(end))
    {
        double startTime = std::get<double>(start);
        double endTime = std::get<double>(end);
        setRangeByTimestamps(startTime, endTime);
    }
    else
    {

        throw std::invalid_argument("Unsupported type for start and end.");
    }
}

void GUIReader::addFilter(std::shared_ptr<FilterBase> filter)
{
    CELUX_INFO("Adding filter: {}", filter->getFilterDescription());
    if (!filter)
    {
        CELUX_ERROR("Cannot add a null filter");
        throw std::invalid_argument("Filter cannot be null.");
    }
    // Check if the filter is already in the list
    for (const auto& existingFilter : filters_)
    {
        if (existingFilter->getFilterDescription() == filter->getFilterDescription())
        {
            CELUX_WARN("Filter {} is already added, skipping.", filter->getFilterDescription());
            return; // Filter already exists, skip adding
        }
    }
    // Add the new filter
    filters_.push_back(filter);
    decoder->addFilter(filter);

}

void GUIReader::setRangeByFrames(int startFrame, int endFrame)
{
    CELUX_INFO("Setting frame range: start={}, end={}", startFrame, endFrame);

    // Handle negative indices by converting them to positive frame numbers
    if (startFrame < 0)
    {
        startFrame = properties.totalFrames + startFrame;
        CELUX_INFO("Adjusted start_frame to {}", startFrame);
    }
    if (endFrame < 0)
    {
        endFrame = properties.totalFrames + endFrame;
        CELUX_INFO("Adjusted end_frame to {}", endFrame);
    }

    // Validate the adjusted frame range
    if (startFrame < 0 || endFrame < 0)
    {
        CELUX_ERROR("Frame indices out of range after adjustment: start={}, end={}",
                    startFrame, endFrame);
        throw std::runtime_error("Frame indices out of range after adjustment.");
    }
    if (endFrame <= startFrame)
    {
        CELUX_ERROR("Invalid frame range: end_frame ({}) must be greater than "
                    "start_frame ({}) after adjustment.",
                    endFrame, startFrame);
        throw std::runtime_error(
            "end_frame must be greater than start_frame after adjustment.");
    }

    // Make end_frame exclusive by subtracting one
    endFrame = endFrame - 1;
    CELUX_INFO("Adjusted end_frame to be exclusive: {}", endFrame);

    start_frame = startFrame;
    end_frame = endFrame;
    CELUX_INFO("Frame range set: start_frame={}, end_frame={}", start_frame, end_frame);
}

void GUIReader::setRangeByTimestamps(double startTime, double endTime)
{
    CELUX_INFO("Setting timestamp range: start={}, end={}", startTime, endTime);

    // Validate the timestamp range
    if (startTime < 0 || endTime < 0)
    {
        CELUX_ERROR("Timestamps cannot be negative: start={}, end={}", startTime,
                    endTime);
        throw std::invalid_argument("Timestamps cannot be negative.");
    }
    if (endTime <= startTime)
    {
        CELUX_ERROR("Invalid timestamp range: end ({}) must be greater than start ({})",
                    endTime, startTime);
        throw std::invalid_argument("end must be greater than start.");
    }

    // Set the timestamp range
    start_time = startTime;
    end_time = endTime;
    CELUX_INFO("Timestamp range set: start_time={}, end_time={}", start_time, end_time);
}

torch::Tensor GUIReader::readFrame()
{
    CELUX_TRACE("readFrame() called");

    double frame_timestamp = 0.0;
    bool success = decoder->decodeNextFrame(tensor.data_ptr(), &frame_timestamp);
    if (!success)
    {
        CELUX_WARN("Decoding failed or no more frames available");
        return torch::Tensor(); // Return an empty tensor if decoding failed
    }

    // Update current timestamp
    current_timestamp = frame_timestamp;

    CELUX_TRACE("Frame decoded successfully at timestamp: {}", current_timestamp);
    return tensor;
}

void GUIReader::close()
{
    CELUX_INFO("Closing GUIReader");
    // Clean up decoder and other resources
    if (decoder)
    {
        CELUX_INFO("Closing decoder");
        decoder->close();
        decoder.reset();
        CELUX_INFO("Decoder closed and reset successfully");
    }
    else
    {
        CELUX_INFO("Decoder already closed or was never initialized");
    }
}

bool GUIReader::seek(double timestamp)
{
    CELUX_TRACE("Seeking to timestamp: {}", timestamp);

    if (timestamp < 0 || timestamp > properties.duration)
    {
        CELUX_ERROR("Timestamp out of range: {}", timestamp);
        return false;
    }

    bool success = decoder->seekToNearestKeyframe(timestamp);
    if (!success)
    {
        CELUX_WARN("Seek to keyframe failed at timestamp {}", timestamp);
        return false;
    }

    // Decode frames until reaching the exact timestamp
    while (current_timestamp < timestamp)
    {
        readFrame();
    }

    CELUX_TRACE("Exact seek to timestamp {} successful", timestamp);
    return true;
}

std::vector<std::string> GUIReader::supportedCodecs()
{
    CELUX_TRACE("supportedCodecs() called");
    std::vector<std::string> codecs = decoder->listSupportedDecoders();
    CELUX_INFO("Number of supported decoders: {}", codecs.size());
    for (const auto& codec : codecs)
    {
        CELUX_TRACE("Supported decoder: {}", codec);
    }
    return codecs;
}

void GUIReader::reset()
{
    CELUX_INFO("Resetting GUIReader to the beginning");
    bool success = seek(0.0);
    if (success)
    {
        currentIndex = 0;
        CELUX_INFO("GUIReader reset successfully");
    }
    else
    {
        CELUX_WARN("Failed to reset GUIReader to the beginning");
    }
}

bool GUIReader::seekToFrame(int frame_number)
{
    CELUX_INFO("Seeking to frame number: {}", frame_number);

    if (frame_number < 0 || frame_number >= properties.totalFrames)
    {
        CELUX_ERROR("Frame number {} is out of range (0 to {})", frame_number,
                    properties.totalFrames);
        return false;
    }
    double seek_timestamp = frame_number / properties.fps;


    // Seek to the closest keyframe first
    bool success = decoder->seekToNearestKeyframe(seek_timestamp);
    if (!success)
    {
        CELUX_WARN("Seek to keyframe for frame {} failed", frame_number);
        return false;
    }

    // Decode frames until reaching the exact requested frame
    int current_frame = static_cast<int>(current_timestamp * properties.fps);
    while (current_frame < frame_number)
    {
        readFrame();
        current_frame++;
    }

    CELUX_INFO("Exact seek to frame {} successful", frame_number);
    return true;
}


GUIReader& GUIReader::iter()
{
    CELUX_TRACE("iter() called: Preparing GUIReader for iteration");

    // Reset iterator state
    currentIndex = 0;
    current_timestamp = 0.0;
    bufferedFrame = torch::Tensor(); // Clear any old buffered frame
    hasBufferedFrame = false;

    if (start_time >= 0.0 && end_time > 0.0)
    {
        // Using timestamp range
        CELUX_INFO("Using timestamp range for iteration: start_time={}, end_time={}",
                   start_time, end_time);

        bool success = seek(start_time);
        if (!success)
        {
            CELUX_ERROR("Failed to seek to start_time: {}", start_time);
            throw std::runtime_error("Failed to seek to start_time.");
        }

        // -------------------------------------------------------
        // 1) DECODING + DISCARD loop
        // -------------------------------------------------------
        // Keep reading frames, discarding them, until we hit >= start_time.
        while (true)
        {
            // Attempt to decode a frame
            torch::Tensor f = readFrame();
            if (!f.defined() || f.numel() == 0)
            {
                // No more frames, or decode error
                CELUX_WARN("Ran out of frames while discarding up to start_time={}",
                           start_time);
                break;
            }

            // current_timestamp was updated in readFrame().
            if (current_timestamp >= start_time)
            {
                // We have reached or passed start_time
                // --> store this frame for later return in next()
                bufferedFrame = f;
                hasBufferedFrame = true;
                CELUX_DEBUG("Discard loop found first frame at timestamp {}",
                            current_timestamp);
                break;
            }
            // else discard and loop again
        }
        // -------------------------------------------------------

        current_timestamp = std::max(current_timestamp, start_time);
    }
    else if (start_frame >= 0 && end_frame >= 0)
    {
        // Using frame range
        CELUX_INFO("Using frame range for iteration: start_frame={}, end_frame={}",
                   start_frame, end_frame);
        bool success = seekToFrame(start_frame);
        if (!success)
        {
            CELUX_ERROR("Failed to seek to start_frame: {}", start_frame);
            throw std::runtime_error("Failed to seek to start_frame.");
        }
        currentIndex = start_frame;
        current_timestamp = static_cast<double>(currentIndex) / properties.fps;
    }
    else
    {
        // No range set; start from the beginning
        CELUX_INFO("No range set; starting from the beginning");
        bool success = seek(0.0);
        if (!success)
        {
            CELUX_ERROR("Failed to seek to the beginning of the video");
            throw std::runtime_error("Failed to seek to the beginning of the video.");
        }
        current_timestamp = 0.0;
    }

    // Return self for iterator protocol
    return *this;
}

torch::Tensor GUIReader::next()
{
    CELUX_TRACE("next() called: Retrieving next frame");

    // If we have a buffered frame from the discard loop, consume it first.
    torch::Tensor frame;
    if (hasBufferedFrame)
    {
        frame = bufferedFrame;
        hasBufferedFrame = false;
        // current_timestamp is already set by readFrame() earlier.
    }
    else
    {
        // Otherwise decode the next frame
        frame = readFrame();
        if (!frame.defined() || frame.numel() == 0)
        {
            CELUX_INFO("No more frames available (decode returned empty).");

        }
    }

    // -- Now check if we exceeded the time range AFTER decoding.
    if (start_time >= 0.0 && end_time > 0.0)
    {
        // If the current frame's timestamp is >= end_time, skip/stop. end time + 1 frame
        if (current_timestamp > end_time + 1/properties.fps)
        {
            CELUX_DEBUG("Frame timestamp {} >= end_time {}, skipping frame.",
                        current_timestamp, end_time);

        }
    }
    else if (start_frame >= 0 && end_frame >= 0)
    {
        if (currentIndex > end_frame)
        {
            CELUX_DEBUG("Frame range exhausted: currentIndex={}, end_frame={}",
                        currentIndex, end_frame);

        }
    }

    currentIndex++;
    CELUX_TRACE("Returning frame index={}, timestamp={}", currentIndex - 1,
                current_timestamp);
    return frame;
}


int GUIReader::length() const
{
    CELUX_TRACE("length() called: Returning totalFrames = {}", properties.totalFrames);
    return properties.totalFrames;
}