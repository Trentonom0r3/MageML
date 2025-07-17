// Decoder.cpp
#include "Decoder.hpp"
#include <Factory.hpp>

using namespace celux::error;

namespace celux
{

Decoder::Decoder(int numThreads, std::vector<std::shared_ptr<FilterBase>> filters)
    : converter(nullptr), formatCtx(nullptr), codecCtx(nullptr), pkt(nullptr),
      videoStreamIndex(-1), numThreads(numThreads), filters_(filters)
{
    CELUX_DEBUG("BASE DECODER: Decoder constructed");
}

Decoder::~Decoder()
{
    CELUX_DEBUG("BASE DECODER: Decoder destructor called");
    close();
}

Decoder::Decoder(Decoder&& other) noexcept
    : formatCtx(std::move(other.formatCtx)), codecCtx(std::move(other.codecCtx)),
      pkt(std::move(other.pkt)), videoStreamIndex(other.videoStreamIndex),
      properties(std::move(other.properties)), frame(std::move(other.frame)),
      converter(std::move(other.converter))
{
    CELUX_DEBUG("BASE DECODER: Decoder move constructor called");
    other.videoStreamIndex = -1;
    // Reset other members if necessary
}

Decoder& Decoder::operator=(Decoder&& other) noexcept
{
    CELUX_DEBUG("BASE DECODER: Decoder move assignment operator called");
    if (this != &other)
    {
        close();

        formatCtx = std::move(other.formatCtx);
        codecCtx = std::move(other.codecCtx);
        pkt = std::move(other.pkt);
        videoStreamIndex = other.videoStreamIndex;
        properties = std::move(other.properties);
        frame = std::move(other.frame);
        converter = std::move(other.converter);

        other.videoStreamIndex = -1;
        // Reset other members if necessary
    }
    return *this;
}

void Decoder::setProperties()
{
    // Set basic video properties
    properties.codec = codecCtx->codec->name;
    properties.width = codecCtx->width;
    properties.height = codecCtx->height;

    // Frame rate calculation
    properties.fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);
    properties.min_fps = properties.fps; // Initialize min fps
    properties.max_fps = properties.fps; // Initialize max fps

    // Ensure duration is calculated properly
    if (formatCtx->streams[videoStreamIndex]->duration != AV_NOPTS_VALUE)
    {
        properties.duration =
            static_cast<double>(formatCtx->streams[videoStreamIndex]->duration) *
            av_q2d(formatCtx->streams[videoStreamIndex]->time_base);
    }
    else if (formatCtx->duration != AV_NOPTS_VALUE)
    {
        properties.duration = static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
    }
    else
    {
        properties.duration = 0.0; // Unknown duration
    }

    // Set pixel format and bit depth
    properties.pixelFormat = codecCtx->pix_fmt;
    properties.bitDepth = getBitDepth();

    // Check for audio stream
    properties.hasAudio = false; // Initialize as false
    for (int i = 0; i < formatCtx->nb_streams; ++i)
    {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
        {
            properties.hasAudio = true; // Set to true if an audio stream is found
            properties.audioBitrate = formatCtx->streams[i]->codecpar->bit_rate;
            properties.audioChannels =
                formatCtx->streams[i]->codecpar->ch_layout.nb_channels;
            properties.audioSampleRate = formatCtx->streams[i]->codecpar->sample_rate;
            properties.audioCodec =
                avcodec_get_name(formatCtx->streams[i]->codecpar->codec_id);
            break; // Stop after finding the first audio stream
        }
    }

    // Calculate total frames
    if (formatCtx->streams[videoStreamIndex]->nb_frames > 0)
    {
        properties.totalFrames = formatCtx->streams[videoStreamIndex]->nb_frames;
    }
    else if (properties.fps > 0 && properties.duration > 0)
    {
        properties.totalFrames = static_cast<int>(properties.fps * properties.duration);
    }
    else
    {
        properties.totalFrames = 0; // Unknown total frames
    }

    // Calculate aspect ratio
    if (properties.width > 0 && properties.height > 0)
    {
        properties.aspectRatio =
            static_cast<double>(properties.width) / properties.height;
    }
    else
    {
        properties.aspectRatio = 0.0; // Unknown aspect ratio
    }

    // Log the video properties
    CELUX_INFO(
        "Video properties: width={}, height={}, fps={}, duration={}, totalFrames={}, "
        "audioBitrate={}, audioChannels={}, audioSampleRate={}, audioCodec={}, "
        "aspectRatio={}",
        properties.width, properties.height, properties.fps, properties.duration,
        properties.totalFrames, properties.audioBitrate, properties.audioChannels,
        properties.audioSampleRate, properties.audioCodec, properties.aspectRatio);
}

void Decoder::initialize(const std::string& filePath)
{
    CELUX_DEBUG("BASE DECODER: Initializing decoder with file: {}", filePath);
    openFile(filePath);
    findVideoStream();
    initCodecContext();
    setProperties();

    converter = celux::Factory::createConverter(torch::kCPU, properties.pixelFormat);

    CELUX_DEBUG("BASE DECODER: Decoder initialization completed");

    CELUX_INFO("BASE DECODER: Decoder using codec: {}, and pixel format: {}",
               codecCtx->codec->name, av_get_pix_fmt_name(codecCtx->pix_fmt));

    if (filters_.size() > 0)
    {
        CELUX_INFO("Applying filters to the decoder");
        initFilterGraph();
    }
}

bool Decoder::initFilterGraph()
{
    char args[512];
    int ret = 0;

    // Create the filter graph
    filter_graph_.reset(avfilter_graph_alloc());
    if (!filter_graph_)
    {
        CELUX_DEBUG("Cannot create filter graph");
        return false;
    }

    // Create buffer source
    const AVFilter* buffersrc = avfilter_get_by_name("buffer");
    AVFilterContext* buffersrc_ctx_local = nullptr;

    // Prepare buffer source arguments
    snprintf(args, sizeof(args),
             "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=1/1",
             codecCtx->width, codecCtx->height, codecCtx->pix_fmt,
             formatCtx->streams[videoStreamIndex]->time_base.num,
             formatCtx->streams[videoStreamIndex]->time_base.den);

    ret = avfilter_graph_create_filter(&buffersrc_ctx_local, buffersrc, "in", args,
                                       nullptr, filter_graph_.get());
    if (ret < 0)
    {
        CELUX_CRITICAL("Cannot create buffer source: {} ", celux::errorToString(ret));
        return false;
    }
    buffersrc_ctx_ = buffersrc_ctx_local;

    // Create buffer sink
    const AVFilter* buffersink = avfilter_get_by_name("buffersink");
    AVFilterContext* buffersink_ctx_local = nullptr;

    ret = avfilter_graph_create_filter(&buffersink_ctx_local, buffersink, "out",
                                       nullptr, nullptr, filter_graph_.get());
    if (ret < 0)
    {
        CELUX_CRITICAL("Cannot create buffer sink: {}", celux::errorToString(ret));
        return false;
    }
    buffersink_ctx_ = buffersink_ctx_local;

    std::string filter_desc;
    for (const auto& filter : filters_)
    {
        filter_desc += filter->getFilterDescription() + ",";
    }
    CELUX_DEBUG("Filter command/description being used: {}", filter_desc);
    // Parse and create the filter graph
    AVFilterInOut* inputs = avfilter_inout_alloc();
    AVFilterInOut* outputs = avfilter_inout_alloc();
    if (!inputs || !outputs)
    {
        std::cerr << "Cannot allocate filter in/out.\n";
        return false;
    }

    outputs->name = av_strdup("in");
    outputs->filter_ctx = buffersrc_ctx_;
    outputs->pad_idx = 0;
    outputs->next = nullptr;

    inputs->name = av_strdup("out");
    inputs->filter_ctx = buffersink_ctx_;
    inputs->pad_idx = 0;
    inputs->next = nullptr;

    ret = avfilter_graph_parse_ptr(filter_graph_.get(), filter_desc.c_str(), &inputs,
                                   &outputs, nullptr);
    if (ret < 0)
    {
        CELUX_ERROR("Error parsing filter graph: {}", celux::errorToString(ret));
        return false;
    }

    ret = avfilter_graph_config(filter_graph_.get(), nullptr);
    if (ret < 0)
    {
        CELUX_ERROR("Error configuring filter graph: {}", celux::errorToString(ret));
        return false;
    }

    // Free the in/out structures
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);

    return true;
}

void Decoder::openFile(const std::string& filePath)
{
    CELUX_DEBUG("BASE DECODER: Opening file: {}", filePath);
    // Open input file
    frame = Frame(); // Fallback to CPU Frame

    AVFormatContext* fmt_ctx = nullptr;
    FF_CHECK_MSG(avformat_open_input(&fmt_ctx, filePath.c_str(), nullptr, nullptr),
                 std::string("Failure Opening Input:"));

    formatCtx.reset(fmt_ctx); // Wrap in unique_ptr
    CELUX_DEBUG("BASE DECODER: Input file opened successfully");

    // Retrieve stream information
    FF_CHECK_MSG(avformat_find_stream_info(formatCtx.get(), nullptr),
                 std::string("Failure Finding Stream Info:"));

    pkt.reset(av_packet_alloc()); // Allocate packet
    CELUX_DEBUG("BASE DECODER: Stream information retrieved successfully");
}

void Decoder::findVideoStream()
{
    CELUX_DEBUG("BASE DECODER: Finding best video stream");

    int ret =
        av_find_best_stream(formatCtx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ret < 0)
    {
		CELUX_ERROR("No video stream found");
		throw CxException("No video stream found");
	}

    videoStreamIndex = ret;
    CELUX_DEBUG("BASE DECODER: Video stream found at index {}", videoStreamIndex);
}

void Decoder::initCodecContext()
{
    const AVCodec* codec =
        avcodec_find_decoder(formatCtx->streams[videoStreamIndex]->codecpar->codec_id);

    CELUX_DEBUG("BASE DECODER: Initializing codec context");
    if (!codec)
    {
        CELUX_ERROR("Unsupported codec!");
        throw CxException("Unsupported codec!");
    }

    // Allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        CELUX_ERROR("Could not allocate codec context");
        throw CxException("Could not allocate codec context");
    }
    codecCtx.reset(codec_ctx);
    CELUX_DEBUG("BASE DECODER: Codec context allocated");

    // Copy codec parameters from input stream to codec context
    FF_CHECK_MSG(avcodec_parameters_to_context(
                     codecCtx.get(), formatCtx->streams[videoStreamIndex]->codecpar),
                 std::string("Failed to copy codec parameters:"));

    CELUX_DEBUG("BASE DECODER: Codec parameters copied to codec context");

    codecCtx->thread_count = numThreads;
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    CELUX_DEBUG("BASE DECODER: Codec context threading configured: thread_count={}, "
                "thread_type={}",
                codecCtx->thread_count, codecCtx->thread_type);
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;
    // Open codec
    FF_CHECK_MSG(avcodec_open2(codecCtx.get(), codec, nullptr),
                 std::string("Failed to open codec:"));

    CELUX_DEBUG("BASE DECODER: Codec opened successfully");
}

// Decoder.cpp

bool Decoder::decodeNextFrame(void* buffer, double* frame_timestamp)
{
    CELUX_TRACE("Decoding next frame");
    int ret;

    if (buffer == nullptr)
    {
        CELUX_ERROR("Buffer is null");
        throw CxException("Buffer is null");
    }

    while (true)
    {
        // Attempt to receive a decoded frame
        ret = avcodec_receive_frame(codecCtx.get(), frame.get());
        if (ret == AVERROR(EAGAIN))
        {
            CELUX_DEBUG("Decoder needs more packets, reading next packet");
            // Need to feed more packets
            // Proceed to read the next packet
        }
        else if (ret == AVERROR_EOF)
        {
            CELUX_TRACE("No more frames to decode");
            // No more frames to decode
            return false;
        }
        else if (ret < 0)
        {
            CELUX_ERROR("Error during decoding");
            throw CxException("Error during decoding");
        }
        else
        {
            // Successfully received a frame
            CELUX_DEBUG("Frame decoded successfully");

            if (filters_.size() > 0)
            {
                // Push the decoded frame into the filter graph's buffer source
                ret = av_buffersrc_add_frame(buffersrc_ctx_, frame.get());
                if (ret < 0)
                {
                    CELUX_ERROR("Error while feeding the filter graph: {}",
                                celux::errorToString(ret));
                    return false;
                }

                // Retrieve the filtered frame from the buffer sink
                ret = av_buffersink_get_frame(buffersink_ctx_, frame.get());
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                {
                    break;
                }
                else if (ret < 0)
                {
                    CELUX_ERROR("Error during filtering: {}",
                                celux::errorToString(ret));
                    return false;
                }
            }

            // **Retrieve the frame timestamp**
            if (frame_timestamp)
            {
                *frame_timestamp = getFrameTimestamp(frame.get());
                CELUX_DEBUG("Frame timestamp retrieved: {}", *frame_timestamp);
            }

            // Pass the (possibly filtered) frame to the converter
            converter->convert(frame, buffer);
            CELUX_DEBUG("Frame converted");

            return true;
        }

        // Read the next packet from the video file
        ret = av_read_frame(formatCtx.get(), pkt.get());
        if (ret == AVERROR_EOF)
        {
            CELUX_TRACE("End of file reached, flushing decoder");
            // End of file: flush the decoder
            FF_CHECK(avcodec_send_packet(codecCtx.get(), nullptr));
        }
        else if (ret < 0)
        {
            CELUX_ERROR("Error reading frame");
            throw CxException("Error reading frame");
        }
        else
        {
            CELUX_TRACE("Packet read from file, stream_index={}", pkt->stream_index);
            // If the packet belongs to the video stream, send it to the decoder
            if (pkt->stream_index == videoStreamIndex)
            {
                FF_CHECK(avcodec_send_packet(codecCtx.get(), pkt.get()));
                CELUX_DEBUG("Packet sent to decoder");
            }
            else
            {
                CELUX_DEBUG("Packet does not belong to video stream, skipping");
            }
            // Release the packet back to FFmpeg
            av_packet_unref(pkt.get());
        }
    }
}

bool Decoder::seekFrame(int frameIndex)
{
    CELUX_TRACE("Seeking to frame index: {}", frameIndex);

    if (frameIndex < 0 || frameIndex > properties.totalFrames)
    {
        CELUX_WARN("Frame index out of bounds: {}", frameIndex);
        return false;
    }

    int64_t target_pts = av_rescale_q(frameIndex, {1, static_cast<int>(properties.fps)},
                                      formatCtx->streams[videoStreamIndex]->time_base);
    return seek(target_pts * av_q2d(formatCtx->streams[videoStreamIndex]->time_base));
}

bool Decoder::seek(double timestamp)
{
    CELUX_TRACE("Seeking to timestamp: {}", timestamp);
    if (timestamp < 0 || timestamp > properties.duration)
    {
        CELUX_WARN("Timestamp out of bounds: {}", timestamp);
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    CELUX_DEBUG("Converted timestamp for seeking: {}", ts);
    int ret =
        av_seek_frame(formatCtx.get(), videoStreamIndex, ts, AVSEEK_FLAG_BACKWARD);

    if (ret < 0)
    {
        CELUX_ERROR("Seek failed to timestamp: {}", timestamp);
        return false;
    }

    // Flush codec buffers
    avcodec_flush_buffers(codecCtx.get());
    CELUX_TRACE("Seek successful, codec buffers flushed");

    return true;
}

Decoder::VideoProperties Decoder::getVideoProperties() const
{
    CELUX_TRACE("Retrieving video properties");
    return properties;
}

bool Decoder::isOpen() const
{
    bool open = formatCtx != nullptr && codecCtx != nullptr;
    CELUX_DEBUG("BASE DECODER: Decoder isOpen check: {}", open);
    return open;
}

void Decoder::close()
{
    CELUX_DEBUG("BASE DECODER: Closing decoder");
    if (codecCtx)
    {
        codecCtx.reset();
        CELUX_DEBUG("BASE DECODER: Codec context reset");
    }
    if (formatCtx)
    {
        formatCtx.reset();
        CELUX_DEBUG("BASE DECODER: Format context reset");
    }
    if (converter)
    {
        CELUX_DEBUG("BASE DECODER: Synchronizing converter in Decoder close");
        converter->synchronize();
        converter.reset();
    }
    videoStreamIndex = -1;
    properties = VideoProperties{};
    CELUX_DEBUG("BASE DECODER: Decoder closed");
}


std::vector<std::string> Decoder::listSupportedDecoders() const
{
    CELUX_DEBUG("BASE DECODER: Listing supported decoders");
    std::vector<std::string> decoders;
    void* iter = nullptr;
    const AVCodec* codec = nullptr;

    while ((codec = av_codec_iterate(&iter)) != nullptr)
    {
        if (av_codec_is_decoder(codec))
        {
            std::string codecInfo = std::string(codec->name);

            // Append the long name if available
            if (codec->long_name)
            {
                codecInfo += " - " + std::string(codec->long_name);
            }

            decoders.push_back(codecInfo);
            CELUX_TRACE("Supported decoder found: {}", codecInfo);
        }
    }

    return decoders;
}

AVCodecContext* Decoder::getCtx()
{
    CELUX_TRACE("Getting codec context");
    return codecCtx.get();
}

int64_t Decoder::convertTimestamp(double timestamp) const
{
    CELUX_TRACE("Converting timestamp: {}", timestamp);
    AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
    int64_t ts = static_cast<int64_t>(timestamp * time_base.den / time_base.num);
    CELUX_TRACE("Converted timestamp: {}", ts);
    return ts;
}
int Decoder::getBitDepth() const
{
    CELUX_TRACE("Getting bit depth");
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(
        AVPixelFormat(formatCtx->streams[videoStreamIndex]->codecpar->format));
    if (!desc)
    {
        CELUX_WARN("Unknown pixel format, defaulting to NV12ToRGB");
    }

    int bitDepth = desc->comp[0].depth;
    CELUX_TRACE("Bit depth: {}", bitDepth);
    return bitDepth;
}

bool Decoder::seekToNearestKeyframe(double timestamp)
{
    CELUX_TRACE("Seeking to the nearest keyframe for timestamp: {}", timestamp);
    if (timestamp < 0 || timestamp > properties.duration)
    {
        CELUX_WARN("Timestamp out of bounds: {}", timestamp);
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    CELUX_DEBUG("Converted timestamp for keyframe seeking: {}", ts);

    // Perform seek operation to the nearest keyframe before the timestamp
    int ret =
        av_seek_frame(formatCtx.get(), videoStreamIndex, ts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0)
    {
        CELUX_ERROR("Keyframe seek failed for timestamp: {}", timestamp);
        return false;
    }

    // Flush codec buffers to reset decoding from the keyframe
    avcodec_flush_buffers(codecCtx.get());
    CELUX_TRACE("Keyframe seek successful, codec buffers flushed");

    return true;
}

double Decoder::getFrameTimestamp(AVFrame* frame)
{
    if (!frame)
    {
        CELUX_WARN("Received a null frame pointer.");
        return -1.0;
    }

    // Define a lambda to convert AV_TIME_BASE to seconds
    auto convert_to_seconds = [&](int64_t timestamp, AVRational time_base) -> double
    { return static_cast<double>(timestamp) * av_q2d(time_base); };

    // Attempt to retrieve the best_effort_timestamp first
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
    {
        AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
        double timestamp = convert_to_seconds(frame->best_effort_timestamp, time_base);
        CELUX_DEBUG("Using best_effort_timestamp: {}", timestamp);
        return timestamp;
    }

    // Fallback to frame->pts
    if (frame->pts != AV_NOPTS_VALUE)
    {
        AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
        double timestamp = convert_to_seconds(frame->pts, time_base);
        CELUX_DEBUG("Using frame->pts: {}", timestamp);
        return timestamp;
    }

    // Fallback to frame->pkt_dts if available
    if (frame->pkt_dts != AV_NOPTS_VALUE)
    {
        AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
        double timestamp = convert_to_seconds(frame->pkt_dts, time_base);
        CELUX_DEBUG("Using frame->pkt_dts: {}", timestamp);
        return timestamp;
    }

    // If all timestamp fields are invalid, log a warning and handle accordingly
    CELUX_WARN("Frame has no valid timestamp. Returning -1.0");
    return -1.0;
}


void Decoder::addFilter(std::shared_ptr<FilterBase> filter)
{

    filters_.push_back(filter);
    CELUX_DEBUG("BASE DECODER: Added filter: {}", filter->getFilterDescription());

    initFilterGraph();
}

void Decoder::removeFilter(std::shared_ptr<FilterBase> filter)
{
    auto it = std::remove(filters_.begin(), filters_.end(), filter);
    if (it != filters_.end())
    {
        filters_.erase(it, filters_.end());
        CELUX_DEBUG("BASE DECODER: Removed filter: {}", filter->getFilterDescription());
    }
    else
    {
        CELUX_DEBUG("BASE DECODER: Filter not found: {}", filter->getFilterDescription());
    }
    initFilterGraph();
}

} // namespace celux
