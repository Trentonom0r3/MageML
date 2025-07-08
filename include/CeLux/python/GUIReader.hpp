#ifndef GUIREADER_HPP
#define GUIREADER_HPP

#include "Celux/backends/Decoder.hpp" // Ensure this includes the Filter class
#include "Celux/Factory.hpp"


static void PrintTensorDebugInfo(const torch::Tensor& tensor,
                                 std::ostream& os = std::cout)
{
    if (tensor.defined())
    {
        os << "Tensor Info:\n";
        os << "  Shape: " << tensor.sizes() << "\n";
        os << "  Dtype: " << tensor.dtype() << "\n";
        os << " Min: " << tensor.min().item<float>() << "\n";
        os << " Max: " << tensor.max().item<float>() << "\n";
    }
    else
    {
        os << "Tensor is undefined.\n";
    }

}
struct FramePacket
{
    torch::Tensor tensor;
    double timestamp;

    FramePacket() = default;

    FramePacket(torch::Tensor _frame, double _timestamp)
        : tensor(_frame), timestamp(_timestamp)
    {
    }

};

class GUIReader
{
  public:
    /**
     * @brief Constructs a GUIReader for a given input file.
     *
     * @param filePath Path to the input video file.
     * @param numThreads Number of threads to use for decoding.
     * @param device Processing device ("cpu" or "cuda").
     */
    GUIReader(const std::string& filePath,
                int numThreads = static_cast<int>(std::thread::hardware_concurrency() /
                                                  2)
               , std::string tensorShape = "HWC");

    /**
     * @brief Destructor for GUIReader.
     */
    ~GUIReader();

    /**
     * @brief Read a frame from the video.
     *
     * Depending on the configuration, returns either a torch::Tensor or a
     * py::array<uint8_t>. Shape is always HWC. If batch size is specified in Reader
     * config, output shape will be BHWC for Tensors.
     *
     * @return torch::Tensor The next frame as torch::Tensor.
     */
    torch::Tensor readFrame();

    FramePacket readFramePacket();
    /**
     * @brief Seek to a specific timestamp in the video.
     *
     * @param timestamp The timestamp to seek to (in seconds).
     * @return true if seek was successful, false otherwise.
     */
    bool seek(double timestamp);

    /**
     * @brief Get a list of supported codecs.
     *
     * @return std::vector<std::string> List of supported codec names.
     */
    std::vector<std::string> supportedCodecs();


    /**
     * @brief Reset the video reader to the beginning.
     */
    void reset();

    /**
     * @brief Iterator initialization for Python.
     *
     * @return GUIReader& Reference to the GUIReader object.
     */
    GUIReader& iter();

    /**
     * @brief Get the next frame in iteration.
     *
     * @return torch::Tensor Next frame as torch::Tensor.
     */
    FramePacket next();

    /**
     * @brief Get the total number of frames.
     *
     * @return int Total frame count.
     */
    int length() const;

    /**
     * @brief Set the range of frames or timestamps to read.
     *
     * If the start and end values are integers, they are interpreted as frame numbers.
     * If they are floating-point numbers, they are interpreted as timestamps in
     * seconds.
     *
     * @param start Starting frame number or timestamp.
     * @param end Ending frame number or timestamp.
     */
    void setRange(std::variant<int, double> start, std::variant<int, double> end);

    /**
     * @brief Add a filter to the decoder's filter pipeline.
     *
     * @param filterName Name of the filter (e.g., "scale").
     * @param filterOptions Options for the filter (e.g., "1280:720").
     */
    void addFilter(std::shared_ptr<FilterBase> filter);

    /**
     * @brief Initialize the decoder after adding all desired filters.
     *
     * This separates filter addition from decoder initialization, allowing
     * users to configure filters before starting the decoding process.
     *
     * @return true if initialization is successful.
     * @return false otherwise.
     */
    bool initialize();
    /**
     * @brief Set the range of frames to read (helper function).
     *
     * @param startFrame Starting frame index.
     * @param endFrame Ending frame index (-1 for no limit).
     */
    void setRangeByFrames(int startFrame, int endFrame);

    /**
     * @brief Set the range of timestamps to read (helper function).
     *
     * @param startTime Starting timestamp.
     * @param endTime Ending timestamp (-1 for no limit).
     */
    void setRangeByTimestamps(double startTime, double endTime);


    bool seekToFrame(int frame_number);
    torch::ScalarType findTypeFromBitDepth();

    /**
     * @brief Close the video reader and release resources.
     */
    void close();

    // Member variables
    std::shared_ptr<celux::Decoder> decoder;
    celux::Decoder::VideoProperties properties;

    torch::Tensor tensor;

    // Variables for frame range
    int start_frame = 0;
    int end_frame = -1; // -1 indicates no limit

    // Variables for timestamp range
    double start_time = 0.0;
    double end_time = -1.0; // -1 indicates no limit

    // Iterator state
    int currentIndex;
    double current_timestamp; // Add this line

    torch::Tensor bufferedFrame; // The "first valid" frame, if we found it early
    bool hasBufferedFrame = false;
};

#endif // GUIReader_HPP
