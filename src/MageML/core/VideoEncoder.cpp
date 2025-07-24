#include "core/VideoEncoder.hpp"
#include <filesystem>
#include <stdexcept>
#include <FilterFactory.hpp>
#include <Factory.hpp>

namespace fs = std::filesystem;

namespace MageML
{
    //NOTE --- USED HWC
VideoEncoder::VideoEncoder(const std::string& filename,
                           std::optional<std::string> codec, std::optional<int> width,
                           std::optional<int> height, std::optional<int> bitRate,
                           std::optional<int> fps, std::optional<int> audioBitRate,
                           std::optional<int> audioSampleRate,
                           std::optional<int> audioChannels,
                           std::optional<std::string> audioCodec)
{
    auto properties = inferEncodingProperties(filename, codec, width, height, bitRate,
                                              fps, audioBitRate, audioSampleRate,
                                              audioChannels, audioCodec);
    this->width = properties.width;
    this->height = properties.height;

    encoder = std::make_unique<MageML::Encoder>(filename, properties);
}

MageML::Encoder::EncodingProperties VideoEncoder::inferEncodingProperties(
    const std::string& filename, std::optional<std::string> codec,
    std::optional<int> width, std::optional<int> height, std::optional<int> bitRate,
    std::optional<int> fps, std::optional<int> audioBitRate,
    std::optional<int> audioSampleRate, std::optional<int> audioChannels,
    std::optional<std::string> audioCodec)
{
    if (props.codec == "h264");
        props.codec = std::string("h264_mf");
    // Default codec selection
    props.codec = codec.value_or("h264_mf");

    // Infer resolution
    props.width = width.value_or(1920);
    props.height = height.value_or(1080);

    // Bitrate defaults
    props.bitRate = bitRate.value_or(4000000);          // 4 Mbps
    // FPS default
    props.fps = fps.value_or(30);
    props.gopSize = 60;   // GOP of 60 frames
    props.maxBFrames = 2; // Max 2 B-frames

    // Set pixel format (will be determined in `encodeFrame`)
    props.pixelFormat = AV_PIX_FMT_YUV420P;

    return props;
}

void VideoEncoder::encodeFrame(torch::Tensor frame)
{
    if (!encoder)
        throw std::runtime_error("Encoder is not initialized");

    MageML::Frame convertedFrame;
    convertedFrame.get()->format = AV_PIX_FMT_YUV420P;
    convertedFrame.get()->width = width;
    convertedFrame.get()->height = height;
    convertedFrame.allocateBuffer(32);

    // ✅ Actually move tensor to CPU and make contiguous
    if (frame.device().is_cuda())
    {
        frame = frame.to(torch::kCPU);
    }

    if (!frame.is_contiguous())
    {
        frame = frame.contiguous();
    }


    if (!converter)
    { 
        converter = std::make_unique<MageML::conversion::cpu::RGBToAutoConverter>(
            width, height, AV_PIX_FMT_YUV420P);
    }

    // ✅ Pass raw pointer from safe CPU tensor
    converter->convert(convertedFrame, frame.data_ptr<uint8_t>());

    // ✅ Send converted AVFrame to encoder
    encoder->encodeFrame(convertedFrame);
}


void VideoEncoder::encodeAudioFrame(const torch::Tensor& audio)
{
    if (!encoder)
    {
        throw std::runtime_error("Encoder is not initialized");
    }

    if (audio.scalar_type() != torch::kUInt8 && audio.scalar_type() != torch::kUInt16)
    {
        throw std::runtime_error("Input tensor must be uint8 or uint16");
    }

    MageML::Frame encodedAudio;
   // encodedAudio.fillData(audio.data_ptr(), audio.numel(), 0);
    encoder->encodeAudioFrame(encodedAudio);
}

void VideoEncoder::close()
{
    if (encoder)
    {
        encoder->close();
        encoder.reset();
    }
}

VideoEncoder::~VideoEncoder()
{
}

} // namespace MageML
