#pragma once
#ifndef VIDEO_ENCODER_HPP
#define VIDEO_ENCODER_HPP

#include "Encoder.hpp"
#include <filesystem>
#include <optional>

namespace MageML
{

class VideoEncoder
{
  public:
    // Simplified constructor with optional arguments
    VideoEncoder(const std::string& filename,
                 std::optional<std::string> codec = std::nullopt,
                 std::optional<int> width = std::nullopt,
                 std::optional<int> height = std::nullopt,
                 std::optional<int> bitRate = std::nullopt,
                 std::optional<int> fps = std::nullopt,
                 std::optional<int> audioBitRate = std::nullopt,
                 std::optional<int> audioSampleRate = std::nullopt,
                 std::optional<int> audioChannels = std::nullopt,
                 std::optional<std::string> audioCodec = std::nullopt);

    ~VideoEncoder();

    void encodeFrame(torch::Tensor frame);
    void encodeAudioFrame(const torch::Tensor& audio);
    void close();
    MageML::Encoder::EncodingProperties props;
  private:
    std::unique_ptr<MageML::Encoder> encoder;
    int width, height;
    std::unique_ptr<MageML::conversion::IConverter> converter;
    MageML::Encoder::EncodingProperties inferEncodingProperties(
        const std::string& filename, std::optional<std::string> codec,
        std::optional<int> width, std::optional<int> height, std::optional<int> bitRate,
        std::optional<int> fps, std::optional<int> audioBitRate,
        std::optional<int> audioSampleRate, std::optional<int> audioChannels,
        std::optional<std::string> audioCodec);
};

} // namespace MageML

#endif // VIDEO_ENCODER_HPP
