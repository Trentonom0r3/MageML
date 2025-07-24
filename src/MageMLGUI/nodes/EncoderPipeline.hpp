// EncoderPipeline.hpp

#pragma once

#include "core/VideoEncoder.hpp" // Or just VideoEncoder.hpp, depending on path
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <torch/torch.h>

class EncoderPipeline
{
  public:
    EncoderPipeline();
    ~EncoderPipeline();

    // Initialize with output file and optional params
    bool initialize(const std::string& outFile, std::optional<std::string> codec = {},
                    std::optional<int> width = {}, std::optional<int> height = {},
                    std::optional<int> bitRate = {}, std::optional<int> fps = {});

    // Write a single frame (tensor: HWC uint8, RGB)
    bool writeFrame(const torch::Tensor& frame);

    // Flush and close the encoder
    void finalize();

    // Properties for external use (query for UI, progress, etc.)
    int getWidth() const
    {
        return width_;
    }
    int getHeight() const
    {
        return height_;
    }
    int getFps() const
    {
        return fps_;
    }
    size_t getFrameCount() const
    {
        return frameCount_;
    }
    std::string getFilePath() const
    {
        return filePath_;
    }

  private:
    std::string filePath_;
    int width_ = 0, height_ = 0, fps_ = 0;
    size_t frameCount_ = 0;
    std::unique_ptr<MageML::VideoEncoder> encoder_;
    std::mutex encoderMutex_;
};
