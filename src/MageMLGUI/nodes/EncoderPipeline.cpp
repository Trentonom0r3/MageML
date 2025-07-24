// EncoderPipeline.cpp

#include "EncoderPipeline.hpp"
#include <iostream>

EncoderPipeline::EncoderPipeline() = default;
EncoderPipeline::~EncoderPipeline()
{
    finalize();
}

bool EncoderPipeline::initialize(
    const std::string& outFile, std::optional<std::string> codec,
    std::optional<int> width, std::optional<int> height, std::optional<int> bitRate,
    std::optional<int> fps)
{
    std::lock_guard<std::mutex> lock(encoderMutex_);
    try
    {
        encoder_ = std::make_unique<MageML::VideoEncoder>(
            outFile, codec, width, height, bitRate, fps);

        filePath_ = outFile;
        width_ = width.value_or(1920);
        height_ = height.value_or(1080);
        fps_ = fps.value_or(30);
        frameCount_ = 0;
        return true;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[EncoderPipeline] Initialize failed: " << ex.what() << "\n";
        encoder_ = nullptr;
        return false;
    }
}

bool EncoderPipeline::writeFrame(const torch::Tensor& frame)
{
    std::lock_guard<std::mutex> lock(encoderMutex_);
    if (!encoder_)
        return false;

    try
    {
        encoder_->encodeFrame(frame);
        ++frameCount_;
        return true;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[EncoderPipeline] writeFrame error: " << ex.what() << "\n";
        return false;
    }
}

void EncoderPipeline::finalize()
{
    std::lock_guard<std::mutex> lock(encoderMutex_);
    if (encoder_)
    {
        encoder_->close();
        encoder_.reset();
    }
}
