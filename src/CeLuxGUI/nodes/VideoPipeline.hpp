#pragma once

#include "python/GUIReader.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

static std::string normalizePath(const std::string& raw)
{
    try
    {
        std::string s = raw;
        if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                              (s.front() == '\'' && s.back() == '\'')))
            s = s.substr(1, s.size() - 2);
        std::error_code ec;
        auto abs = std::filesystem::absolute(s, ec);
        return ec ? s : abs.string();
    }
    catch (...)
    {
        return raw;
    }
}

enum class StepResult
{
    NewFrame,
    EndOfStream,
    Error
};

class VideoPipeline
{
  public:
    VideoPipeline();
    explicit VideoPipeline(const std::string& filePath);
    ~VideoPipeline();

    bool initialize();
    bool seek(double seconds);
    StepResult step();
    torch::Tensor getCurrentFrame() const;
    // Optionally, provide a FramePacket version:
    std::shared_ptr<FramePacket> getCurrentPacket() const;
    double currentTime() const;
    double duration() const;
    double getVideoFps() const;
    size_t totalFrames() const;

    // For live playback, always just use getNextFrame()
    std::shared_ptr<FramePacket> getNextFrame();
    // For stepping, also just use getNextFrame()
    std::shared_ptr<FramePacket>
    getFrameAtTime(double t); // Only for random-access (seek + read)

    int width() const;
    int height() const;

    std::shared_ptr<FramePacket> lastPacket_;
    mutable std::mutex frameMutex_;

    // Core state
    std::string filePath_;
    std::unique_ptr<GUIReader> reader_;

    double playbackBaseTime_{0.0};
    std::chrono::steady_clock::time_point playStartReal_;
    double currentTime_{0.0};
    mutable std::mutex readerMutex_;
};
