#include "VideoPipeline.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>

VideoPipeline::VideoPipeline() : filePath_("")
{
}
VideoPipeline::VideoPipeline(const std::string& fp) : filePath_(normalizePath(fp))
{
}
VideoPipeline::~VideoPipeline() = default;

bool VideoPipeline::initialize()
{
    std::lock_guard<std::mutex> lock(readerMutex_);
    try
    {
        reader_ = std::make_unique<GUIReader>(filePath_);
        reader_->iter();
        reader_->seek(0.0);
        currentTime_ = 0.0;
        playbackBaseTime_ = 0.0;
        return true;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[VideoPipeline] initialize failed: " << ex.what() << "\n";
        reader_ = nullptr;
        return false;
    }
}

bool VideoPipeline::seek(double seconds)
{
    // Clamp to valid range
    seconds = std::min(seconds, duration());
    seconds = std::max(seconds, 0.0);

    {    std::lock_guard<std::mutex> lock(readerMutex_);
  //      std::lock_guard<std::mutex> lock(readerMutex_);
        if (!reader_ || !reader_->seek(seconds))
            return false;
    }
    currentTime_ = seconds;
    playbackBaseTime_ = seconds;
    playStartReal_ = std::chrono::steady_clock::now();
    return true;
}

// MAIN method for live playback (NO seek): just get next frame.
std::shared_ptr<FramePacket> VideoPipeline::getNextFrame()
{
    std::lock_guard<std::mutex> lock(readerMutex_);
    if (!reader_)
        return std::make_shared<FramePacket>();

    lastPacket_ = std::make_shared<FramePacket>(reader_->readFramePacket());
    if (lastPacket_->tensor.defined())
        currentTime_ = lastPacket_->timestamp;
    
    return lastPacket_;
}

torch::Tensor VideoPipeline::getCurrentFrame() const
{
    std::lock_guard<std::mutex> lock(frameMutex_);
    return lastPacket_->tensor;
}

std::shared_ptr<FramePacket> VideoPipeline::getCurrentPacket() const
{
    std::lock_guard<std::mutex> lock(frameMutex_);
    return lastPacket_;
}

// For random-access/seek, jump to a time, read the next frame (rarely used!)
std::shared_ptr<FramePacket> VideoPipeline::getFrameAtTime(double t)
{
    std::lock_guard<std::mutex> lock(readerMutex_);
    if (!reader_)
        return std::make_shared<FramePacket>();

    if (reader_->seek(t))
    {
        lastPacket_ = std::make_shared<FramePacket>(reader_->readFramePacket());
        if (lastPacket_->tensor.defined())
        {
            currentTime_ = lastPacket_->timestamp;
            return lastPacket_;
        }
    }
    return std::make_shared<FramePacket>();
}

StepResult VideoPipeline::step()
{
    // For stepping, just decode one frame forward.
    auto pkt = getNextFrame();

    if (pkt->tensor.defined())
    {

        return StepResult::NewFrame;
    }
    else
    {
        // No more frames available, return EndOfStream.
        return StepResult::EndOfStream;
    }
}
 


double VideoPipeline::currentTime() const
{

    return currentTime_;
}

double VideoPipeline::duration() const
{
    std::lock_guard<std::mutex> lock(readerMutex_);
    return reader_ ? reader_->properties.duration : 0.0;
}

double VideoPipeline::getVideoFps() const
{
    std::lock_guard<std::mutex> lock(readerMutex_);
    return reader_ ? reader_->properties.fps : 0.0;
}

size_t VideoPipeline::totalFrames() const
{
    std::lock_guard<std::mutex> lock(readerMutex_);
    if (!reader_)
        return 0;
    const auto& P = reader_->properties;
    if (P.totalFrames > 0)
        return P.totalFrames;
    return size_t(std::ceil(P.duration * getVideoFps()));
}

int VideoPipeline::width() const
{
    std::lock_guard<std::mutex> lock(readerMutex_);
    return reader_ ? reader_->properties.width : 0;
}
int VideoPipeline::height() const
{
    std::lock_guard<std::mutex> lock(readerMutex_);
    return reader_ ? reader_->properties.height : 0;
}
