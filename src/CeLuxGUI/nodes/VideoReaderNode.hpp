// VideoReaderNode.hpp

#pragma once

#include "Celux/python/GUIReader.hpp"
#include "CeluxGUI/core/Node.hpp"
#include <any>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class FilterBase;
///
/// A node that reads video frames and outputs them as a torch::Tensor.
/// Supports dynamic “Filters”: you can add them at runtime via addFilter().
///
class VideoReaderNode : public Node
{
  public:
    VideoReaderNode(const std::string& path = "");

    void compute() override;

    std::vector<PortInfo> inputs() const override;
    std::vector<PortInfo> outputs() const override;
    std::vector<ParamInfo> params() const override;
    void setParam(const std::string& name, std::any val) override;

    std::string typeName() const override
    {
        return "VideoReader";
    }

    void drawUI(const std::string& uid) override;

    // Playback controls
    void play();
    void pause();
    void rewind();
    void step();
    void seekToTime(float t);

    float getCurrentTime() const
    {
        return currentTimeSec;
    }
    float getDuration() const;


  private:
    void updatePlayback();

    // ——— internal state ———
    std::string videoPath;
    bool initialized = false;
    bool isPlaying = false;
    float playbackSpeed = 1.0f;
    float currentTimeSec = 0.0f;

    std::unique_ptr<GUIReader> reader;
    std::chrono::steady_clock::time_point lastFrameTime;

};
