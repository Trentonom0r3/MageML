// VideoReaderNode.cpp

#include "CeluxGUI/nodes/VideoReaderNode.hpp"

#include <chrono>
#include <imgui.h>
#include <imgui_stdlib.h>
#include <iostream>
#include <torch/torch.h>

//
// -- Registration
//
namespace
{
const bool registered = []()
{
    NodeFactory::instance().registerType(
        "VideoReader", []() { return std::make_shared<VideoReaderNode>(""); });
    return true;
}();
} // namespace

//
// -- Constructor
//
VideoReaderNode::VideoReaderNode(const std::string& path) : videoPath(path)
{
    if (!videoPath.empty())
    {
        try
        {
            reader = std::make_unique<GUIReader>(videoPath);
            reader->iter();
            initialized = true;
        }
        catch (const std::exception& ex)
        {
            std::cerr << "[VideoReaderNode] Failed to open " << videoPath << ": "
                      << ex.what() << "\n";
        }
    }
}

//
// -- compute()
//
void VideoReaderNode::compute()
{
    if (!initialized || videoPath.empty())
    {
        tensorOutputs["frame"] = torch::Tensor();
        return;
    }
    if (isPlaying)
        updatePlayback();
}

void VideoReaderNode::updatePlayback()
{
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - lastFrameTime).count();
    float fps = static_cast<float>(reader->properties.fps);

    if (dt >= (1.0f / fps) / playbackSpeed)
    {
        lastFrameTime = now;
        try
        {
            auto frame = reader->next();
            if (frame.defined() && frame.numel() > 0)
            {
                tensorOutputs["frame"] = frame;
                currentTimeSec += 1.0f / fps;
            }
        }
        catch (const std::exception& ex)
        {
            std::cerr << "[VideoReaderNode] next() error: " << ex.what() << "\n";
            tensorOutputs["frame"] = torch::Tensor();
        }
    }
}

//
// -- Playback controls
//
void VideoReaderNode::play()
{
    if (initialized)
    {
        isPlaying = true;
        lastFrameTime = std::chrono::steady_clock::now();
    }
}
void VideoReaderNode::pause()
{
    isPlaying = false;
}
void VideoReaderNode::rewind()
{
    if (!initialized)
        return;
    reader->reset();
    currentTimeSec = 0.0f;
    tensorOutputs["frame"] = reader->readFrame();
}
void VideoReaderNode::step()
{
    if (!initialized)
        return;
    try
    {
        auto frame = reader->next();
        if (frame.defined() && frame.numel() > 0)
        {
            tensorOutputs["frame"] = frame;
            currentTimeSec += 1.0f / reader->properties.fps;
        }
    }
    catch (...)
    {
        tensorOutputs["frame"] = torch::Tensor();
    }
}

#include <algorithm>
#include <filesystem>
#include <iostream>

static std::string normalisePath(std::string raw)
{
    /* 0) trim outer whitespace */
    auto ltrim = [](std::string& s)
    {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
            s.erase(s.begin());
    };
    auto rtrim = [](std::string& s)
    {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
            s.pop_back();
    };
    ltrim(raw);
    rtrim(raw);

    /* 1) strip paired quotes */
    auto isQuote = [](char c) { return c == '"' || c == '\''; };
    if (raw.size() >= 2 && isQuote(raw.front()) && isQuote(raw.back()))
    {
        raw.erase(raw.begin());
        raw.pop_back();
    }

#ifdef _WIN32
    std::replace(raw.begin(), raw.end(), '\\', '/'); // unify early
#endif

    /* 2) canonicalise or absolute */
    std::filesystem::path p;
    try
    {
        p = std::filesystem::weakly_canonical(raw); // may throw if nothing exists
    }
    catch (...)
    {
        p = std::filesystem::absolute(raw); // still gives full path
    }

    /* 3) back to string, flip slashes on Windows */
    std::string out = p.make_preferred().string();
#ifdef _WIN32
    std::replace(out.begin(), out.end(), '\\', '/');
#endif

    /* 4) add trailing '/' only for directories */
    const bool rawHadSlash = !raw.empty() && (raw.back() == '/' || raw.back() == '\\');
    const bool isExistingDir =
        std::filesystem::exists(p) && std::filesystem::is_directory(p);

    if (!p.has_extension() && (rawHadSlash || isExistingDir))
        if (out.empty() || out.back() != '/')
            out.push_back('/');

    return out;
}

void VideoReaderNode::seekToTime(float t)
{
    if (!initialized)
        return;
    reader->seek(t);
    tensorOutputs["frame"] = reader->readFrame();
    currentTimeSec = t;
}

float VideoReaderNode::getDuration() const
{
    return reader ? static_cast<float>(reader->properties.duration) : 0.0f;
}

//
// -- Params
//
// -- replacement body for VideoReaderNode::setParam ---------------------------
void VideoReaderNode::setParam(const std::string& name, std::any val)
{
    if (name == "path")
    {
        try
        {
            // 1) fetch and normalise
            auto raw = std::any_cast<std::string>(val);
            videoPath = normalisePath(raw);

            // 2) re-open
            reader = std::make_unique<GUIReader>(videoPath);
            reader->iter();

            // 3) reset state
            initialized = true;
            currentTimeSec = 0.0f;
            isPlaying = false;
            tensorOutputs.clear();
        }
        catch (const std::exception& ex)
        {
            std::cerr << "[VideoReaderNode] setParam(path) failed: " << ex.what()
                      << '\n';
            initialized = false;
        }
    }
}

std::vector<PortInfo> VideoReaderNode::inputs() const
{
    std::vector<PortInfo> ports;
    return ports;
}
std::vector<PortInfo> VideoReaderNode::outputs() const
{
    return {{"frame", PortInfo::Tensor}};
}
std::vector<ParamInfo> VideoReaderNode::params() const
{
    return {{"path", ParamInfo::String, std::any(videoPath)}};
}


//
// -- UI rendering
//
void VideoReaderNode::drawUI(const std::string& uid)
{
 

    // 1) Path & playback
    {
        std::string p = videoPath;
        if (ImGui::InputText(("Path##" + uid).c_str(), &p))
            setParam("path", p);
    }
    ImGui::SameLine();
    if (ImGui::Button((isPlaying ? ("Pause##" + uid) : ("Play##" + uid)).c_str()))
        isPlaying ? pause() : play();
    ImGui::SameLine();
    if (ImGui::Button(("Rewind##" + uid).c_str()))
        rewind();
    ImGui::SameLine();
    if (ImGui::Button(("Step##" + uid).c_str()))
        step();

    // 2) Seek slider
    float dur = getDuration();
    if (dur > 0.0f)
    {
        if (ImGui::SliderFloat(("Time##" + uid).c_str(), &currentTimeSec, 0.0f, dur))
            seekToTime(currentTimeSec);
        ImGui::Text("Duration: %.2fs", dur);
    }
    else
    {
        ImGui::Text("No video loaded");
    }

    ImGuiStyle& style = ImGui::GetStyle();
    float btnW = ImGui::CalcTextSize("×").x + style.FramePadding.x * 2;
    float spacing = style.ItemSpacing.x;

}
