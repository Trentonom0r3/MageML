#pragma once
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include <any>
#include <algorithm>
#include <cstdint> // For fixed-width integer types
#include <exception>
#include <ostream>   // For std::ostream
#include <stdexcept> // For std::runtime_error
#include <Cache.hpp>
#include <CeLuxGUI/nodes/VideoPipeline.hpp> // Assuming this is the header for your VideoReader class
#include <imgui_stdlib.h>

#include <imgui-node-editor/imgui_node_editor.h>

namespace ed = ax::NodeEditor;

struct PortInfo
{
    std::string name;
    bool use_in_header = false; // If true, this port is displayed in the header row
    enum Type
    {
        Tensor,
        Float,
        Int /*…*/
    } type;
};

struct ParamInfo
{
    std::string name;
    enum Type
    {
        Float,
        Int,
        Bool,
        String,
        Double,
        Group, // NEW: Group of sub-params, useful for permute/clamp/etc.
    } type;

    std::any defaultValue;

    /// For Group: key = label, value = ParamInfo
    std::unordered_map<std::string, ParamInfo> children;

    // Optional metadata
    float min = 0.0f;
    float max = 1.0f;
    bool useSlider = false;
    std::string tooltip;
};


// Base class for all nodes
class Node : public std::enable_shared_from_this<Node>
{
  public:
    Node(ed::NodeId id) : _nodeId(id)
    {
    }

    virtual ~Node() = default;

    // Unique ID for this node
    ed::NodeId getId() const
    {
        return _nodeId;
    }


    virtual void Node::draw()
    {
        // ─── Push node‐level styling ───────────────────────────────────────────────
        ed::PushStyleColor(ed::StyleColor_NodeBg, ImColor(45, 45, 48, 220));
        ed::PushStyleColor(ed::StyleColor_NodeBorder, ImColor(62, 62, 66, 200));
        ed::PushStyleVar(ed::StyleVar_NodeRounding, 4.0f);
        ed::PushStyleVar(ed::StyleVar_NodePadding, ImVec4(8, 8, 8, 8));

        ed::BeginNode(_nodeId);

        // ─── Temporarily tighten spacing for header row ────────────────────────
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 2));

        auto inputs = this->inputs();
        auto outputs = this->outputs();

        // 1) Draw inputs on the left
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const auto& port = inputs[i];
            ed::BeginPin(pinId(port.name), ed::PinKind::Input);
            // circle style
            auto drawList = ImGui::GetWindowDrawList();
            auto center = ImGui::GetCursorScreenPos();
            center.x += 6;
            center.y += ImGui::GetTextLineHeight() * 0.5f;
            float r = 6.0f;
            drawList->AddCircleFilled(center, r, IM_COL32(200, 100, 100, 255), 12);
            drawList->AddCircle(center, r, IM_COL32(40, 40, 40, 255), 12, 1.5f);
            ImGui::Dummy(ImVec2(r * 2, ImGui::GetTextLineHeight()));
            ed::EndPin();

            // same line if more to come
            if (i + 1 < inputs.size())
                ImGui::SameLine();
        }

        // 2) Title in the center
        if (!inputs.empty())
            ImGui::SameLine();

        ImGui::SetWindowFontScale(1.2f);
        ImGui::TextUnformatted(typeName().c_str());
        ImGui::SetWindowFontScale(1.0f);

        // 3) Draw outputs on the right
        if (!outputs.empty())
            ImGui::SameLine();

        for (size_t i = 0; i < outputs.size(); ++i)
        {
            const auto& port = outputs[i];
            ed::BeginPin(pinId(port.name), ed::PinKind::Output);
            // square style
            auto drawList = ImGui::GetWindowDrawList();
            auto center = ImGui::GetCursorScreenPos();
            center.x += 6;
            center.y += ImGui::GetTextLineHeight() * 0.5f;
            float r = 6.0f;
            drawList->AddRectFilled(ImVec2(center.x - r, center.y - r),
                                    ImVec2(center.x + r, center.y + r),
                                    IM_COL32(100, 200, 100, 255), 2.0f);
            drawList->AddRect(ImVec2(center.x - r, center.y - r),
                              ImVec2(center.x + r, center.y + r),
                              IM_COL32(40, 40, 40, 255), 2.0f, 0, 1.5f);
            ImGui::Dummy(ImVec2(r * 2, ImGui::GetTextLineHeight()));
            ed::EndPin();

            if (i + 1 < outputs.size())
                ImGui::SameLine();
        }

        // ─── Restore spacing & move below header ────────────────────────────────
        ImGui::PopStyleVar(); // ImGuiStyleVar_ItemSpacing
        ImGui::NewLine();


        this->drawExtraUI();

        // ─── Pop node styling ─────────────────────────────────────────────────────
        ed::EndNode();
        ed::PopStyleVar(2);   // NodePadding, NodeRounding
        ed::PopStyleColor(2); // NodeBg, NodeBorder
    }


    // Node-specific compute logic
    virtual void compute() = 0;

    // Node metadata
    virtual std::string typeName() const = 0;
    virtual std::vector<PortInfo> inputs() const = 0;
    virtual std::vector<PortInfo> outputs() const = 0;
    virtual std::vector<ParamInfo> params() const = 0;
    virtual void setParam(const std::string& name, std::any value) = 0;
    std::map<std::string, torch::Tensor> tensorInputs, tensorOutputs;


   

    // Optional extra UI in derived nodes
    virtual bool drawExtraUI()
    {
        return false;
    }

    // Generate a stable PinId per node/port name
    ed::PinId pinId(const std::string& name) const
    {
        // Combine hash of name with pointer
        void* ptr = getId().AsPointer();
        size_t h = std::hash<std::string>{}(name);
        h ^= reinterpret_cast<size_t>(ptr);
        return ed::PinId(h);
    }

  private:
    ed::NodeId _nodeId;
};


#include <functional>

using NodeConstructor = std::function<std::shared_ptr<Node>(ed::NodeId)>;

class NodeFactory
{
  public:
    static NodeFactory& instance()
    {
        static NodeFactory f;
        return f;
    }

    void registerType(std::string typeName, NodeConstructor ctor)
    {
        ctors[typeName] = ctor;
    }

    std::vector<std::string> availableTypes() const
    {
        std::vector<std::string> names;
        for (auto& kv : ctors)
            names.push_back(kv.first);
        return names;
    }

    std::shared_ptr<Node> create(const std::string& typeName, ed::NodeId uuid) const
    {
        auto it = ctors.find(typeName);
        if (it == ctors.end())
            return nullptr;
        return it->second(uuid); // Pass uuid to the registered lambda
    }


  private:
    std::map<std::string, std::function<std::shared_ptr<Node>(ed::NodeId)>> ctors;
};

class VideoReaderNode : public Node
{
  public:
    VideoReaderNode(ed::NodeId id) : Node(id) {};
    ~VideoReaderNode() = default;

    std::string typeName() const override
    {
        return "VideoReader";
    }

    // Node interface
    // VideoReaderNode.cpp

    void compute() override
    {
        try
        {
            std::cout << "VideoReaderNode compute called" << std::endl;
            if (!pipeline_)
            {
                tensorOutputs["frame"] = torch::Tensor();
                return;
            }

                // On "play", advance the pipeline one frame (decode next)
                auto result = pipeline_->getNextFrame();
                if (result->timestamp == -1)
                {
                    std::cout << "End of file" << std::endl;
                    // Optionally set frame to empty at EOS
                    tensorOutputs["frame"] = torch::Tensor();
                    return;
                }
                // If we got a valid frame, update the output tensor
                tensorOutputs["frame"] = result->tensor;
               
                std::cout << "Updated Valid Frame" << std::endl;
                PrintTensorDebugInfo(tensorOutputs["frame"]);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error during compute: " << e.what() << std::endl;
            tensorOutputs["frame"] = torch::Tensor(); // Emit an empty tensor on error
        }
    }


    std::vector<PortInfo> inputs() const override
    {
        return {};
    }

    std::vector<PortInfo> outputs() const override
    {
        return {{"frame", PortInfo::Tensor}};
    }
    std::vector<ParamInfo> params() const override
    {
        return {{"path", ParamInfo::String, path_}};
    }

    void setParam(const std::string& name, std::any val) override
    {
        if (name == "path")
        {   
         
            std::string newPath = std::any_cast<std::string>(val);
            path_ = newPath;
            openVideoPath(); // Always reload, even if the path didn't change
        }
    }

    void openVideoPath()
    {

        pipeline_ = std::make_shared<VideoPipeline>(path_);
        if (pipeline_)
        {
           
            pipeline_->initialize();
        }
        else
        {
            std::cerr << "Failed to initialize video pipeline." << std::endl;
        }
    }



    std::shared_ptr<VideoPipeline> getPipeline() const
    {
        return pipeline_;
    }

  bool drawExtraUI() override
    {
        // Path entry
        ImGui::InputText("Path", &path_);
        ImGui::SameLine();
        if (ImGui::Button("Load Video"))
        {
            openVideoPath();
            lastTime_ = 0.0;
            tensorOutputs["frame"] = torch::Tensor();
        }
        if (!pipeline_)
            return false; // No pipeline, nothing to draw

        double dur = pipeline_->duration();
        lastTime_ = std::clamp(lastTime_, 0.0, dur);
        ImGui::Text("Position: %.3f / %.3f s", lastTime_, dur);
        float t = (float)lastTime_, tMax = (float)dur;
        if (ImGui::SliderFloat("Scrub", &t, 0.0f, tMax, "%.3fs"))
        {
            lastTime_ = (double)t;
            pipeline_->seek(lastTime_);
            tensorOutputs["frame"] = pipeline_->getCurrentPacket()->tensor;
        }
        return true; // Indicate we drew something extra
    }


  private:
    std::shared_ptr<VideoPipeline> pipeline_;
    std::string path_;
    double lastTime_ = -1.0;
};
