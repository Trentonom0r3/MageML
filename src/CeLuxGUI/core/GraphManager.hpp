#pragma once
#include "CeluxGUI/core/Node.hpp"
#include <imgui.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <imgui-node-editor/imgui_node_editor.h>

class GraphManager;

static GraphManager* gCurrentGraphManager = nullptr;

namespace ed = ax::NodeEditor;
struct NodeUIState
{
    ImVec2 position{100, 100};
    ImVec2 size{100, 100}; // ← add this!
    bool open = true;
};


struct Connection
{
    std::shared_ptr<Node> src;
    std::string srcPort;
    std::shared_ptr<Node> dst;
    std::string dstPort;
};

class GraphManager
{
  public:
    GraphManager()
    {
        // Create a fresh NodeEditor context for *this* graph
        _ctx = ed::CreateEditor();
        gCurrentGraphManager = this;
    }

    ~GraphManager()
    {
        // Clean up when we're done
        ed::DestroyEditor(_ctx);
        gCurrentGraphManager = nullptr;
    }
    bool GraphManager::hasPath(std::shared_ptr<Node> start,
                               std::shared_ptr<Node> target) const;

    int _nextId = 1;
    ed::NodeId nextNodeId()
    {
        return ed::NodeId(_nextId++);
    }
    void addNode(std::shared_ptr<Node> node);
    void connect(std::shared_ptr<Node> src, const std::string& srcPort,
                 std::shared_ptr<Node> dst, const std::string& dstPort);
    void evaluate();
    void drawUI();
    void setSelected(const std::shared_ptr<Node>& node)
    {
        selectedNode = node;
    }
    std::shared_ptr<Node> getSelected() const
    {
        return selectedNode;
    }
    // right after your existing addNode:
    void addNodeAt(std::shared_ptr<Node> node, ImVec2 pos);

    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<Connection> connections;
    void runPipelineAsync();
    void pausePipeline();
    void stopPipeline();
    bool isPipelineRunning() const
    {
        return running_;
    }
    int getCurrentFrameIndex() const
    {
        return currentFrameIndex_;
    }
    int getTotalFrames() const
    {
        return totalFrames_;
    }
    // Use this to scope all ed:: calls
    ed::EditorContext* getEditorContext() const
    {
        return _ctx;
    }


    ed::EditorContext* _ctx = nullptr; // the node-editor context
    std::thread pipelineThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::mutex pipelineMutex_;
    std::condition_variable cv_;
    int currentFrameIndex_ = 0;
    int totalFrames_ = 0;
    std::shared_ptr<Node> selectedNode = nullptr;
    std::map<std::shared_ptr<Node>, NodeUIState> uiStates;

    // per-frame storage of pin positions
    std::map<std::pair<Node*, std::string>, ImVec2> pinPositions;

    // click-and-drag state
    struct DragState
    {
        std::shared_ptr<Node> node;
        std::string port;
        ImVec2 startPos;
        bool active = false;
    } drag;
};
