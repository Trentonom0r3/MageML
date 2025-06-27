#pragma once
#include "CeluxGUI/core/Node.hpp"
#include <imgui.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct NodeUIState
{
    ImVec2 position{100, 100};
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

  private:
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
