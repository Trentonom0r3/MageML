#include "CeluxGUI/core/GraphManager.hpp"
#include <imgui.h>
#include <imgui_internal.h> // for ImDrawList
#include <iostream>
#include <CeLuxGUI/nodes/VideoReaderNode.hpp>

void GraphManager::addNode(std::shared_ptr<Node> node)
{
    nodes.push_back(std::move(node));
}

void GraphManager::connect(std::shared_ptr<Node> src, const std::string& srcPort,
                           std::shared_ptr<Node> dst, const std::string& dstPort)
{
    try
    {

        // 1) record the connection
        connections.push_back({src, srcPort, dst, dstPort});
        std::cout << "[GraphManager] Connected " << src->typeName() << "." << srcPort
                  << " → " << dst->typeName() << "." << dstPort << "\n";

    }
    catch (const std::exception& e)
    {
        std::cerr << "[GraphManager] Error connecting nodes: " << e.what() << "\n";
    }
}


void GraphManager::evaluate()
{
    // 1) clear all tensor maps
    for (auto& n : nodes)
    {
        n->tensorInputs.clear();
        n->tensorOutputs.clear();
    }
    // 2) drive every node in insertion order
    for (auto& n : nodes)
    {
        // determine if this node should run:
        bool isSource =
            n->inputs().empty(); // nodes that declare no inputs drive themselves
        bool hasInput = false;
        for (auto& c : connections)
        {
            if (c.dst == n)
            {
                hasInput = true;
                break;
            }
        }
        if (!isSource && !hasInput)
            continue; // skip compute() until we’ve wired something in

        // gather its tensorInputs from connections, as before…
        for (auto& c : connections)
        {
            if (c.dst == n)
            {
                auto it = c.src->tensorOutputs.find(c.srcPort);
                if (it != c.src->tensorOutputs.end())
                    n->tensorInputs[c.dstPort] = it->second;
            }
        }
        n->compute();
    }
}
void GraphManager::addNodeAt(std::shared_ptr<Node> node, ImVec2 pos)
{
    // 1) add to list
    nodes.push_back(node);
    // 2) seed its UI‐state position
    uiStates[node].position = pos;
}

void GraphManager::drawUI()
{
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    std::vector<std::shared_ptr<Node>> toRemove;
    pinPositions.clear();

    // 1) Draw each node window & record pin positions
    for (auto& node : nodes)
    {
        auto& ui = uiStates[node];
        std::string uid = std::to_string((intptr_t)node.get());
        bool isPreview = (node->typeName() == "RenderPreview");
        bool isVideoReader = (node->typeName() == "VideoReader");

        // Constrain VideoReader height so it scrolls instead of expanding endlessly
        if (isVideoReader)
        {
            ImGui::SetNextWindowSizeConstraints(ImVec2(300, 0),      // min width 300px
                                                ImVec2(FLT_MAX, 400) // max height 400px
            );
        }

        // Choose window flags
        ImGuiWindowFlags flags = ImGuiWindowFlags_None;
        if (!isPreview && !isVideoReader)
            flags = ImGuiWindowFlags_AlwaysAutoResize;

        // First‐use position + size for preview windows
        ImGui::SetNextWindowPos(ui.position, ImGuiCond_FirstUseEver);
        if (isPreview)
            ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

        // Begin the node window
        if (!ImGui::Begin((node->typeName() + "##" + uid).c_str(), &ui.open, flags))
        {
            ImGui::End();
            continue;
        }

        // If closed by user, mark for removal
        if (!ui.open)
        {
            ImGui::End();
            toRemove.push_back(node);
            continue;
        }

        // 1a) Draw the node's internal UI
        node->drawUI(uid);

        // 1b) Click‐to‐select
        if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            setSelected(node);

        // 1c) Capture this window's geometry
        ui.position = ImGui::GetWindowPos();
        ImVec2 wpos = ui.position;
        ImVec2 wsize = ImGui::GetWindowSize();

        // 2) OUTPUT pins (green) on right edge
        auto& outs = node->outputs();
        for (int i = 0; i < (int)outs.size(); ++i)
        {
            float t = (i + 1) / float(outs.size() + 1);
            ImVec2 pin{wpos.x + wsize.x + 4.0f, wpos.y + wsize.y * t};
            dl->AddCircleFilled(pin, 5.0f, IM_COL32(100, 200, 100, 255));
            pinPositions[{node.get(), outs[i].name}] = pin;

            // start drag detection
            ImVec2 mp = ImGui::GetIO().MousePos;
            float dx = mp.x - pin.x, dy = mp.y - pin.y;
            if (!drag.active && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                dx * dx + dy * dy < 64.0f)
            {
                drag.node = node;
                drag.port = outs[i].name;
                drag.startPos = pin;
                drag.active = true;
                std::cout << "[GraphManager] Drag start: " << node->typeName() << "."
                          << drag.port << "\n";
            }
        }

        // 3) INPUT pins (red) on left edge
        auto& ins = node->inputs();
        if (isVideoReader)
        {
           
        }
        else
        {
            // evenly spaced fallback
            for (int i = 0; i < (int)ins.size(); ++i)
            {
                float t = (i + 1) / float(ins.size() + 1);
                ImVec2 pin{wpos.x - 4.0f, wpos.y + wsize.y * t};
                dl->AddCircleFilled(pin, 5.0f, IM_COL32(200, 100, 100, 255));
                pinPositions[{node.get(), ins[i].name}] = pin;
            }
        }

        ImGui::End();
    }

    // 4) Draw all established connections (yellow Beziers)
    for (auto& c : connections)
    {
        auto kS = std::make_pair(c.src.get(), c.srcPort);
        auto kD = std::make_pair(c.dst.get(), c.dstPort);
        auto itS = pinPositions.find(kS);
        auto itD = pinPositions.find(kD);
        if (itS != pinPositions.end() && itD != pinPositions.end())
        {
            ImVec2 p1 = itS->second, p2 = itD->second;
            ImVec2 m1{p1.x + 50, p1.y}, m2{p2.x - 50, p2.y};
            dl->AddBezierCubic(p1, m1, m2, p2, IM_COL32(200, 200, 100, 200), 2.0f);
        }
    }

    // 5) Global drop‐to‐connect on mouse release
    if (drag.active && ImGui::IsMouseReleased(ImGuiMouseButton_Left))
    {
        ImVec2 mp = ImGui::GetIO().MousePos;
        float bestD2 = FLT_MAX;
        Node* bestN = nullptr;
        std::string bestPort;

        for (auto& node : nodes)
        {
            for (auto& pi : node->inputs())
            {
                auto key = std::make_pair(node.get(), pi.name);
                auto it = pinPositions.find(key);
                if (it == pinPositions.end())
                    continue;
                ImVec2 pin = it->second;
                float dx = mp.x - pin.x, dy = mp.y - pin.y;
                float d2 = dx * dx + dy * dy;
                if (d2 < bestD2)
                {
                    bestD2 = d2;
                    bestN = node.get();
                    bestPort = pi.name;
                }
            }
        }
        if (bestN && bestD2 < 256.0f) // within 16px
        {
            std::shared_ptr<Node> dstPtr;
            for (auto& n : nodes)
                if (n.get() == bestN)
                {
                    dstPtr = n;
                    break;
                }
            connect(drag.node, drag.port, dstPtr, bestPort);
        }
        drag.active = false;
    }

    // 6) Draw rubber‐band line while dragging
    if (drag.active)
    {
        ImVec2 mp = ImGui::GetIO().MousePos;
        dl->AddBezierCubic(drag.startPos, ImVec2(drag.startPos.x + 50, drag.startPos.y),
                           ImVec2(mp.x - 50, mp.y), mp, IM_COL32(200, 200, 100, 255),
                           2.5f);
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) ||
            ImGui::IsKeyPressed(ImGuiKey_Escape))
        {
            drag.active = false;
        }
    }

    // 7) Remove any nodes the user closed (and their connections)
    for (auto& rem : toRemove)
    {
        connections.erase(std::remove_if(connections.begin(), connections.end(),
                                         [&](auto& c)
                                         { return c.src == rem || c.dst == rem; }),
                          connections.end());
        uiStates.erase(rem);
        nodes.erase(std::remove(nodes.begin(), nodes.end(), rem), nodes.end());
        std::cout << "[GraphManager] Removed node " << rem->typeName() << "\n";
    }
}

