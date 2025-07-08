#include "CeluxGUI/core/GraphManager.hpp"
#include <imgui.h>
#include <imgui_internal.h> // for ImDrawList
#include <iostream>
#include <CeLuxGUI/nodes/VideoPipeline.hpp>

// Fix for E0349: no operator "+" matches these operands  
// The issue arises because `ImVec2` does not have a defined operator+ for addition.  
// We need to explicitly define the addition operation for `ImVec2`.  
// Fix for E0349: no operator "+" matches these operands  
// The issue arises because `ImVec2` does not have a defined operator+ for addition.  
// We need to explicitly define the addition operation for `ImVec2`.  

inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) {  
    return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y);  
}  

void GraphManager::addNode(std::shared_ptr<Node> node)
{
    nodes.push_back(std::move(node));
}

void GraphManager::connect(std::shared_ptr<Node> src, const std::string& srcPort,
                           std::shared_ptr<Node> dst, const std::string& dstPort)
{
    try
    {
        // Remove any duplicate connection that already exists!
        connections.erase(std::remove_if(connections.begin(), connections.end(),
                                         [&](const Connection& c)
                                         {
                                             return c.src == src &&
                                                    c.srcPort == srcPort &&
                                                    c.dst == dst &&
                                                    c.dstPort == dstPort;
                                         }),
                          connections.end());

        // Now record the new connection
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
        std::cout << "[GraphManager] Evaluating node: " << n->typeName() << "\n";
        n->compute();
    }
}

void GraphManager::addNodeAt(std::shared_ptr<Node> node, ImVec2 screenPos)
{
    // 1) Add to our master list
    nodes.push_back(node);

    // 2) Figure out its canvas position
    ImVec2 canvasPos = ed::ScreenToCanvas(screenPos);
    ed::SetNodePosition(node->getId(), canvasPos);

    // 3) Initialise its UI state (with a sensible default size!)
    NodeUIState st;
    st.position = canvasPos;
    st.size = ImVec2(200.0f, 100.0f); // ← default starting size
    st.open = true;
    uiStates[node] = st;
}



bool GraphManager::hasPath(std::shared_ptr<Node> start,
                           std::shared_ptr<Node> target) const
{
    // Fast‐path: same node
    if (start == target)
        return true;

    // Track which raw Node* we’ve already visited
    std::unordered_set<Node*> visited;

    // Define a recursive lambda
    std::function<bool(std::shared_ptr<Node>)> dfs =
        [&](std::shared_ptr<Node> cur) -> bool
    {
        // If we reached the target, we have a path
        if (cur == target)
            return true;

        // Mark this node as visited
        visited.insert(cur.get());

        // Explore every outgoing connection
        for (auto& c : connections)
        {
            if (c.src == cur)
            {
                Node* nextRaw = c.dst.get();
                // Skip if we’ve already been here
                if (visited.count(nextRaw))
                    continue;

                // Recurse
                if (dfs(c.dst))
                    return true;
            }        }

        return false;
    };

    // Kick off the search
    return dfs(start);
}

void GraphManager::drawUI()
{
    // 1) Draw & size each node
    // 1) Draw all nodes first (this registers all pins, etc)
    for (auto& n : nodes)
    {
        n->draw();

    }


    // ─── 1a) Show right‐click menus ─────────────────────────────────────────
    static ed::NodeId nodeCtx = 0;
    static ed::PinId pinCtx = 0;
    static ed::LinkId linkCtx = 0;

    ed::Suspend(); // suspend the editor so ImGui popups work properly

    if (ed::ShowNodeContextMenu(&nodeCtx))
        ImGui::OpenPopup("NodeContextMenu");
    if (ed::ShowPinContextMenu(&pinCtx))
        ImGui::OpenPopup("PinContextMenu");
    if (ed::ShowLinkContextMenu(&linkCtx))
        ImGui::OpenPopup("LinkContextMenu");
    if (ed::ShowBackgroundContextMenu())
        ImGui::OpenPopup("BackgroundMenu");

    ed::Resume();

    // ─── 1b) Build the actual ImGui popups ───────────────────────────────────
    ed::Suspend();

    // Node menu
    if (ImGui::BeginPopup("NodeContextMenu"))
    {
        if (ImGui::MenuItem("Delete Node"))
        {
            nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                                       [&](auto& n) { return n->getId() == nodeCtx; }),
                        nodes.end());
        }
        ImGui::EndPopup();
    }

    // Pin menu
    if (ImGui::BeginPopup("PinContextMenu"))
    {
        if (ImGui::MenuItem("Disconnect All"))
        {
            connections.erase(
                std::remove_if(connections.begin(), connections.end(),
                               [&](auto& c)
                               {
                                   return c.src->pinId(c.srcPort) == pinCtx ||
                                          c.dst->pinId(c.dstPort) == pinCtx;
                               }),
                connections.end());
        }
        ImGui::EndPopup();
    }

    // Link menu
    if (ImGui::BeginPopup("LinkContextMenu"))
    {
        if (ImGui::MenuItem("Delete Link"))
        {
            connections.erase(
                std::remove_if(connections.begin(), connections.end(),
                               [&](auto& c)
                               {
                                   size_t h = std::hash<size_t>{}(
                                       reinterpret_cast<size_t>(c.src.get()) ^
                                       reinterpret_cast<size_t>(c.dst.get()) ^
                                       std::hash<std::string>{}(c.srcPort + "|" +
                                                                c.dstPort));
                                   return static_cast<ed::LinkId>(h) == linkCtx;
                               }),
                connections.end());
        }
        ImGui::EndPopup();
    }

    // Background menu
    if (ImGui::BeginPopup("BackgroundMenu"))
    {
        if (ImGui::MenuItem("Cancel Link Drag"))
            ImGui::ClearActiveID();
        ImGui::EndPopup();
    }

    ed::Resume();
    // ────────────────────────────────────────────────────────────────────────

    // 2) Link‐creation pass with green/red preview
    ed::BeginCreate();
    {
        struct Match
        {
            std::shared_ptr<Node> node;
            std::string port;
            PortInfo::Type type;
            bool isInput;
            bool isOutput;
        };

        auto findPin = [&](ed::PinId pid) -> Match
        {
            for (auto& n : nodes)
            {
                for (auto& p : n->inputs())
                    if (n->pinId(p.name) == pid)
                        return {n, p.name, p.type, true, false};
                for (auto& p : n->outputs())
                    if (n->pinId(p.name) == pid)
                        return {n, p.name, p.type, false, true};
            }
            return {nullptr, "", PortInfo::Tensor, false, false};
        };

        ed::PinId startPin = 0, endPin = 0;
        if (ed::QueryNewLink(&startPin, &endPin))
        {
            // Determine which is output vs input
            auto mA = findPin(startPin), mB = findPin(endPin);
            Match out = mA.isOutput ? mA : (mB.isOutput ? mB : Match{});
            Match in = mA.isInput ? mA : (mB.isInput ? mB : Match{});

            // Check validity
            bool valid = mA.node && mB.node &&         // both ends exist
                         out.node != in.node &&        // no self‐connect
                         out.isOutput && in.isInput && // correct directions
                         out.type == in.type &&        // same data type
                         !hasPath(in.node, out.node);  // no cycle

            // Colors for preview
            ImColor validCol(0, 200, 0, 255);
            ImColor invalidCol(200, 0, 0, 255);
            float previewThick = 3.0f;

            if (valid)
            {
                // Draw preview in green; returns true on mouse‐release over a valid
                // spot
                if (ed::AcceptNewItem(validCol, previewThick))
                {
                    // On release: tear off old input link
                    connections.erase(
                        std::remove_if(
                            connections.begin(), connections.end(), [&](auto& c)
                            { return c.dst == in.node && c.dstPort == in.port; }),
                        connections.end());

                    // Guard against duplicates
                    connections.erase(std::remove_if(connections.begin(),
                                                     connections.end(),
                                                     [&](auto& c)
                                                     {
                                                         return c.src == out.node &&
                                                                c.srcPort == out.port &&
                                                                c.dst == in.node &&
                                                                c.dstPort == in.port;
                                                     }),
                                      connections.end());

                    // Record the new link
                    connect(out.node, out.port, in.node, in.port);
                }
            }
            else
            {
                // Draw preview in red (always reject on invalid)
                ed::RejectNewItem(invalidCol, previewThick);
            }
        }
    }
    ed::EndCreate();

    // 3) Link‐deletion pass
    if (ed::BeginDelete())
    {
        ed::LinkId lid;
        if (ed::QueryDeletedLink(&lid))
        {
            connections.erase(
                std::remove_if(connections.begin(), connections.end(),
                               [&](auto& c)
                               {
                                   size_t h = std::hash<size_t>{}(
                                       reinterpret_cast<size_t>(c.src.get()) ^
                                       reinterpret_cast<size_t>(c.dst.get()) ^
                                       std::hash<std::string>{}(c.srcPort + "|" +
                                                                c.dstPort));
                                   return static_cast<ed::LinkId>(h) == lid;
                               }),
                connections.end());
            ed::AcceptDeletedItem();
        }
    }
    ed::EndDelete();

    // 4) Draw all surviving links (static wires)
    for (auto& c : connections)
    {
        if (!c.src || !c.dst)
            continue;

        size_t h =
            std::hash<size_t>{}(reinterpret_cast<size_t>(c.src.get()) ^
                                reinterpret_cast<size_t>(c.dst.get()) ^
                                std::hash<std::string>{}(c.srcPort + "|" + c.dstPort));
        ed::Link(static_cast<ed::LinkId>(h), c.src->pinId(c.srcPort),
                 c.dst->pinId(c.dstPort));
    }
}



void GraphManager::runPipelineAsync()
{

    if (running_)
    {

        return;
    }
    running_ = true;
    paused_ = false;
    currentFrameIndex_ = 0;

    // Get totalFrames from VideoReaderNode
    for (auto& n : nodes)
    {

        if (n->typeName() == "VideoReader")
        {
            auto vrNode = std::dynamic_pointer_cast<VideoReaderNode>(n);
            if (vrNode && vrNode->getPipeline())
            {
                totalFrames_ = int(vrNode->getPipeline()->totalFrames());

                break;
            }
        }
    }
    // Seek to 0
    for (auto& n : nodes)
    {

        if (n->typeName() == "VideoReader")
        {
            auto vrNode = std::dynamic_pointer_cast<VideoReaderNode>(n);
            if (vrNode && vrNode->getPipeline())
                vrNode->getPipeline()->seek(0.0);

        }
    }

    pipelineThread_ = std::thread(
        [this]()
        {
            while (running_ && currentFrameIndex_ < totalFrames_)
            {
                {
                    std::unique_lock<std::mutex> lk(pipelineMutex_);
                    if (paused_)
                    {


                    }
                    cv_.wait(lk, [this]() { return !paused_ || !running_; });
                }
                if (!running_)
                {

                    break;
                }
              //  this->evaluate();
              //  currentFrameIndex_++;
               

            }
            running_ = false;
        });

}


void GraphManager::pausePipeline()
{
    std::cout << "[GraphManager] Pipeline paused at frame " << currentFrameIndex_
              << "\n";
    paused_ = !paused_;
    if (!paused_)
        cv_.notify_all(); // Resume
}

void GraphManager::stopPipeline()
{
    std::cout << "[GraphManager] Stopping pipeline\n";
    running_ = false;
    paused_ = false;
    cv_.notify_all();
    if (pipelineThread_.joinable())
        pipelineThread_.join();
}
