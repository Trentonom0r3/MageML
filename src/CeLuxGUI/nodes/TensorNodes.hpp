#pragma once

#include "CeluxGUI/core/Node.hpp" // adjust include path as needed
#include <algorithm>
#include <imgui.h>
#include <iostream>
#include <torch/torch.h>
#include <vector>
//------------------------------------------------------------------------------
// helper: move tensor to `dst` only if devices differ (async if pinned)
//------------------------------------------------------------------------------
inline torch::Tensor maybe_to(torch::Tensor t, torch::Device dst,
                              bool non_blocking = true)
{
    try
    {
        if (!t.defined() || t.device() == dst)
            return t;
#if TORCH_VERSION_MAJOR >= 2
        torch::TensorOptions opts =
            torch::TensorOptions().dtype(t.scalar_type()).device(dst);
        return t.to(opts, non_blocking, /*copy=*/false, t.suggest_memory_format());
#else
        return t.to(dst, t.scalar_type(), non_blocking, /*copy=*/false);
#endif
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[maybe_to] " << ex.what() << '\n';
        return torch::Tensor();
    }
}

//==============================================================================
// UnaryTensorNode — base: 1 tensor IN  → 1 tensor OUT
//==============================================================================
class UnaryTensorNode : public Node
{
  protected:
    torch::Device dstDev_{torch::kCPU};
    std::string kind_;
    bool dirty_ = true; // set when UI params change

  public:
    explicit UnaryTensorNode(std::string kind, ed::NodeId id) : Node(id), kind_(std::move(kind))
    {
    }

    std::vector<PortInfo> inputs() const override
    {
        return {{"in", PortInfo::Tensor}};
    }
    std::vector<PortInfo> outputs() const override
    {
        return {{"frame", PortInfo::Tensor}};
    }

    std::vector<ParamInfo> params() const override
    {
        return {};
    }
    void setParam(const std::string&, std::any) override
    {
    }
    std::string typeName() const override
    {
        return kind_;
    }

torch::Tensor inTensor()
{
    auto it = tensorInputs.find("in");
    return it == tensorInputs.end() ? torch::Tensor() : maybe_to(it->second, dstDev_);
}

void publish(torch::Tensor t)
{
    tensorOutputs["frame"] = maybe_to(std::move(t), torch::kCPU);
    std::cout << '[' << kind_ << "] compute] published tensor with shape: "
              << tensorOutputs["frame"].sizes() << '\n';
    dirty_ = false;
}
}
;

//------------------------------------------------------------------------------
#define SAFE_COMPUTE(body, name)                                                       \
    try                                                                                \
    {                                                                                  \
        body                                                                           \
    }                                                                                  \
    catch (const std::exception& ex)                                                   \
    {                                                                                  \
        std::cerr << '[' << name << "] compute] " << ex.what() << '\n';                \
        publish(torch::Tensor());                                                      \
    }

//==============================================================================
// PermuteNode — unique‑dim permutation for 2‑4D tensors
//==============================================================================
class PermuteNode final : public UnaryTensorNode
{
    std::array<int64_t, 4> order_{0, 1, 2, 3};

  public:
    PermuteNode(ed::NodeId id) : UnaryTensorNode("Permute", id)
    {
        // Set default param values
        setParam("O0", static_cast<int>(order_[0]));
        setParam("O1", static_cast<int>(order_[1]));
        setParam("O2", static_cast<int>(order_[2]));
        setParam("O3", static_cast<int>(order_[3]));
    }

    /// Expose each dimension as an editable int param
    std::vector<ParamInfo> params() const override
    {
        return {
            {"O0", ParamInfo::Int, static_cast<int>(order_[0])},
            {"O1", ParamInfo::Int, static_cast<int>(order_[1])},
            {"O2", ParamInfo::Int, static_cast<int>(order_[2])},
            {"O3", ParamInfo::Int, static_cast<int>(order_[3])},
        };
    }

    /// Update internal state when a param is edited
    void setParam(const std::string& name, std::any value) override
    {
        if (name == "O0")
            order_[0] = std::any_cast<int>(value);
        if (name == "O1")
            order_[1] = std::any_cast<int>(value);
        if (name == "O2")
            order_[2] = std::any_cast<int>(value);
        if (name == "O3")
            order_[3] = std::any_cast<int>(value);
    }

    void compute() override
    {
        SAFE_COMPUTE(
            {
                auto x = inTensor();
                if (!x.defined())
                {
                    publish({});
                    return;
                }

                const int64_t rank = x.dim();
                if (rank < 2 || rank > 4)
                {
                    std::cerr << "[PermuteNode] rank " << rank << " unsupported\n";
                    publish({});
                    return;
                }

                // Build unique permutation vector
                std::vector<int64_t> used;
                used.reserve(rank);
                for (size_t i = 0; i < static_cast<size_t>(rank); ++i)
                {
                    int64_t idx = std::clamp<int64_t>(order_[i], 0, rank - 1);
                    if (std::find(used.begin(), used.end(), idx) == used.end())
                        used.push_back(idx);
                }
                for (int64_t d = 0; d < rank; ++d)
                    if (std::find(used.begin(), used.end(), d) == used.end())
                        used.push_back(d);

                if (static_cast<int64_t>(used.size()) != rank)
                {
                    std::cerr << "[PermuteNode] internal error building permutation\n";
                    publish({});
                    return;
                }

                publish(x.permute(used).contiguous());
            },
            "PermuteNode");
    }

bool drawExtraUI() override
    {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        auto x = inTensor();

        if (!x.defined())
        {
            ImGui::Text("No input tensor defined");
            return false;
        }

        int rank = int(x.dim());
        auto sizes = x.sizes();
        static const char* names[] = {"N", "C", "H", "W"};

        // Explicitly sized ImGui table
        ImVec2 tableSize = ImVec2(160, (rank + 1) * 30.0f);

        if (ImGui::BeginTable("permuteTable", 2,
                              ImGuiTableFlags_BordersInnerV |
                                  ImGuiTableFlags_SizingFixedFit,
                              tableSize))
        {
            ImGui::TableSetupColumn("Dims", ImGuiTableColumnFlags_WidthFixed, 50.0f);
            ImGui::TableSetupColumn("Shape", ImGuiTableColumnFlags_WidthFixed, 90.0f);
            ImGui::TableHeadersRow();

            for (int i = 0; i < rank; ++i)
            {
                ImGui::TableNextRow(ImGuiTableRowFlags_None, 26.0f);

                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(i);

                // Make each button a drag source AND drop target
                ImGui::Button(names[order_[i]], ImVec2(40, 20));

                // Drag source
                if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None))
                {
                    ImGui::SetDragDropPayload("DND_PERMUTE_DIM", &i, sizeof(int));
                    ImGui::Text("Move %s", names[order_[i]]);
                    ImGui::EndDragDropSource();
                }

                // Drop target
                if (ImGui::BeginDragDropTarget())
                {
                    if (const ImGuiPayload* payload =
                            ImGui::AcceptDragDropPayload("DND_PERMUTE_DIM"))
                    {
                        int src = *(const int*)payload->Data;
                        std::swap(order_[src], order_[i]);
                        // No setParam here!
                        // Just update order_ directly.
                        std::cout << "[PermuteNode] Swapped: ";
                        for (auto v : order_)
                            std::cout << v << " ";
                        std::cout << std::endl;
                    }


                    ImGui::EndDragDropTarget();
                }

                ImGui::PopID();

                // Show dimension size
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%lld", (long long)sizes[order_[i]]);
            }

            ImGui::EndTable();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        return true;
    }


};

//==============================================================================
// TypeCastNode
//==============================================================================
class TypeCastNode final : public UnaryTensorNode
{
    int idx_ = 0;
    bool combo_open_ = false; // Track if our fake combo is expanded

    static constexpr struct
    {
        const char* lab;
        torch::ScalarType tp;
    } opts[] = {{"F32", torch::kFloat32},
                {"F16", torch::kFloat16},
                {"U8", torch::kUInt8},
                {"I32", torch::kInt32}};

  public:
    TypeCastNode(ed::NodeId id) : UnaryTensorNode("TypeCast", id)
    {
    }

    bool drawExtraUI() override
    {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextDisabled("[TypeCastNode UI]");

        auto ptr = reinterpret_cast<intptr_t>(getId().AsPointer());
        std::string comboId = "TypeFakeCombo##" + std::to_string(ptr);
        const char* labels[] = {"F32", "F16", "U8", "I32"};

        // Render a button that "opens" our fake dropdown
        ImVec2 btnSize = ImVec2(110, 0);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(64, 90, 180, 200));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(90, 120, 240, 220));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(44, 60, 120, 240));

        bool changed = false;
        // Draw the fake combo "header" (current selection)
        if (ImGui::Button((std::string(labels[idx_]) + "  \xE2\x96\xBC").c_str(),
                          btnSize))
        {
            combo_open_ = !combo_open_; // Toggle open/close
        }
        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar();

        // Draw the "dropdown" if open
        if (combo_open_)
        {
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(48, 60, 130, 220));
            ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(120, 130, 200, 150));
            ImGui::BeginChild((comboId + "Menu").c_str(), ImVec2(btnSize.x, 100), true,
                              ImGuiChildFlags_FrameStyle |
                                  ImGuiWindowFlags_NoScrollbar);
            for (int i = 0; i < IM_ARRAYSIZE(labels); ++i)
            {
                // Highlight selection
                if (ImGui::Selectable(labels[i], idx_ == i, ImGuiSelectableFlags_None,
                                      ImVec2(btnSize.x - 8, 0)))
                {
                    idx_ = i;
                    combo_open_ = false;
                    changed = true;
                }
            }
            ImGui::EndChild();
            ImGui::PopStyleColor(2);
            ImGui::PopStyleVar();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Optional: Close dropdown if click outside (primitive hit test)
        if (combo_open_ && !ImGui::IsItemHovered(ImGuiHoveredFlags_AnyWindow) &&
            ImGui::IsMouseClicked(0))
        {
            combo_open_ = false;
        }

        return changed;
    }

    void compute() override
    {
        SAFE_COMPUTE(
            {
                auto x = inTensor();
                publish(x.defined() ? x.to(opts[idx_].tp, true, false)
                                    : torch::Tensor());
            },
            "TypeCastNode");
    }
};


//==============================================================================
// ClampNode
//==============================================================================
class ClampNode final : public UnaryTensorNode
{
    float lo_ = 0.f, hi_ = 1.f;

  public:
    ClampNode(ed::NodeId id) : UnaryTensorNode("Clamp", id)
    {
    }
    bool drawExtraUI() override
    {
        auto ptr = reinterpret_cast<intptr_t>(getId().AsPointer());
        return ImGui::DragFloatRange2(("Range##" + std::to_string(ptr)).c_str(), &lo_,
                                      &hi_, 0.001f, -10.f, 10.f);
    }

    bool changed = false;
    void compute() override
    {
        SAFE_COMPUTE(
            {
                std::cout << "[ClampNode] compute] lo: " << lo_ << ", hi: " << hi_
                          << '\n';
                auto x = inTensor();
                publish(x.defined() ? x.clamp(lo_, hi_) : torch::Tensor());
            },
            "ClampNode");
    }
};

class RingBufferNode : public Node
{
  public:
    RingBufferNode(ed::NodeId id = 0) : Node(id), maxSize_(0)
    {
        buffer_.resize(0);
    }

    std::string typeName() const override
    {
        return "RingBuffer";
    }

    void compute() override
    {
        // 1. Get input frame
        if (tensorInputs.count("input") == 0 || !tensorInputs["input"].defined())
            return;

        auto newFrame = tensorInputs["input"];

        // 2. Add to buffer
        buffer_.push_back(newFrame);
        if (buffer_.size() > maxSize_)
            buffer_.pop_front();

        // 3. Set outputs
        // Output 0: most recent (current)
        for (size_t i = 0; i < buffer_.size(); ++i)
        {
            tensorOutputs["frame_" + std::to_string(i)] = buffer_[buffer_.size() - 1 - i];
            // out0: newest, out1: prev, out2: 2nd prev, etc
        }
    }

    std::vector<PortInfo> inputs() const override
    {
        return {{"input", PortInfo::Tensor}};
    }
    std::vector<PortInfo> outputs() const override
    {
        std::vector<PortInfo> outs;
        for (size_t i = 0; i < maxSize_; ++i)
            outs.push_back({"frame_" + std::to_string(i), PortInfo::Tensor});
        return outs;
    }
    std::vector<ParamInfo> params() const override
    {
        return {};
    }
    void setParam(const std::string& name, std::any val) override
    {
        if (name == "size")
        {
            int s = std::any_cast<int>(val);
            if (s > 1 && s <= 32)
                maxSize_ = s;
            while (buffer_.size() > maxSize_)
                buffer_.pop_front();
        }
    }
    bool drawExtraUI() override
    {
        auto ptr = reinterpret_cast<intptr_t>(getId().AsPointer());
        // allow resizing the buffer
        return ImGui::InputInt(("Size##" + std::to_string(ptr)).c_str(),
                               reinterpret_cast<int*>(&maxSize_));
    }

  private:
    std::deque<torch::Tensor> buffer_;
    size_t maxSize_;
};

namespace
{
const bool registered = []()
{
    NodeFactory::instance().registerType("Permute", [](ed::NodeId id)
                                         { return std::make_shared<PermuteNode>(id); });
    NodeFactory::instance().registerType("TypeCast", [](ed::NodeId id)
                                         { return std::make_shared<TypeCastNode>(id); });

    NodeFactory::instance().registerType("Clamp", [](ed::NodeId id)
                                         { return std::make_shared<ClampNode>(id); });

    NodeFactory::instance().registerType(
        "RingBuffer", [](ed::NodeId id) { return std::make_shared<RingBufferNode>(id); });
    NodeFactory::instance().registerType(
        "VideoReader",
        [](ed::NodeId id) { return std::make_shared<VideoReaderNode>(id); });
    return true;
}();
}