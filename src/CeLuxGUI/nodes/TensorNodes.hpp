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
    explicit UnaryTensorNode(std::string kind) : kind_(std::move(kind))
    {
    }

    std::vector<PortInfo> inputs() const override
    {
        return {{"in", PortInfo::Tensor}};
    }
    std::vector<PortInfo> outputs() const override
    {
        return {{"out", PortInfo::Tensor}};
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

    void drawUI(const std::string& uid) override
    {
        bool uiChanged = false;
        if (torch::cuda::is_available())
        {
            bool useGpu = dstDev_.is_cuda();
            if (ImGui::Checkbox(("GPU##" + uid).c_str(), &useGpu))
            {
                dstDev_ = useGpu ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
                uiChanged = true;
            }
            ImGui::SameLine();
        }
        uiChanged |= drawExtraUI(uid);
        if (uiChanged)
        {
            dirty_ = true;
            tensorOutputs.clear(); // invalidate cached ptr → downstream refresh
        }
    }
protected :
    virtual bool drawExtraUI(const std::string&)
{
        return false;
    }

torch::Tensor inTensor()
{
    auto it = tensorInputs.find("in");
    return it == tensorInputs.end() ? torch::Tensor() : maybe_to(it->second, dstDev_);
}
void publish(torch::Tensor t)
{
    tensorOutputs["out"] = maybe_to(std::move(t), torch::kCPU);
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
    PermuteNode() : UnaryTensorNode("Permute")
    {
    }

    bool drawExtraUI(const std::string& uid) override
    {
        bool changed = false;
        changed |= ImGui::InputInt4(("Order##" + uid).c_str(),
                                    reinterpret_cast<int*>(order_.data()));
        ImGui::SameLine();
        ImGui::TextUnformatted("(first N values used)");
        return changed;
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

                // Build unique permutation
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
};

//==============================================================================
// TypeCastNode
//==============================================================================
class TypeCastNode final : public UnaryTensorNode {
    int idx_ = 0;
    static constexpr struct {
        const char* lab;
        torch::ScalarType tp;
    } opts[] = {{"F32", torch::kFloat32},
                {"F16", torch::kFloat16},
                {"U8", torch::kUInt8},
                {"I32", torch::kInt32}};

  public:
    TypeCastNode() : UnaryTensorNode("TypeCast") {}

    bool drawExtraUI(const std::string& uid) override {
        const char* labels[] = {"F32", "F16", "U8", "I32"};
        return ImGui::Combo(("Type##" + uid).c_str(), &idx_, labels, IM_ARRAYSIZE(labels));
    }

    void compute() override {
        SAFE_COMPUTE(
            {
                auto x = inTensor();
                publish(x.defined() ? x.to(opts[idx_].tp, true, false) : torch::Tensor());
            },
            "TypeCastNode");
    }
};

//==============================================================================
// NormalizeNode
//==============================================================================
class NormalizeNode final : public UnaryTensorNode
{
    float scale_ = 1.f / 255.f, shift_ = 0.f;

  public:
    NormalizeNode() : UnaryTensorNode("Normalize")
    {
    }
    bool drawExtraUI(const std::string& uid) override
    {
        bool ch1 = ImGui::DragFloat(("Scale##" + uid).c_str(), &scale_, 0.0001f);
        ImGui::SameLine();
        bool ch2 = ImGui::DragFloat(("Shift##" + uid).c_str(), &shift_, 0.0001f);
        return ch1 || ch2;
    }
    bool changed = false;
    void compute() override
    {
        SAFE_COMPUTE(
            {
                auto x = inTensor();
                publish(x.defined() ? x.to(torch::kFloat32).mul(scale_).add_(shift_)
                                    : torch::Tensor());
            },
            "NormalizeNode");
    }
};

//==============================================================================
// ClampNode
//==============================================================================
class ClampNode final : public UnaryTensorNode
{
    float lo_ = 0.f, hi_ = 1.f;

  public:
    ClampNode() : UnaryTensorNode("Clamp")
    {
    }
    bool drawExtraUI(const std::string& uid) override
    {
        return ImGui::DragFloatRange2(("Range##" + uid).c_str(), &lo_, &hi_, 0.001f,
                                      -10.f, 10.f);
    }
    bool changed = false;
    void compute() override
    {
        SAFE_COMPUTE(
            {
                auto x = inTensor();
                publish(x.defined() ? x.clamp(lo_, hi_) : torch::Tensor());
            },
            "ClampNode");
    }
};

//==============================================================================
// ScaleShiftNode
//==============================================================================
class ScaleShiftNode final : public UnaryTensorNode
{
    float s_ = 1.f, b_ = 0.f;

  public:
    ScaleShiftNode() : UnaryTensorNode("ScaleShift")
    {
    }
    bool drawExtraUI(const std::string& uid) override
    {
        bool ch1 = ImGui::DragFloat(("Scale##" + uid).c_str(), &s_, 0.001f);
        ImGui::SameLine();
        bool ch2 = ImGui::DragFloat(("Shift##" + uid).c_str(), &b_, 0.001f);
        return ch1 || ch2;
    }
    bool changed = false;
    void compute() override
    {
        SAFE_COMPUTE(
            {
                auto x = inTensor();
                publish(x.defined() ? x.mul(s_).add_(b_) : torch::Tensor());
            },
            "ScaleShiftNode");
    }
};
namespace
{
const bool registered = []()
{
    NodeFactory::instance().registerType("Permute", []()
                                         { return std::make_shared<PermuteNode>(); });
    NodeFactory::instance().registerType("TypeCast", []()
                                         { return std::make_shared<TypeCastNode>(); });
    NodeFactory::instance().registerType("Normalize", []()
                                         { return std::make_shared<NormalizeNode>(); });
    NodeFactory::instance().registerType("Clamp", []()
                                         { return std::make_shared<ClampNode>(); });
    NodeFactory::instance().registerType(
        "ScaleShift", []() { return std::make_shared<ScaleShiftNode>(); });
    return true;
}();
}