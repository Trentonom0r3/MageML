#define IMGUI_DEFINE_MATH_OPERATORS
#include "utilities/builders.h"
#include "utilities/widgets.h"
#include <application.h>

#include <MageMLGUI/nodes/VideoPipeline.hpp>
#include <MageMLGUI/nodes/EncoderPipeline.hpp>
#include <algorithm>
#include <any>
#include <commdlg.h>
#include <glad/glad.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_node_editor.h>
#include <map>
#include <onnxruntime_cxx_api.h>
#include <sstream> // Include for std::ostringstream
#include <string>
#include <utility>
#include <vector>
#include <torch/torch.h>
using namespace std::string_literals;
using namespace MageML;

#if defined(_WIN32)
#include <windows.h>
std::filesystem::path getExecutableDir()
{
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    return std::filesystem::path(buffer).parent_path();
}
#else
#include <limits.h>
#include <unistd.h>
std::filesystem::path getExecutableDir()
{
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return std::filesystem::path(std::string(result, (count > 0) ? count : 0))
        .parent_path();
}
#endif
// at top of your cpp
// … at top of your Application.cpp …
static bool openFileDialog = false;
static bool saveFileDialog = false;
static char filePathBuf[260] = ""; // MAX_PATH

// Helper wrappers:
bool DoOpenFileDialog(char* outPath, size_t bufSize)
{
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "MageML Graph (*.json)\0*.json\0All Files\0*.*\0";
    ofn.lpstrFile = outPath;
    ofn.nMaxFile = (DWORD)bufSize;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    return GetOpenFileNameA(&ofn) == TRUE;
}

bool DoSaveFileDialog(char* outPath, size_t bufSize)
{
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "MageML Graph (*.json)\0*.json\0All Files\0*.*\0";
    ofn.lpstrFile = outPath;
    ofn.nMaxFile = (DWORD)bufSize;
    ofn.Flags = OFN_OVERWRITEPROMPT;
    return GetSaveFileNameA(&ofn) == TRUE;
}
#pragma region BASEHELPERS
namespace std
{
template <> struct hash<ax::NodeEditor::PinId>
{
    size_t operator()(ax::NodeEditor::PinId id) const noexcept
    {
        return std::hash<decltype(id.Get())>()(id.Get());
    }
};
// `std::equal_to` will just call `operator==` on PinId, which you already have.
} // namespace std

static int m_NextId = 1; // Global ID counter for nodes, links, etc.
// === THREADING MEMBERS ===
std::thread processingThread;
std::atomic<bool> stopThread{false};
std::atomic<bool> playRequested{false};
std::atomic<bool> scrubbing{false};
std::condition_variable playCv;

std::mutex playMutex;
int GetNextId()
{
    return m_NextId++;
}

static inline ImRect ImGui_GetItemRect()
{
    return ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
}

static inline ImRect ImRect_Expanded(const ImRect& rect, float x, float y)
{
    auto result = rect;
    result.Min.x -= x;
    result.Min.y -= y;
    result.Max.x += x;
    result.Max.y += y;
    return result;
}

namespace ed = ax::NodeEditor;
namespace util = ax::NodeEditor::Utilities;

using namespace ax;

using ax::Widgets::IconType;

static ed::EditorContext* m_Editor = nullptr;

// Helper function to convert a pointer to a string
template <typename T> std::string PointerToString(T* ptr)
{
    std::ostringstream oss;
    oss << ptr;
    return oss.str();
}

enum class PinType
{
    Flow,
    Bool,
    Int,
    Float,
    String,
    Object,
    Function,
    Delegate,
    Tensor,
    Tuple,
    Variant // Typically, will hold Tensor, Float, or Int values, but can be used for
            // any type
};

enum class PinKind
{
    Output,
    Input
};

enum class NodeType
{
    Blueprint,
    Simple,
    Tree,
    Comment,
    Houdini,
    VideoReader,
    VideoWriter,
    ONNX, // For ML models like ONNX
};

struct Node;
using PinValue = std::variant<std::monostate, bool, int, float, std::string,
                              torch::Tensor, uint64_t>;

// our one-and-only “overloaded” template
template <class... Ts> struct overloaded : Ts...
{
    using Ts::operator()...;
};
// CTAD so you can write overloaded{…} without spelling out Ts…
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;


// helper to turn a torch dtype into a readable string
inline const char* DTypeToString(torch::Dtype dt)
{
    switch (dt)
    {
    case torch::kFloat32:
        return "float32";
    case torch::kFloat64:
        return "float64";
    case torch::kInt32:
        return "int32";
    case torch::kInt64:
        return "int64";
    case torch::kUInt8:
        return "uint8";
    case torch::kFloat16:
        return "float16";
    default:
        return "unknown";
    }
}

inline std::string variantToString(const PinValue& v)
{
    return std::visit(overloaded{[](const std::monostate&) { return "<none>"s; },
                                 [](bool b) { return b ? "true"s : "false"s; },
                                 [](int i) { return std::to_string(i); },
                                 [](float f) { return std::to_string(f); },
                                 [](const std::string& s) { return s; },
                                 [](uint64_t u) { return std::to_string(u); },
                                 [&](const torch::Tensor& t)
                                 {
                                     std::ostringstream oss;
                                     oss << "Tensor(";
                                     for (auto d : t.sizes())
                                         oss << d << 'x';
                                     if (!t.sizes().empty())
                                         oss.seekp(-1, std::ios::end);
                                     oss << ") ";
                                     // <— use your mapping helper here:
                                     oss << DTypeToString(t.scalar_type());
                                     if (t.defined() && t.numel() <= 16)
                                     {
                                         oss << " [";
                                         auto c = t.contiguous().cpu();
                                         for (int i = 0; i < c.numel(); ++i)
                                             oss << c.data_ptr<float>()[i] << ',';
                                         oss.seekp(-1, std::ios::end);
                                         oss << "]";
                                     }
                                     return oss.str();
                                 }},
                      v);
}


struct Pin
{
    ed::PinId ID;
    Node* Node; // Pointer to the node this pin belongs to
    std::string Name;
    PinType Type;
    PinKind Kind;
    bool showUI = true;
    PinValue Value;

    Pin(int id, const char* name, PinType type)
        : ID(id), Node(nullptr), Name(name), Type(type), Kind(PinKind::Input)
    {
        switch (type)
        {
        case PinType::Float:
            Value = 0.0f;
            break;
        case PinType::Int:
            Value = 0;
            break;
        case PinType::Bool:
            Value = false;
            break;
        case PinType::String:
            Value = std::string("");
            break;
        case PinType::Object:
            Value = std::string("");
            break;
        case PinType::Function:
            Value = std::string("");
            break;
        case PinType::Delegate:
            Value = std::string("");
            break;
        case PinType::Tensor:
            Value = torch::Tensor(); // Default to an empty tensor
            break;
        }
    }
    Pin(int id, const char* name, PinType type, bool ShowUI)
        : ID(id), Node(nullptr), Name(name), Type(type), Kind(PinKind::Input),
          showUI(ShowUI)
    {
        switch (type)
        {
        case PinType::Float:
            Value = 0.0f;
            break;
        case PinType::Int:
            Value = 0;
            break;
        case PinType::Bool:
            Value = false;
            break;
        case PinType::String:
            Value = std::string("");
            break;
        case PinType::Object:
            Value = std::string("");
            break;
        case PinType::Function:
            Value = std::string("");
            break;
        case PinType::Delegate:
            Value = std::string("");
            break;
        case PinType::Tensor:
            Value = torch::Tensor(); // Default to an empty tensor
            break;
        }
    }
};

struct Link
{
    ed::LinkId ID;

    ed::PinId StartPinID;
    ed::PinId EndPinID;

    ImColor Color;

    Link(ed::LinkId id, ed::PinId startPinId, ed::PinId endPinId)
        : ID(id), StartPinID(startPinId), EndPinID(endPinId), Color(255, 255, 255)
    {
    }
};

struct NodeIdLess
{
    bool operator()(const ed::NodeId& lhs, const ed::NodeId& rhs) const
    {
        return lhs.AsPointer() < rhs.AsPointer();
    }
};

static bool Splitter(bool split_vertically, float thickness, float* size1, float* size2,
                     float min_size1, float min_size2,
                     float splitter_long_axis_size = -1.0f)
{
    using namespace ImGui;
    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = g.CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;
    bb.Min = window->DC.CursorPos +
             (split_vertically ? ImVec2(*size1, 0.0f) : ImVec2(0.0f, *size1));
    bb.Max = bb.Min + CalcItemSize(split_vertically
                                       ? ImVec2(thickness, splitter_long_axis_size)
                                       : ImVec2(splitter_long_axis_size, thickness),
                                   0.0f, 0.0f);
    return SplitterBehavior(bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, size1,
                            size2, min_size1, min_size2, 0.0f);
}
#pragma endregion

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Node class definition
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma region NODES
enum class NodeCategory
{
    Basic,
    Arithmetic,
    Logic,
    TensorOps,
    MachineLearning, // Nodes for machine learning tasks like ONNX models, etc.
    Utility,         // Utility nodes that don't fit into other categories
};

class Node
{
  public:
    ed::NodeId ID;
    std::string Name;
    std::vector<Pin> Inputs;
    std::vector<Pin> Outputs;
    ImColor Color;
    NodeType Type;
    ImVec2 Size;
    NodeCategory Category = NodeCategory::Basic; // Default category

    // Store previous input/output for dirty check.
    // These must be the same size as Inputs/Outputs after node is set up.
    std::vector<PinValue> prevInputs;
    std::vector<PinValue> prevOutputs;
    std::vector<int64_t> prevTensorVersions;
    bool dirty = true;
    void SetDirty()
    {
        dirty = true;
    }
    bool IsDirty() 
    {
        return dirty;
    }
    void ClearDirty()
    {
        dirty = false;
    }

virtual void UpdateDirtyFlag()
    {
        dirty = false;

        if (prevInputs.size() != Inputs.size())
        {
            prevInputs.resize(Inputs.size());
            dirty = true; // New inputs → mark dirty
        }

        if (prevOutputs.size() != Outputs.size())
            prevOutputs.resize(Outputs.size());

        for (size_t i = 0; i < Inputs.size(); ++i)
        {
            if (!pinvalue_equal(Inputs[i].Value, prevInputs[i]))
            {
                dirty = true;
                break;
            }
        }
    }


inline bool pinvalue_equal(const PinValue& a, const PinValue& b)
    {
        if (a.index() != b.index())
            return false;

        return std::visit(
            [](auto&& va, auto&& vb) -> bool
            {
                using A = std::decay_t<decltype(va)>;
                using B = std::decay_t<decltype(vb)>;

                if constexpr (std::is_same_v<A, torch::Tensor> &&
                              std::is_same_v<B, torch::Tensor>)
                {
                    if (!va.defined() && !vb.defined())
                        return true;
                    if (!va.defined() || !vb.defined())
                        return false;

                    return va._version() ==
                           vb._version(); // feel free to change this logic!
                }
                else if constexpr (std::is_same_v<A, B>)
                {
                    return va == vb;
                }
                else
                {
                    return false;
                }
            },
            a, b);
    }

    virtual void PostComputeUpdate()
    {
        prevInputs = {};
        prevOutputs = {};
        prevInputs.resize(Inputs.size());
        prevOutputs.resize(Outputs.size());

        for (size_t i = 0; i < Inputs.size(); ++i)
            prevInputs[i] = Inputs[i].Value;

        for (size_t i = 0; i < Outputs.size(); ++i)
            prevOutputs[i] = Outputs[i].Value;
    }


    // Pass-through: Inputs → Outputs (matching index only)
    virtual void PassThrough()
    {
        size_t cnt = std::min(Inputs.size(), Outputs.size());
        for (size_t i = 0; i < cnt; ++i)
            Outputs[i].Value = Inputs[i].Value;
        ClearDirty();
    }

    std::string State;
    std::string SavedState;

    Node(int id, const char* name, ImColor color = ImColor(255, 255, 255))
        : ID(id), Name(name), Color(color), Type(NodeType::Blueprint), Size(0, 0)
    {
    }
    virtual ~Node() = default;

    virtual void compute() {};
    virtual void PanelUI() {};
    // Add pass-through logic to compute()
  
    virtual void OnBegin()
    {
        // This method can be overridden by subclasses to perform initialization
        // tasks when the node is created or loaded.
    }
    // helper functions for adding params (for subclasses)
    void AddInput(const char* name, PinType type, bool showUI = true)
    {
        // Fix: Use reinterpret_cast to ensure proper pointer arithmetic
        Inputs.emplace_back(GetNextId(), name, type, showUI);
    }

    void AddOutput(const char* name, PinType type, bool showUI = true)
    {
        // Fix: Use reinterpret_cast to ensure proper pointer arithmetic
        Outputs.emplace_back(GetNextId(), name, type, showUI);
    }


    virtual void serialize(nlohmann::json& j) const
    {
        // Base class writes its generic fields:
        j["state"] = State;
        // (and any other truly common bits)
    }

    /// Called by LoadGraph *after* the Node has been constructed (and generic fields
    /// have already been read from j).  Each subclass pulls out whatever it needs.
    virtual void deserialize(nlohmann::json const& j)
    {
        if (j.contains("state"))
            State = j["state"].get<std::string>();
    }
};

class TensorNode : public Node
{
  public:
    using Node::Node; // Inherit constructors from Node

    int tensorRank = 4;
    int tensorDims[6] = {1, 3, 224, 224, 1, 1};
    int tensorDTypeIdx = 0;
    int tensorFillIdx = 0;

    void OnBegin() override
    {

        AddOutput("Tensor", PinType::Tensor);
    }

    void compute() override
    {
    }
    void PanelUI() override
    {

        //  isolate all ImGui IDs for this node
        ImGui::PushID(ID.AsPointer());

        ImGui::TextUnformatted("Tensor Input Properties");
        ImGui::Spacing();

        static const char* dtypeLabels[] = {"float32", "int32", "uint8", "float16"};
        static torch::Dtype dtypeEnums[] = {torch::kFloat32, torch::kInt32,
                                            torch::kUInt8, torch::kFloat16};
        static const char* fillLabels[] = {"Zeros", "Ones", "Random"};

        // ───────── DataType & FillMode (two columns) ──────────
        ImGui::Columns(2, nullptr, false);

        ImGui::TextUnformatted("Data Type:");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::Combo("##dtype", &tensorDTypeIdx, dtypeLabels,
                     IM_ARRAYSIZE(dtypeLabels));
        ImGui::NextColumn();

        ImGui::TextUnformatted("Fill Mode:");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::Combo("##fill", &tensorFillIdx, fillLabels, IM_ARRAYSIZE(fillLabels));
        ImGui::Columns(1);

        ImGui::Spacing();

        // ───────── Rank slider ────────────────────────────────
        ImGui::Text("Rank (Dims):");
        ImGui::SameLine(120);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::SliderInt("##rank", &tensorRank, 1, 6);

        ImGui::Spacing();

        // ───────── Dimension editors ──────────────────────────
        ImGui::Text("Shape:");
        ImGui::SameLine(120);

        float avail = ImGui::GetContentRegionAvail().x;
        float itemW =
            (avail - (tensorRank - 1) * ImGui::GetStyle().ItemSpacing.x) / tensorRank;

        for (int i = 0; i < tensorRank; ++i)
        {
            ImGui::PushID(i); // unique per dim
            ImGui::SetNextItemWidth(itemW);
            ImGui::DragInt("##dim", &tensorDims[i], 1, 1, 8192);
            ImGui::PopID();
            if (i + 1 < tensorRank)
                ImGui::SameLine();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ───────── Preview & Apply ────────────────────────────
        std::string shapeStr = "[";
        for (int i = 0; i < tensorRank; ++i)
        {
            shapeStr += std::to_string(tensorDims[i]);
            if (i + 1 < tensorRank)
                shapeStr += ", ";
        }
        shapeStr += "]";

        ImGui::Text("Final Shape: %s", shapeStr.c_str());
        ImGui::SameLine(250);

        if (ImGui::Button("Apply"))
        {
            std::vector<int64_t> shape(tensorDims, tensorDims + tensorRank);

            torch::Tensor t;
            switch (tensorFillIdx)
            {
            case 0:
                t = torch::zeros(
                    shape, torch::TensorOptions().dtype(dtypeEnums[tensorDTypeIdx]));
                break;
            case 1:
                t = torch::ones(
                    shape, torch::TensorOptions().dtype(dtypeEnums[tensorDTypeIdx]));
                break;
            case 2:
                t = torch::rand(
                    shape, torch::TensorOptions().dtype(dtypeEnums[tensorDTypeIdx]));
                break;
            }
            t = t.contiguous(); // Ensure contiguous memory layout
            Outputs[0].Value = t;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::PopID(); // <── done: restores ImGui ID stack
    }

  void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j); // write base fields
        j["tensor_rank"] = tensorRank;
        j["tensor_dims"] = std::vector<int>(tensorDims, tensorDims + tensorRank);
        j["tensor_dtype_idx"] = tensorDTypeIdx;
        j["tensor_fill_idx"] = tensorFillIdx;
    }

   void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j); // read base fields
        if (j.contains("tensor_rank"))
            tensorRank = j["tensor_rank"].get<int>();
        if (j.contains("tensor_dims"))
            std::copy(j["tensor_dims"].begin(), j["tensor_dims"].end(), tensorDims);
        if (j.contains("tensor_dtype_idx"))
            tensorDTypeIdx = j["tensor_dtype_idx"].get<int>();
        if (j.contains("tensor_fill_idx"))
            tensorFillIdx = j["tensor_fill_idx"].get<int>();
    }
    };

// OnnxNode: dynamically loads an ONNX model and exposes its inputs/outputs as pins
    class OnnxNode : public Node
    {
      public:
        OnnxNode(int id, const char* name);
        ~OnnxNode() override = default;

        // (Re)initialize ORT session, rebuild pins
        void initONNX();

        // Called each frame / graph-tick
        void compute() override;

        // UI for loading model, selecting device, and editing scalar inputs
        void PanelUI() override;

        // Serialization
        void serialize(nlohmann::json& j) const override;
        void deserialize(nlohmann::json const& j) override;

      private:
        // Model path & buffer for InputText
        std::string modelPath;
        char pathBuffer[256]{};

        bool initialized = false;
        int deviceIndex = 0; // 0=CPU, 1=CUDA
        bool useCuda = false;

        // ORT core objects
        Ort::Env env;
        std::unique_ptr<Ort::Session> session;
        Ort::MemoryInfo memInfo;

        // Meta for dynamic pins
        std::vector<std::string> inputNames, outputNames;
        std::vector<std::vector<int64_t>> inputShapes, outputShapes;
        std::vector<ONNXTensorElementDataType> inputTypes, outputTypes;

        // Cached input/output C-string pointers for performance
        std::vector<const char*> cachedInNames, cachedOutNames;

                 std::vector<std::string> availableModels;
         int selectedModelIndex = -1;

        // Helper for name caching
        void cacheNames();
    };

    inline OnnxNode::OnnxNode(int id, const char* name)
        : Node(id, name), env(ORT_LOGGING_LEVEL_WARNING, "OnnxNode"),
          memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        pathBuffer[0] = '\0';
    }

    inline void OnnxNode::serialize(nlohmann::json& j) const
    { /*        static std::vector<std::string> availableModels;
        static int selectedModelIndex = -1;*/
        Node::serialize(j);
        j["model_path"] = modelPath;
        j["device_index"] = deviceIndex;
        j["use_cuda"] = useCuda;
        j["available_models"] = availableModels;
        j["selected_model"] = selectedModelIndex;
    }

    inline void OnnxNode::deserialize(nlohmann::json const& j)
    {
        Node::deserialize(j);
        if (j.contains("model_path"))
            modelPath = j["model_path"].get<std::string>();
        if (j.contains("device_index"))
            deviceIndex = j["device_index"].get<int>();
        if (j.contains("use_cuda"))
            useCuda = j["use_cuda"].get<bool>();
        if (j.contains("available_models"))
            availableModels = j["available_models"].get<std::vector<std::string>>();
        if (j.contains("selected_model"))
            deviceIndex = j["selected_model"].get<int>();

        if (!modelPath.empty())
            initONNX();
    }

    inline void OnnxNode::initONNX()
    {
        initialized = false;
        Inputs.clear();
        Outputs.clear();
        inputNames.clear();
        inputShapes.clear();
        inputTypes.clear();
        outputNames.clear();
        outputShapes.clear();
        outputTypes.clear();
        cachedInNames.clear();
        cachedOutNames.clear();
        availableModels.clear();

        // File existence check
#ifdef _WIN32
        std::wstring wpath(modelPath.begin(), modelPath.end());
        std::wifstream test(wpath);
        if (!test)
        {
            std::wcerr << L"[OnnxNode] ERROR opening: " << wpath << "\n";
            return;
        }
        test.close();
#else
        std::ifstream test(modelPath);
        if (!test)
        {
            std::cerr << "[OnnxNode] ERROR opening: " << modelPath << "\n";
            return;
        }
        test.close();
#endif

        try
        {
            // --- Session options ---
            Ort::SessionOptions opts;
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (useCuda)
            {

                OrtCUDAProviderOptions cuda_opts;
                opts.AppendExecutionProvider_CUDA(cuda_opts);

            }

            // --- Session creation ---
#ifdef _WIN32
            session = std::make_unique<Ort::Session>(
                env, std::wstring(modelPath.begin(), modelPath.end()).c_str(), opts);
#else
            session = std::make_unique<Ort::Session>(env, modelPath.c_str(), opts);
#endif

            Ort::AllocatorWithDefaultOptions alloc;

            // --- Build input pins from ONNX meta ---
            size_t nIn = session->GetInputCount();
            inputNames.resize(nIn);
            inputShapes.resize(nIn);
            inputTypes.resize(nIn);
            for (size_t i = 0; i < nIn; ++i)
            {
                auto namePtr = session->GetInputNameAllocated(i, alloc);
                inputNames[i] = namePtr.get();

                auto info = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
                inputTypes[i] = info.GetElementType();
                inputShapes[i] = info.GetShape();

                bool isScalar = true;
                for (auto d : inputShapes[i])
                {
                    if (d != 1)
                    {
                        isScalar = false;
                        break;
                    }
                }

                if (isScalar)
                {
                    Inputs.emplace_back(GetNextId(), inputNames[i].c_str(),
                                        PinType::Float);
                    Inputs.back().Value = 0.0f;
                }
                else
                {
                    Inputs.emplace_back(GetNextId(), inputNames[i].c_str(),
                                        PinType::Tensor);
                }
            }

            // --- Build output pins ---
            size_t nOut = session->GetOutputCount();
            outputNames.resize(nOut);
            outputShapes.resize(nOut);
            outputTypes.resize(nOut);
            for (size_t i = 0; i < nOut; ++i)
            {
                auto namePtr = session->GetOutputNameAllocated(i, alloc);
                outputNames[i] = namePtr.get();
                auto info = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
                outputTypes[i] = info.GetElementType();
                outputShapes[i] = info.GetShape();
                Outputs.emplace_back(GetNextId(), outputNames[i].c_str(),
                                     PinType::Tensor);
            }

            cacheNames(); // For performance: avoid string allocs in compute()

            // MemoryInfo is device-specific
            memInfo =
                useCuda
                    ? Ort::MemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault)
                    : Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            initialized = true;
        }
        catch (const Ort::Exception& e)
        {
            std::cerr << "[OnnxNode] ORT Error: " << e.what() << "\n";
        }
        catch (const std::exception& e)
        {
            std::cerr << "[OnnxNode] std::exception: " << e.what() << "\n";
        }
        catch (...)
        {
            std::cerr << "[OnnxNode] Unknown init error.\n";
        }
    }

    // Helper: cache name pointers for ONNX API (perf!)
    inline void OnnxNode::cacheNames()
    {
        cachedInNames.clear();
        cachedOutNames.clear();
        for (auto& s : inputNames)
            cachedInNames.push_back(s.c_str());
        for (auto& s : outputNames)
            cachedOutNames.push_back(s.c_str());
    }

    // Perform ONNX inference
    inline void OnnxNode::compute()
    {
        if (!initialized)
            return;

        try
        {
            // Track scalar type (defaults to float32 for now)
            auto scalartype = torch::kFloat32;
            std::vector<Ort::Value> ortInputs;
            ortInputs.reserve(Inputs.size());

            for (size_t i = 0; i < Inputs.size(); ++i)
            {
                auto& pin = Inputs[i];
                if (pin.Type == PinType::Float)
                {
                    float scalar = std::get<float>(pin.Value);
                    auto cpuInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                              OrtMemTypeDefault);
                    ortInputs.push_back(Ort::Value::CreateTensor<float>(
                        cpuInfo, &scalar, 1, nullptr, 0));
                }
                else
                {
                    // Tensor input
                    auto t = std::get<torch::Tensor>(pin.Value);
                    auto targetDevice = useCuda ? torch::kCUDA : torch::kCPU;

                    // Only move if not already correct device
                    if (t.device().type() != targetDevice)
                        t = t.to(targetDevice);
                    // Always contiguous for safety
                    t = t.contiguous();

                    // ONNX type mapping (expand as needed)
                    ONNXTensorElementDataType ortType;
                    scalartype = t.scalar_type();
                    switch (scalartype)
                    {
                    case torch::kFloat32:
                        ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
                        break;
                    case torch::kFloat64:
                        ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
                        break;
                    case torch::kInt32:
                        ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
                        break;
                    case torch::kInt64:
                        ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
                        break;
                    case torch::kUInt8:
                        ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
                        break;
                    case torch::kFloat16:
                        ortType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
                        break;
                    default:
                        std::cerr
                            << "[OnnxNode] Unsupported tensor type: " << t.scalar_type()
                            << "\n";
                        continue;
                    }

                    std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
                    ortInputs.push_back(Ort::Value::CreateTensor(
                        memInfo, t.data_ptr(), t.numel() * t.element_size(),
                        shape.data(), shape.size(), ortType));
                }
            }

            // -- Inference --
            auto ortOutputs = session->Run(
                Ort::RunOptions{nullptr}, cachedInNames.data(), ortInputs.data(),
                ortInputs.size(), cachedOutNames.data(), cachedOutNames.size());

           // -- Convert outputs back to torch::Tensor --
            for (size_t i = 0; i < Outputs.size(); ++i)
            {
                auto& val = ortOutputs[i];
                auto shape = val.GetTensorTypeAndShapeInfo().GetShape();
                auto outType = outputTypes[i];

                torch::Tensor outT;
                switch (outType)
                {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                {
                    float* dataPtr = val.GetTensorMutableData<float>();
                    outT = torch::from_blob(dataPtr, torch::IntArrayRef(shape),
                                            torch::kFloat32)
                               ;
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                {
                    double* dataPtr = val.GetTensorMutableData<double>();
                    outT = torch::from_blob(dataPtr, torch::IntArrayRef(shape),
                                            torch::kFloat64)
                              ;
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                {
                    int32_t* dataPtr = val.GetTensorMutableData<int32_t>();
                    outT = torch::from_blob(dataPtr, torch::IntArrayRef(shape),
                                            torch::kInt32)
                               ;
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                {
                    int64_t* dataPtr = val.GetTensorMutableData<int64_t>();
                    outT = torch::from_blob(dataPtr, torch::IntArrayRef(shape),
                                            torch::kInt64)
                               ;
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                {
                    uint8_t* dataPtr = val.GetTensorMutableData<uint8_t>();
                    outT = torch::from_blob(dataPtr, torch::IntArrayRef(shape),
                                            torch::kUInt8)
                               ;
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                {
                    // PyTorch doesn't have native float16 CPU, so cast to float32.
                    void* rawData = val.GetTensorMutableData<void>();
                    size_t numel = 1;
                    for (auto d : shape)
                        numel *= d;

                    outT = torch::from_blob(rawData, torch::IntArrayRef(shape),
                                            torch::kFloat16)
                              ;
                    break;
                }
                default:
                    std::cerr << "[OnnxNode] Unsupported output type: " << outType
                              << " for output " << i << std::endl;
                    // Make a dummy tensor (empty)
                    outT = torch::empty({0});
                    break;
                }
              //  Outputs[i].Value = outT;
                if (useCuda)
                {
                    // Move to CUDA if needed
                    outT = outT.to(torch::kCUDA);
                }
                else
                {
                    // Ensure it's on CPU
                    outT = outT.to(torch::kCPU);
                }
                // Store the output tensor in the node's output pin
                Outputs[i].Value = outT;
            }

        }
        catch (const Ort::Exception& e)
        {
            std::cerr << "[OnnxNode] Inference error: " << e.what() << "\n";
        }
        catch (const c10::Error& e)
        {
            std::cerr << "[OnnxNode] C10 error: " << e.what() << "\n";
        }
        catch (const std::exception& e)
        {
            std::cerr << "[OnnxNode] std exception: " << e.what() << "\n";
        }
        catch (...)
        {
            std::cerr << "[OnnxNode] Unknown inference error.\n";
        }
    }

    // ImGui UI for ONNX node configuration and input editing
    inline void OnnxNode::PanelUI()
    {
        ImGui::PushID((void*)ID.AsPointer());
        ImGui::TextUnformatted("🔄 ONNX Inference Node");
        ImGui::Spacing();

        namespace fs = std::filesystem;
        fs::path templateDir = getExecutableDir() / "models";

        // 🧠 Rebuild model list (could be cached if slow)
        availableModels.clear();
        if (fs::exists(templateDir) && fs::is_directory(templateDir))
        {
            for (auto& entry : fs::directory_iterator(templateDir))
            {
                if (entry.path().extension() == ".onnx")
                    availableModels.push_back(entry.path().filename().string());
            }
        }

        // 🧠 Sync modelPath → buffer on first draw
        if (pathBuffer[0] == '\0' && !modelPath.empty())
        {
            std::strncpy(pathBuffer, modelPath.c_str(), sizeof(pathBuffer) - 1);
            pathBuffer[sizeof(pathBuffer) - 1] = '\0';

            // Try to match dropdown index
            fs::path mp = fs::path(modelPath);
            auto it = std::find(availableModels.begin(), availableModels.end(),
                                mp.filename().string());
            if (it != availableModels.end())
                selectedModelIndex =
                    static_cast<int>(std::distance(availableModels.begin(), it));
            else
                selectedModelIndex = -1;
        }

        // Dropdown with model selection
        if (!availableModels.empty())
        {
            if (ImGui::BeginCombo("Available Models",
                                  selectedModelIndex >= 0
                                      ? availableModels[selectedModelIndex].c_str()
                                      : "Choose"))
            {
                for (int i = 0; i < availableModels.size(); ++i)
                {
                    bool isSelected = (selectedModelIndex == i);
                    if (ImGui::Selectable(availableModels[i].c_str(), isSelected))
                    {
                        selectedModelIndex = i;
                        modelPath = (templateDir / availableModels[i]).string();
                        std::strncpy(pathBuffer, modelPath.c_str(),
                                     sizeof(pathBuffer) - 1);
                        pathBuffer[sizeof(pathBuffer) - 1] = '\0';
                        initONNX(); // Load immediately on selection
                    }
                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

        // Manual path entry
        bool enter = ImGui::InputText("Model Path", pathBuffer, sizeof(pathBuffer),
                                      ImGuiInputTextFlags_EnterReturnsTrue);

        if (enter)
        {
            modelPath = normalizePath(pathBuffer);
            selectedModelIndex = -1; // Custom input → no dropdown selection
            initONNX();
        }

        ImGui::SameLine();
        if (ImGui::Button(initialized ? "📥 Reload" : "📂 Load") || enter)
        {
            modelPath = normalizePath(pathBuffer);
            initONNX();
        }

        // Device selector
        const char* devices[] = {"CPU", "CUDA"};
        if (ImGui::Combo("Device", &deviceIndex, devices, 2))
        {
            useCuda = (deviceIndex == 1);
            initONNX(); // Re-init on device change
        }

        ImGui::Separator();
        if (!initialized)
        {
            ImGui::TextDisabled("Model not loaded.");
            ImGui::PopID();
            return;
        }

        // Input info
        ImGui::Text("Inputs:");
        for (size_t i = 0; i < Inputs.size(); ++i)
        {
            auto& pin = Inputs[i];
            ImGui::PushID((void*)(intptr_t)i);

            if (pin.Type == PinType::Float)
            {
                float v = std::get<float>(pin.Value);
                if (ImGui::InputFloat(pin.Name.c_str(), &v))
                    pin.Value = v;
            }
            else
            {
                std::ostringstream oss;
                oss << pin.Name << " : [";
                for (auto d : inputShapes[i])
                    oss << d << ",";
                oss << "]";
                ImGui::BulletText("%s", oss.str().c_str());
            }

            ImGui::PopID();
        }

        ImGui::Separator();
        ImGui::Text("Outputs:");
        for (size_t i = 0; i < Outputs.size(); ++i)
        {
            std::ostringstream oss;
            oss << outputNames[i] << " : [";
            for (auto d : outputShapes[i])
                oss << d << ",";
            oss << "]";
            ImGui::BulletText("%s", oss.str().c_str());
        }

        ImGui::PopID();
    }

    class DeviceNode : public Node
{
  public:
    using Node::Node;
    // ─── Internal state ──────────────────────────────────────
    int deviceIndex = 0; // 0=CPU, 1=CUDA
    void OnBegin() override
    {
        AddOutput("Out", PinType::Tensor);
        AddInput("In", PinType::Tensor);
    }

    void compute() override
    {
        // Get input tensor
        auto& inVal = Inputs[0].Value;
        if (auto* t = std::get_if<torch::Tensor>(&inVal))
        {
            if (t->defined() && t->numel() > 0)
            {
                // Move to selected device
                torch::Device device = (deviceIndex == 1) ? torch::kCUDA : torch::kCPU;
                Outputs[0].Value = t->to(device).contiguous();
            }
            else
            {
                Outputs[0].Value = torch::Tensor(); // empty tensor
            }
        }

        else
        {
            Outputs[0].Value = torch::Tensor(); // empty tensor if input is not a tensor
        }
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        // Device selection combo
        const char* devices[] = {"CPU", "CUDA"};
        if (ImGui::Combo("Device", &deviceIndex, devices, 2))
        {
        }
        // Show current device
        ImGui::Text("Current Device: %s", devices[deviceIndex]);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["device_index"] = deviceIndex;
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("device_index"))
            deviceIndex = j["device_index"].get<int>();
    }

    };


class DelayNode : public Node
{
  public:
    using Node::Node;

    // ─── User-configurable cache size ───────────────────────
    float maxCacheGB = 0.5f;                            // how many GB to hold
    size_t maxCacheBytes = size_t(0.5f * (1ull << 30)); // = GB × 2³⁰

    // ─── Internal state ────────────────────────────────────
    std::deque<torch::Tensor> cache;
    size_t cachedBytes = 0;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddInput("Offset", PinType::Int);
        AddOutput("Out", PinType::Tensor);
    }

    void compute() override
    {
        // 1) push new frame into cache (as a deep copy)
        auto& inVal = Inputs[0].Value;
        if (auto* t = std::get_if<torch::Tensor>(&inVal))
        {
            if (t->defined() && t->numel() > 0)
            {
                auto frame = t->clone(); // deep copy
                size_t sz = frame.numel() * frame.element_size();

                cache.push_back(std::move(frame));
                cachedBytes += sz;

                // evict oldest until under byte budget
                while (cachedBytes > maxCacheBytes && !cache.empty())
                {
                    auto& old = cache.front();
                    cachedBytes -= old.numel() * old.element_size();
                    cache.pop_front();
                }
            }
        }

        // 2) compute delayed output
        int offset = 0;
        if (auto* o = std::get_if<int>(&Inputs[1].Value))
            offset = *o;

        torch::Tensor out;
        {
            // clamp offset and grab from back
            if (!cache.empty())
            {
                int maxOff = int(cache.size()) - 1;
                offset = std::clamp(offset, 0, maxOff);
                out = cache[cache.size() - 1 - offset];
            }
        }
        Outputs[0].Value = std::move(out);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());

        // Slider to choose cache size in GB
        if (ImGui::SliderFloat("Cache Size (GB)", &maxCacheGB, 0.1f, 10.0f, "%.1f"))
        {
            maxCacheBytes = size_t(maxCacheGB * (1ull << 30));
            // immediately trim if over new budget
            while (cachedBytes > maxCacheBytes && !cache.empty())
            {
                auto& old = cache.front();
                cachedBytes -= old.numel() * old.element_size();
                cache.pop_front();
            }
        }

        // Offset control
        int off = std::get<int>(Inputs[1].Value);
        int maxOff = cache.empty() ? 0 : int(cache.size()) - 1;
        if (ImGui::DragInt("Offset", &off, 1, 0, maxOff))
            Inputs[1].Value = off;

        // Stats
        ImGui::Text("Frames cached: %zu", cache.size());
        double usedGB = double(cachedBytes) / double(1ull << 30);
        ImGui::Text("Using %.2f / %.1f GB", usedGB, maxCacheGB);

        ImGui::PopID();
    }

  void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["max_cache_gb"] = maxCacheGB;
        j["max_cache_bytes"] = maxCacheBytes;
    }

  void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("max_cache_gb"))
            maxCacheGB = j["max_cache_gb"].get<float>();
        if (j.contains("max_cache_bytes"))
            maxCacheBytes = j["max_cache_bytes"].get<size_t>();
        // recalculate cached bytes based on new max size
        cachedBytes = 0;
        for (const auto& t : cache)
            cachedBytes += t.numel() * t.element_size();
        // trim cache if needed
        while (cachedBytes > maxCacheBytes && !cache.empty())
        {
            auto& old = cache.front();
            cachedBytes -= old.numel() * old.element_size();
            cache.pop_front();
        }
    }

    };

class VideoReaderNode : public Node
{
  public:
    using Node::Node;

    // ─── Internal state ──────────────────────────────────────
    std::string videoPath;
    std::shared_ptr<VideoPipeline> pipeline;
    bool eof = false;

    // decode timing
    std::vector<double> frameTimes;
    static constexpr size_t MAX_SAMPLES = 200;

    // UI buffer
    char pathBuffer[256] = "";

    void OnBegin() override
    {
        AddOutput("Frame", PinType::Tensor);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Video Reader");
        ImGui::Spacing();

        // — Path entry —
        if (pathBuffer[0] == '\0' && !videoPath.empty())
        {
            std::strncpy(pathBuffer, videoPath.c_str(), sizeof(pathBuffer) - 1);
            pathBuffer[sizeof(pathBuffer) - 1] = '\0';
        }
        bool enterPressed = ImGui::InputText("Path", pathBuffer, sizeof(pathBuffer),
                                             ImGuiInputTextFlags_EnterReturnsTrue);
        videoPath = pathBuffer;
        ImGui::SameLine();
        if (ImGui::Button(pipeline ? "Re-open" : "Open") || enterPressed)
            openPipeline();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (!pipeline)
        {
            ImGui::TextDisabled("No video loaded.");
            ImGui::PopID();
            return;
        }

        // — Timeline slider —
        float cur = pipeline->currentTime();
        double dur = pipeline->duration();


            if (ImGui::SliderFloat("Time", &cur, 0.0f, (float)dur, "%.3f s"))
            {

                pipeline->seek(cur);

                eof = false;
            }
        

      
         //debug info on fps

        if (eof)
            ImGui::TextColored({1, 0.4f, 0.4f, 1}, "End-of-stream");

        ImGui::PopID();
    }

    void compute() override
    {
        if (!pipeline || eof)
            return;

        // 1) decode frame
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = pipeline->step();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // record timing
        frameTimes.push_back(ms);
        if (frameTimes.size() > MAX_SAMPLES)
            frameTimes.erase(frameTimes.begin());

        if (res == StepResult::NewFrame)
        {
            // get the new frame
            auto frame = pipeline->getCurrentFrame();
            Outputs[0].Value = frame;
        }
        else // EndOfStream
        {
            playRequested = false;
            eof = true;
        }
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j); // write base fields
        j["video_path"] = videoPath;
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j); // read base fields
        if (j.contains("video_path"))
            videoPath = j["video_path"].get<std::string>();
        // re-open pipeline if path is valid
        if (!videoPath.empty())
            openPipeline();
    }

  private:
    void openPipeline()
    {
        pipeline = std::make_shared<VideoPipeline>(videoPath);
        if (!pipeline->initialize())
        {
            pipeline.reset();
            ImGui::OpenPopup("VideoOpenError");
            return;
        }
        playRequested = eof = false;
    }
};
#pragma once

class EncoderNode : public Node
{
  public:
    using Node::Node;

    // ─── Internal state ─────────────────────────────
    std::string outPath;              // output file path
    std::optional<std::string> codec; // e.g. "h264"
    std::optional<int> width, height; // frame size
    std::optional<int> bitRate;       // bps
    std::optional<int> fps;           // frames/sec

    std::shared_ptr<EncoderPipeline> pipeline;
    bool pipelineOpen = false;

    // Timing/statistics
    std::vector<double> frameTimes;
    static constexpr size_t MAX_SAMPLES = 200;

    // UI buffer
    char pathBuffer[256] = "";
    char codecBuffer[64] = "";
    int widthBuffer = 1920;
    int heightBuffer = 1080;
    int bitRateBuffer = 4'000'000;
    int fpsBuffer = 30;
    int lastFrameW = -1, lastFrameH = -1;


    void OnBegin() override
    {
        AddInput("Frame", PinType::Tensor);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Video Encoder");
        ImGui::Spacing();

        // — Output path entry —
        if (pathBuffer[0] == '\0' && !outPath.empty())
        {
            std::strncpy(pathBuffer, outPath.c_str(), sizeof(pathBuffer) - 1);
            pathBuffer[sizeof(pathBuffer) - 1] = '\0';
        }
        bool enterPressed =
            ImGui::InputText("Output Path", pathBuffer, sizeof(pathBuffer),
                             ImGuiInputTextFlags_EnterReturnsTrue);
        outPath = pathBuffer;

        ImGui::Spacing();

        // — Codec, resolution, etc —
        if (codecBuffer[0] == '\0' && codec.has_value())
        {
            std::strncpy(codecBuffer, codec->c_str(), sizeof(codecBuffer) - 1);
            codecBuffer[sizeof(codecBuffer) - 1] = '\0';
        }
        // Show an InputText with placeholder inside the field if it's empty.
        ImGui::PushID("CodecInput");
        ImVec2 cursorPos = ImGui::GetCursorScreenPos();
        float w = ImGui::CalcItemWidth();

        if (codecBuffer[0] == '\0')
        {
            // Draw placeholder text over the input box (greyed out)
            ImGui::SetCursorScreenPos(cursorPos);
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            ImGui::TextUnformatted("e.g. h264, hevc, mjpeg");
            ImGui::PopStyleColor();
        }

        ImGui::SetCursorScreenPos(cursorPos); // Reset cursor for input box
        ImGui::InputText("##CodecInput", codecBuffer, sizeof(codecBuffer));
        ImGui::SameLine();
        ImGui::Text("Codec");
        ImGui::PopID();

        ImGui::DragInt("Width", &widthBuffer);
        ImGui::DragInt("Height", &heightBuffer);
        ImGui::DragInt("Bitrate (bps)", &bitRateBuffer);
        ImGui::DragInt("FPS", &fpsBuffer);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // — Open/close/reinit pipeline —
        if (!pipelineOpen)
        {
            if (ImGui::Button("Initialize Encoder") || enterPressed)
                openPipeline();
        }
        else
        {
            if (ImGui::Button("Re-initialize"))
                openPipeline();
            ImGui::SameLine();
            if (ImGui::Button("Finalize/Close"))
            {
                pipeline->finalize();
                pipeline.reset();
                pipelineOpen = false;
            }

        }

        ImGui::Spacing();
        if (!pipelineOpen)
        {
            ImGui::TextDisabled("Encoder not initialized.");
            ImGui::PopID();
            return;
        }

       
        ImGui::PopID();
    }

void compute() override
    {
        if (Inputs.empty())
            return;
        auto const& v = Inputs[0].Value;
        if (!std::holds_alternative<torch::Tensor>(v))
            return;
        const torch::Tensor& frame = std::get<torch::Tensor>(v);

        // --- Get frame shape ---
        int inH = 0, inW = 0;
        if (frame.dim() == 3)
        {
            if (frame.size(2) == 3 || frame.size(2) == 1)
            {
                // HWC
                inH = frame.size(0);
                inW = frame.size(1);
            }
            else if (frame.size(0) == 3 || frame.size(0) == 1)
            {
                // CHW
                inH = frame.size(1);
                inW = frame.size(2);
            }
        }
        if (!inH || !inW)
            return; // skip bad shape

        // --- If width/height are not set, use incoming shape (and sync UI buffer) ---
        if (lastFrameW == -1)
        {
            width = inW;
            widthBuffer = inW;
        }
        if (lastFrameH == -1)
        {
            height = inH;
            heightBuffer = inH;
        }

        // --- Detect shape mismatch or uninitialized pipeline ---
        bool shapeChanged = (lastFrameW != inW) || (lastFrameH != inH);
        if (!pipelineOpen || !pipeline || shapeChanged)
        {
            // Update shape tracking
            lastFrameW = inW;
            lastFrameH = inH;

            // If user has overridden, always use their value; otherwise use detected
            width = width.value_or(inW);
            height = height.value_or(inH);
            widthBuffer = width.value();
            heightBuffer = height.value();

            // Reinit pipeline
            openPipeline(); // will use latest width/height/codec etc
            if (!pipeline || !pipelineOpen)
                return;
        }

        // Encode the frame
        auto t0 = std::chrono::high_resolution_clock::now();
        pipeline->writeFrame(frame);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        frameTimes.push_back(ms);
        if (frameTimes.size() > MAX_SAMPLES)
            frameTimes.erase(frameTimes.begin());
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["outPath"] = outPath;
        if (codec)
            j["codec"] = *codec;
        if (width)
            j["width"] = *width;
        if (height)
            j["height"] = *height;
        if (bitRate)
            j["bitRate"] = *bitRate;
        if (fps)
            j["fps"] = *fps;
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("outPath"))
            outPath = j["outPath"].get<std::string>();
        if (j.contains("codec"))
            codec = j["codec"].get<std::string>();
        if (j.contains("width"))
            width = j["width"].get<int>();
        if (j.contains("height"))
            height = j["height"].get<int>();
        if (j.contains("bitRate"))
            bitRate = j["bitRate"].get<int>();
        if (j.contains("fps"))
            fps = j["fps"].get<int>();

        // restore UI buffers
        std::strncpy(pathBuffer, outPath.c_str(), sizeof(pathBuffer) - 1);
        if (codec)
            std::strncpy(codecBuffer, codec->c_str(), sizeof(codecBuffer) - 1);
        widthBuffer = width.value_or(1920);
        heightBuffer = height.value_or(1080);
        bitRateBuffer = bitRate.value_or(4'000'000);
        fpsBuffer = fps.value_or(30);

        pipelineOpen = false;
        pipeline.reset();
        openPipeline();
    }

    ~EncoderNode() override
    {
        if (pipeline)
            pipeline->finalize();
        pipeline.reset();
        lastFrameW = -1, lastFrameH = -1;
    }

  private:
    void openPipeline()
    {
        // Update config from UI:
        outPath = pathBuffer;
        codec = (std::strlen(codecBuffer) > 0 ? std::optional<std::string>(codecBuffer)
                                              : std::nullopt);
        width = (widthBuffer > 0 ? std::optional<int>(widthBuffer) : std::nullopt);
        height = (heightBuffer > 0 ? std::optional<int>(heightBuffer) : std::nullopt);
        bitRate =
            (bitRateBuffer > 0 ? std::optional<int>(bitRateBuffer) : std::nullopt);
        fps = (fpsBuffer > 0 ? std::optional<int>(fpsBuffer) : std::nullopt);

        pipeline = std::make_shared<EncoderPipeline>();
        bool ok = pipeline->initialize(outPath, codec, width, height, bitRate, fps);
        pipelineOpen = ok;
        if (!ok)
        {
            pipeline.reset();
            // Optionally: ImGui::OpenPopup("EncoderOpenError");
        }
    }
};

class FloatValueNode : public Node
{
  public:
    using Node::Node;

    void OnBegin() override
    {
        AddOutput("Value", PinType::Float);
    }

    void compute() override
    {
    }

    void PanelUI() override
    {

        if (float* value = std::get_if<float>(&Outputs[0].Value))
        {
            ImGui::DragFloat("##float", value, 0.1f, 0.0f, 100.0f);
        }
        else
        {
            ImGui::TextUnformatted("[Invalid float!]");
        }
    }

};

class IntValueNode : public Node
{
  public:
    using Node::Node;

    void OnBegin() override
    {
        AddOutput("Value", PinType::Int);
    }
    void compute() override
    {
    }
    void PanelUI() override
    {
        if (int* value = std::get_if<int>(&Outputs[0].Value))
        {
            ImGui::DragInt("##int", value, 1.0f, 0, 100);
        }
        else
        {
            ImGui::TextUnformatted("[Invalid int!]");
        }

    }
};

class BoolValueNode : public Node
{
  public:
    using Node::Node;

    void OnBegin() override
    {
        AddOutput("Value", PinType::Bool);
    }
    void compute() override
    {
    }
    void PanelUI() override
    {
        if (bool* value = std::get_if<bool>(&Outputs[0].Value))
        {
            ImGui::Checkbox("##bool", value);
            ImGui::Spring(0);
        }
        else
        {
            ImGui::TextUnformatted("[Invalid bool!]");
        }
    }
};

class StringValueNode : public Node
{
  public:
    using Node::Node;

    char stringBuffer[256] = "";
    std::string stringValue = "";

    void OnBegin() override
    {
        AddOutput("Value", PinType::String);
    }
    void compute() override
    {
        Outputs[0].Value = stringValue;
    }
    void PanelUI() override
    {
        // — Path entry —
        if (stringBuffer[0] == '\0' && !stringValue.empty())
        {
            std::strncpy(stringBuffer, stringValue.c_str(), sizeof(stringBuffer) - 1);
            stringBuffer[sizeof(stringBuffer) - 1] = '\0';
        }
        bool enterPressed = ImGui::InputText("Text", stringBuffer, sizeof(stringBuffer),
                                             ImGuiInputTextFlags_EnterReturnsTrue);
        stringValue = stringBuffer;
    }
};

// helper to turn a torch dtype into a readable string

class MultiplyNode : public Node
{
  public:
    using Node::Node;

    // keep a sliding window of recent compute times
    std::vector<double> timings;
    static constexpr size_t MAX_SAMPLES = 100;

    // scratch buffer for Tensor×Tensor multiplications
    torch::Tensor scratch_;

    void OnBegin() override
    {
        AddInput("A", PinType::Variant);
        AddInput("B", PinType::Variant);
        AddOutput("Result", PinType::Variant);
    }

    void compute() override
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        auto const& inA = Inputs[0].Value;
        auto const& inB = Inputs[1].Value;

        PinValue out = std::visit(
            overloaded{// Tensor × Tensor → at::mul_out into scratch_
                       [&](const torch::Tensor& a, const torch::Tensor& b) -> PinValue
                       {
                           // allocate scratch_ once or resize if shape changed
                           if (!scratch_.defined() || scratch_.sizes() != a.sizes())
                               scratch_ = torch::empty_like(a);
                           at::mul_out(scratch_, a, b);
                           return scratch_;
                       },
                       // Tensor × int
                       [&](const torch::Tensor& a, int b) -> PinValue
                       { return a.mul(static_cast<double>(b)); },
                       // Tensor × float
                       [&](const torch::Tensor& a, float b) -> PinValue
                       { return a.mul(static_cast<double>(b)); },
                       // int × Tensor
                       [&](int a, const torch::Tensor& b) -> PinValue
                       { return b.mul(static_cast<double>(a)); },
                       // float × Tensor
                       [&](float a, const torch::Tensor& b) -> PinValue
                       { return b.mul(static_cast<double>(a)); },
                       // float × float
                       [&](float a, float b) -> PinValue { return a * b; },
                       // int × int
                       [&](int a, int b) -> PinValue { return a * b; },
                       // everything else → empty
                       [](auto&&, auto&&) -> PinValue { return PinValue{}; }},
            inA, inB);

        Outputs[0].Value = std::move(out);

        // record timing
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timings.push_back(ms);
        if (timings.size() > MAX_SAMPLES)
            timings.erase(timings.begin());
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Multiply Node");
        ImGui::Spacing();
        ImGui::Text("Multiplies two inputs using libtorch ops.");

        if (!timings.empty())
        {
            double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
            double avg = sum / timings.size();
            ImGui::Text("Compute: %.3f ms (%.1f FPS)", avg, 1000.0 / avg);
        }
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["scratch_shape"] = scratch_.sizes();
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("scratch_shape"))
        {
            auto shape = j["scratch_shape"].get<std::vector<int64_t>>();
            scratch_ = torch::empty(shape, torch::kFloat32);
        }
    }
};
class AddNode : public Node
{
  public:
    using Node::Node;
    void OnBegin() override
    {
        AddInput("A", PinType::Variant);
        AddInput("B", PinType::Variant);
        AddOutput("Result", PinType::Variant);
    }
    void compute() override
    {
        auto const& inA = Inputs[0].Value;
        auto const& inB = Inputs[1].Value;

        PinValue out = std::visit(
            overloaded{
                [](torch::Tensor const& a, torch::Tensor const& b) -> PinValue
                { return a + b; }, [](torch::Tensor const& a, int b) -> PinValue
                { return a + b; }, [](torch::Tensor const& a, float b) -> PinValue
                { return a + b; }, [](int a, torch::Tensor const& b) -> PinValue
                { return a + b; }, [](float a, torch::Tensor const& b) -> PinValue
                { return a + b; }, [](float a, float b) -> PinValue { return a + b; },
                [](int a, float b) -> PinValue { return static_cast<float>(a) + b; },
                [](float a, int b) -> PinValue { return a + static_cast<float>(b); },
                [](int a, int b) -> PinValue { return a + b; },
                [](auto&&, auto&&) -> PinValue { return PinValue{}; }},
            inA, inB);

        Outputs[0].Value = std::move(out);
    }
    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Add Node");
        ImGui::Spacing();
        ImGui::Text("Adds two inputs together.");
        ImGui::PopID();
    }
};

class SubtractNode : public Node
{
  public:
    using Node::Node;
    void OnBegin() override
    {
        AddInput("A", PinType::Variant);
        AddInput("B", PinType::Variant);
        AddOutput("Result", PinType::Variant);
    }
    void compute() override
    {
        auto const& inA = Inputs[0].Value;
        auto const& inB = Inputs[1].Value;

        PinValue out = std::visit(
            overloaded{
                [](torch::Tensor const& a, torch::Tensor const& b) -> PinValue
                { return a - b; }, [](torch::Tensor const& a, int b) -> PinValue
                { return a - b; }, [](torch::Tensor const& a, float b) -> PinValue
                { return a - b; }, [](int a, torch::Tensor const& b) -> PinValue
                { return a - b; }, [](float a, torch::Tensor const& b) -> PinValue
                { return a - b; }, [](float a, float b) -> PinValue { return a - b; },
                [](int a, float b) -> PinValue { return static_cast<float>(a) - b; },
                [](float a, int b) -> PinValue { return a - static_cast<float>(b); },
                [](int a, int b) -> PinValue { return a - b; },
                [](auto&&, auto&&) -> PinValue { return PinValue{}; }},
            inA, inB);

        Outputs[0].Value = std::move(out);
    }
    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Subtract Node");
        ImGui::Spacing();
        ImGui::Text("Subtracts two inputs.");
        ImGui::PopID();
    }
};

class DivideNode : public Node
{
  public:
    using Node::Node;
    void OnBegin() override
    {
        AddInput("A", PinType::Variant);
        AddInput("B", PinType::Variant);
        AddOutput("Result", PinType::Variant);
    }
    void compute() override
    {
        auto const& inA = Inputs[0].Value;
        auto const& inB = Inputs[1].Value;

        PinValue out = std::visit(
            overloaded{
                [](torch::Tensor const& a, torch::Tensor const& b) -> PinValue
                { return a / b; }, [](torch::Tensor const& a, int b) -> PinValue
                { return a / b; }, [](torch::Tensor const& a, float b) -> PinValue
                { return a / b; }, [](int a, torch::Tensor const& b) -> PinValue
                { return a / b; }, [](float a, torch::Tensor const& b) -> PinValue
                { return a / b; }, [](float a, float b) -> PinValue
                { return b != 0.f ? a / b : std::numeric_limits<float>::infinity(); },
                [](int a, float b) -> PinValue
                {
                    return b != 0.f ? static_cast<float>(a) / b
                                    : std::numeric_limits<float>::infinity();
                },
                [](float a, int b) -> PinValue
                {
                    return b != 0 ? a / static_cast<float>(b)
                                  : std::numeric_limits<float>::infinity();
                },
                [](int a, int b) -> PinValue
                {
                    return b != 0 ? a / b : 0; // integer‐divide or fallback to 0
                },
                [](auto&&, auto&&) -> PinValue { return PinValue{}; }},
            inA, inB);

        Outputs[0].Value = std::move(out);
    }
    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Divide Node");
        ImGui::Spacing();
        ImGui::Text("Divides two inputs.");
        ImGui::PopID();
    }
};

// tensor op nodes
//  ─────────────────────────────────────────────────────────────────────────────
//  Tensor-ops: full classes with compute() implemented
//  ─────────────────────────────────────────────────────────────────────────────

//
// ClampNode: clamp each element of a tensor into [min, max]
//
class ClampNode : public Node
{
  public:
    using Node::Node;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddInput("Min", PinType::Float, false);
        AddInput("Max", PinType::Float, false);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);
        auto minv = std::get<float>(Inputs[1].Value);
        auto maxv = std::get<float>(Inputs[2].Value);
        Outputs[0].Value = t.clamp(minv, maxv);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Clamp");
        ImGui::PopID();
    }
};

class PermuteNode : public Node
{
  public:
    using Node::Node;
    std::string orderStr = ""; // e.g. "2,1,0"
    std::vector<int64_t> order;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        try
        {
            auto& t = std::get<torch::Tensor>(Inputs[0].Value);
            // parse on the fly if needed:
            if ((int64_t)order.size() != t.dim())
            {
                // default: reverse
                order.resize(t.dim());
                std::iota(order.begin(), order.end(), 0);
                std::reverse(order.begin(), order.end());
            }
            Outputs[0].Value = t.permute(order);
        } catch (const std::exception& e)
        {
            std::cerr << "[PermuteNode] Error: " << e.what() << "\n";
            Outputs[0].Value = torch::Tensor(); // return empty tensor on error
        }
        catch (...)
        {
            std::cerr << "[PermuteNode] Unknown error.\n";
            Outputs[0].Value = torch::Tensor(); // return empty tensor on unknown error
        }
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Order (csv):");

        char orderBuffer[256] = "";

        // Copy the string content to the buffer
        // 🔄 Sync string → buffer on first draw (buffer empty)
        if (orderBuffer[0] == '\0' && !orderStr.empty())
        {
            std::strncpy(orderBuffer, orderStr.c_str(), sizeof(orderBuffer) - 1);
            orderBuffer[sizeof(orderBuffer) - 1] = '\0';
        }

        // 🖊 Editable path input (pressing Enter opens it)
        bool enterPressed = ImGui::InputText("Path", orderBuffer, sizeof(orderBuffer),
                                             ImGuiInputTextFlags_EnterReturnsTrue);

        // 🔄 Always sync buffer → string (for click-away behavior)
        orderStr = orderBuffer;
        if (ImGui::IsItemDeactivatedAfterEdit())
        {
            // re-parse
            order.clear();
            std::istringstream iss(orderStr);
            int64_t v;
            char c;
            while (iss >> v)
            {
                order.push_back(v);
                iss >> c; // eat comma
            }
        }
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["order"] = orderStr; // save the order string
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("order"))
            orderStr = j["order"].get<std::string>();
        // re-parse order string into vector
        order.clear();
        std::istringstream iss(orderStr);
        int64_t v;
        char c;
        while (iss >> v)
        {
            order.push_back(v);
            iss >> c; // eat comma
        }
    }

};

// ─────────────────────────────────────────────────────────────────────────────
// TypeNode: choose one of four dtypes from a combo.
// ─────────────────────────────────────────────────────────────────────────────
class TypeNode : public Node
{
  public:
    using Node::Node;
    int dtypeIdx = 2; // 0=uint8,1=int32,2=float32,3=float16
    static constexpr const char* labels[4] = {"uint8", "int32", "float32", "float16"};

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);
        static const torch::Dtype DT[4] = {torch::kUInt8, torch::kInt32,
                                           torch::kFloat32, torch::kFloat16};
        Outputs[0].Value = t.to(DT[dtypeIdx]);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Output Type:");
        ImGui::Combo("##dtype", &dtypeIdx, labels, IM_ARRAYSIZE(labels));
        ImGui::PopID();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SqueezeNode: remove ALL size-1 dims if dim<0, otherwise that one.
// ─────────────────────────────────────────────────────────────────────────────
class SqueezeNode : public Node
{
  public:
    using Node::Node;
    int dim = -1; // -1 => squeeze all

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        try
        {

            auto& t = std::get<torch::Tensor>(Inputs[0].Value);
            Outputs[0].Value = (dim < 0) ? t.squeeze() : t.squeeze(dim);
        }         catch (const std::exception& e)
        {
            std::cerr << "[SqueezeNode] Error: " << e.what() << "\n";
        }
        catch (...)
        {
            std::cerr << "[SqueezeNode] Unknown error.\n";
        }
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Dim to squeeze (-1=all):");
        ImGui::DragInt("##sqz", &dim, 1.0f, -1, 10);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["dim"] = dim; // save the squeeze dimension
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// UnsqueezeNode: insert a size-1 axis at dim.
// ─────────────────────────────────────────────────────────────────────────────
class UnsqueezeNode : public Node
{
  public:
    using Node::Node;
    int dim = 0;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        try
        {

            auto& t = std::get<torch::Tensor>(Inputs[0].Value);
            Outputs[0].Value = t.unsqueeze(dim);
        } catch (const std::exception& e)
        {
            std::cerr << "[UnsqueezeNode] Error: " << e.what() << "\n";
        }
        catch (...)
        {
            std::cerr << "[UnsqueezeNode] Unknown error.\n";
        }
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Insert at dim:");
        ImGui::DragInt("##unsqz", &dim, 1.0f, 0, 10);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["dim"] = dim; // save the unsqueeze dimension
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
    }


};

// ─────────────────────────────────────────────────────────────────────────────
// ReshapeNode: type in a CSV shape, parsed on Enter.
// ─────────────────────────────────────────────────────────────────────────────
class ReshapeNode : public Node
{
  public:
    using Node::Node;
    char buf[128] = "1,1";
    std::vector<int64_t> targetShape;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);
        if (!targetShape.empty())
            Outputs[0].Value = t.reshape(targetShape);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Shape (csv):");
        if (ImGui::InputText("##shape", buf, sizeof(buf),
                             ImGuiInputTextFlags_EnterReturnsTrue))
        {
            targetShape.clear();
            std::istringstream iss(buf);
            int64_t v;
            char c;
            while (iss >> v)
            {
                targetShape.push_back(v);
                iss >> c;
            }
        }
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["shape"] = std::string(buf); // save the shape string
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("shape"))
        {
            std::string shapeStr = j["shape"].get<std::string>();
            std::strncpy(buf, shapeStr.c_str(), sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0'; // ensure null-termination
            // re-parse shape string into vector
            targetShape.clear();
            std::istringstream iss(buf);
            int64_t v;
            char c;
            while (iss >> v)
            {
                targetShape.push_back(v);
                iss >> c; // eat comma
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// TransposeNode: pick two axes to swap.
// ─────────────────────────────────────────────────────────────────────────────
class TransposeNode : public Node
{
  public:
    using Node::Node;
    int dim1 = 0, dim2 = 1;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);
        Outputs[0].Value = t.transpose(dim1, dim2);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Swap dims:");
        ImGui::DragInt("##d1", &dim1, 1.0f, 0, 10);
        ImGui::SameLine();
        ImGui::Text("<->");
        ImGui::SameLine();
        ImGui::DragInt("##d2", &dim2, 1.0f, 0, 10);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["dim1"] = dim1; // save the first dimension
        j["dim2"] = dim2; // save the second dimension
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("dim1"))
            dim1 = j["dim1"].get<int>();
        if (j.contains("dim2"))
            dim2 = j["dim2"].get<int>();
    }

};

// ─────────────────────────────────────────────────────────────────────────────
// ConcatNode: concatenate along chosen dim.
// ─────────────────────────────────────────────────────────────────────────────
class ConcatNode : public Node
{
  public:
    using Node::Node;
    int dim = 0;

    void OnBegin() override
    {
        AddInput("In1", PinType::Tensor);
        AddInput("In2", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        auto& A = std::get<torch::Tensor>(Inputs[0].Value);
        auto& B = std::get<torch::Tensor>(Inputs[1].Value);
        Outputs[0].Value = torch::cat({A, B}, dim);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Dim to join:");
        ImGui::DragInt("##cat", &dim, 1.0f, 0, 10);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["dim"] = dim; // save the concatenation dimension
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
    }

};

//
// ContiguousNode: make tensor memory contiguous
//
class ContiguousNode : public Node
{
  public:
    using Node::Node;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);
        Outputs[0].Value = t.contiguous();
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Contiguous");
        ImGui::PopID();
    }
};

class FlattenNode : public Node
{
  public:
    using Node::Node;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);
        Outputs[0].Value = t.view(-1);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Flatten");
        ImGui::PopID();
    }
};

//
// SplitNode: split into N chunks along chosen dim
//
class SplitNode : public Node
{
  public:
    using Node::Node;
    int chunks = 2;
    int dim = 0;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out1", PinType::Tensor, false);
        AddOutput("Out2", PinType::Tensor, false);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);
        auto pieces = torch::chunk(t, chunks, dim);
        // clamp in case user gave >2 or <2
        Outputs[0].Value = pieces.size() > 0 ? pieces[0] : t;
        Outputs[1].Value = pieces.size() > 1 ? pieces[1] : torch::Tensor();
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Split");
        ImGui::DragInt("##chunks", &chunks, 1.0f, 1, 16, "Chunks: %d");
        ImGui::DragInt("##dim", &dim, 1.0f, 0, 4, "Dim: %d");
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["chunks"] = chunks; // save the number of chunks
        j["dim"] = dim;       // save the split dimension
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("chunks"))
            chunks = j["chunks"].get<int>();
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
    }
};

//
// StackNode: stack two tensors along a new axis
//
class StackNode : public Node
{
  public:
    using Node::Node;
    int dim = 0;

    void OnBegin() override
    {
        AddInput("In1", PinType::Tensor);
        AddInput("In2", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        auto& A = std::get<torch::Tensor>(Inputs[0].Value);
        auto& B = std::get<torch::Tensor>(Inputs[1].Value);
        Outputs[0].Value = torch::stack({A, B}, dim);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Stack");
        ImGui::DragInt("##dim", &dim, 1.0f, 0, 4, "Dim: %d");
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["dim"] = dim; // save the stacking dimension
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
    }
};

//
// SliceNode: crop a 2D (H×W) or 3D (C×H×W) tensor
//
class SliceNode : public Node
{
  public:
    using Node::Node;
    int cropX = 0, cropY = 0, cropW = 64, cropH = 64;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);
        // assume at least 2D, channel-first if 3D
        if (t.dim() == 2)
        {
            Outputs[0].Value = t.index({torch::indexing::Slice(cropY, cropY + cropH),
                                        torch::indexing::Slice(cropX, cropX + cropW)});
        }
        else if (t.dim() == 3)
        {
            Outputs[0].Value = t.index({torch::indexing::Slice(), // channels
                                        torch::indexing::Slice(cropY, cropY + cropH),
                                        torch::indexing::Slice(cropX, cropX + cropW)});
        }
        else
        {
            // fallback: no-op
            Outputs[0].Value = t;
        }
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Slice");
        ImGui::DragInt("X", &cropX, 1.0f, 0, 4096);
        ImGui::SameLine();
        ImGui::DragInt("Y", &cropY, 1.0f, 0, 4096);
        ImGui::Text("Width/Height:");
        ImGui::DragInt("##w", &cropW, 1.0f, 1, 4096);
        ImGui::SameLine();
        ImGui::DragInt("##h", &cropH, 1.0f, 1, 4096);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["cropX"] = cropX; // save the crop X position
        j["cropY"] = cropY; // save the crop Y position
        j["cropW"] = cropW; // save the crop width
        j["cropH"] = cropH; // save the crop height
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("cropX"))
            cropX = j["cropX"].get<int>();
        if (j.contains("cropY"))
            cropY = j["cropY"].get<int>();
        if (j.contains("cropW"))
            cropW = j["cropW"].get<int>();
        if (j.contains("cropH"))
            cropH = j["cropH"].get<int>();
    }

};

class SumNode : public Node
{
  public:
    using Node::Node;
    int dim = -1;
    bool keepdim = false;

    void OnBegin() override
    {
        AddInput("In", PinType::Variant);
        AddOutput("Out", PinType::Variant);
    }

    void compute() override
    {
        auto const& in = Inputs[0].Value;
        // visitor handles Tensor vs int vs float
        PinValue out =
            std::visit(overloaded{[&](torch::Tensor const& t) -> PinValue
                                  {
                                      if (dim < 0)
                                          return t.sum();
                                      return torch::sum(t, {dim}, keepdim);
                                  },
                                  [&](int x) -> PinValue
                                  {
                                      // sum of scalar is itself
                                      return x;
                                  },
                                  [&](float x) -> PinValue { return x; },
                                  [&](auto&&...) -> PinValue { return PinValue{}; }},
                       in);
        Outputs[0].Value = std::move(out);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Sum");
        ImGui::SliderInt("Dim", &dim, -1, 5);
        ImGui::Checkbox("Keep Dim", &keepdim);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["dim"] = dim;         // save the sum dimension
        j["keepdim"] = keepdim; // save the keepdim flag
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
        if (j.contains("keepdim"))
            keepdim = j["keepdim"].get<bool>();
    }

};

class MeanNode : public Node
{
  public:
    using Node::Node;
    int dim = -1;
    bool keepdim = false;

    void OnBegin() override
    {
        AddInput("In", PinType::Variant);
        AddOutput("Out", PinType::Variant);
    }

    void compute() override
    {
        auto const& in = Inputs[0].Value;
        PinValue out = std::visit(
            overloaded{[&](torch::Tensor const& t) -> PinValue
                       {
                           if (dim < 0)
                               return t.mean();
                           return torch::mean(t, {dim}, keepdim);
                       },
                       [&](int x) -> PinValue { return static_cast<float>(x); },
                       [&](float x) -> PinValue { return x; },
                       [&](auto&&...) -> PinValue { return PinValue{}; }},
            in);
        Outputs[0].Value = std::move(out);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Mean");
        ImGui::SliderInt("Dim", &dim, -1, 5);
        ImGui::Checkbox("Keep Dim", &keepdim);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const
        {
        Node::serialize(j);
        j["dim"] = dim;         // save the mean dimension
        j["keepdim"] = keepdim; // save the keepdim flag
        }

    void deserialize(nlohmann::json const& j) override
        {
        Node::deserialize(j);
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
        if (j.contains("keepdim"))
            keepdim = j["keepdim"].get<bool>();
        }

};

class MaxNode : public Node
{
  public:
    using Node::Node;
    int dim = -1;
    bool keepdim = false;

    void OnBegin() override
    {
        AddInput("In", PinType::Variant);
        AddOutput("Out", PinType::Variant);
    }

    void compute() override
    {
        auto const& in = Inputs[0].Value;
        PinValue out =
            std::visit(overloaded{[&](torch::Tensor const& t) -> PinValue
                                  {
                                      if (dim < 0)
                                          return torch::amax(t);
                                      return torch::amax(t, {dim}, keepdim);
                                  },
                                  [&](int x) -> PinValue { return x; },
                                  [&](float x) -> PinValue { return x; },
                                  [&](auto&&...) -> PinValue { return PinValue{}; }},
                       in);
        Outputs[0].Value = std::move(out);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Max");
        ImGui::SliderInt("Dim", &dim, -1, 5);
        ImGui::Checkbox("Keep Dim", &keepdim);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["dim"] = dim;         // save the max dimension
        j["keepdim"] = keepdim; // save the keepdim flag
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
        if (j.contains("keepdim"))
            keepdim = j["keepdim"].get<bool>();
    }
};

class MinNode : public Node
{
  public:
    using Node::Node;
    int dim = -1;
    bool keepdim = false;

    void OnBegin() override
    {
        AddInput("In", PinType::Variant);
        AddOutput("Out", PinType::Variant);
    }

    void compute() override
    {
        auto const& in = Inputs[0].Value;
        PinValue out =
            std::visit(overloaded{[&](torch::Tensor const& t) -> PinValue
                                  {
                                      if (dim < 0)
                                          return torch::amin(t);
                                      return torch::amin(t, {dim}, keepdim);
                                  },
                                  [&](int x) -> PinValue { return x; },
                                  [&](float x) -> PinValue { return x; },
                                  [&](auto&&...) -> PinValue { return PinValue{}; }},
                       in);
        Outputs[0].Value = std::move(out);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Min");
        ImGui::SliderInt("Dim", &dim, -1, 5);
        ImGui::Checkbox("Keep Dim", &keepdim);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["dim"] = dim;         // save the min dimension
        j["keepdim"] = keepdim; // save the keepdim flag
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
        if (j.contains("keepdim"))
            keepdim = j["keepdim"].get<bool>();
    }
};

// —————————————————————————
// ArgmaxNode (Tensor in → Tensor out)
// —————————————————————————

class ArgmaxNode : public Node
{
  public:
    using Node::Node;
    int dim = 0;
    bool keepdim = false;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);
        auto out = torch::argmax(t, dim);
        if (keepdim)
            out = out.unsqueeze(dim);
        Outputs[0].Value = out;
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Argmax");
        ImGui::SliderInt("Dim", &dim, 0, 5);
        ImGui::Checkbox("Keep Dim", &keepdim);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["dim"] = dim;         // save the argmax dimension
        j["keepdim"] = keepdim; // save the keepdim flag
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
        if (j.contains("keepdim"))
            keepdim = j["keepdim"].get<bool>();
    }
};

// —————————————————————————
// CompareNode (Variant A, Variant B → Tensor bool mask)
// —————————————————————————
class RepeatNode : public Node
{
  public:
    using Node::Node;
    int repeats = 1; // how many times to repeat
    int dim = 0;     // which dimension to repeat along

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor, false);
    }

    void compute() override
    {
        // Ensure input tensor is valid
        if (Inputs.empty())
        {
            std::cerr << "[RepeatNode] Error: No input tensor provided.\n";
            return;
        }

        auto& t = std::get<torch::Tensor>(Inputs[0].Value);

        // Clamp dim into valid range
        int64_t clampedDim = std::clamp(dim, 0, static_cast<int>(t.dim()) - 1);

        // Construct repeat vector: 1 for all dims except the one we're
        // repeating
        std::vector<int64_t> repeat_vector(t.dim(), 1);
        repeat_vector[clampedDim] = std::max(1, repeats);

        // Compute repeated tensor
        Outputs[0].Value = t.repeat(repeat_vector);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Repeat");
        ImGui::DragInt("Repeats", &repeats, 1.0f, 1, 16);
        ImGui::DragInt("Dim", &dim, 1.0f, 0, 8);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["repeats"] = repeats; // save the number of repeats
        j["dim"] = dim;         // save the repeat dimension
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("repeats"))
            repeats = j["repeats"].get<int>();
        if (j.contains("dim"))
            dim = j["dim"].get<int>();
    }
};

class CompareNode : public Node
{
  public:
    using Node::Node;
    enum CompareOp
    {
        Equal,
        NotEqual,
        Greater,
        Less,
        GreaterEqual,
        LessEqual
    };
    int opIndex = Equal;

    void OnBegin() override
    {
        AddInput("A", PinType::Variant);
        AddInput("B", PinType::Variant);
        AddOutput("Result", PinType::Tensor);
    }

    void compute() override
    {
        auto const& a = Inputs[0].Value;
        auto const& b = Inputs[1].Value;

        // helper to wrap a bool into a 0-d Tensor
        auto toTensor = [&](bool f) { return torch::tensor(f); };

        // For any combination, produce a Tensor
        torch::Tensor result = std::visit(
            overloaded{[&](torch::Tensor const& t1, torch::Tensor const& t2)
                       {
                           switch (opIndex)
                           {
                           case Equal:
                               return t1.eq(t2);
                           case NotEqual:
                               return t1.ne(t2);
                           case Greater:
                               return t1.gt(t2);
                           case Less:
                               return t1.lt(t2);
                           case GreaterEqual:
                               return t1.ge(t2);
                           case LessEqual:
                               return t1.le(t2);
                           }
                           return t1.eq(t2);
                       },
                       [&](torch::Tensor const& t1, int x)
                       {
                           switch (opIndex)
                           {
                           case Equal:
                               return t1.eq(x);
                           case NotEqual:
                               return t1.ne(x);
                           case Greater:
                               return t1.gt(x);
                           case Less:
                               return t1.lt(x);
                           case GreaterEqual:
                               return t1.ge(x);
                           case LessEqual:
                               return t1.le(x);
                           }
                           return t1.eq(x);
                       },
                       [&](torch::Tensor const& t1, float x)
                       {
                           switch (opIndex)
                           {
                           case Equal:
                               return t1.eq(x);
                           case NotEqual:
                               return t1.ne(x);
                           case Greater:
                               return t1.gt(x);
                           case Less:
                               return t1.lt(x);
                           case GreaterEqual:
                               return t1.ge(x);
                           case LessEqual:
                               return t1.le(x);
                           }
                           return t1.eq(x);
                       },
                       [&](int x, torch::Tensor const& t2)
                       {
                           switch (opIndex)
                           {
                           case Equal:
                               return t2.eq(x);
                           case NotEqual:
                               return t2.ne(x);
                           case Greater:
                               return t2.lt(x);
                           case Less:
                               return t2.gt(x);
                           case GreaterEqual:
                               return t2.le(x);
                           case LessEqual:
                               return t2.ge(x);
                           }
                           return t2.eq(x);
                       },
                       [&](float x, torch::Tensor const& t2)
                       {
                           switch (opIndex)
                           {
                           case Equal:
                               return t2.eq(x);
                           case NotEqual:
                               return t2.ne(x);
                           case Greater:
                               return t2.lt(x);
                           case Less:
                               return t2.gt(x);
                           case GreaterEqual:
                               return t2.le(x);
                           case LessEqual:
                               return t2.ge(x);
                           }
                           return t2.eq(x);
                       },
                       [&](int x, int y)
                       {
                           return toTensor(opIndex == Equal          ? x == y
                                           : opIndex == NotEqual     ? x != y
                                           : opIndex == Greater      ? x > y
                                           : opIndex == Less         ? x < y
                                           : opIndex == GreaterEqual ? x >= y
                                                                     : x <= y);
                       },
                       [&](int x, float y)
                       {
                           return toTensor(opIndex == Equal          ? x == y
                                           : opIndex == NotEqual     ? x != y
                                           : opIndex == Greater      ? x > y
                                           : opIndex == Less         ? x < y
                                           : opIndex == GreaterEqual ? x >= y
                                                                     : x <= y);
                       },
                       [&](float x, int y)
                       {
                           return toTensor(opIndex == Equal          ? x == y
                                           : opIndex == NotEqual     ? x != y
                                           : opIndex == Greater      ? x > y
                                           : opIndex == Less         ? x < y
                                           : opIndex == GreaterEqual ? x >= y
                                                                     : x <= y);
                       },
                       [&](float x, float y)
                       {
                           return toTensor(opIndex == Equal          ? x == y
                                           : opIndex == NotEqual     ? x != y
                                           : opIndex == Greater      ? x > y
                                           : opIndex == Less         ? x < y
                                           : opIndex == GreaterEqual ? x >= y
                                                                     : x <= y);
                       },
                       [&](auto&&...) -> torch::Tensor
                       {
                           // unsupported types
                           return torch::Tensor();
                       }},
            a, b);
        Outputs[0].Value = result;
    }

    void PanelUI() override
    {
        static const char* ops[] = {"==", "!=", ">", "<", ">=", "<="};
        ImGui::PushID(ID.AsPointer());
        ImGui::Text("Compare A vs B");
        ImGui::Combo("Operation", &opIndex, ops, IM_ARRAYSIZE(ops));
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["opIndex"] = opIndex; // save the comparison operation index
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("opIndex"))
            opIndex = j["opIndex"].get<int>();
    }
};

// —————————————————————————
// WhereNode (Tensor bool → select between two Variants)
// —————————————————————————

class WhereNode : public Node
{
  public:
    using Node::Node;

    void OnBegin() override
    {
        AddInput("Condition", PinType::Tensor);
        AddInput("A", PinType::Variant);
        AddInput("B", PinType::Variant);
        AddOutput("Result", PinType::Variant);
    }

    void compute() override
    {
        auto cond = std::get<torch::Tensor>(Inputs[0].Value);
        auto const& a = Inputs[1].Value;
        auto const& b = Inputs[2].Value;

        // Only handle Tensor→Tensor
        if (std::holds_alternative<torch::Tensor>(a) &&
            std::holds_alternative<torch::Tensor>(b))
        {
            auto ta = std::get<torch::Tensor>(a);
            auto tb = std::get<torch::Tensor>(b);
            Outputs[0].Value = torch::where(cond, ta, tb);
        }
        else
        {
            // unsupported combos
            Outputs[0].Value = PinValue{};
        }
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Where");
        ImGui::TextUnformatted("Select A where true, else B");
        ImGui::PopID();
    }
};

// —————————————————————————
// MaskNode (Variant in → masked Variant out)
// —————————————————————————

class MaskNode : public Node
{
  public:
    using Node::Node;

    void OnBegin() override
    {
        AddInput("In", PinType::Variant);
        AddInput("Mask", PinType::Tensor);
        AddOutput("Out", PinType::Variant);
    }

    void compute() override
    {
        auto mask = std::get<torch::Tensor>(Inputs[1].Value);
        auto const& in = Inputs[0].Value;

        PinValue out = std::visit(overloaded{[&](torch::Tensor const& t) -> PinValue
                                             { return t.masked_select(mask); },
                                             [&](auto&& x) -> PinValue
                                             {
                                                 // do not mask scalars — just pass
                                                 // through
                                                 return PinValue{x};
                                             }},
                                  in);

        Outputs[0].Value = std::move(out);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Mask");
        ImGui::TextUnformatted("Apply boolean mask to Tensor");
        ImGui::PopID();
    }
};

// —————————————————————————
// PadNode, CropNode, ResizeNode (Tensor → Tensor)
// —————————————————————————

class PadNode : public Node
{
  public:
    using Node::Node;
    int padLeft = 0, padRight = 0, padTop = 0, padBottom = 0;
    float padValue = 0.0f;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);
        // padding order: {left, right, top, bottom}
        Outputs[0].Value =
            torch::constant_pad_nd(t, {padLeft, padRight, padTop, padBottom}, padValue);
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Pad");
        ImGui::DragInt("Left", &padLeft, 1, 0, 1024);
        ImGui::DragInt("Right", &padRight, 1, 0, 1024);
        ImGui::DragInt("Top", &padTop, 1, 0, 1024);
        ImGui::DragInt("Bottom", &padBottom, 1, 0, 1024);
        ImGui::DragFloat("Value", &padValue, 0.1f);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["padLeft"] = padLeft;     // save left padding
        j["padRight"] = padRight;   // save right padding
        j["padTop"] = padTop;       // save top padding
        j["padBottom"] = padBottom; // save bottom padding
        j["padValue"] = padValue;   // save padding value
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("padLeft"))
            padLeft = j["padLeft"].get<int>();
        if (j.contains("padRight"))
            padRight = j["padRight"].get<int>();
        if (j.contains("padTop"))
            padTop = j["padTop"].get<int>();
        if (j.contains("padBottom"))
            padBottom = j["padBottom"].get<int>();
        if (j.contains("padValue"))
            padValue = j["padValue"].get<float>();
    }
};

class CropNode : public Node
{
  public:
    using Node::Node;
    int cropX = 0, cropY = 0, cropW = 224, cropH = 224;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor);
    }

    void compute() override
    {
        auto& t = std::get<torch::Tensor>(Inputs[0].Value);

        if (t.dim() == 2)
        {
            t = t.index({torch::indexing::Slice(cropY, cropY + cropH),
                         torch::indexing::Slice(cropX, cropX + cropW)});
            Outputs[0].Value = t;
        }
        else if (t.dim() == 3)
        {
            // HWC layout: [height, width, channels]
            t = t.index({
                torch::indexing::Slice(cropY, cropY + cropH), // height
                torch::indexing::Slice(cropX, cropX + cropW), // width
                torch::indexing::Slice()                      // channels (all)
            });
            Outputs[0].Value = t;
        }
        else
        {
            Outputs[0].Value = t;
        }
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Crop");
        ImGui::DragInt("X", &cropX, 1, 0, 4096);
        ImGui::DragInt("Y", &cropY, 1, 0, 4096);
        ImGui::DragInt("Width", &cropW, 1, 1, 4096);
        ImGui::DragInt("Height", &cropH, 1, 1, 4096);
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["cropX"] = cropX; // save the crop X position
        j["cropY"] = cropY; // save the crop Y position
        j["cropW"] = cropW; // save the crop width
        j["cropH"] = cropH; // save the crop height
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("cropX"))
            cropX = j["cropX"].get<int>();
        if (j.contains("cropY"))
            cropY = j["cropY"].get<int>();
        if (j.contains("cropW"))
            cropW = j["cropW"].get<int>();
        if (j.contains("cropH"))
            cropH = j["cropH"].get<int>();
    }
};
template <typename VariantType> bool has_value(const VariantType& variant)
{
    return variant.index() != 0; // std::monostate is at index 0
}
class ResizeNode : public Node
{
  public:
    using Node::Node;
    int modeIndex = 0; // 0=bilinear,1=nearest,2=bicubic
    int targetW = 224, targetH = 224;

    void OnBegin() override
    {
        AddInput("In", PinType::Tensor);
        AddOutput("Out", PinType::Tensor);
    }
    void compute() override
    {
        try
        {
            if (Inputs.empty())
            {
                std::cerr << "[ResizeNode] No input tensor!\n";
                return;
            }
            auto& t = std::get<torch::Tensor>(Inputs[0].Value);

            // — Convert HWC → CHW if needed
            torch::Tensor t_chw = t;
            if (t.dim() == 3 && t.size(2) <= 4)
            {
                t_chw = t.permute({2, 0, 1}).contiguous();
            }

            // — Add batch if needed, and make sure it's contiguous
            torch::Tensor inp = (t_chw.dim() == 3) ? t_chw.unsqueeze(0).contiguous()
                                                   : t_chw.contiguous();

            // — Always work in float
            auto inp_f = inp.to(torch::kFloat32);

            // — Perform the resize with the exact kernel you want:
            torch::Tensor out;
            auto size = std::vector<int64_t>{targetH, targetW};
            switch (modeIndex)
            {
            case 0: // bilinear
                out = at::upsample_bilinear2d(inp_f, size,
                                              /*align_corners=*/false);
                break;

            case 2: // bicubic
                out = at::upsample_bicubic2d(inp_f, size,
                                             /*align_corners=*/false);
                break;
            default:
                out = at::upsample_bilinear2d(inp_f, size, false);
            }

            // — Remove the batch dim if we added one
            if (t_chw.dim() == 3)
                out = out.squeeze(0);

            // — Convert back to HWC if original was HWC
            if (t.dim() == 3 && t.size(2) <= 4)
                out = out.permute({1, 2, 0}).contiguous().to(torch::kUInt8);

            Outputs[0].Value = out;
        }
        catch (const c10::Error& e)
        {
            std::cerr << "[ResizeNode] PyTorch error: " << e.what() << "\n";
        }
        catch (const std::exception& e)
        {
            std::cerr << "[ResizeNode] std::exception: " << e.what() << "\n";
        }
        catch (...)
        {
            std::cerr << "[ResizeNode] Unknown exception in compute()\n";
        }
    }

    void PanelUI() override
    {
        static const char* modes[] = {"Bilinear", "Nearest", "Bicubic"};
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Resize");
        ImGui::DragInt("W", &targetW, 1, 1, 8192);
        ImGui::DragInt("H", &targetH, 1, 1, 8192);
        ImGui::Combo("Mode", &modeIndex, modes, IM_ARRAYSIZE(modes));
        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["targetW"] = targetW; // save target width
        j["targetH"] = targetH; // save target height
        j["modeIndex"] = modeIndex; // save resize mode index
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("targetW"))
            targetW = j["targetW"].get<int>();
        if (j.contains("targetH"))
            targetH = j["targetH"].get<int>();
        if (j.contains("modeIndex"))
            modeIndex = j["modeIndex"].get<int>();
    }
};

// ────────────────────────────────────────────────────────────────
// Simple CHW Image-Loader Node
//   • Loads an image file from disk
//   • Converts it to a torch::Tensor shaped [C, H, W] (no batch dim)
//   • Always returns 3-channel RGB, float32 in [0,1]  (for now)
//   • Real decoding logic goes in compute()
// ────────────────────────────────────────────────────────────────
class LoadImageNode : public Node
{
  public:
    using Node::Node;

    std::string filePath; // absolute / relative path
    char pathBuffer[256]{};

    // ───────── Initialization ─────────
    void OnBegin() override
    {
        AddOutput("Image", PinType::Tensor);
    }

    // ───────── Compute (TBD) ──────────
    void compute() override
    {
    }

    // ───────── UI ─────────────────────
    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::Text("Load Image");

        // path text box
        if (pathBuffer[0] == '\0' && !filePath.empty())
            strncpy(pathBuffer, filePath.c_str(), sizeof(pathBuffer) - 1);

        if (ImGui::InputText("Path", pathBuffer, sizeof(pathBuffer)))
            filePath = pathBuffer;

        // load button
        if (ImGui::Button("Load"))
        {
            // trigger compute() on next evaluation tick
            // (or call immediately if you prefer)
        }

        ImGui::PopID();
    }

    // ───────── Serialization ─────────
    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
        j["filePath"] = filePath; // save the file path
    }

    void deserialize(nlohmann::json const& j) override
    {
        Node::deserialize(j);
        if (j.contains("filePath"))
            filePath = j["filePath"].get<std::string>();
    }
};
using NodeCreator = std::function<std::unique_ptr<Node>(int)>;

struct NodeInfo
{
    NodeCreator creator;
    NodeCategory category;
};

class NodeFactory
{
  public:
    static NodeFactory& instance()
    {
        static NodeFactory f;
        return f;
    }

    // now takes a category
    void registerType(const std::string& key, NodeCategory cat, NodeCreator c)
    {
        infos_[key] = {std::move(c), cat};
    }

    std::unique_ptr<Node> create(const std::string& key, int id) const
    {
        auto it = infos_.find(key);
        if (it == infos_.end())
            return nullptr;
        return it->second.creator(id);
    }

    // get all keys for a given category
    std::vector<std::string> typesForCategory(NodeCategory cat) const
    {
        std::vector<std::string> out;
        for (auto& [key, info] : infos_)
        {
            if (info.category == cat)
                out.push_back(key);
        }
        return out;
    }

  private:
    std::unordered_map<std::string, NodeInfo> infos_;
};

// Macro to simplify registration
// instead of old REGISTER_NODE:
#define REGISTER_NODE(Category, ClassType, key)                                        \
    namespace                                                                          \
    {                                                                                  \
    const bool _reg_##ClassType = []()                                                 \
    {                                                                                  \
        NodeFactory::instance().registerType(                                          \
            key, Category,                                                             \
            [](int id) { return std::make_unique<ClassType>(id, key); });              \
        return true;                                                                   \
    }();                                                                               \
    }


class RenderPreviewNode : public Node
{
  public:
    using Node::Node;

    ~RenderPreviewNode() override;
    // Frame data
    int H = 0, W = 0, C = 0;
    int texW = 0, texH = 0;
    std::vector<uint8_t> rgbBuffer;
    std::atomic<bool> newFrameAvailable{false};

    // UI state
    ImVec2 nodeSize = ImVec2(320, 240);
    ImVec2 previewSize = ImVec2(0, 0);
    bool aspectLocked = true;

    // GL
    GLuint textureID = 0;
    bool allocated = false;
    int lastW = 0, lastH = 0;
    float nodeSizeVal = nodeSize.x; // Or whatever default you want

    // Utility
    inline ImVec2 FitSizeWithAspect(const ImVec2& b, float a)
    {
        ImVec2 s = b;
        float ba = b.x / b.y;
        if (ba > a)
            s.x = b.y * a;
        else
            s.y = b.x / a;
        return s;
    }
    void OnBegin() override
    {
        // one input pin
        AddInput("In", PinType::Tensor);
    }

    void compute() override
    {
        try
        {

            if (Inputs.empty())
                return;
            auto const& v = Inputs[0].Value;
            if (!std::holds_alternative<torch::Tensor>(v))
                return;

            torch::Tensor img = std::get<torch::Tensor>(v);
            if (!img.defined() || img.numel() == 0)
                return;
            // put tensor on cpu)
            // img = img.to(torch::kCPU);
            // normalize to uint8 [0,255], bring to CPU contiguous:
            if (img.scalar_type() == torch::kFloat32 ||
                img.scalar_type() == torch::kFloat64 ||
                img.scalar_type() == torch::kFloat16)
                img = (img * 255.f).clamp(0, 255).to(torch::kUInt8);
            else if (img.scalar_type() != torch::kUInt8)
                img = img.to(torch::kUInt8);

            img = img.flip({0}); // flip height and width

            img = img.contiguous().cpu();

            // first time, initialize size
            H = int(img.size(0));
            W = int(img.size(1));
            C = int(img.size(2)); // channels

            size_t expected = size_t(W) * H * std::max(3, C);
            if (rgbBuffer.size() < expected)
            {
                // resize the buffer to hold the image data
                rgbBuffer.resize(expected);
            }
            std::memcpy(rgbBuffer.data(), img.data_ptr<uint8_t>(), expected);

            // publish new size & flag for UI thread
            texW = W;
            texH = H;
            newFrameAvailable.store(true, std::memory_order_release);
        }
        catch (const c10::Error& e)
        {
            std::cerr << "[RenderPreviewNode] PyTorch error: " << e.what() << "\n";
        }
        catch (const std::exception& e)
        {
            std::cerr << "[RenderPreviewNode] std::exception: " << e.what() << "\n";
        }
        catch (...)
        {
            std::cerr << "[RenderPreviewNode] Unknown exception in compute()\n";
        }
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());

        ImGui::Checkbox("Lock Aspect", &aspectLocked);

        if (ImGui::SliderFloat("Node Size", &nodeSizeVal, 64.0f, 2048.0f, "%.0f"))
        {
            nodeSize = ImVec2(nodeSizeVal, nodeSizeVal); // Keep square aspect ratio
        }



        ImGui::PopID();
    }

    void DrawPreviewFrame()
    {
        // 1) Upload new texture if needed
        if (newFrameAvailable.exchange(false))
        {
            if (!textureID)
            {
                glGenTextures(1, &textureID);
                glBindTexture(GL_TEXTURE_2D, textureID);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
            glBindTexture(GL_TEXTURE_2D, textureID);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

            // reallocate if size changed
            if (!allocated || texW != lastW || texH != lastH)
            {
                GLenum fmt =
                    (rgbBuffer.size() >= size_t(texW) * texH * 3) ? GL_RGB : GL_RED;
                glTexImage2D(GL_TEXTURE_2D, 0, (fmt == GL_RGB ? GL_RGB8 : GL_R8), texW,
                             texH, 0, fmt, GL_UNSIGNED_BYTE, nullptr);
                lastW = texW;
                lastH = texH;
                allocated = true;
            }
            GLenum fmt =
                (rgbBuffer.size() >= size_t(texW) * texH * 3) ? GL_RGB : GL_RED;
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texW, texH, fmt, GL_UNSIGNED_BYTE,
                            rgbBuffer.data());
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        // 2) Compute previewSize
        float aspect = (texH > 0 ? float(texW) / texH : 1.0f);
        previewSize = aspectLocked ? FitSizeWithAspect(nodeSize, aspect) : nodeSize;

        // 3) Reserve space and draw
        ImVec2 topLeft = ImGui::GetCursorScreenPos();
        ImGui::Dummy(previewSize);
        ImGui::SetCursorScreenPos(topLeft);
        if (textureID && allocated)
            ImGui::Image((ImTextureID)(intptr_t)textureID, previewSize, ImVec2(0, 1),
                         ImVec2(1, 0));
        else
            ImGui::TextDisabled("No input tensor");
    }
};

/**
 * @brief Converts image tensor value ranges and optionally data type:
 *
 * Modes:
 *  - Normalize:    uint8 [0..255] → float [0..1]
 *  - Denormalize:  float  [0..1]   → uint8  [0..255]
 *
 * Precision (for Normalize):
 *  - Float16
 *  - Float32
 *
 * Use two of these to bracket your ONNX super-res node.
 */
class NormalizationNode : public Node
{
  public:
    enum class Mode
    {
        Normalize = 0,
        Denormalize = 1
    };
    enum class Precision
    {
        Float16 = 0,
        Float32 = 1
    };

    NormalizationNode(int id, const char* name)
        : Node(id, name), mode(Mode::Normalize), precision(Precision::Float16)
    {
        Inputs.emplace_back(GetNextId(), "In", PinType::Tensor);
        Outputs.emplace_back(GetNextId(), "Out", PinType::Tensor);
    }

    void compute() override
    {
        try
        {

            if (Inputs.empty())
                return;
            auto inTensor = std::get<torch::Tensor>(Inputs[0].Value);
            torch::Tensor outTensor;

            if (mode == Mode::Normalize)
            {
                // cast to selected float type and normalize to [0..1]
                if (precision == Precision::Float16)
                {
                    outTensor = inTensor.to(torch::kFloat16).div(255.0f);
                }
                else
                {
                    outTensor = inTensor.to(torch::kFloat32).div(255.0f);
                }
            }
            else
            {
                // Denormalize: clamp [0..1], scale to [0..255], cast to uint8
                outTensor = inTensor.clamp(0.0f, 1.0f).mul(255.0f).to(torch::kUInt8);
            }

            Outputs[0].Value = outTensor;
        } catch (const c10::Error& e)
        {
            std::cerr << "[NormalizationNode] PyTorch error: " << e.what() << "\n";
        }
        catch (const std::exception& e)
        {
            std::cerr << "[NormalizationNode] std::exception: " << e.what() << "\n";
        }
        catch (...)
        {
            std::cerr << "[NormalizationNode] Unknown exception in compute()\n";
        }
    }

    void PanelUI() override
    {
        ImGui::PushID((void*)ID.AsPointer());
        ImGui::TextUnformatted("🔄 Range Converter");
        ImGui::Spacing();

        // Mode selector
        static constexpr const char* modeNames[] = {"Normalize", "Denormalize"};
        int m = static_cast<int>(mode);
        if (ImGui::Combo("Mode", &m, modeNames, IM_ARRAYSIZE(modeNames)))
            mode = static_cast<Mode>(m);

        // Precision selector (only for Normalize)
        if (mode == Mode::Normalize)
        {
            static constexpr const char* precNames[] = {"Float16", "Float32"};
            int p = static_cast<int>(precision);
            if (ImGui::Combo("Precision", &p, precNames, IM_ARRAYSIZE(precNames)))
                precision = static_cast<Precision>(p);
        }

        ImGui::Separator();
        // Show current I/O types
        if (mode == Mode::Normalize)
        {
            ImGui::Text("Input : uint8/float [0..255]");
            ImGui::Text("Output: %s [0..1]",
                        precision == Precision::Float16 ? "float16" : "float32");
        }
        else
        {
            ImGui::Text("Input : float [0..1]");
            ImGui::Text("Output: uint8 [0..255]");
        }

        ImGui::PopID();
    }

    // Node state
  void serialize(nlohmann::json& j) const override 
    {
        Node::serialize(j);
        j["mode"] = static_cast<int>(mode);
        j["precision"] = static_cast<int>(precision);
    }

  void deserialize(const nlohmann::json& j) override
    {
        Node::deserialize(j);
        if (j.contains("mode"))
            mode = static_cast<Mode>(j["mode"].get<int>());
        if (j.contains("precision"))
            precision = static_cast<Precision>(j["precision"].get<int>());
    }


  private:
    Mode mode;
    Precision precision;
};

// DebugPrintNode: prints a debug message (optional) and a variant value to the console
// then passes the value through unchanged.
// ------------------------------------------------------------
class DebugPrintNode : public Node
{
  public:
    using Node::Node;
    // Show current input value
    std::string display = "";
    void OnBegin() override
    {
        AddInput("Message", PinType::String);
        AddInput("Value", PinType::Variant);
        AddOutput("Out", PinType::Variant);
    }

    void compute() override
    {
        // Pass-through
        Outputs[0].Value = Inputs[1].Value;
    }

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("🔍 Debug Print");
        ImGui::Separator();

        // Message input
        ImGui::PushID("msg");
        char buf[256] = "";
        if (auto strPtr = std::get_if<std::string>(&Inputs[0].Value))
            std::strncpy(buf, strPtr->c_str(), sizeof(buf) - 1);
        if (ImGui::InputText("Message", buf, sizeof(buf)))
            Inputs[0].Value = std::string(buf);
        ImGui::PopID();

        // reuse same visitor logic to format
        display = variantToString(Inputs[1].Value);
        ImGui::Text("Value: %s", display.c_str());

        ImGui::PopID();
    }

    void serialize(nlohmann::json& j) const override
    {
        Node::serialize(j);
    }

    void deserialize(const nlohmann::json& j) override
    {
        Node::deserialize(j);
    }
};

// register it once at app startup:
REGISTER_NODE(NodeCategory::Utility, DelayNode, "Delay");

// Register the node under the Utility category
REGISTER_NODE(NodeCategory::Utility, DebugPrintNode, "Debug");

// Register in your Utility category under the key "Preview"
REGISTER_NODE(NodeCategory::Utility, RenderPreviewNode, "Preview");
REGISTER_NODE(NodeCategory::Utility, OnnxNode, "Onnx");

// ─── Basic (value & simple input) ─────────────────────────
REGISTER_NODE(NodeCategory::Basic, TensorNode, "Tensor");
REGISTER_NODE(NodeCategory::Basic, FloatValueNode, "Float");
REGISTER_NODE(NodeCategory::Basic, IntValueNode, "Int");
REGISTER_NODE(NodeCategory::Basic, BoolValueNode, "Bool");
REGISTER_NODE(NodeCategory::Basic, StringValueNode, "String");

// ─── Arithmetic ───────────────────────────────────────────
REGISTER_NODE(NodeCategory::Arithmetic, AddNode, "Add");
REGISTER_NODE(NodeCategory::Arithmetic, MultiplyNode, "Multiply");
REGISTER_NODE(NodeCategory::Arithmetic, SubtractNode, "Subtract");
REGISTER_NODE(NodeCategory::Arithmetic, DivideNode, "Divide");

// ─── Logic ────────────────────────────────────────────────
REGISTER_NODE(NodeCategory::Logic, CompareNode, "Compare");
REGISTER_NODE(NodeCategory::Logic, WhereNode, "Where");
REGISTER_NODE(NodeCategory::Logic, MaskNode, "Mask");

// ─── TensorOps (shape, layout, reduction, image‐prep) ────
// Layout & shape
REGISTER_NODE(NodeCategory::TensorOps, NormalizationNode, "Normalize");
REGISTER_NODE(NodeCategory::TensorOps, ClampNode, "Clamp");
REGISTER_NODE(NodeCategory::TensorOps, PermuteNode, "Permute");
REGISTER_NODE(NodeCategory::TensorOps, TypeNode, "Type");
REGISTER_NODE(NodeCategory::TensorOps, SqueezeNode, "Squeeze");
REGISTER_NODE(NodeCategory::TensorOps, UnsqueezeNode, "Unsqueeze");
REGISTER_NODE(NodeCategory::TensorOps, ReshapeNode, "Reshape");
REGISTER_NODE(NodeCategory::TensorOps, TransposeNode, "Transpose");
REGISTER_NODE(NodeCategory::TensorOps, ConcatNode, "Concat");
REGISTER_NODE(NodeCategory::TensorOps, ContiguousNode, "Contiguous");
REGISTER_NODE(NodeCategory::TensorOps, FlattenNode, "Flatten");
REGISTER_NODE(NodeCategory::TensorOps, SplitNode, "Split");
REGISTER_NODE(NodeCategory::TensorOps, StackNode, "Stack");
REGISTER_NODE(NodeCategory::TensorOps, SliceNode, "Slice");
// Repeat node
REGISTER_NODE(NodeCategory::TensorOps, RepeatNode, "Repeat");
// Reductions
REGISTER_NODE(NodeCategory::TensorOps, SumNode, "Sum");
REGISTER_NODE(NodeCategory::TensorOps, MeanNode, "Mean");
REGISTER_NODE(NodeCategory::TensorOps, MaxNode, "Max");
REGISTER_NODE(NodeCategory::TensorOps, MinNode, "Min");
REGISTER_NODE(NodeCategory::TensorOps, ArgmaxNode, "Argmax");
// Image ops
REGISTER_NODE(NodeCategory::TensorOps, PadNode, "Pad");
REGISTER_NODE(NodeCategory::TensorOps, CropNode, "Crop");
REGISTER_NODE(NodeCategory::TensorOps, ResizeNode, "Resize");

// ─── Utility ───────────────────────────────────────────────
REGISTER_NODE(NodeCategory::Utility, VideoReaderNode, "Read Video");
REGISTER_NODE(NodeCategory::Utility, LoadImageNode, "Load Image");
REGISTER_NODE(NodeCategory::Utility, EncoderNode, "Export Video");

#pragma endregion

RenderPreviewNode::~RenderPreviewNode()
{
    if (textureID)
        glDeleteTextures(1, &textureID);
}

#pragma region APPLICATION
struct Example : public Application
{
    using Application::Application;

    enum class EvalMode
    {
        Live,  /* runs inference on all nodes (if dirty) as frames are played
                  Run/Stop Button (will adjust) controls if it is PLAYING (IE looping) or
                  not.  if stop, can mess with single frame, and edit other params to see
                  changes on single frame
                  */
        Basic, // runs only VideoReader and Preview nodes, others pass data through
               // run/stop button controls if it is PLAYING (IE looping) or not.

    };

    EvalMode g_EvalMode = EvalMode::Basic;

    // Active VideoReader (index into vector of nodes)
    int g_ActiveVideoIdx = 0;

    // Topo‐sorted compute order
    std::vector<Node*> topoOrder;

    // Latest frame produced by the pipeline
    std::mutex frameMutex;

    const int m_PinIconSize = 24;
    std::vector<std::unique_ptr<Node>> m_Nodes;
    bool graphDirty = true;
    std::vector<Link> m_Links;
    ImTextureID m_HeaderBackground = nullptr;
    ImTextureID m_SaveIcon = nullptr;
    ImTextureID m_RestoreIcon = nullptr;
    const float m_TouchTime = 1.0f;
    std::map<ed::NodeId, float, NodeIdLess> m_NodeTouchTime;
    bool m_ShowOrdinals = false;
    bool pipelineRunning = false;
    ed::NodeId renamingNode = 0;
    std::map<ed::NodeId, std::vector<char>, NodeIdLess> renameBuffers;

    std::unordered_map<ed::PinId, int> pinToNodeIndex;
    // in Example’s private section

    std::unique_ptr<std::atomic<int>[]> indegree;
    std::vector<std::thread> graphWorkers;
    std::atomic<bool> stopWorkers{false};
    std::mutex graphMutex;
    std::condition_variable graphCv;
    bool workReady = false;

    // barrier to let ComputeAll() wait for all N nodes
    std::mutex barrierMutex;
    std::condition_variable barrierCv;

    // === STATIC TOPOLOGY ===
    std::vector<int> baseIndegree;
    std::vector<std::vector<int>> adj;
    // instead of vector<vector<ed::LinkId>>
    std::vector<std::vector<Link*>> outEdges;

    // === PER‐RUN COUNTERS ===
    int N{0};
    std::queue<int> readyQueue;
    std::atomic<int> processedCount{0};

    std::atomic<bool> done;

    std::string m_CurrentFilePath; // full path, empty → untitled
    // ed::NodeId GetNextNodeId()
    //{
    //     return ed::NodeId(GetNextId());
    // }
    void DrawGlobalPlaybackControls()
    {
        // Find all VideoReader nodes
        std::vector<VideoReaderNode*> videos;
        for (auto& n : m_Nodes)
            if (auto* vr = dynamic_cast<VideoReaderNode*>(n.get()))
                videos.push_back(vr);

        if (videos.empty())
        {
            ImGui::TextDisabled("No VideoReader node in graph.");
            return;
        }

        std::vector<const char*> videoLabels;
        for (auto* vr : videos)
            videoLabels.push_back(vr->videoPath.empty() ? "(unnamed)"
                                                        : vr->videoPath.c_str());

        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f);

        // --- Row 1: Mode and Run Inference ---
        ImGui::BeginChild("Row1_ModeRun", ImVec2(0, 32), false);
        {
            ImGui::SetNextItemWidth(120.f);
            const char* evalModes[] = {"Live", "Basic"};
            int modeIdx = int(g_EvalMode);
            ImGui::Combo("##mode", &modeIdx, evalModes, IM_ARRAYSIZE(evalModes));
            g_EvalMode = EvalMode(modeIdx);

            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip(
                    "Playback Mode:\n"
                    "  • Live: All nodes run full inference (if dirty) while playing "
                    "or scrubbing. Use this for full, accurate pipeline results.\n"
                    "  • Basic: Only VideoReader and Preview nodes compute; all others "
                    "simply pass data through. Fast for previewing source/outputs "
                    "only.");
            }

            ImGui::SameLine(0, 10);

            if (ImGui::Button(playRequested ? "Stop" : "Run Inference"))
            {
                {
                    std::lock_guard<std::mutex> lk(playMutex);
                    playRequested = !playRequested;
                }
                playCv.notify_all();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Run or stop the full inference pipeline for this video.");
        }
        ImGui::EndChild();

        // --- Row 2: Video Dropdown ---
        ImGui::BeginChild("Row2_ActiveVideo", ImVec2(0, 28), false);
        {
            ImGui::SetNextItemWidth(240.f);
            ImGui::Combo("##activevideo", &g_ActiveVideoIdx, videoLabels.data(),
                         int(videoLabels.size()));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Select which VideoReader node to control.");
        }
        ImGui::EndChild();

        auto* g_ActiveVideo = videos[g_ActiveVideoIdx];
        if (!g_ActiveVideo)
        {
            ImGui::TextDisabled("No active video selected.");
            ImGui::PopStyleVar();
            return;
        }

        // --- Row 3: Timeline Slider + Reset ---
        ImGui::BeginChild("Row3_SliderReset", ImVec2(0, 32), false);
        {
            float cur =
                g_ActiveVideo->pipeline ? g_ActiveVideo->pipeline->currentTime() : 0.0f;
            double dur =
                g_ActiveVideo->pipeline ? g_ActiveVideo->pipeline->duration() : 1.0;
            bool changed = false;
            ImGui::SetNextItemWidth(140.f);

            changed = ImGui::SliderFloat("##time", &cur, 0.0f, (float)dur, "%.3f s",
                                         ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Scrub or seek through the video.");

            ImGui::SameLine(0, 10);

            if (ImGui::Button("⏮##reset", ImVec2(34, 0)))
            {
                {
                    std::lock_guard<std::mutex> lk(playMutex);
                }
                playCv.notify_all();
                g_ActiveVideo->pipeline->seek(0.0);
                ComputeAll();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Reset video to start.");

            if (changed && g_ActiveVideo->pipeline)
            {
                if (g_ActiveVideo->pipeline)
                {
                    std::lock_guard<std::mutex> lk(playMutex);

                    playRequested = false;
                    scrubbing = true;
                }
                playCv.notify_all();
                g_ActiveVideo->pipeline->seek(cur);
                g_ActiveVideo->SetDirty();
            }
        }

        ImGui::EndChild();

        ImGui::PopStyleVar();
    }



    ed::LinkId GetNextLinkId()
    {
        return ed::LinkId(GetNextId());
    }

    void TouchNode(ed::NodeId id)
    {
        m_NodeTouchTime[id] = m_TouchTime;
    }

    float GetTouchProgress(ed::NodeId id)
    {
        auto it = m_NodeTouchTime.find(id);
        if (it != m_NodeTouchTime.end() && it->second > 0.0f)
            return (m_TouchTime - it->second) / m_TouchTime;
        else
            return 0.0f;
    }

    void UpdateTouch()
    {
        const auto deltaTime = ImGui::GetIO().DeltaTime;
        for (auto& entry : m_NodeTouchTime)
        {
            if (entry.second > 0.0f)
                entry.second -= deltaTime;
        }
    }

    Node* FindNode(ed::NodeId id)
    {
        for (auto& node : m_Nodes)
            if (node->ID == id)
                return node.get(); // <-- Returns a raw pointer, does not move ownership
        return nullptr;
    }

    Link* FindLink(ed::LinkId id)
    {
        for (auto& link : m_Links)
            if (link.ID == id)
                return &link;

        return nullptr;
    }

    Pin* FindPin(ed::PinId id)
    {
        if (!id)
            return nullptr;

        for (auto& node : m_Nodes)
        {
            for (auto& pin : node->Inputs)
                if (pin.ID == id)
                    return &pin;

            for (auto& pin : node->Outputs)
                if (pin.ID == id)
                    return &pin;
        }

        return nullptr;
    }

    bool IsPinLinked(ed::PinId id)
    {
        if (!id)
            return false;

        for (auto& link : m_Links)
            if (link.StartPinID == id || link.EndPinID == id)
                return true;

        return false;
    }

    bool CanCreateLink(Pin* a, Pin* b)
    {
        if (!a || !b || a == b || a->Kind == b->Kind || a->Type != b->Type ||
            a->Node == b->Node)
            return false;

        return true;
    }

    // void DrawItemRect(ImColor color, float expand = 0.0f)
    //{
    //     ImGui::GetWindowDrawList()->AddRect(
    //         ImGui::GetItemRectMin() - ImVec2(expand, expand),
    //         ImGui::GetItemRectMax() + ImVec2(expand, expand),
    //         color);
    // };

    // void FillItemRect(ImColor color, float expand = 0.0f, float rounding = 0.0f)
    //{
    //     ImGui::GetWindowDrawList()->AddRectFilled(
    //         ImGui::GetItemRectMin() - ImVec2(expand, expand),
    //         ImGui::GetItemRectMax() + ImVec2(expand, expand),
    //         color, rounding);
    // };

    void BuildNode(Node* node)
    {
        for (auto& input : node->Inputs)
        {
            input.Node = node;
            input.Kind = PinKind::Input;
        }

        for (auto& output : node->Outputs)
        {
            output.Node = node;
            output.Kind = PinKind::Output;
        }
    }

    /*    Node* SpawnTensorInputNode()
    {
        auto node = std::make_unique<TensorNode>(GetNextId(), "Tensor",
                                                 ImColor(120, 210, 110));

        node->Outputs.emplace_back(GetNextId(), "Tensor", PinType::Tensor, true);

        BuildNode(node.get());

        Node* out = node.get(); // raw, non-owning handle
        m_Nodes.push_back(std::move(node));
        return out;
    }*/

    Node* CreateNode(const std::string& type)
    {
        auto node = NodeFactory::instance().create(type, GetNextId());
        if (!node)
            return nullptr;
        node->OnBegin();

        BuildNode(node.get());

        Node* out = node.get(); // raw, non-owning handle
        m_Nodes.push_back(std::move(node));
        graphDirty = true; // mark graph as dirty
        return out;
    }

    void BuildNodes()
    {
        for (auto& node : m_Nodes)
            BuildNode(node.get());
    }

    void OnStart() override
    {
        m_Editor = ed::CreateEditor();
        ed::SetCurrentEditor(m_Editor);
        ed::EnableShortcuts(true);
        // … in your Example::OnStart() or constructor, after the window is created:
        m_CurrentFilePath.clear(); // no file yet
        UpdateWindowTitle();
        BuildNodes();
        rebuildStructure(); // Build the initial graph structure
        graphDirty = false;
        // Launch the worker
        StartProcessingThread();
        m_HeaderBackground = LoadTexture("data/BlueprintBackground.png");
        m_SaveIcon = LoadTexture("data/ic_save_white_24dp.png");
        m_RestoreIcon = LoadTexture("data/ic_restore_white_24dp.png");

        // auto& io = ImGui::GetIO();
    }
    void Example::UpdateWindowTitle()
    {
        namespace fs = std::filesystem;

        // figure out “basename” or fall back to “Untitled Graph”
        std::string base;
        if (m_CurrentFilePath.empty())
            base = "Untitled Graph";
        else
            base = fs::path(m_CurrentFilePath).stem().string();

        // build your full title
        std::string title = base + "  MageML GUI";

        SetTitle(title.c_str());
    }
    void OnStop() override
    {
        auto releaseTexture = [this](ImTextureID& id)
        {
            if (id)
            {
                DestroyTexture(id);
                id = nullptr;
            }
        };

        releaseTexture(m_RestoreIcon);
        releaseTexture(m_SaveIcon);
        releaseTexture(m_HeaderBackground);
        StopProcessingThread();
        if (m_Editor)
        {
            ed::DestroyEditor(m_Editor);
            m_Editor = nullptr;
        }
    }

    ImColor GetIconColor(PinType type)
    {
        switch (type)
        {
        default:
        case PinType::Flow:
            return ImColor(255, 255, 255);
        case PinType::Bool:
            return ImColor(220, 48, 48);
        case PinType::Int:
            return ImColor(68, 201, 156);
        case PinType::Float:
            return ImColor(147, 226, 74);
        case PinType::String:
            return ImColor(124, 21, 153);
        case PinType::Object:
            return ImColor(51, 150, 215);
        case PinType::Function:
            return ImColor(218, 0, 183);
        case PinType::Delegate:
            return ImColor(255, 48, 48);
        }
    };

    void DrawPinIcon(const Pin& pin, bool connected, int alpha)
    {
        IconType iconType;
        ImColor color = GetIconColor(pin.Type);
        color.Value.w = alpha / 255.0f;
        switch (pin.Type)
        {
        case PinType::Flow:
            iconType = IconType::Flow;
            break;
        case PinType::Bool:
            iconType = IconType::Circle;
            break;
        case PinType::Int:
            iconType = IconType::Circle;
            break;
        case PinType::Float:
            iconType = IconType::Circle;
            break;
        case PinType::String:
            iconType = IconType::Circle;
            break;
        case PinType::Object:
            iconType = IconType::Circle;
            break;
        case PinType::Function:
            iconType = IconType::Circle;
            break;
        case PinType::Delegate:
            iconType = IconType::Square;
            break;
        case PinType::Tensor:
            iconType = IconType::Grid;
            break;
        case PinType::Variant:
            iconType = IconType::Diamond;
            break;
        default:
            return;
        }

        ax::Widgets::Icon(ImVec2(static_cast<float>(m_PinIconSize),
                                 static_cast<float>(m_PinIconSize)),
                          iconType, connected, color, ImColor(32, 32, 32, alpha));
    };

    void ShowStyleEditor(bool* show = nullptr)
    {
        if (!ImGui::Begin("Style", show))
        {
            ImGui::End();
            return;
        }

        auto paneWidth = ImGui::GetContentRegionAvail().x;

        auto& editorStyle = ed::GetStyle();

        ImGui::BeginHorizontal("Style buttons", ImVec2(paneWidth, 0), 1.0f);
        ImGui::TextUnformatted("Values");
        ImGui::Spring();
        if (ImGui::Button("Reset to defaults"))
            editorStyle = ed::Style();
        ImGui::EndHorizontal();
        ImGui::Spacing();
        ImGui::DragFloat4("Node Padding", &editorStyle.NodePadding.x, 0.1f, 0.0f,
                          40.0f);
        ImGui::DragFloat("Node Rounding", &editorStyle.NodeRounding, 0.1f, 0.0f, 40.0f);
        ImGui::DragFloat("Node Border Width", &editorStyle.NodeBorderWidth, 0.1f, 0.0f,
                         15.0f);
        ImGui::DragFloat("Hovered Node Border Width",
                         &editorStyle.HoveredNodeBorderWidth, 0.1f, 0.0f, 15.0f);
        ImGui::DragFloat("Hovered Node Border Offset",
                         &editorStyle.HoverNodeBorderOffset, 0.1f, -40.0f, 40.0f);
        ImGui::DragFloat("Selected Node Border Width",
                         &editorStyle.SelectedNodeBorderWidth, 0.1f, 0.0f, 15.0f);
        ImGui::DragFloat("Selected Node Border Offset",
                         &editorStyle.SelectedNodeBorderOffset, 0.1f, -40.0f, 40.0f);
        ImGui::DragFloat("Pin Rounding", &editorStyle.PinRounding, 0.1f, 0.0f, 40.0f);
        ImGui::DragFloat("Pin Border Width", &editorStyle.PinBorderWidth, 0.1f, 0.0f,
                         15.0f);
        ImGui::DragFloat("Link Strength", &editorStyle.LinkStrength, 1.0f, 0.0f,
                         500.0f);
        // ImVec2  SourceDirection;
        // ImVec2  TargetDirection;
        ImGui::DragFloat("Scroll Duration", &editorStyle.ScrollDuration, 0.001f, 0.0f,
                         2.0f);
        ImGui::DragFloat("Flow Marker Distance", &editorStyle.FlowMarkerDistance, 1.0f,
                         1.0f, 200.0f);
        ImGui::DragFloat("Flow Speed", &editorStyle.FlowSpeed, 1.0f, 1.0f, 2000.0f);
        ImGui::DragFloat("Flow Duration", &editorStyle.FlowDuration, 0.001f, 0.0f,
                         5.0f);
        // ImVec2  PivotAlignment;
        // ImVec2  PivotSize;
        // ImVec2  PivotScale;
        // float   PinCorners;
        // float   PinRadius;
        // float   PinArrowSize;
        // float   PinArrowWidth;
        ImGui::DragFloat("Group Rounding", &editorStyle.GroupRounding, 0.1f, 0.0f,
                         40.0f);
        ImGui::DragFloat("Group Border Width", &editorStyle.GroupBorderWidth, 0.1f,
                         0.0f, 15.0f);

        ImGui::Separator();

        static ImGuiColorEditFlags edit_mode = ImGuiColorEditFlags_DisplayRGB;
        ImGui::BeginHorizontal("Color Mode", ImVec2(paneWidth, 0), 1.0f);
        ImGui::TextUnformatted("Filter Colors");
        ImGui::Spring();
        ImGui::RadioButton("RGB", &edit_mode, ImGuiColorEditFlags_DisplayRGB);
        ImGui::Spring(0);
        ImGui::RadioButton("HSV", &edit_mode, ImGuiColorEditFlags_DisplayHSV);
        ImGui::Spring(0);
        ImGui::RadioButton("HEX", &edit_mode, ImGuiColorEditFlags_DisplayHex);
        ImGui::EndHorizontal();

        static ImGuiTextFilter filter;
        filter.Draw("##filter", paneWidth);

        ImGui::Spacing();

        ImGui::PushItemWidth(-160);
        for (int i = 0; i < ed::StyleColor_Count; ++i)
        {
            auto name = ed::GetStyleColorName((ed::StyleColor)i);
            if (!filter.PassFilter(name))
                continue;

            ImGui::ColorEdit4(name, &editorStyle.Colors[i].x, edit_mode);
        }
        ImGui::PopItemWidth();

        ImGui::End();
    }

    void ShowLeftPane(float paneWidth)
    {
        auto& io = ImGui::GetIO();

        ImGui::BeginChild("SelectionPane", ImVec2(paneWidth, 0), true);

        // ─── Top controls (unchanged) ─────────────────────────────────
        {
            ImGui::BeginHorizontal("PaneTop", ImVec2(paneWidth, 0));
            ImGui::Spring(0.0f, 0.0f);
            if (ImGui::Button("Zoom to Content"))
                ed::NavigateToContent();
            ImGui::Spring(0.0f);
            if (ImGui::Button("Show Flow"))
                for (auto& l : m_Links)
                    ed::Flow(l.ID);
            ImGui::Spring();
            ImGui::EndHorizontal();
            ImGui::Separator();
        }

        ImGui::BeginChild("Playback Controls", ImVec2(paneWidth, 100), true);
        DrawGlobalPlaybackControls();
        ImGui::EndChild();


        ImGui::BeginChild("NodeList", ImVec2(0, 200), true,
                          ImGuiWindowFlags_HorizontalScrollbar);
        for (auto& node : m_Nodes)
        {
            ImGui::PushID(node->ID.AsPointer());
            bool isSel = ed::IsNodeSelected(node->ID);
            if (ImGui::Selectable(node->Name.c_str(), isSel))
            {
                if (io.KeyCtrl)
                {
                    if (isSel)
                        ed::DeselectNode(node->ID);
                    else
                        ed::SelectNode(node->ID, true);
                }
                else
                    ed::SelectNode(node->ID, false);
                ed::NavigateToSelection();
            }
            ImGui::PopID();
        }
        ImGui::EndChild();

        ImGui::Separator();
        ImGui::Spacing();

        // ─── Detail panel for a single node ───────────────
        {
            int totalSelected = ed::GetSelectedObjectCount();
            std::vector<ed::NodeId> selectedNodes(totalSelected);
            int nodeCount = ed::GetSelectedNodes(selectedNodes.data(), totalSelected);
            selectedNodes.resize(nodeCount);

            if (nodeCount == 1)
            {
                // find the Node
                ed::NodeId selId = selectedNodes[0];
                auto it = std::find_if(m_Nodes.begin(), m_Nodes.end(),
                                       [&](const std::unique_ptr<Node>& n)
                                       {
                                           return n && n->ID == selId; // note n->ID
                                       });

                if (it != m_Nodes.end())
                {

                        it->get()->PanelUI(); // call the node's UI method
                    
                }
            }
        }
        ImGui::EndChild();
    }

    static int RenameCallback(ImGuiInputTextCallbackData* data)
    {
        if (data->EventFlag == ImGuiInputTextFlags_CallbackResize)
        {
            auto& buf = *static_cast<std::vector<char>*>(data->UserData);
            buf.resize(data->BufTextLen + 1);
            data->Buf = buf.data();
        }
        return 0;
    }

#include <nlohmann/json.hpp>
    using json = nlohmann::json;

void Example::SaveGraph(const char* path)
    {
        using json = nlohmann::json;

        json root;
        root["nodes"] = json::array();
        root["links"] = json::array();

        // --- Serialize nodes ---
        for (auto& up : m_Nodes)
        {
            Node* n = up.get();

            // 1) basic placement & identity
            ImVec2 pos = ed::GetNodePosition(n->ID);
            ImVec2 sz = ed::GetNodeSize(n->ID);

            json nj;
            nj["id"] = (uintptr_t)n->ID.AsPointer();
            nj["type"] = n->Name;
            nj["pos"] = {pos.x, pos.y};
            nj["size"] = {sz.x, sz.y};

            // 2) let base+subclass write their own fields
            n->serialize(nj);

            // 3) pins & scalar values (unchanged)
            json inPins = json::array(), inVals = json::array(),
                 outPins = json::array(), outVals = json::array();

            for (auto& pin : n->Inputs)
            {
                inPins.push_back((uintptr_t)pin.ID.AsPointer());
                switch (pin.Type)
                {
                case PinType::Bool:
                    inVals.push_back(std::get<bool>(pin.Value));
                    break;
                case PinType::Int:
                    inVals.push_back(std::get<int>(pin.Value));
                    break;
                case PinType::Float:
                    inVals.push_back(std::get<float>(pin.Value));
                    break;
                case PinType::String:
                    inVals.push_back(std::get<std::string>(pin.Value));
                    break;
                default:
                    inVals.push_back(nullptr);
                    break;
                }
            }
            for (auto& pin : n->Outputs)
            {
                outPins.push_back((uintptr_t)pin.ID.AsPointer());
                switch (pin.Type)
                {
                case PinType::Bool:
                    outVals.push_back(std::get<bool>(pin.Value));
                    break;
                case PinType::Int:
                    outVals.push_back(std::get<int>(pin.Value));
                    break;
                case PinType::Float:
                    outVals.push_back(std::get<float>(pin.Value));
                    break;
                case PinType::String:
                    outVals.push_back(std::get<std::string>(pin.Value));
                    break;
                default:
                    outVals.push_back(nullptr);
                    break;
                }
            }

            nj["inPins"] = std::move(inPins);
            nj["inVals"] = std::move(inVals);
            nj["outPins"] = std::move(outPins);
            nj["outVals"] = std::move(outVals);

            root["nodes"].push_back(std::move(nj));
        }

        // --- Serialize links (unchanged) ---
        for (auto& link : m_Links)
        {
            json lj;
            lj["id"] = (uintptr_t)link.ID.AsPointer();
            lj["start"] = (uintptr_t)link.StartPinID.AsPointer();
            lj["end"] = (uintptr_t)link.EndPinID.AsPointer();
            root["links"].push_back(std::move(lj));
        }
        m_CurrentFilePath = path;
        // --- Write file ---
        std::ofstream ofs(path);
        ofs << std::setw(4) << root << "\n";

        UpdateWindowTitle();
    }

void Example::LoadGraph(const char* path)
    {
        using json = nlohmann::json;
        std::ifstream ifs(path);
        if (!ifs.is_open())
            return;
        m_CurrentFilePath = path;
        UpdateWindowTitle();
        json root;
        ifs >> root;

        m_Nodes.clear();
        m_Links.clear();

        // maps old pointer-IDs → new ed:: IDs
        std::unordered_map<uintptr_t, ed::NodeId> nodeMap;
        std::unordered_map<uintptr_t, ed::PinId> pinMap;

        // 1) Recreate every node
        for (auto const& nj : root["nodes"])
        {
            // read back id & type so we can hand them off to deserialize()
            uintptr_t oldId = nj["id"].get<uintptr_t>();
            std::string type = nj["type"].get<std::string>();

            // create and let subclass restore its own fields + base pins/state
            auto n = CreateNode(type);
            n->deserialize(nj);

            // restore editor position & size
            float px = nj["pos"][0].get<float>();
            float py = nj["pos"][1].get<float>();
            float sx = nj["size"][0].get<float>();
            float sy = nj["size"][1].get<float>();
            ed::SetNodePosition(n->ID, {px, py});
            ed::SetGroupSize(n->ID, {sx, sy});

            // build pinMap: map every old-pin-pointer → new PinId
            auto const& inPins = nj["inPins"];
            auto const& outPins = nj["outPins"];
            for (size_t i = 0; i < inPins.size() && i < n->Inputs.size(); ++i)
            {
                pinMap[inPins[i].get<uintptr_t>()] = n->Inputs[i].ID;
            }
            for (size_t i = 0; i < outPins.size() && i < n->Outputs.size(); ++i)
            {
                pinMap[outPins[i].get<uintptr_t>()] = n->Outputs[i].ID;
            }

            nodeMap[oldId] = n->ID;
        }

        // 2) Rebuild links using our pinMap
        for (auto const& lj : root["links"])
        {
            uintptr_t oldStart = lj["start"].get<uintptr_t>();
            uintptr_t oldEnd = lj["end"].get<uintptr_t>();
            if (pinMap.find(oldStart) == pinMap.end() ||
                pinMap.find(oldEnd) == pinMap.end())
            {
                std::cerr << "[LoadGraph] Pin missing! Start: " << std::hex << oldStart
                          << " in pinMap? " << (pinMap.find(oldStart) != pinMap.end())
                          << " | End: " << oldEnd << " in pinMap? "
                          << (pinMap.find(oldEnd) != pinMap.end()) << std::endl;
                continue;
            }

            ed::PinId newStart = pinMap.at(oldStart);
            ed::PinId newEnd = pinMap.at(oldEnd);
            ed::LinkId newLink = GetNextLinkId();
            m_Links.emplace_back(newLink, newStart, newEnd);
        }


        graphDirty = true;

        BuildNodes();

        rebuildStructure(); 

        int maxId = m_NextId;
        for (auto& up : m_Nodes)
        {
            maxId = std::max(maxId, (int)up->ID.AsPointer());
            for (auto& p : up->Inputs)
                maxId = std::max(maxId, (int)p.ID.AsPointer());
            for (auto& p : up->Outputs)
                maxId = std::max(maxId, (int)p.ID.AsPointer());
        }
        for (auto& l : m_Links)
            maxId = std::max(maxId, (int)l.ID.AsPointer());
        m_NextId = maxId + 1;
    }

void Example::CopySelectionToClipboard()
    {
        using json = nlohmann::json;

        json clip;
        clip["nodes"] = json::array();
        clip["links"] = json::array();

        // 1a) serialize each selected node
        for (auto& up : m_Nodes)
        {
            Node* n = up.get();
            if (!ed::IsNodeSelected(n->ID))
                continue;

            json nj;
            nj["id"] = (uintptr_t)n->ID.AsPointer();
            nj["type"] = n->Name;
            nj["state"] = n->State;
            ImVec2 p = ed::GetNodePosition(n->ID);
            nj["pos"] = {p.x, p.y};

            // --- scalar pin values (just like SaveGraph) ---
            auto packPins = [&](auto& pins, const char* prefix)
            {
                json arrPins = json::array(), arrVals = json::array();
                for (auto& pin : pins)
                {
                    arrPins.push_back((uintptr_t)pin.ID.AsPointer());
                    switch (pin.Type)
                    {
                    case PinType::Bool:
                        arrVals.push_back(std::get<bool>(pin.Value));
                        break;
                    case PinType::Int:
                        arrVals.push_back(std::get<int>(pin.Value));
                        break;
                    case PinType::Float:
                        arrVals.push_back(std::get<float>(pin.Value));
                        break;
                    case PinType::String:
                        arrVals.push_back(std::get<std::string>(pin.Value));
                        break;
                    default:
                        arrVals.push_back(nullptr);
                        break;
                    }
                }
                nj[std::string(prefix) + "Pins"] = std::move(arrPins);
                nj[std::string(prefix) + "Vals"] = std::move(arrVals);
            };
            packPins(n->Inputs, "in");
            packPins(n->Outputs, "out");

            clip["nodes"].push_back(nj);
        }

        // 1b) serialize every link *between* selected nodes
        for (auto& link : m_Links)
        {
            // only copy links whose start+end nodes are both selected
            ed::PinId sPin = link.StartPinID, ePin = link.EndPinID;
            ed::NodeId sNode, eNode;
            // look up the node for each pin:
            for (auto& up : m_Nodes)
            {
                Node* n = up.get();
                for (auto& p : n->Inputs)
                    if (p.ID == sPin)
                        sNode = n->ID;
                for (auto& p : n->Outputs)
                    if (p.ID == sPin)
                        sNode = n->ID;
                for (auto& p : n->Inputs)
                    if (p.ID == ePin)
                        eNode = n->ID;
                for (auto& p : n->Outputs)
                    if (p.ID == ePin)
                        eNode = n->ID;
            }
            if (!ed::IsNodeSelected(sNode) || !ed::IsNodeSelected(eNode))
                continue;

            json lj;
            lj["start"] = (uintptr_t)sPin.AsPointer();
            lj["end"] = (uintptr_t)ePin.AsPointer();
            clip["links"].push_back(lj);
        }

        // 2) dump to string & shove into clipboard
        std::string s = clip.dump();
        ImGui::SetClipboardText(s.c_str());
    }

void StopIfRunning()
    {
        if (pipelineRunning)
        {
            std::lock_guard<std::mutex> lk(playMutex);
            playRequested = false;
            playCv.notify_all();
        }
    }
    // 2) Paste from clipboard
    void Example::PasteClipboardAt(const ImVec2& mousePos)
    {
        using json = nlohmann::json;

        // 1) Grab raw clipboard and bail early if empty
        const char* raw = ImGui::GetClipboardText();
        if (!raw || !*raw)
            return;
        std::string txt(raw);

        // 2) Quick sanity‐check: must at least look like JSON object/array
        //    (optional but very cheap)
        char first = txt.front(), last = txt.back();
        if ((first != '{' || last != '}') && (first != '[' || last != ']'))
            return;

        // 3) Validate without throwing
        if (!json::accept(txt))
            return;

        // 4) Now safe to parse
        json clip;
        try
        {
            clip = json::parse(txt);
        }
        catch (const json::parse_error& e)
        {
            // invalid JSON after all—ignore it
            return;
        }

        // 5) Compute average position of the copied nodes
        ImVec2 avg{0, 0};
        int count = int(clip["nodes"].size());
        for (auto& nj : clip["nodes"])
        {
            avg.x += nj["pos"][0].get<float>();
            avg.y += nj["pos"][1].get<float>();
        }
        if (count > 0)
        {
            avg.x /= count;
            avg.y /= count;
        }
        ImVec2 offset = mousePos - avg;

        // 6) Recreate nodes & pins
        std::unordered_map<uintptr_t, ed::NodeId> nodeMap;
        std::unordered_map<uintptr_t, ed::PinId> pinMap;

        for (auto& nj : clip["nodes"])
        {
            uintptr_t oldId = nj["id"].get<uintptr_t>();
            std::string type = nj["type"].get<std::string>();
            std::string state = nj["state"].get<std::string>();

            ImVec2 p{nj["pos"][0], nj["pos"][1]};
            p.x += offset.x;
            p.y += offset.y;

            Node* n = CreateNode(type);
            n->State = state;
            ed::SetNodePosition(n->ID, p);

            // restore scalar pins
            auto unpackPins = [&](auto& pins, const char* prefix)
            {
                auto& arrPins = nj[std::string(prefix) + "Pins"];
                auto& arrVals = nj[std::string(prefix) + "Vals"];
                for (size_t i = 0; i < pins.size() && i < arrPins.size(); ++i)
                {
                    uintptr_t oldPin = arrPins[i].get<uintptr_t>();
                    pinMap[oldPin] = pins[i].ID;
                    auto& v = arrVals[i];
                    switch (pins[i].Type)
                    {
                    case PinType::Bool:
                        if (!v.is_null())
                            pins[i].Value = v.get<bool>();
                        break;
                    case PinType::Int:
                        if (!v.is_null())
                            pins[i].Value = v.get<int>();
                        break;
                    case PinType::Float:
                        if (!v.is_null())
                            pins[i].Value = v.get<float>();
                        break;
                    case PinType::String:
                        if (!v.is_null())
                            pins[i].Value = v.get<std::string>();
                        break;
                    default:
                        break;
                    }
                }
            };
            unpackPins(n->Inputs, "in");
            unpackPins(n->Outputs, "out");

            nodeMap[oldId] = n->ID;
        }

        // 7) Recreate links
        for (auto& lj : clip["links"])
        {
            uintptr_t oldStart = lj["start"].get<uintptr_t>();
            uintptr_t oldEnd = lj["end"].get<uintptr_t>();
            ed::PinId s = pinMap.at(oldStart);
            ed::PinId e = pinMap.at(oldEnd);
            ed::LinkId id = GetNextLinkId();
            m_Links.emplace_back(id, s, e);
        }

        // 8) Mark graph dirty so OnFrame() will rebuild
        graphDirty = true;
    }

    void RemoveLinksForNode(Node* node)
    {
        // Gather all PinIDs from this node
        std::unordered_set<ed::PinId> pins;
        for (const auto& p : node->Inputs)
            pins.insert(p.ID);
        for (const auto& p : node->Outputs)
            pins.insert(p.ID);

        // Remove any link with a matching StartPinID or EndPinID
        m_Links.erase(
            std::remove_if(
                m_Links.begin(), m_Links.end(), [&](const Link& l)
                { return pins.count(l.StartPinID) || pins.count(l.EndPinID); }),
            m_Links.end());
    }



    void OnFrame(float deltaTime) override
    {
        UpdateTouch();

        auto& io = ImGui::GetIO();

        ImGui::Text("FPS: %.2f (%.2gms)", io.Framerate,
                    io.Framerate ? 1000.0f / io.Framerate : 0.0f);

        ed::SetCurrentEditor(m_Editor);
        // ––– at top of your Application class or cpp file –––––––––––––
        static bool openFileDialog = false;
        static bool saveFileDialog = false;
        static char filePathBuf[256] = ""; // reuse for both open/save

        // auto& style = ImGui::GetStyle();

#if 0
        {
            for (auto x = -io.DisplaySize.y; x < io.DisplaySize.x; x += 10.0f)
            {
                ImGui::GetWindowDrawList()->AddLine(ImVec2(x, 0), ImVec2(x + io.DisplaySize.y, io.DisplaySize.y),
                    IM_COL32(255, 255, 0, 255));
            }
        }
#endif

        static ed::NodeId contextNodeId = 0;
        static ed::LinkId contextLinkId = 0;
        static ed::PinId contextPinId = 0;
        static bool createNewNode = false;
        static Pin* newNodeLinkPin = nullptr;
        static Pin* newLinkPin = nullptr;

        static float leftPaneWidth = 300.0f;
        static float rightPaneWidth = 900.0f;
        Splitter(true, 4.0f, &leftPaneWidth, &rightPaneWidth, 50.0f, 50.0f);

        ShowLeftPane(leftPaneWidth - 4.0f);

        ImGui::SameLine(0.0f, 12.0f);

        if (graphDirty)
        {
            rebuildStructure();
            BuildNodes();
            graphDirty = false;
        }

        static bool showStyleEditor = false; 
        ed::Begin("Node editor");
        {

            ed::Suspend();
            if (ImGui::BeginMainMenuBar())
            {
                if (ImGui::BeginMenu("File"))
                {
                    if (ImGui::MenuItem("New", "Ctrl+N"))
                    {
                        StopIfRunning();
                        m_Nodes.clear();
                        m_Links.clear();
                        m_NextId = 1;
                        ed::ClearSelection();
                        graphDirty = true;

                        m_CurrentFilePath.clear(); // mark as untitled
                        UpdateWindowTitle();
                    }

                    if (ImGui::MenuItem("Open…", "Ctrl+O"))
                    {
                        StopIfRunning(); // stop processing if running
                        openFileDialog = true;
                    }
                    if (ImGui::MenuItem("Save As…", "Ctrl+S"))
                    {
                        StopIfRunning(); // stop processing if running
                        saveFileDialog = true;
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("Exit"))
                        OnStop();
                    ImGui::Separator();

                    // — Templates submenu —
                    if (ImGui::BeginMenu("Templates"))
                    {
                        namespace fs = std::filesystem;

                        fs::path templateDir = getExecutableDir() / "templates";

                        if (fs::exists(templateDir) && fs::is_directory(templateDir))
                        {

                            for (auto& entry : fs::directory_iterator(templateDir))
                            {
                                if (!entry.is_regular_file())
                                    continue;
                                if (entry.path().extension() != ".json")
                                    continue;

                                // show just the file-stem as the menu label:
                                std::string tmplName = entry.path().stem().string();

                                if (ImGui::MenuItem(tmplName.c_str()))
                                {
                                    // clear current graph
                                    StopIfRunning();
                                    m_Nodes.clear();
                                    m_Links.clear();
                                    m_NextId = 1;
                                    ed::ClearSelection();
                                    graphDirty = true;

                                    // load the template you just clicked
                                    LoadGraph(entry.path().string().c_str());
                                }
                            }
                        }
                        else
                        {
                            ImGui::TextDisabled("No templates folder found");
                        }

                        ImGui::EndMenu();
                    }
                    ImGui::EndMenu(); // ← HERE
                }

                if (ImGui::BeginMenu("Edit"))
                {
                    if (!io.WantCaptureKeyboard) // only when no text input is active
                    {
                        // copy -- ctrl + c
                        if (io.KeyCtrl &&
                            ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_C)))
                            CopySelectionToClipboard();

                        // paste -- ctrl + v
                        if (io.KeyCtrl &&
                            ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_V)))
                            PasteClipboardAt(ImGui::GetMousePos());
                    }

                      if (ImGui::MenuItem("Select All", "Ctrl+A"))
                    {
                        for (auto& up : m_Nodes)
                            ed::SelectNode(up->ID, true);
                    }

                    if (ImGui::MenuItem("Deselect All", "Ctrl+Shift+A"))
                    {
                        ed::ClearSelection();
                    }

                    if (ImGui::MenuItem("Paste", "Ctrl+V"))
                    {
                        PasteClipboardAt(ImGui::GetIO().MousePos);
                    }

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("View"))
                {
                    ImGui::Checkbox("Show Ordinals", &m_ShowOrdinals);
                    if (ImGui::MenuItem("Edit Style"))
                        showStyleEditor = true;
                    ImGui::EndMenu(); // ← AND HERE
                }

                ImGui::EndMainMenuBar();
            }
            ed::Resume();

            ed::Suspend();
            // now, outside the menu bar, pop up your style editor if requested
            if (showStyleEditor)
                ShowStyleEditor(&showStyleEditor);
            ed::Resume();
            // flick the flag and actually call the Win32 dialog
            if (openFileDialog)
            {
                openFileDialog = false;
                if (DoOpenFileDialog(filePathBuf, sizeof(filePathBuf)))
                    LoadGraph(filePathBuf);
            }

            if (saveFileDialog)
            {
                saveFileDialog = false;
                // prefill with last path or default:
                strncpy(filePathBuf, "graph.json", sizeof(filePathBuf));
                if (DoSaveFileDialog(filePathBuf, sizeof(filePathBuf)))
                    SaveGraph(filePathBuf);
            }
            auto cursorTopLeft = ImGui::GetCursorScreenPos();

            util::BlueprintNodeBuilder builder(m_HeaderBackground,
                                               GetTextureWidth(m_HeaderBackground),
                                               GetTextureHeight(m_HeaderBackground));

            for (auto& node : m_Nodes)
            {
                if (node->Type != NodeType::Blueprint && node->Type != NodeType::Simple)
                {

                    std::cout << "Continuing through nodes" << std::endl;
                    continue;
                }

                const auto isSimple = node->Type == NodeType::Simple;

                bool hasOutputDelegates = false;
                for (auto& output : node->Outputs)
                    if (output.Type == PinType::Delegate)
                        hasOutputDelegates = true;

                builder.Begin(node->ID);
                if (!isSimple)
                {
                    builder.Header(node->Color);
                    ImGui::Spring(0);
                    if (node->ID == renamingNode)
                    {
                        auto& buf = renameBuffers[node->ID];

                        // 1) Measure the current text
                        ImVec2 textSize = ImGui::CalcTextSize(buf.data());
                        float paddingX = ImGui::GetStyle().FramePadding.x * 2.0f;
                        float extra = 10.0f; // extra room for cursor/click area
                        float boxW = textSize.x + paddingX + extra;

                        // 2) Tell ImGui to size the next widget to that width
                        ImGui::SetNextItemWidth(boxW);

                        // 3) Call InputText with the resize‐callback as before
                        if (ImGui::InputText("##rename", buf.data(), (int)buf.size(),
                                             ImGuiInputTextFlags_EnterReturnsTrue |
                                                 ImGuiInputTextFlags_AutoSelectAll |
                                                 ImGuiInputTextFlags_CallbackResize,
                                             RenameCallback, &buf))
                        {
                            // commit
                            node->Name = buf.data();
                            renamingNode = 0;
                            // optional: shrink the vector to trim unused capacity
                            buf.shrink_to_fit();
                        }
                    }
                    else
                    {
                        ImGui::TextUnformatted(node->Name.c_str());
                        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
                        {
                            renamingNode = node->ID;
                            auto& buf = renameBuffers[node->ID];
                            buf.resize(node->Name.size() + 1);
                            memcpy(buf.data(), node->Name.c_str(), buf.size());
                            buf.reserve(buf.size() + 128);
                        }
                    }

                    ImGui::Spring(1);
                    ImGui::Dummy(ImVec2(0, 28));
                    if (hasOutputDelegates)
                    {
                        ImGui::BeginVertical("delegates", ImVec2(0, 28));
                        ImGui::Spring(1, 0);
                        for (auto& output : node->Outputs)
                        {
                            if (output.Type != PinType::Delegate)
                                continue;

                            auto alpha = ImGui::GetStyle().Alpha;
                            if (newLinkPin && !CanCreateLink(newLinkPin, &output) &&
                                &output != newLinkPin)
                                alpha = alpha * (48.0f / 255.0f);

                            ed::BeginPin(output.ID, ed::PinKind::Output);
                            ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
                            ed::PinPivotSize(ImVec2(0, 0));
                            ImGui::BeginHorizontal(output.ID.AsPointer());
                            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
                            if (!output.Name.empty())
                            {
                                ImGui::TextUnformatted(output.Name.c_str());
                                ImGui::Spring(0);
                            }
                            DrawPinIcon(output, IsPinLinked(output.ID),
                                        (int)(alpha * 255));
                            ImGui::Spring(0, ImGui::GetStyle().ItemSpacing.x / 2);
                            ImGui::EndHorizontal();
                            ImGui::PopStyleVar();
                            ed::EndPin();

                            // DrawItemRect(ImColor(255, 0, 0));
                        }
                        ImGui::Spring(1, 0);
                        ImGui::EndVertical();
                        ImGui::Spring(0, ImGui::GetStyle().ItemSpacing.x / 2);
                    }
                    else
                        ImGui::Spring(0);
                    builder.EndHeader();
                }

                for (auto& input : node->Inputs)
                {
                    auto alpha = ImGui::GetStyle().Alpha;
                    if (newLinkPin && !CanCreateLink(newLinkPin, &input) &&
                        &input != newLinkPin)
                        alpha = alpha * (48.0f / 255.0f);

                    builder.Input(input.ID);
                    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
                    DrawPinIcon(input, IsPinLinked(input.ID), (int)(alpha * 255));
                    ImGui::Spring(0);
                    if (!input.Name.empty())
                    {
                        ImGui::TextUnformatted(input.Name.c_str());
                        ImGui::Spring(0);
                    }
                    if (input.Type == PinType::Bool)
                    {
                        ImGui::Button("Hello");
                        ImGui::Spring(0);
                    }
                    ImGui::PopStyleVar();
                    builder.EndInput();
                }

                if (!isSimple)
                {
                    builder.Middle();

                    ImGui::Spring(0, 1);
                    // if node is videoreadernode
                    // cast to VideoReaderNode to check
                    if (auto videoNode = dynamic_cast<RenderPreviewNode*>(node.get()))
                    {
                        videoNode->DrawPreviewFrame();
                    }
                    else if (auto debugNode = dynamic_cast<DebugPrintNode*>(node.get()))
                    {
                        ImGui::TextColored(ImVec4(0,1,0,1), "Value: %s",
                                           debugNode->display.c_str());
                    }
                    ImGui::Spring(1, 0);
                }

                for (auto& output : node->Outputs)
                {
                    if (!isSimple && output.Type == PinType::Delegate)
                        continue;

                    auto alpha = ImGui::GetStyle().Alpha;
                    if (newLinkPin && !CanCreateLink(newLinkPin, &output) &&
                        &output != newLinkPin)
                        alpha = alpha * (48.0f / 255.0f);

                    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
                    builder.Output(output.ID);

                    if (output.showUI)
                    {
                        if (output.Type == PinType::Float)
                        {
                            ImGui::PushItemWidth(100.0f);

                            if (float* value = std::get_if<float>(&output.Value))
                            {
                                ImGui::DragFloat("##float", value, 0.1f, 0.0f, 100.0f);
                            }
                            else
                            {
                                ImGui::TextUnformatted("[Invalid float!]");
                            }

                            ImGui::PopItemWidth();
                            ImGui::Spring(0);
                        }
                        else if (output.Type == PinType::Int)
                        {
                            ImGui::PushItemWidth(100.0f);

                            if (int* value = std::get_if<int>(&output.Value))
                            {
                                ImGui::DragInt("##int", value, 1.0f, 0, 100);
                            }
                            else
                            {
                                ImGui::TextUnformatted("[Invalid int!]");
                            }

                            ImGui::PopItemWidth();
                            ImGui::Spring(0);
                        }
                        else if (output.Type == PinType::Bool)
                        {
                            if (bool* value = std::get_if<bool>(&output.Value))
                            {
                                ImGui::Checkbox("##bool", value);
                                ImGui::Spring(0);
                            }
                            else
                            {
                                ImGui::TextUnformatted("[Invalid bool!]");
                            }
                        }
                    }

                    if (!output.Name.empty())
                    {
                        ImGui::Spring(0);
                        ImGui::TextUnformatted(output.Name.c_str());
                    }
                    ImGui::Spring(0);
                    DrawPinIcon(output, IsPinLinked(output.ID), (int)(alpha * 255));
                    ImGui::PopStyleVar();
                    builder.EndOutput();
                }

                builder.End();
            }

            for (auto& node : m_Nodes)
            {
                if (node->Type != NodeType::Tree)
                    continue;

                const float rounding = 5.0f;
                const float padding = 12.0f;

                const auto pinBackground = ed::GetStyle().Colors[ed::StyleColor_NodeBg];

                ed::PushStyleColor(ed::StyleColor_NodeBg, ImColor(128, 128, 128, 200));
                ed::PushStyleColor(ed::StyleColor_NodeBorder, ImColor(32, 32, 32, 200));
                ed::PushStyleColor(ed::StyleColor_PinRect, ImColor(60, 180, 255, 150));
                ed::PushStyleColor(ed::StyleColor_PinRectBorder,
                                   ImColor(60, 180, 255, 150));

                ed::PushStyleVar(ed::StyleVar_NodePadding, ImVec4(0, 0, 0, 0));
                ed::PushStyleVar(ed::StyleVar_NodeRounding, rounding);
                ed::PushStyleVar(ed::StyleVar_SourceDirection, ImVec2(0.0f, 1.0f));
                ed::PushStyleVar(ed::StyleVar_TargetDirection, ImVec2(0.0f, -1.0f));
                ed::PushStyleVar(ed::StyleVar_LinkStrength, 0.0f);
                ed::PushStyleVar(ed::StyleVar_PinBorderWidth, 1.0f);
                ed::PushStyleVar(ed::StyleVar_PinRadius, 5.0f);
                ed::BeginNode(node->ID);

                ImGui::BeginVertical(node->ID.AsPointer());
                ImGui::BeginHorizontal("inputs");
                ImGui::Spring(0, padding * 2);

                ImRect inputsRect;
                int inputAlpha = 200;
                if (!node->Inputs.empty())
                {
                    auto& pin = node->Inputs[0];
                    ImGui::Dummy(ImVec2(0, padding));
                    ImGui::Spring(1, 0);
                    inputsRect = ImGui_GetItemRect();

                    ed::PushStyleVar(ed::StyleVar_PinArrowSize, 10.0f);
                    ed::PushStyleVar(ed::StyleVar_PinArrowWidth, 10.0f);
#if IMGUI_VERSION_NUM > 18101
                    ed::PushStyleVar(ed::StyleVar_PinCorners,
                                     ImDrawFlags_RoundCornersBottom);
#else
                    ed::PushStyleVar(ed::StyleVar_PinCorners, 12);
#endif
                    ed::BeginPin(pin.ID, ed::PinKind::Input);
                    ed::PinPivotRect(inputsRect.GetTL(), inputsRect.GetBR());
                    ed::PinRect(inputsRect.GetTL(), inputsRect.GetBR());
                    ed::EndPin();
                    ed::PopStyleVar(3);

                    if (newLinkPin && !CanCreateLink(newLinkPin, &pin) &&
                        &pin != newLinkPin)
                        inputAlpha =
                            (int)(255 * ImGui::GetStyle().Alpha * (48.0f / 255.0f));
                }
                else
                    ImGui::Dummy(ImVec2(0, padding));

                ImGui::Spring(0, padding * 2);
                ImGui::EndHorizontal();

                ImGui::BeginHorizontal("content_frame");
                ImGui::Spring(1, padding);

                ImGui::BeginVertical("content", ImVec2(0.0f, 0.0f));
                ImGui::Dummy(ImVec2(160, 0));
                ImGui::Spring(1);
                ImGui::TextUnformatted(node->Name.c_str());
                ImGui::Spring(1);
                ImGui::EndVertical();
                auto contentRect = ImGui_GetItemRect();

                ImGui::Spring(1, padding);
                ImGui::EndHorizontal();

                ImGui::BeginHorizontal("outputs");
                ImGui::Spring(0, padding * 2);

                ImRect outputsRect;
                int outputAlpha = 200;
                if (!node->Outputs.empty())
                {
                    auto& pin = node->Outputs[0];
                    ImGui::Dummy(ImVec2(0, padding));
                    ImGui::Spring(1, 0);
                    outputsRect = ImGui_GetItemRect();

#if IMGUI_VERSION_NUM > 18101
                    ed::PushStyleVar(ed::StyleVar_PinCorners,
                                     ImDrawFlags_RoundCornersTop);
#else
                    ed::PushStyleVar(ed::StyleVar_PinCorners, 3);
#endif
                    ed::BeginPin(pin.ID, ed::PinKind::Output);
                    ed::PinPivotRect(outputsRect.GetTL(), outputsRect.GetBR());
                    ed::PinRect(outputsRect.GetTL(), outputsRect.GetBR());
                    ed::EndPin();
                    ed::PopStyleVar();

                    if (newLinkPin && !CanCreateLink(newLinkPin, &pin) &&
                        &pin != newLinkPin)
                        outputAlpha =
                            (int)(255 * ImGui::GetStyle().Alpha * (48.0f / 255.0f));
                }
                else
                    ImGui::Dummy(ImVec2(0, padding));

                ImGui::Spring(0, padding * 2);
                ImGui::EndHorizontal();

                ImGui::EndVertical();

                ed::EndNode();
                ed::PopStyleVar(7);
                ed::PopStyleColor(4);

                auto drawList = ed::GetNodeBackgroundDrawList(node->ID);

                // const auto fringeScale = ImGui::GetStyle().AntiAliasFringeScale;
                // const auto unitSize    = 1.0f / fringeScale;

                // const auto ImDrawList_AddRect = [](ImDrawList* drawList, const
                // ImVec2& a, const ImVec2& b, ImU32 col, float rounding, int
                // rounding_corners, float thickness)
                //{
                //     if ((col >> 24) == 0)
                //         return;
                //     drawList->PathRect(a, b, rounding, rounding_corners);
                //     drawList->PathStroke(col, true, thickness);
                // };

#if IMGUI_VERSION_NUM > 18101
                const auto topRoundCornersFlags = ImDrawFlags_RoundCornersTop;
                const auto bottomRoundCornersFlags = ImDrawFlags_RoundCornersBottom;
#else
                const auto topRoundCornersFlags = 1 | 2;
                const auto bottomRoundCornersFlags = 4 | 8;
#endif

                drawList->AddRectFilled(
                    inputsRect.GetTL() + ImVec2(0, 1), inputsRect.GetBR(),
                    IM_COL32((int)(255 * pinBackground.x), (int)(255 * pinBackground.y),
                             (int)(255 * pinBackground.z), inputAlpha),
                    4.0f, bottomRoundCornersFlags);
                // ImGui::PushStyleVar(ImGuiStyleVar_AntiAliasFringeScale, 1.0f);
                drawList->AddRect(inputsRect.GetTL() + ImVec2(0, 1), inputsRect.GetBR(),
                                  IM_COL32((int)(255 * pinBackground.x),
                                           (int)(255 * pinBackground.y),
                                           (int)(255 * pinBackground.z), inputAlpha),
                                  4.0f, bottomRoundCornersFlags);
                // ImGui::PopStyleVar();
                drawList->AddRectFilled(
                    outputsRect.GetTL(), outputsRect.GetBR() - ImVec2(0, 1),
                    IM_COL32((int)(255 * pinBackground.x), (int)(255 * pinBackground.y),
                             (int)(255 * pinBackground.z), outputAlpha),
                    4.0f, topRoundCornersFlags);
                // ImGui::PushStyleVar(ImGuiStyleVar_AntiAliasFringeScale, 1.0f);
                drawList->AddRect(
                    outputsRect.GetTL(), outputsRect.GetBR() - ImVec2(0, 1),
                    IM_COL32((int)(255 * pinBackground.x), (int)(255 * pinBackground.y),
                             (int)(255 * pinBackground.z), outputAlpha),
                    4.0f, topRoundCornersFlags);
                // ImGui::PopStyleVar();
                drawList->AddRectFilled(contentRect.GetTL(), contentRect.GetBR(),
                                        IM_COL32(24, 64, 128, 200), 0.0f);
                // ImGui::PushStyleVar(ImGuiStyleVar_AntiAliasFringeScale, 1.0f);
                drawList->AddRect(contentRect.GetTL(), contentRect.GetBR(),
                                  IM_COL32(48, 128, 255, 100), 0.0f);
                // ImGui::PopStyleVar();
            }

            for (auto& node : m_Nodes)
            {
                if (node->Type != NodeType::Houdini)
                    continue;

                const float rounding = 10.0f;
                const float padding = 12.0f;

                ed::PushStyleColor(ed::StyleColor_NodeBg, ImColor(229, 229, 229, 200));
                ed::PushStyleColor(ed::StyleColor_NodeBorder,
                                   ImColor(125, 125, 125, 200));
                ed::PushStyleColor(ed::StyleColor_PinRect, ImColor(229, 229, 229, 60));
                ed::PushStyleColor(ed::StyleColor_PinRectBorder,
                                   ImColor(125, 125, 125, 60));

                const auto pinBackground = ed::GetStyle().Colors[ed::StyleColor_NodeBg];

                ed::PushStyleVar(ed::StyleVar_NodePadding, ImVec4(0, 0, 0, 0));
                ed::PushStyleVar(ed::StyleVar_NodeRounding, rounding);
                ed::PushStyleVar(ed::StyleVar_SourceDirection, ImVec2(0.0f, 1.0f));
                ed::PushStyleVar(ed::StyleVar_TargetDirection, ImVec2(0.0f, -1.0f));
                ed::PushStyleVar(ed::StyleVar_LinkStrength, 0.0f);
                ed::PushStyleVar(ed::StyleVar_PinBorderWidth, 1.0f);
                ed::PushStyleVar(ed::StyleVar_PinRadius, 6.0f);
                ed::BeginNode(node->ID);

                ImGui::BeginVertical(node->ID.AsPointer());
                if (!node->Inputs.empty())
                {
                    ImGui::BeginHorizontal("inputs");
                    ImGui::Spring(1, 0);

                    ImRect inputsRect;
                    int inputAlpha = 200;
                    for (auto& pin : node->Inputs)
                    {
                        ImGui::Dummy(ImVec2(padding, padding));
                        inputsRect = ImGui_GetItemRect();
                        ImGui::Spring(1, 0);
                        inputsRect.Min.y -= padding;
                        inputsRect.Max.y -= padding;

#if IMGUI_VERSION_NUM > 18101
                        const auto allRoundCornersFlags = ImDrawFlags_RoundCornersAll;
#else
                        const auto allRoundCornersFlags = 15;
#endif
                        // ed::PushStyleVar(ed::StyleVar_PinArrowSize, 10.0f);
                        // ed::PushStyleVar(ed::StyleVar_PinArrowWidth, 10.0f);
                        ed::PushStyleVar(ed::StyleVar_PinCorners, allRoundCornersFlags);

                        ed::BeginPin(pin.ID, ed::PinKind::Input);
                        ed::PinPivotRect(inputsRect.GetCenter(),
                                         inputsRect.GetCenter());
                        ed::PinRect(inputsRect.GetTL(), inputsRect.GetBR());
                        ed::EndPin();
                        // ed::PopStyleVar(3);
                        ed::PopStyleVar(1);

                        auto drawList = ImGui::GetWindowDrawList();
                        drawList->AddRectFilled(inputsRect.GetTL(), inputsRect.GetBR(),
                                                IM_COL32((int)(255 * pinBackground.x),
                                                         (int)(255 * pinBackground.y),
                                                         (int)(255 * pinBackground.z),
                                                         inputAlpha),
                                                4.0f, allRoundCornersFlags);
                        drawList->AddRect(inputsRect.GetTL(), inputsRect.GetBR(),
                                          IM_COL32((int)(255 * pinBackground.x),
                                                   (int)(255 * pinBackground.y),
                                                   (int)(255 * pinBackground.z),
                                                   inputAlpha),
                                          4.0f, allRoundCornersFlags);

                        if (newLinkPin && !CanCreateLink(newLinkPin, &pin) &&
                            &pin != newLinkPin)
                            inputAlpha =
                                (int)(255 * ImGui::GetStyle().Alpha * (48.0f / 255.0f));
                    }

                    // ImGui::Spring(1, 0);
                    ImGui::EndHorizontal();
                }

                ImGui::BeginHorizontal("content_frame");
                ImGui::Spring(1, padding);

                ImGui::BeginVertical("content", ImVec2(0.0f, 0.0f));
                ImGui::Dummy(ImVec2(160, 0));
                ImGui::Spring(1);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
                ImGui::TextUnformatted(node->Name.c_str());
                ImGui::PopStyleColor();
                ImGui::Spring(1);
                ImGui::EndVertical();
                auto contentRect = ImGui_GetItemRect();

                ImGui::Spring(1, padding);
                ImGui::EndHorizontal();

                if (!node->Outputs.empty())
                {
                    ImGui::BeginHorizontal("outputs");
                    ImGui::Spring(1, 0);

                    ImRect outputsRect;
                    int outputAlpha = 200;
                    for (auto& pin : node->Outputs)
                    {
                        ImGui::Dummy(ImVec2(padding, padding));
                        outputsRect = ImGui_GetItemRect();
                        ImGui::Spring(1, 0);
                        outputsRect.Min.y += padding;
                        outputsRect.Max.y += padding;

#if IMGUI_VERSION_NUM > 18101
                        const auto allRoundCornersFlags = ImDrawFlags_RoundCornersAll;
                        const auto topRoundCornersFlags = ImDrawFlags_RoundCornersTop;
#else
                        const auto allRoundCornersFlags = 15;
                        const auto topRoundCornersFlags = 3;
#endif

                        ed::PushStyleVar(ed::StyleVar_PinCorners, topRoundCornersFlags);
                        ed::BeginPin(pin.ID, ed::PinKind::Output);
                        ed::PinPivotRect(outputsRect.GetCenter(),
                                         outputsRect.GetCenter());
                        ed::PinRect(outputsRect.GetTL(), outputsRect.GetBR());
                        ed::EndPin();
                        ed::PopStyleVar();

                        auto drawList = ImGui::GetWindowDrawList();
                        drawList->AddRectFilled(
                            outputsRect.GetTL(), outputsRect.GetBR(),
                            IM_COL32((int)(255 * pinBackground.x),
                                     (int)(255 * pinBackground.y),
                                     (int)(255 * pinBackground.z), outputAlpha),
                            4.0f, allRoundCornersFlags);
                        drawList->AddRect(outputsRect.GetTL(), outputsRect.GetBR(),
                                          IM_COL32((int)(255 * pinBackground.x),
                                                   (int)(255 * pinBackground.y),
                                                   (int)(255 * pinBackground.z),
                                                   outputAlpha),
                                          4.0f, allRoundCornersFlags);

                        if (newLinkPin && !CanCreateLink(newLinkPin, &pin) &&
                            &pin != newLinkPin)
                            outputAlpha =
                                (int)(255 * ImGui::GetStyle().Alpha * (48.0f / 255.0f));
                    }

                    ImGui::EndHorizontal();
                }

                ImGui::EndVertical();

                ed::EndNode();
                ed::PopStyleVar(7);
                ed::PopStyleColor(4);

                // auto drawList = ed::GetNodeBackgroundDrawList(node->ID);

                // const auto fringeScale = ImGui::GetStyle().AntiAliasFringeScale;
                // const auto unitSize    = 1.0f / fringeScale;

                // const auto ImDrawList_AddRect = [](ImDrawList* drawList, const
                // ImVec2& a, const ImVec2& b, ImU32 col, float rounding, int
                // rounding_corners, float thickness)
                //{
                //     if ((col >> 24) == 0)
                //         return;
                //     drawList->PathRect(a, b, rounding, rounding_corners);
                //     drawList->PathStroke(col, true, thickness);
                // };

                // drawList->AddRectFilled(inputsRect.GetTL() + ImVec2(0, 1),
                // inputsRect.GetBR(),
                //     IM_COL32((int)(255 * pinBackground.x), (int)(255 *
                //     pinBackground.y), (int)(255 * pinBackground.z),
                //     inputAlpha), 4.0f, 12);
                // ImGui::PushStyleVar(ImGuiStyleVar_AntiAliasFringeScale, 1.0f);
                // drawList->AddRect(inputsRect.GetTL() + ImVec2(0, 1),
                // inputsRect.GetBR(),
                //     IM_COL32((int)(255 * pinBackground.x), (int)(255 *
                //     pinBackground.y), (int)(255 * pinBackground.z),
                //     inputAlpha), 4.0f, 12);
                // ImGui::PopStyleVar();
                // drawList->AddRectFilled(outputsRect.GetTL(), outputsRect.GetBR() -
                // ImVec2(0, 1),
                //     IM_COL32((int)(255 * pinBackground.x), (int)(255 *
                //     pinBackground.y), (int)(255 * pinBackground.z),
                //     outputAlpha), 4.0f, 3);
                ////ImGui::PushStyleVar(ImGuiStyleVar_AntiAliasFringeScale, 1.0f);
                // drawList->AddRect(outputsRect.GetTL(), outputsRect.GetBR() -
                // ImVec2(0, 1),
                //     IM_COL32((int)(255 * pinBackground.x), (int)(255 *
                //     pinBackground.y), (int)(255 * pinBackground.z),
                //     outputAlpha), 4.0f, 3);
                ////ImGui::PopStyleVar();
                // drawList->AddRectFilled(contentRect.GetTL(), contentRect.GetBR(),
                // IM_COL32(24, 64, 128, 200), 0.0f);
                // ImGui::PushStyleVar(ImGuiStyleVar_AntiAliasFringeScale, 1.0f);
                // drawList->AddRect(
                //     contentRect.GetTL(),
                //     contentRect.GetBR(),
                //     IM_COL32(48, 128, 255, 100), 0.0f);
                // ImGui::PopStyleVar();
            }

            for (auto& node : m_Nodes)
            {
                if (node->Type != NodeType::Comment)
                    continue;

                const float commentAlpha = 0.75f;

                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, commentAlpha);
                ed::PushStyleColor(ed::StyleColor_NodeBg, ImColor(255, 255, 255, 64));
                ed::PushStyleColor(ed::StyleColor_NodeBorder,
                                   ImColor(255, 255, 255, 64));
                ed::BeginNode(node->ID);
                ImGui::PushID(node->ID.AsPointer());
                ImGui::BeginVertical("content");
                ImGui::BeginHorizontal("horizontal");
                ImGui::Spring(1);
                ImGui::TextUnformatted(node->Name.c_str());
                ImGui::Spring(1);
                ImGui::EndHorizontal();
                ed::Group(node->Size);
                ImGui::EndVertical();
                ImGui::PopID();
                ed::EndNode();
                ed::PopStyleColor(2);
                ImGui::PopStyleVar();

                if (ed::BeginGroupHint(node->ID))
                {
                    // auto alpha   = static_cast<int>(commentAlpha *
                    // ImGui::GetStyle().Alpha * 255);
                    auto bgAlpha = static_cast<int>(ImGui::GetStyle().Alpha * 255);

                    // ImGui::PushStyleVar(ImGuiStyleVar_Alpha, commentAlpha *
                    // ImGui::GetStyle().Alpha);

                    auto min = ed::GetGroupMin();
                    // auto max = ed::GetGroupMax();

                    ImGui::SetCursorScreenPos(
                        min - ImVec2(-8, ImGui::GetTextLineHeightWithSpacing() + 4));
                    ImGui::BeginGroup();
                    ImGui::TextUnformatted(node->Name.c_str());
                    ImGui::EndGroup();

                    auto drawList = ed::GetHintBackgroundDrawList();

                    auto hintBounds = ImGui_GetItemRect();
                    auto hintFrameBounds = ImRect_Expanded(hintBounds, 8, 4);

                    drawList->AddRectFilled(
                        hintFrameBounds.GetTL(), hintFrameBounds.GetBR(),
                        IM_COL32(255, 255, 255, 64 * bgAlpha / 255), 4.0f);

                    drawList->AddRect(hintFrameBounds.GetTL(), hintFrameBounds.GetBR(),
                                      IM_COL32(255, 255, 255, 128 * bgAlpha / 255),
                                      4.0f);

                    // ImGui::PopStyleVar();
                }
                ed::EndGroupHint();
            }

            for (auto& link : m_Links)
                ed::Link(link.ID, link.StartPinID, link.EndPinID, link.Color, 2.0f);

            if (!createNewNode)
            {
                if (ed::BeginCreate(ImColor(255, 255, 255), 2.0f))
                {
                    auto showLabel = [](const char* label, ImColor color)
                    {
                        ImGui::SetCursorPosY(ImGui::GetCursorPosY() -
                                             ImGui::GetTextLineHeight());
                        auto size = ImGui::CalcTextSize(label);

                        auto padding = ImGui::GetStyle().FramePadding;
                        auto spacing = ImGui::GetStyle().ItemSpacing;

                        ImGui::SetCursorPos(ImGui::GetCursorPos() +
                                            ImVec2(spacing.x, -spacing.y));

                        auto rectMin = ImGui::GetCursorScreenPos() - padding;
                        auto rectMax = ImGui::GetCursorScreenPos() + size + padding;

                        auto drawList = ImGui::GetWindowDrawList();
                        drawList->AddRectFilled(rectMin, rectMax, color,
                                                size.y * 0.15f);
                        ImGui::TextUnformatted(label);
                    };

                    ed::PinId startPinId = 0, endPinId = 0;
                    if (ed::QueryNewLink(&startPinId, &endPinId))
                    {
                        auto startPin = FindPin(startPinId);
                        auto endPin = FindPin(endPinId);
                        newLinkPin = startPin ? startPin : endPin;

                        // ─────────── Early reject for invalid pins ─────────────
                        if (!startPin || !endPin)
                        {
                            ed::RejectNewItem(ImColor(200, 0, 0), 2.0f);
                            graphDirty = true;
                            return;
                        }

                        // Ensure start is always the output, end the input
                        if (startPin->Kind == PinKind::Input)
                        {
                            std::swap(startPin, endPin);
                            std::swap(startPinId, endPinId);
                        }

                        bool valid = true;

                        // ─────────── Rejection Checks ──────────────────────────

                        if (startPin == endPin)
                        {
                            showLabel("x Can't connect to self",
                                      ImColor(60, 20, 20, 180));
                            ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
                            valid = false;
                            graphDirty = true;
                        }
                        else if (startPin->Kind == endPin->Kind)
                        {
                            showLabel("x Incompatible Pin Kind",
                                      ImColor(60, 20, 20, 180));
                            ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
                            valid = false;
                            graphDirty = true;
                        }
                        else
                        {
                            bool compatible = startPin->Type == endPin->Type ||
                                              startPin->Type == PinType::Variant ||
                                              endPin->Type == PinType::Variant;

                            if (!compatible)
                            {
                                showLabel("x Incompatible Pin Type",
                                          ImColor(60, 20, 20, 180));
                                ed::RejectNewItem(ImColor(255, 128, 128), 2.0f);
                                valid = false;
                                graphDirty = true;
                            }
                        }

                        // ─────────── Input Pin Already Linked? ────────────────
                        Link* oldLink = nullptr;
                        if (valid && endPin->Kind == PinKind::Input)
                        {
                            for (auto& link : m_Links)
                            {
                                if (link.EndPinID == endPin->ID)
                                {
                                    oldLink = &link;
                                    break;
                                }
                            }
                        }

                        // ─────────── Accept or Replace ────────────────────────
                        if (valid)
                        {
                            const char* label =
                                oldLink ? "↻ Replace Link" : "+ Create Link";
                            showLabel(label, ImColor(20, 60, 20, 180));

                            if (ed::AcceptNewItem(ImColor(128, 255, 128), 4.0f))
                            {
                                // If replacing an existing link on input pin
                                if (oldLink)
                                {
                                    m_Links.erase(
                                        std::remove_if(m_Links.begin(), m_Links.end(),
                                                       [&](const Link& l)
                                                       { return l.ID == oldLink->ID; }),
                                        m_Links.end());
                                }

                                // Add new link
                                m_Links.emplace_back(
                                    Link(GetNextId(), startPinId, endPinId));
                                m_Links.back().Color = GetIconColor(startPin->Type);
                                graphDirty = true;
                            }
                        }
                    }

                    ed::PinId pinId = 0;
                    if (ed::QueryNewNode(&pinId))
                    {
                        newLinkPin = FindPin(pinId);
                        if (newLinkPin)
                            showLabel("+ Create Node", ImColor(32, 45, 32, 180));

                        if (ed::AcceptNewItem())
                        {
                            createNewNode = true;
                            newNodeLinkPin = FindPin(pinId);
                            newLinkPin = nullptr;
                            ed::Suspend();
                            ImGui::OpenPopup("Create New Node");
                            ed::Resume();
                            graphDirty = true;
                        }
                    }
                }
                else
                    newLinkPin = nullptr;

                ed::EndCreate();

                if (ed::BeginDelete())
                {
                    ed::NodeId nodeId = 0;
                    while (ed::QueryDeletedNode(&nodeId))
                    {
                        if (ed::AcceptDeletedItem())
                        {
                            auto id = std::find_if(m_Nodes.begin(), m_Nodes.end(),
                                                   [nodeId](auto& node)
                                                   { return node->ID == nodeId; });
                            if (id != m_Nodes.end())
                                m_Nodes.erase(id);
                            graphDirty = true;
                        }
                    }

                    ed::LinkId linkId = 0;
                    while (ed::QueryDeletedLink(&linkId))
                    {
                        if (ed::AcceptDeletedItem())
                        {
                            auto id = std::find_if(m_Links.begin(), m_Links.end(),
                                                   [linkId](auto& link)
                                                   { return link.ID == linkId; });
                            if (id != m_Links.end())
                                m_Links.erase(id);
                            graphDirty = true;
                        }
                    }
                }
                ed::EndDelete();
            }

            ImGui::SetCursorScreenPos(cursorTopLeft);
        }

#if 1 // In your context‐menu drawing (inside BeginPopup("Node Context Menu")):

        auto openPopupPosition = ImGui::GetMousePos();
        ed::Suspend();
        if (ed::ShowNodeContextMenu(&contextNodeId))
            ImGui::OpenPopup("Node Context Menu");
        else if (ed::ShowPinContextMenu(&contextPinId))
            ImGui::OpenPopup("Pin Context Menu");
        else if (ed::ShowLinkContextMenu(&contextLinkId))
            ImGui::OpenPopup("Link Context Menu");
        else if (ed::ShowBackgroundContextMenu())
        {
            ImGui::OpenPopup("Create New Node");
            newNodeLinkPin = nullptr;
        }
        ed::Resume();

        ed::Suspend();
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
        if (ImGui::BeginPopup("Node Context Menu"))
        {
            auto node = FindNode(contextNodeId);

            ImGui::TextUnformatted("Node Context Menu");
            ImGui::Separator();
            if (node)
            {
                ImGui::Text("ID: %p", node->ID.AsPointer());
                ImGui::Text("Type: %s",
                            node->Type == NodeType::Blueprint
                                ? "Blueprint"
                                : (node->Type == NodeType::Tree ? "Tree" : "Comment"));
                ImGui::Text("ID: %p", node->ID.AsPointer());
                ImGui::Text("Type: %s",
                            node->Type == NodeType::Blueprint
                                ? "Blueprint"
                                : (node->Type == NodeType::Tree ? "Tree" : "Comment"));

                // --- Pin Value Display Section ---
                ImGui::Separator();
                ImGui::Text("Input Pins:");
                for (size_t i = 0; i < node->Inputs.size(); ++i)
                {
                    auto& pin = node->Inputs[i];
                    ImGui::Text("  [%zu] %s: %s", i, pin.Name.c_str(),
                                variantToString(pin.Value).c_str());
                }
                ImGui::Separator();
                ImGui::Text("Output Pins:");
                for (size_t i = 0; i < node->Outputs.size(); ++i)
                {
                    auto& pin = node->Outputs[i];
                    ImGui::Text("  [%zu] %s: %s", i, pin.Name.c_str(),
                                variantToString(pin.Value).c_str());
                }

            }
            else
                ImGui::Text("Unknown node: %p", contextNodeId.AsPointer());
            ImGui::Separator();
            if (ImGui::MenuItem("Delete"))
            {
                ed::DeleteNode(contextNodeId);
                graphDirty = true; // mark graph as dirty
            }
            else if (ImGui::MenuItem("Rename"))
            {
                renamingNode = contextNodeId;
                if (auto node = FindNode(contextNodeId))
                {
                    auto& buf = renameBuffers[contextNodeId];
                    // start with capacity for the old name + some extra
                    size_t init = node->Name.size();
                    buf.resize(init + 1); // size = old name + '\0'
                    memcpy(buf.data(), node->Name.c_str(), init + 1);
                    buf.reserve(init + 1 + 128); // allow 128 more chars before callback
                }
            }

            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("Pin Context Menu"))
        {
            auto pin = FindPin(contextPinId);

            ImGui::TextUnformatted("Pin Context Menu");
            ImGui::Separator();
            if (pin)
            {
                ImGui::Text("ID: %p", pin->ID.AsPointer());
                if (pin->Node)
                    ImGui::Text("Node: %p", pin->Node->ID.AsPointer());
                else
                    ImGui::Text("Node: %s", "<none>");
            }
            else
                ImGui::Text("Unknown pin: %p", contextPinId.AsPointer());

            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("Link Context Menu"))
        {
            auto link = FindLink(contextLinkId);

            ImGui::TextUnformatted("Link Context Menu");
            ImGui::Separator();
            if (link)
            {
                ImGui::Text("ID: %p", link->ID.AsPointer());
                ImGui::Text("From: %p", link->StartPinID.AsPointer());
                ImGui::Text("To: %p", link->EndPinID.AsPointer());
            }
            else
                ImGui::Text("Unknown link: %p", contextLinkId.AsPointer());
            ImGui::Separator();
            if (ImGui::MenuItem("Delete"))
            {

                ed::DeleteLink(contextLinkId);
                graphDirty = true; // mark graph as dirty
            }
            ImGui::EndPopup();
        }

        // 2) Your popup — *no* other context menus should be invoked here:
        if (ImGui::BeginPopup("Create New Node"))
        {
            auto newNodePosition = openPopupPosition;
            Node* node = nullptr;

            // ─── Category menus ───────────────────────────────────────
            for (auto cat : {NodeCategory::Basic, NodeCategory::Arithmetic,
                             NodeCategory::Logic, NodeCategory::TensorOps,
                             NodeCategory::MachineLearning, NodeCategory::Utility})
            {
                const char* catLabel = [&]()
                {
                    switch (cat)
                    {
                    case NodeCategory::Basic:
                        return "Basic";
                    case NodeCategory::Arithmetic:
                        return "Arithmetic";
                    case NodeCategory::Logic:
                        return "Logic";
                    case NodeCategory::TensorOps:
                        return "Tensor Ops";
                    case NodeCategory::MachineLearning:
                        return "ML Models";
                    case NodeCategory::Utility:
                        return "Utility";
                    }
                    return "Unknown";
                }();

                auto types = NodeFactory::instance().typesForCategory(cat);
                if (types.empty())
                    continue;

                if (ImGui::BeginMenu(catLabel))
                {
                    for (auto& key : types)
                    {
                        if (ImGui::MenuItem(key.c_str()))
                        {
                            node = CreateNode(key);
                            ImGui::CloseCurrentPopup();
                        }
                    }
                    ImGui::EndMenu();
                }
            }

            ImGui::Separator();
            // … any one‐off items here …

            ImGui::EndPopup();

            // 3) After we close the popup, position & auto‐link the new node:
            if (node)
            {
                BuildNodes();
                ed::SetNodePosition(node->ID, newNodePosition);
                ed::Suspend();
                ed::SelectNode(node->ID);
                ed::Resume();
                if (auto startPin = newNodeLinkPin)
                {
                    // choose the opposite side on the new node
                    auto& pins = (startPin->Kind == PinKind::Input) ? node->Outputs
                                                                    : node->Inputs;

                    for (auto& pin : pins)
                    {
                        // allow link if either side is Variant or types match
                        bool variantEither = (startPin->Type == PinType::Variant ||
                                              pin.Type == PinType::Variant);
                        if (variantEither || CanCreateLink(startPin, &pin))
                        {
                            Pin* endPin = &pin;
                            // ensure startPin is output
                            if (startPin->Kind == PinKind::Input)
                                std::swap(startPin, endPin);

                            // create the link
                            m_Links.emplace_back(
                                Link(GetNextId(), startPin->ID, endPin->ID));
                            m_Links.back().Color = GetIconColor(startPin->Type);
                            graphDirty = true; // mark graph as dirty
                            break;
                        }
                    }
                }
            }
        }
        else
            createNewNode = false;
        ImGui::PopStyleVar();
        ed::Resume();
#endif

        bool ctrl = io.KeyCtrl;
        bool shift = io.KeyShift;
        bool pressedA = ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_A), false);
        bool pressedC = ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_C), false);
        bool pressedV = ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_V), false);
        bool pressedX = ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_X), false);
        bool pressedS = ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_S), false);
        bool pressedO = ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_O), false);

        // Only run node editor shortcuts if no ImGui input field/widget is active or
        // focused
        if (!ImGui::IsAnyItemActive() && !ImGui::IsAnyItemFocused())
        {
            // Ctrl+Shift+A → Deselect all
            if (ctrl && shift && pressedA)
            {
                ed::ClearSelection();
            }
            // Ctrl+A → Select all nodes
            else if (ctrl && pressedA)
            {
                for (auto& up : m_Nodes)
                    ed::SelectNode(up->ID, true); // true = append
            }
            // Ctrl+X → Cut (copy + delete selected)
            else if (ctrl && pressedX)
            {
                CopySelectionToClipboard();
                for (auto& up : m_Nodes)
                {
                    if (ed::IsNodeSelected(up->ID))
                    {
                        ed::DeleteNode(up->ID);
                        graphDirty = true; // mark graph as dirty
                    }
                }
            }
            // Ctrl+C → Copy
            else if (ctrl && pressedC)
            {
                CopySelectionToClipboard();
            }
            // Ctrl+V → Paste
            else if (ctrl && pressedV)
            {
                PasteClipboardAt(ImGui::GetMousePos());
            }
            // Ctrl+S → Save
            else if (ctrl && pressedS)
            {
                saveFileDialog = true;
            }
            // Ctrl+O → Open
            else if (ctrl && pressedO)
            {
                openFileDialog = true;
            }
        }

        ed::End();

        auto editorMin = ImGui::GetItemRectMin();
        auto editorMax = ImGui::GetItemRectMax();

        if (m_ShowOrdinals)
        {
            int nodeCount = ed::GetNodeCount();
            std::vector<ed::NodeId> orderedNodeIds;
            orderedNodeIds.resize(static_cast<size_t>(nodeCount));
            ed::GetOrderedNodeIds(orderedNodeIds.data(), nodeCount);

            auto drawList = ImGui::GetWindowDrawList();
            drawList->PushClipRect(editorMin, editorMax);

            int ordinal = 0;
            for (auto& nodeId : orderedNodeIds)
            {
                auto p0 = ed::GetNodePosition(nodeId);
                auto p1 = p0 + ed::GetNodeSize(nodeId);
                p0 = ed::CanvasToScreen(p0);
                p1 = ed::CanvasToScreen(p1);

                ImGuiTextBuffer builder;
                builder.appendf("#%d", ordinal++);

                auto textSize = ImGui::CalcTextSize(builder.c_str());
                auto padding = ImVec2(2.0f, 2.0f);
                auto widgetSize = textSize + padding * 2;

                auto widgetPosition = ImVec2(p1.x, p0.y) + ImVec2(0.0f, -widgetSize.y);

                drawList->AddRectFilled(widgetPosition, widgetPosition + widgetSize,
                                        IM_COL32(100, 80, 80, 190), 3.0f,
                                        ImDrawFlags_RoundCornersAll);
                drawList->AddRect(widgetPosition, widgetPosition + widgetSize,
                                  IM_COL32(200, 160, 160, 190), 3.0f,
                                  ImDrawFlags_RoundCornersAll);
                drawList->AddText(widgetPosition + padding,
                                  IM_COL32(255, 255, 255, 255), builder.c_str());
            }

            drawList->PopClipRect();
        }

        // ImGui::ShowTestWindow();
        // ImGui::ShowMetricsWindow();
    }

    // in Example (alongside topoOrder, etc.)
    std::vector<std::vector<int>> chains;

    // Call anytime nodes or links change:
    void Example::OnGraphChanged()
    {
        graphDirty = true;

        StopGraphWorkers();  // Kill old worker threads tied to the old graph
        rebuildStructure();  // Rebuild chains
        StartGraphWorkers(); // Spawn workers per-chain again
    }

    // Rebuild both the usual DAG tables and our "chains" decomposition
    void Example::rebuildStructure()
    {
        // — build basic graph data —
        N = int(m_Nodes.size());
        pinToNodeIndex.clear();
        // Remove links whose pins no longer exist
        std::unordered_set<ed::PinId> allPins;
        for (auto& n : m_Nodes)
            for (auto& p : n->Inputs)
                allPins.insert(p.ID);
        for (auto& n : m_Nodes)
            for (auto& p : n->Outputs)
                allPins.insert(p.ID);

        m_Links.erase(std::remove_if(m_Links.begin(), m_Links.end(),
                                     [&](const Link& l)
                                     {
                                         return allPins.count(l.StartPinID) == 0 ||
                                                allPins.count(l.EndPinID) == 0;
                                     }),
                      m_Links.end());
        for (int i = 0; i < N; ++i)
        {
            for (auto& p : m_Nodes[i]->Inputs)
                pinToNodeIndex[p.ID] = i;
            for (auto& p : m_Nodes[i]->Outputs)
                pinToNodeIndex[p.ID] = i;
        }

        baseIndegree.assign(N, 0);
        adj.clear();
        adj.resize(N);
        outEdges.clear();
        outEdges.resize(N);

        for (auto& L : m_Links)
        {
            int u = pinToNodeIndex[L.StartPinID];
            int v = pinToNodeIndex[L.EndPinID];
            if (u == v)
                continue;
            adj[u].push_back(v);
            baseIndegree[v] += 1;
            outEdges[u].push_back(&L);
        }

        // — decompose into maximal linear chains —
        chains.clear();
        std::vector<bool> used(N, false);
        for (int i = 0; i < N; ++i)
        {
            // start a chain at any node with indegree != 1
            if (baseIndegree[i] != 1 && !used[i])
            {
                int cur = i;
                std::vector<int> chain;
                while (true)
                {
                    used[cur] = true;
                    chain.push_back(cur);

                    // must have exactly one outgoing edge...
                    if (adj[cur].size() != 1)
                        break;
                    int nxt = adj[cur][0];
                    // ...and that next node must have indegree==1
                    if (baseIndegree[nxt] != 1)
                        break;
                    cur = nxt;
                }
                chains.push_back(std::move(chain));
            }
        }

        graphDirty = false;
    }

    // Spawn one persistent thread per chain:
    void Example::StartGraphWorkers()
    {
        stopWorkers = false;
        for (auto& chain : chains)
        {
            graphWorkers.emplace_back(
                [this, chain]() mutable
                {
                    std::unique_lock<std::mutex> lk(playMutex);
                    while (!stopWorkers)
                    {
                        // wait for play or shutdown
                        playCv.wait(lk, [&] { return playRequested || stopWorkers; });
                        if (stopWorkers)
                            break;

                        // while playing, run our chain in order, as fast as it comes
                        while (playRequested && !stopWorkers)
                        {
                            for (int nodeIdx : chain)
                            {
                                m_Nodes[nodeIdx]->compute();
                                // immediately propagate this node’s outputs:
                                for (auto* link : outEdges[nodeIdx])
                                {
                                    auto* outPin = FindPin(link->StartPinID);
                                    auto* inPin = FindPin(link->EndPinID);
                                    inPin->Value = outPin->Value;
                                }
                            }
                        }
                    }
                });
        }
    }

    // Gracefully tear them down:
    void Example::StopGraphWorkers()
    {
        {
            std::lock_guard<std::mutex> lk(playMutex);
            stopWorkers = true;
        }
        // wake everyone up
        playCv.notify_all();
        for (auto& w : graphWorkers)
            w.join();
        graphWorkers.clear();
    }

void ComputeAll()
    {
        if (graphDirty)
            rebuildStructure();

        indegree = std::make_unique<std::atomic<int>[]>(N);
        for (int i = 0; i < N; ++i)
            indegree[i].store(baseIndegree[i], std::memory_order_relaxed);

        {
            std::lock_guard lk(graphMutex);
            std::queue<int> empty;
            std::swap(readyQueue, empty);
            for (int i = 0; i < N; ++i)
                if (indegree[i].load(std::memory_order_acquire) == 0)
                    readyQueue.push(i);
            workReady = true;
        }

        bool live = (g_EvalMode == EvalMode::Live);
        bool basic = (g_EvalMode == EvalMode::Basic);

        while (true)
        {
            int currentNode = -1;
            {
                std::lock_guard lk(graphMutex);
                if (readyQueue.empty())
                    break;
                currentNode = readyQueue.front();
                readyQueue.pop();
            }

            Node* node = m_Nodes[currentNode].get();
            bool isVideoReader = (dynamic_cast<VideoReaderNode*>(node) != nullptr);
            bool isRenderPreview = (dynamic_cast<RenderPreviewNode*>(node) != nullptr);

           if (live)
            {

               
                    node->compute();
             

            }

            else if (basic)
            {
                if (isVideoReader || isRenderPreview)
                {
                    node->compute();
                }
                else
                {
                    node->PassThrough();
                }
            }

            // Propagate values to downstream nodes via links
            for (auto* link : outEdges[currentNode])
            {
                auto* outPin = FindPin(link->StartPinID);
                auto* inPin = FindPin(link->EndPinID);
                inPin->Value = outPin->Value;

                int dst = pinToNodeIndex[link->EndPinID];
                if (indegree[dst].fetch_sub(1) == 1)
                {
                    std::lock_guard lk(graphMutex);
                    readyQueue.push(dst);
                }
            }

            processedCount.fetch_add(1);
        }

        barrierCv.notify_all();
    }

    // in your application startup:
    void StartProcessingThread()
    {
        // first launch the graph pool
        StartGraphWorkers();

        // then your existing pipeline thread
        stopThread = false;
        processingThread = std::thread(&Example::ProcessingLoop, this);
    }

    void StopProcessingThread()
    {
        // stop graph workers too
        StopGraphWorkers();

        stopThread = true;
        playCv.notify_all();
        if (processingThread.joinable())
            processingThread.join();
    }

    // Worker loop: waits for playRequested, then ComputeAll repeatedly
    void ProcessingLoop()
    {
        while (!stopThread)
        {
            std::unique_lock<std::mutex> lock(playMutex);
            playCv.wait(lock, [&] { return playRequested ||scrubbing|| stopThread; });
            if (stopThread)
                break;

           // while (playRequested && !stopThread) // or scrub bing and not stopthread
            while ((playRequested && !stopThread) || scrubbing)
            {
                // if scrubbing, we need to recompute the whole graph
                if (scrubbing)
                {
                    lock.unlock();
                    ComputeAll();
                    scrubbing = false;
                    lock.lock();
                  
                }
                else
                {
     
                    lock.unlock();
                    ComputeAll();
                    lock.lock();
                }
            }
      

        }
    }
};

#pragma endregion

int main(int argc, char** argv)
{
    Example exampe("MageML", argc, argv);

    if (exampe.Create())
        return exampe.Run();

    return 0;
}