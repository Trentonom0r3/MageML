#define IMGUI_DEFINE_MATH_OPERATORS
#include "utilities/builders.h"
#include "utilities/widgets.h"
#include <application.h>

#include <imgui_internal.h>
#include <imgui_node_editor.h>

#include <CeLuxGUI/nodes/VideoPipeline.hpp>
#include <algorithm>
#include <any>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ed = ax::NodeEditor;
namespace util = ax::NodeEditor::Utilities;

using namespace ax;

using ax::Widgets::IconType;

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
using PinValue =
    std::variant<std::monostate, bool, int, float, std::string, torch::Tensor>;

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

    std::string State;
    std::string SavedState;

    Node(int id, const char* name, ImColor color = ImColor(255, 255, 255))
        : ID(id), Name(name), Color(color), Type(NodeType::Blueprint), Size(0, 0)
    {
    }

    virtual void compute() {}; //called by processing threads.
    virtual void PanelUI() {};
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
