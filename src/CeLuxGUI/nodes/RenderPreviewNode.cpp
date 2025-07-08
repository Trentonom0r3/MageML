#include "CeluxGUI/nodes/RenderPreviewNode.hpp"
#include <iostream>
#include <torch/torch.h>

/// Returns vector of (frame_idx, tensor), sorted by frame_idx ascending.
static std::vector<std::pair<int, torch::Tensor>>
getAllFrameXTensors(const std::map<std::string, torch::Tensor>& tensorInputs)
{
    std::vector<std::pair<int, torch::Tensor>> out;
    for (const auto& kv : tensorInputs)
    {
        const std::string& key = kv.first;
        if (key.rfind("frame_", 0) == 0)
        {
            try
            {
                int idx = std::stoi(key.substr(6));
                out.emplace_back(idx, kv.second);
            }
            catch (...)
            {
                // skip keys with non-integer suffix
            }
        }
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return out;
}


RenderPreviewNode::RenderPreviewNode(ed::NodeId id)
    : Node(id), textureID(0), isAllocated(false), texWidth(0), texHeight(0)
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Linear filtering, clamp-to-edge wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glBindTexture(GL_TEXTURE_2D, 0);
}

RenderPreviewNode::~RenderPreviewNode()
{
    if (textureID)
        glDeleteTextures(1, &textureID);
}

void RenderPreviewNode::compute()
{
    // --- Gather all frame_* tensors ---
    std::vector<std::pair<int, torch::Tensor>> frames;
    for (const auto& kv : tensorInputs)
    {
        const std::string& key = kv.first;
        std::cout << "[RenderPreviewNode] key: " << key << "\n";
        if (key.rfind("frame_", 0) == 0) // key starts with "frame_"
        {
            try
            {
                // Extract index from key, e.g., "frame_0" -> 0
                int idx = std::stoi(key.substr(6));
                frames.emplace_back(idx, kv.second);
            }
            catch (...)
            {
                // skip if not a valid number
            }
        }
    }
    // If none found, fallback to "frame"
    if (frames.empty())
    {
        std::cout
            << "[RenderPreviewNode] No frame_x tensors found, checking for 'frame'\n";
        auto it = tensorInputs.find("frame");
        if (it != tensorInputs.end())
        {
            frames.emplace_back(0, it->second);
        }
    }

    if (frames.empty())
    {
        std::cerr
            << "[RenderPreviewNode] ERROR: no frame_x or 'frame' in tensorInputs\n";
        return;
    }

    // Sort by index and get the latest (highest index) frame
    std::sort(frames.begin(), frames.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    const auto& [frameIdx, frameTensor] = frames.back();

    if (!frameTensor.defined() || frameTensor.numel() == 0)
    {
       
        return;
    }

    // --- Frame preparation (your original logic) ---
    torch::Tensor img = frameTensor;
    PrintTensorDebugInfo(img);
    if (img.scalar_type() == torch::kFloat32 || img.scalar_type() == torch::kFloat64)
    {
        img = img.mul(255.0f).clamp(0, 255).to(torch::kUInt8);
    }
    else if (img.scalar_type() != torch::kUInt8)
    {
        img = img.to(torch::kUInt8);
    }

    // Fix orientation by flipping height axis on CHW
    img = img.flip({0, 2, 1}); // double-check if this is your intended axes

    // Convert from CHW -> HWC, remove last idx
    img = img.contiguous().cpu();

    int height = static_cast<int>(img.size(0));
    int width = static_cast<int>(img.size(1));

    // Upload to OpenGL texture
    glBindTexture(GL_TEXTURE_2D, textureID);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Allocate or resize texture storage
    if (!isAllocated || width != texWidth || height != texHeight)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, nullptr);
        texWidth = width;
        texHeight = height;
        isAllocated = true;
    }

    // Update sub-image
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE,
                    img.data_ptr());
    glBindTexture(GL_TEXTURE_2D, 0);

    // Check for GL errors
    if (GLenum err = glGetError(); err != GL_NO_ERROR)
    {
        std::cerr << "[RenderPreviewNode] glError: 0x" << std::hex << err << std::dec
                  << "\n";
    }
}

bool RenderPreviewNode::drawExtraUI() 
{

    // Fetch our texture info
    GLuint tex = getTextureID();
    int w = getWidth();
    int h = getHeight();

    if (tex != 0 && w > 0 && h > 0)
    {
        // compute a size that preserves aspect ratio
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float aspect = float(w) / float(h);
        if (avail.x / avail.y > aspect)
            avail.x = avail.y * aspect;
        else
            avail.y = avail.x / aspect;

        // draw the preview image
        ImGui::Image((ImTextureID)(intptr_t)tex, avail, ImVec2(0, 1), // UV0
                     ImVec2(1, 0) // UV1 (flip Y if needed)
        );
    }
    else
    {
        ImGui::TextDisabled("No texture");
    }

    // We don’t modify any parameters here, so return false
    return false;
}

std::vector<PortInfo> RenderPreviewNode::inputs() const
{
    return {{"frame", PortInfo::Tensor}};
}

std::vector<PortInfo> RenderPreviewNode::outputs() const
{
    return {}; // side-effect only
}

std::string RenderPreviewNode::typeName() const
{
    return "RenderPreview";
}

GLuint RenderPreviewNode::getTextureID() const
{
    return textureID;
}

std::vector<ParamInfo> RenderPreviewNode::params() const
{
    return {}; // no parameters
}

void RenderPreviewNode::setParam(const std::string&, std::any)
{
}


// Auto-register
namespace
{
const bool registered = []()
{
    NodeFactory::instance().registerType(
        "RenderPreview",
        [](ed::NodeId id) { return std::make_shared<RenderPreviewNode>(id); });
    return true;
}();
} // namespace
