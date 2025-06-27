#include "CeluxGUI/nodes/RenderPreviewNode.hpp"
#include <iostream>
#include <torch/torch.h>

RenderPreviewNode::RenderPreviewNode()
    : textureID(0), isAllocated(false), texWidth(0), texHeight(0)
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

    // Fetch the frame tensor
    auto it = tensorInputs.find("frame");
    if (it == tensorInputs.end())
    {
       // std::cerr << "[RenderPreviewNode] ERROR: no 'frame' key in tensorInputs\n";
        return;
    }
    torch::Tensor frame = it->second;

    if (!frame.defined() || frame.numel() == 0)
    {
       // std::cerr << "[RenderPreviewNode] frame tensor is empty or undefined\n";
        return;
    }

    // Debug: shape & dtype
    std::cerr << "[RenderPreviewNode] raw tensor: dtype=" << frame.scalar_type()
              << " shape=[";
    for (auto d : frame.sizes())
        std::cerr << d << ",";
    std::cerr << "]\n";

    // Ensure uint8
    torch::Tensor img = frame;
    if (img.scalar_type() == torch::kFloat32 || img.scalar_type() == torch::kFloat64)
    {
        img = img.mul(255.0f).clamp(0, 255).to(torch::kUInt8);
    }
    else if (img.scalar_type() != torch::kUInt8)
    {
        img = img.to(torch::kUInt8);
    }

    // Fix orientation by flipping height axis on CHW
    img = img.flip({0,2,1});

    // Convert from CHW -> HWC, remove last idx
    img = img.clone().contiguous().cpu();

    int height = static_cast<int>(img.size(1));
    int width = static_cast<int>(img.size(2));

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

void RenderPreviewNode::drawUI(const std::string& uid)
{
    // We'll embed the texture right in this node's window.
    ImGui::TextUnformatted(
        typeName().c_str()); // window title is already drawn by GraphManager
    ImGui::Separator();

    // Fetch our texture info
    GLuint tex = getTextureID();
    int w = getWidth(), h = getHeight();
    if (tex && w > 0 && h > 0)
    {
        // compute a size that preserves aspect ratio
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float aspect = float(w) / float(h);
        if (avail.x / avail.y > aspect)
            avail.x = avail.y * aspect;
        else
            avail.y = avail.x / aspect;

        // draw it
        ImGui::Image((ImTextureID)(intptr_t)tex, avail, ImVec2(1, 1), ImVec2(0, 0));
    }
    else
    {
        ImGui::TextDisabled("No texture");
    }
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
        "RenderPreview", []() { return std::make_shared<RenderPreviewNode>(); });
    return true;
}();
} // namespace
