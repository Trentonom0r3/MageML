// RenderPreviewNode.hpp
#pragma once

#include "CeluxGUI/core/Node.hpp"

class RenderPreviewNode : public Node
{
  public:
    RenderPreviewNode();
    ~RenderPreviewNode();

    void compute() override;

    std::vector<PortInfo> inputs() const override;
    std::vector<PortInfo> outputs() const override;

    std::vector<ParamInfo> params() const override;

    void setParam(const std::string&, std::any) override;

    std::string typeName() const override;
    void drawUI(const std::string& uid) override;
    GLuint getTextureID() const;
    int getWidth() const
    {
        return texWidth;
    }
    int getHeight() const
    {
        return texHeight;
    }

  private:
    GLuint textureID;
    bool isAllocated = false;
    int texWidth = 0, texHeight = 0;
};
