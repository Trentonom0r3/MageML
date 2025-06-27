#pragma once
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include <any>
#include <algorithm>
#include <cstdint> // For fixed-width integer types
#include <exception>
#include <ostream>   // For std::ostream
#include <stdexcept> // For std::runtime_error


struct PortInfo
{
    std::string name;
    enum
    {
        Tensor,
        Float,
        Int /*…*/
    } type;
};

struct ParamInfo
{
    std::string name;
    enum
    {
        Float,
        Int,
        Bool,
        String,
        Double
    } type;
    std::any defaultValue;
};

class Node
{
  public:
    virtual ~Node() = default;

    /// Draw this node’s window contents (parameters, buttons, etc)
    virtual void drawUI(const std::string& uid) = 0;

    /// Do your compute step
    virtual void compute() = 0;

    /// Describe your pins
    virtual std::vector<PortInfo> inputs() const = 0;
    virtual std::vector<PortInfo> outputs() const = 0;

    /// For an external “inspector” panel
    virtual std::vector<ParamInfo> params() const = 0;
    virtual void setParam(const std::string& name, std::any val) = 0;

    virtual std::string typeName() const = 0;

    std::map<std::string, torch::Tensor> tensorInputs, tensorOutputs;
};

#include <functional>
#include <map>

using NodeConstructor = std::function<std::shared_ptr<Node>()>;

class NodeFactory
{
  public:
    static NodeFactory& instance()
    {
        static NodeFactory f;
        return f;
    }

    void registerType(std::string typeName, NodeConstructor ctor)
    {
        ctors[typeName] = ctor;
    }

    std::vector<std::string> availableTypes() const
    {
        std::vector<std::string> names;
        for (auto& kv : ctors)
            names.push_back(kv.first);
        return names;
    }

    std::shared_ptr<Node> create(const std::string& typeName) const
    {
        auto it = ctors.find(typeName);
        if (it == ctors.end())
            return nullptr;
        return it->second();
    }

  private:
    std::map<std::string, NodeConstructor> ctors;
};
