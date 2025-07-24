//------------------------------------------------------------------------------
// LICENSE
//   This software is dual-licensed to the public domain and under the following
//   license: you are granted a perpetual, irrevocable license to copy, modify,
//   publish, and distribute this file as you see fit.
//
// CREDITS
//   Written by Michal Cichon
//------------------------------------------------------------------------------
# pragma once


//------------------------------------------------------------------------------
#include <imgui-node-editor/imgui_node_editor.h>

#pragma once
#include <imgui.h>

/// A minimal stubbed layout API (horizontal/vertical groups + springs)
/// so that your BlueprintNodeBuilder (or any code) can call
/// BeginHorizontal/EndHorizontal, BeginVertical/EndVertical, Spring, etc.,
/// without touching the ImGui namespace directly.

namespace NodeLayout
{
// --- Horizontal group ---
inline void BeginHorizontal(const char* /*str_id*/,
                            const ImVec2& /*size*/ = ImVec2(0, 0),
                            float /*align*/ = -1.0f)
{
    ImGui::BeginGroup();
}
inline void BeginHorizontal(const void* /*ptr_id*/,
                            const ImVec2& /*size*/ = ImVec2(0, 0),
                            float /*align*/ = -1.0f)
{
    ImGui::BeginGroup();
}
inline void BeginHorizontal(int /*id*/, const ImVec2& /*size*/ = ImVec2(0, 0),
                            float /*align*/ = -1.0f)
{
    ImGui::BeginGroup();
}
inline void EndHorizontal()
{
    ImGui::EndGroup();
}

// --- Vertical group ---
inline void BeginVertical(const char* /*str_id*/, const ImVec2& /*size*/ = ImVec2(0, 0),
                          float /*align*/ = -1.0f)
{
    ImGui::BeginGroup();
}
inline void BeginVertical(const void* /*ptr_id*/, const ImVec2& /*size*/ = ImVec2(0, 0),
                          float /*align*/ = -1.0f)
{
    ImGui::BeginGroup();
}
inline void BeginVertical(int /*id*/, const ImVec2& /*size*/ = ImVec2(0, 0),
                          float /*align*/ = -1.0f)
{
    ImGui::BeginGroup();
}
inline void EndVertical()
{
    ImGui::EndGroup();
}

// --- Spring (spacer) ---
// You can later enhance this to push/pull available space.
inline void Spring(float /*weight*/ = 1.0f, float /*spacing*/ = -1.0f)
{
    // simplest no-op; you could do e.g. ImGui::Spacing();
}
} // namespace NodeLayout

//------------------------------------------------------------------------------
namespace ax {
namespace NodeEditor {
namespace Utilities {


//------------------------------------------------------------------------------
struct BlueprintNodeBuilder
{
    BlueprintNodeBuilder(ImTextureID texture = (ImTextureID)0, int textureWidth = 0, int textureHeight = 0);

    void Begin(NodeId id);
    void End();

    void Header(const ImVec4& color = ImVec4(1, 1, 1, 1));
    void EndHeader();

    void Input(PinId id);
    void EndInput();

    void Middle();

    void Output(PinId id);
    void EndOutput();


private:
    enum class Stage
    {
        Invalid,
        Begin,
        Header,
        Content,
        Input,
        Output,
        Middle,
        End
    };

    bool SetStage(Stage stage);

    void Pin(PinId id, ax::NodeEditor::PinKind kind);
    void EndPin();

    ImTextureID HeaderTextureId;
    int         HeaderTextureWidth;
    int         HeaderTextureHeight;
    NodeId      CurrentNodeId;
    Stage       CurrentStage;
    ImU32       HeaderColor;
    ImVec2      NodeMin;
    ImVec2      NodeMax;
    ImVec2      HeaderMin;
    ImVec2      HeaderMax;
    ImVec2      ContentMin;
    ImVec2      ContentMax;
    bool        HasHeader;
};



//------------------------------------------------------------------------------
} // namespace Utilities
} // namespace Editor
} // namespace ax