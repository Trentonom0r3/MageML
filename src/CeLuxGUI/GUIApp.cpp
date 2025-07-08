#include "GUIApp.hpp"
#include "CeluxGUI/core/GraphManager.hpp"
#include "CeluxGUI/nodes/TensorNodes.hpp"
#include <GLFW/glfw3.h>
#include <imgui-node-editor/imgui_node_editor.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace ed = ax::NodeEditor;

void GUIApp::run()
{
    if (!glfwInit())
        return;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "CeLux GUI", nullptr, nullptr);
    if (!window)
        return;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags &= ~ImGuiConfigFlags_DockingEnable;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    static ImVec2 s_PopupPos = ImVec2(0, 0);


    io.Fonts->Clear();
    if (auto f = io.Fonts->AddFontFromFileTTF("NotoSans_Condensed-Black.ttf", 18.0f))
        io.FontDefault = f;

    GraphManager graph;

    // ─── Install custom link & flow style ───────────────────────────────────────
    ed::SetCurrentEditor(graph.getEditorContext());
    {
        auto& style = ed::GetStyle();

        style.LinkStrength = 100.0f;  // stiffness of the curve

        // Flow animation
        style.Colors[ed::StyleColor_Flow] = ImColor(255, 128, 64, 255);
        style.Colors[ed::StyleColor_FlowMarker] = ImColor(255, 128, 64, 255);
        style.FlowMarkerDistance = 20.0f; // spacing between dots
        style.FlowSpeed = 120.0f;         // speed of the dots
        style.FlowDuration = 1.0f;        // fade‐out time

        // Snap the link tangent to the pin direction
        style.SnapLinkToPinDir = 1.0f; // 1 = enforce, 0 = free curves

        // You can adjust any other style.* field here…
    }
    ed::SetCurrentEditor(nullptr);
    char filterBuf[128] = "";

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ------ FULLSCREEN, BORDERLESS WINDOW ------
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

        // No title, no resize, no move, no scroll, no saved settings, no bring to front
        ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoBringToFrontOnFocus;

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Content", nullptr, flags);
        {
            ImVec2 contentRegion = ImGui::GetContentRegionAvail();
            ed::SetCurrentEditor(graph.getEditorContext());
            // ... inside your main window, inside the node editor Begin/End block:
            ed::Begin("Node editor", contentRegion);

            // Your graph drawing here
            graph.drawUI();
            graph.evaluate();
            ed::Suspend();
               // 2. Open the popup ONLY when right click on the node editor background
            if (ed::ShowBackgroundContextMenu())
            { // stash click pos and open popup
                s_PopupPos = ImGui::GetMousePosOnOpeningCurrentPopup();
                ImGui::OpenPopup("NodeCreation");

            }
            ed::Resume();

            ed::Suspend();
            // 1. Always call BeginPopup, every frame (does nothing if not open)
            if (ImGui::BeginPopup("NodeCreation"))
            {

                ImGui::TextUnformatted("Create Node:");
                ImGui::Separator();
                for (auto& type : NodeFactory::instance().availableTypes())
                {
                    if (ImGui::MenuItem(type.c_str()))
                    {
                        ImVec2 clickPos = s_PopupPos;
                        auto node =
                            NodeFactory::instance().create(type, graph.nextNodeId());
                        graph.addNodeAt(node, clickPos);
                    }
                }
                ImGui::EndPopup();
            }
            ed::Resume();


            // create a button to start pipelineexecution
            if (ImGui::Button("Start Pipeline Execution"))
            {
                graph.runPipelineAsync();
            }
   

            ed::End();

            ed::SetCurrentEditor(nullptr);
        }
        ImGui::End();


        // --- Render everything ---
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.12f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}
