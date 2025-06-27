#include "GUIApp.hpp"
#include "CeluxGUI/core/GraphManager.hpp"
#include "CeluxGUI/nodes/RenderPreviewNode.hpp"
#include "CeluxGUI/nodes/TensorNodes.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_stdlib.h> // for ImGui::InputText(std::string*)

void GUIApp::run()
{
    // 1) GLFW + ImGui init (unchanged)…
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
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    // 2) Our dynamic graph
    GraphManager graph;
    static char filterBuf[128] = "";

    // 3) Main loop
    while (!glfwWindowShouldClose(window))
    {
        // a) New frame
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // b) Node Browser (search & categorized add in a list)
        ImGui::Begin("Node Browser");
        ImGui::InputTextWithHint("##node_filter", "Type to filter nodes...", filterBuf,
                                 sizeof(filterBuf));
        ImGui::Separator();

        // Standard nodes (excluding "Filter")
        for (auto& typeName : NodeFactory::instance().availableTypes())
        {
            if (typeName == "Filter")
                continue;
            if (!*filterBuf || typeName.find(filterBuf) != std::string::npos)
            {
                if (ImGui::Selectable(typeName.c_str()))
                {
                    graph.addNode(NodeFactory::instance().create(typeName));
                }
            }
        }


        ImGui::End();

        // c) Right-click quick-add at mouse location
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !ImGui::IsAnyItemHovered())
            ImGui::OpenPopup("AddNode");

        if (ImGui::BeginPopup("AddNode"))
        {
            ImVec2 spawnPos = ImGui::GetIO().MousePos;

            // Other nodes
            for (auto& t : NodeFactory::instance().availableTypes())
            {
                if (t == "Filter")
                    continue;
                if (ImGui::MenuItem(t.c_str()))
                {
                    auto node = NodeFactory::instance().create(t);
                    graph.addNodeAt(node, spawnPos);
                }
            }

            ImGui::EndPopup();
        }

        // d) Draw & interact with the graph
        graph.drawUI();

        // e) Evaluate all nodes
        graph.evaluate();

        // f) Render & swap
        ImGui::Render();
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.1f, 0.12f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // 4) Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}
