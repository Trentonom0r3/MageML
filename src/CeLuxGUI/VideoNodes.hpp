#include <CeLuxGUI/NodeBase.hpp>

class VideoReaderNode : public Node
{
  public:
    using Node::Node;

    std::string videoPath;
    std::shared_ptr<VideoPipeline> pipeline;
    bool playing = false;
    bool eof = false;

    // 👇 Persistent editable buffer
    char pathBuffer[256] = "";

    void PanelUI() override
    {
        ImGui::PushID(ID.AsPointer());
        ImGui::TextUnformatted("Video Reader");
        ImGui::Spacing();

        // 🔄 Sync string → buffer on first draw (buffer empty)
        if (pathBuffer[0] == '\0' && !videoPath.empty())
        {
            std::strncpy(pathBuffer, videoPath.c_str(), sizeof(pathBuffer) - 1);
            pathBuffer[sizeof(pathBuffer) - 1] = '\0';
        }

        // 🖊 Editable path input (pressing Enter opens it)
        bool enterPressed = ImGui::InputText("Path", pathBuffer, sizeof(pathBuffer),
                                             ImGuiInputTextFlags_EnterReturnsTrue);

        // 🔄 Always sync buffer → string (for click-away behavior)
        videoPath = pathBuffer;

        ImGui::SameLine();
        if (ImGui::Button(pipeline ? "Re-open" : "Open") || enterPressed)
        {
            openPipeline();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (!pipeline)
        {
            ImGui::TextDisabled("No video loaded.");
            ImGui::PopID();
            return;
        }

        // ───────── Timeline slider ─────────
        float cur = pipeline->currentTime();
        double dur = pipeline->duration();
        double fps = pipeline->getVideoFps();
        size_t totalF = pipeline->totalFrames();

        if (ImGui::SliderFloat("Time", &cur, 0.0, dur, "%.3f s"))
        {
            pipeline->seek(cur);
            eof = false;
        }

        // ───────── Playback buttons ────────
        ImGui::BeginGroup();

        if (ImGui::Button(playing ? "Pause" : "Play"))
            playing = !playing;

        ImGui::SameLine();
        if (ImGui::Button("Step"))
            stepFrame();

        ImGui::SameLine();
        if (ImGui::Button("⏮"))
        {
            pipeline->seek(0.0);
            playing = false;
            eof = false;
        }

        ImGui::EndGroup();

        // ───────── Status ──────────────────
        ImGui::Spacing();
        ImGui::Text("FPS: %.2f  |  Frames: %zu", fps, totalF);
        ImGui::Text("Size: %d × %d", pipeline->width(), pipeline->height());

        if (eof)
        {
            ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "End-of-stream");
        }

        ImGui::PopID();
    }

    void compute() override
    {
        if (!pipeline)
            return;
        if (playing && !eof)
            stepFrame();
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
        playing = false;
        eof = false;
    }

    void stepFrame()
    {
        switch (pipeline->step())
        {
        case StepResult::NewFrame:
            Outputs[0].Value = pipeline->getCurrentFrame();
            break;
        case StepResult::EndOfStream:
            playing = false;
            eof = true;
            break;
        }
    }
};
