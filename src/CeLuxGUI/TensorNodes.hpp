#include <CeLuxGUI/NodeBase.hpp>

class TensorNode : public Node
{
  public:
    using Node::Node; // Inherit constructors from Node

    int tensorRank = 4;
    int tensorDims[6] = {1, 3, 224, 224, 1, 1};
    int tensorDTypeIdx = 0;
    int tensorFillIdx = 0;
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

            Outputs[0].Value = t;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::PopID(); // <── done: restores ImGui ID stack
    }
};
