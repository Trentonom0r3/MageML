// CPU Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace MageML::backends::cpu
{
class Decoder : public MageML::Decoder
{
  public:
    Decoder(const std::string& filePath, int numThreads,
            std::vector<std::shared_ptr<FilterBase>> filters)
        : MageML::Decoder( numThreads, filters)
    {
        initialize(filePath);
    }

    // No need to override methods unless specific behavior is needed
};
} // namespace MageML::backends::cpu
