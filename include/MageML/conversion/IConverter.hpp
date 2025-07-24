// IConverter.hpp

#pragma once

#include "MageML/Frame.hpp"

namespace MageML
{
namespace conversion
{

class IConverter
{
  public:
    virtual ~IConverter()
    {
    }
    virtual void convert(MageML::Frame& frame, void* buffer) = 0;
    virtual void synchronize() = 0;
};

} // namespace conversion
} // namespace MageML
