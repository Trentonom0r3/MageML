#ifndef CxException_HPP
#define CxException_HPP

#include "MageML/CxCore.hpp"

#define FF_CHECK(func)                                                                 \
    do                                                                                 \
    {                                                                                  \
        int errorCode = func;                                                          \
        if (errorCode < 0)                                                             \
        {                                                                              \
            throw MageML::error::CxException(errorCode);                                \
        }                                                                              \
    } while (0)

#define FF_CHECK_MSG(func, msg)                                                        \
    do                                                                                 \
    {                                                                                  \
        int errorCode = func;                                                          \
        if (errorCode < 0)                                                             \
        {                                                                              \
            throw MageML::error::CxException(msg + ": " +                               \
                                            MageML::errorToString(errorCode));          \
        }                                                                              \
    } while (0)

namespace MageML
{
namespace error
{
class CxException : public std::exception
{
  public:
    explicit CxException(const std::string& message) : errorMessage(message)
    {
    }

    // Use FFmpeg::errorToString to be explicit
    explicit CxException(int errorCode) : errorMessage(MageML::errorToString(errorCode))
    {
    }

    virtual const char* what() const noexcept override
    {
        return errorMessage.c_str();
    }

  private:
    std::string errorMessage;
};
} // namespace error
} // namespace MageML

#endif // CxException_HPP