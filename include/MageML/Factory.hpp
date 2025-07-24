#pragma once
#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <MageML/backends/Decoders.hpp>

using ConverterKey = std::tuple<bool, AVPixelFormat>;

// Hash function for ConverterKey
struct ConverterKeyHash
{
    std::size_t operator()(const std::tuple<bool, AVPixelFormat>& key) const
    {
        return std::hash<bool>()(std::get<0>(key)) ^
               std::hash<int>()(static_cast<int>(std::get<1>(key)));
    }
};

namespace MageML
{

/**
 * @brief Factory class to create Decoders, Encoders, and Converters based on backend
 * and configuration.
 */
class Factory
{
  public:
    /**
     * @brief Creates a Decoder instance based on the specified backend.
     *
     * @param backend Backend type (CPU or CUDA).
     * @param filename Path to the video file.
     * @param converter Unique pointer to the IConverter instance.
     * @return std::unique_ptr<Decoder> Pointer to the created Decoder.
     */
    static std::shared_ptr<Decoder>
    createDecoder(torch::Device device, const std::string& filename, int numThreads,
                  std::vector<std::shared_ptr<FilterBase>> filters)
    {

        return std::make_shared<MageML::backends::cpu::Decoder>(filename, numThreads,
                                                               filters);
    }

    /**
     * @brief Creates a Converter instance based on the specified backend and pixel
     * format.
     *
     * @param device Device type (CPU or CUDA).
     * @param pixfmt Pixel format.
     * @param  Optional  for CUDA operations.
     * @return std::unique_ptr<MageML::conversion::IConverter> Pointer to the created
     * Converter.
     */
    static std::unique_ptr<MageML::conversion::IConverter>
    createConverter(const torch::Device& device, AVPixelFormat pixfmt)
    {
        using namespace MageML::conversion; // For IConverter
        return std::make_unique<MageML::conversion::cpu::AutoToRGB24Converter>();
    }

};

} // namespace MageML

#endif // FACTORY_HPP
