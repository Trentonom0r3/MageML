﻿#pragma once

#include "error/CxException.hpp"
#include <Conversion.hpp>
#include <filters/FilterFactory.hpp>
#include <Frame.hpp>

namespace MageML
{

class Decoder
{
  public:

    struct VideoProperties
    {
        std::string codec;
        int width;
        int height;
        double fps;
        double duration;
        int totalFrames;
        AVPixelFormat pixelFormat;
        bool hasAudio;
        int bitDepth;
        double aspectRatio;
        int audioBitrate;
        int audioChannels;
        int audioSampleRate;
        std::string audioCodec;
        double min_fps;
        double max_fps;
    };

    Decoder() = default;
    Decoder(int numThreads, std::vector<std::shared_ptr<FilterBase>> filters);
    bool seekToNearestKeyframe(double timestamp);
    virtual ~Decoder();

    // Deleted copy constructor and assignment operator
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    void addFilter(std::shared_ptr<FilterBase> filter);
    void removeFilter(std::shared_ptr<FilterBase> filter);
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;
    bool seekFrame(int frameIndex);
    virtual bool decodeNextFrame(void* buffer, double* frame_timestamp = nullptr);
    virtual bool seek(double timestamp);
    virtual VideoProperties getVideoProperties() const;
    virtual bool isOpen() const;
    virtual void close();
    virtual std::vector<std::string> listSupportedDecoders() const;
    AVCodecContext* getCtx();
    int getBitDepth() const;
    bool extractAudioToFile(const std::string& outputFilePath);
    torch::Tensor getAudioTensor();

  protected:
    void initialize(const std::string& filePath);
    void setProperties();
    virtual void openFile(const std::string& filePath);
    virtual void findVideoStream();
    virtual void initCodecContext();
    virtual int64_t convertTimestamp(double timestamp) const;
    void populateProperties();
    void setFormatFromBitDepth();
    double getFrameTimestamp(AVFrame* frame);

    std::vector<std::shared_ptr<FilterBase>> filters_;

    bool initFilterGraph();
    void set_sw_pix_fmt(AVCodecContextPtr& codecCtx, int bitDepth);

    AVFilterGraphPtr filter_graph_;
    AVFilterContext* buffersrc_ctx_;
    AVFilterContext* buffersink_ctx_;
    AVFormatContextPtr formatCtx;
    AVCodecContextPtr codecCtx;
    AVPacketPtr pkt;
    int videoStreamIndex;
    VideoProperties properties;
    Frame frame;
    std::unique_ptr<MageML::conversion::IConverter> converter;
    int numThreads;

    // Audio-specific members
    int audioStreamIndex = -1;
    AVCodecContextPtr audioCodecCtx;
    Frame audioFrame;
    AVPacketPtr audioPkt;
    SwrContextPtr swrCtx;


    void closeAudio();
};
} // namespace MageML
