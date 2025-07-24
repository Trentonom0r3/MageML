// FrameCache.hpp
#pragma once
#include <Logger.hpp>
#include <mutex>
#include <string>
#include <torch/torch.h>
#include <unordered_map>

class FrameCache
{
  public:
    void saveChunk(const std::string& key, torch::Tensor frame)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cache_[key] = std::move(frame);
        //   CELUX_INFO("SAVING CHUNK WITH KEY {}", key);
    }

    bool hasChunk(const std::string& key)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        return cache_.count(key) != 0;
    }

    torch::Tensor loadChunk(const std::string& key)
    {
        //  CELUX_INFO("Loading CHunk for Key : {}", key);
        std::lock_guard<std::mutex> lk(mutex_);
        return cache_.at(key); // throws if missing
    }

    void clearCache()
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cache_.clear();
        //   CELUX_INFO("Cache CLeared");
    }

    size_t size() 
    {
        std::lock_guard<std::mutex> lk(mutex_);
        return cache_.size();
    }


  private:
    std::unordered_map<std::string, torch::Tensor> cache_;
    std::mutex mutex_;
};