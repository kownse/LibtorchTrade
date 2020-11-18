#ifndef _RANDOM_SAMPLER_H_
#define _RANDOM_SAMPLER_H_

#include "Sampler.h"

class RandomSampler final : public Sampler{

public:
    RandomSampler(const train::TrainOption &option, const ReplayBuffer &buff, torch::Device dev);
    ~RandomSampler() = default;

    void resetIdx() override;
};

#endif // _RANDOM_SAMPLER_H_