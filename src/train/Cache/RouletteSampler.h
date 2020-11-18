#ifndef _ROULETTE_SAMPLER_H_
#define _ROULETTE_SAMPLER_H_

#include "Sampler.h"

class RouletteSampler final : public Sampler{

public:
    RouletteSampler(const train::TrainOption &option, const ReplayBuffer &buff, torch::Device dev);
    ~RouletteSampler() = default;

    void resetIdx() override;
private:
    torch::Tensor _cumsum;
};

#endif // _ROULETTE_SAMPLER_H_