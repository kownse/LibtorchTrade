#include "RandomSampler.h"

RandomSampler::RandomSampler(const train::TrainOption &option, const ReplayBuffer &buff, torch::Device dev)
    : Sampler(option, buff, dev)
{
}

void RandomSampler::resetIdx()
{
    _idx = torch::randperm(_buff.size(), torch::TensorOptions().dtype(at::kLong));
}