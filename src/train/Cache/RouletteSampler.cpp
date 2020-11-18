#include "RouletteSampler.h"

RouletteSampler::RouletteSampler(const train::TrainOption &option, const ReplayBuffer &buff, torch::Device dev)
    : Sampler(option, buff, dev)
{
    // auto expect = (_bufTensor._diffs.abs() - _bufTensor._diffs) / 2 + _bufTensor._diffs;
    _cumsum = torch::cumsum(_bufTensor._diffs.abs(), 0);
}

void RouletteSampler::resetIdx()
{
    auto scores = torch::rand_like(_cumsum) * _cumsum.max();
    _idx = torch::zeros_like(scores, torch::TensorOptions().dtype(at::kLong));
    for (auto i = 0; i < _idx.sizes()[0]; ++i) {
        _idx[i] = (_cumsum > scores[i]).nonzero()[0].item<long>();
    }
}