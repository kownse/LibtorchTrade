#include "Sampler.h"
#include "RandomSampler.h"
#include "RouletteSampler.h"
#include <algorithm>

using torch::indexing::Slice;
using torch::indexing::None;

Sampler *Sampler::create(const train::TrainOption &option, const ReplayBuffer &buff, torch::Device dev)
{
    if (option.sampler == "random") {
        return new RandomSampler(option, buff, dev);
    } else if (option.sampler == "roulette") {
        return new RouletteSampler(option, buff, dev);
    } else {
        std::cout << "wrong sampler type: " << option.sampler << std::endl;
    }
    return nullptr;
}

Sampler::Sampler(const train::TrainOption &option, const ReplayBuffer &buff, torch::Device dev)
        : _buff(buff), _options(option), _device(dev),
        _bufTensor(std::move(buff.toTensor(_device)))
{
}

BatchTensor Sampler::sample() {

    auto idx = _idx.index({Slice(_cur_idx, std::min(_cur_idx + _options.batchSize, (long)_options.trainLen))});
    _cur_idx += _options.batchSize;
    
    std::vector< torch::Tensor > hxs;
    for (auto &hx : _bufTensor._hxs) {
        hxs.emplace_back(hx.index({None, idx, 0, 0, Slice()}));
    }

    return {
        std::move(hxs),
        _cnn ? _bufTensor._states.index({idx, 0, Slice(), Slice()}) : _bufTensor._states.index({None, idx, 0, 0, Slice()}),
        _bufTensor._acts.index(idx).flatten(),
        _bufTensor._rets.index(idx).flatten(),
        _bufTensor._diffs.index(idx).flatten()
    };
}