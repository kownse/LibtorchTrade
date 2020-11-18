#include "PPOSampler.h"
#include "../agent/Agent.h"
#include "../TrainOption.h"
#include "../../utils/Compiler.h"
#include "../Cache/RandomSampler.h"
#include <memory>
#include <algorithm>

using torch::indexing::Slice;
using torch::indexing::None;

namespace train::model {

PPOSampler::PPOSampler(const data::CSVTensor &data,
    train::agent::Agent *agent,
    const train::TrainOption &option,
    torch::optim::Optimizer *optimizer,
    torch::Device device)
    : RNNModel(data, agent, option, optimizer, device)
{
}

PPOSampler::~PPOSampler() {

}

void PPOSampler::mini_batch(BatchTensor &bt) {
    _agent->train();
    _optimizer->zero_grad();

    std::vector<torch::Tensor> params;
    params.emplace_back(std::move(bt._states));
    for (auto &h : bt._hxs)
        params.emplace_back(std::move(h));

    // std::cout << "states.sizes=" << bt._states.sizes() << " h.sizes=" << params[1].sizes() << std::endl;

    auto out = _agent->forward(params);   
    auto actions = out[0].flatten();
    auto values = out[1].flatten();

    auto returns = actions * bt._diffs;
    auto adv = returns - values;
 
    auto clip_loss = -adv.mean();
    auto vf_loss = torch::nn::functional::mse_loss(values, bt._diffs);
    auto loss = clip_loss + vf_loss;
    loss.backward();
    _optimizer->step();
}

float PPOSampler::train(torch::Device device) {
    _agent->eval();
    float cumprod = 1.0f;

    {
        torch::NoGradGuard no_grad_guard;
        std::vector<torch::Tensor> hs(_agent->hxSize(), util::tensor::ToDevice(torch::zeros({1,1,_options.rnnBase}), device));
        for (long i = 0; i < _options.trainLen - 1; ++i) {
            auto state = util::tensor::ToDevice(_zscores.index({None, None, i}), device);
            // std::cout << "state.size=" << state.sizes() << std::endl;
            auto out = trade(state, hs);
            auto action = out[0].flatten();
            auto diff = _diff.index({i + 1});
            auto r = (action * diff)[0];
            cumprod *= (r.item<float>() + 1.0f);
            
            _buff.push_back(state, diff, action, r, hs);
        }
    }
    
    std::unique_ptr<Sampler> sampler(Sampler::create(_options, _buff, device));
    for (int i = 0; i < 4; ++i) {
        sampler->resetIdx();
        while (sampler->hasMore()) {
            auto batchTensor = sampler->sample();
            mini_batch(batchTensor);
        }
    }

    return cumprod;
}

float PPOSampler::test(torch::Device device, std::vector<float> *actOut) {
    torch::NoGradGuard no_grad_guard;
    _agent->eval();

    float cumprod = 1.0f;
    std::vector<torch::Tensor> hs(_agent->hxSize(), util::tensor::ToDevice(torch::zeros({1,1,_options.rnnBase}), device));
    for (long i = _options.trainLen; i < _zscores.sizes()[0] - 1; ++i) {
        auto state = util::tensor::ToDevice(_zscores.index({None, None, i}), device);
        auto out = trade(state, hs);
        auto action = out[0];
        auto r = (action.flatten() * _diff.index({i + 1}))[0].item<float>();
        cumprod *= (r + 1.0f);

        if (unlikely(actOut)) {
            actOut->push_back(action.item<float>());
        }
    }
    return cumprod;
}

};