#include "Model.h"
#include "PPOSampler.h"
#include "../TrainOption.h"
#include "../agent/Agent.h"
#include "../../utils/TensorUtil.h"

using torch::indexing::Slice;
using torch::indexing::None;

namespace train::model {

Model *Model::create(const data::CSVTensor &data,
    train::agent::Agent *agent,
    const train::TrainOption &option,
    torch::optim::Optimizer *optimizer,
    torch::Device device) {
    if (option.modelType == "PPOSampler") {
        return new PPOSampler(data, agent, option, optimizer, device);
    }
    
    return nullptr;
}

Model::Model(const data::CSVTensor &data,
    train::agent::Agent *agent,
    const train::TrainOption &option,
    torch::optim::Optimizer *optimizer,
    torch::Device device)
    :_data(data),
    _agent(agent),
    _options(option),
    _optimizer(optimizer),
    _buff(_options.trainLen, _agent->hxSize()),
    _device(device),
    _diff(util::tensor::ToDevice(data.getCol("diff").index({Slice(option.normalizeLen, None)}),device))
    // _high_diff(util::tensor::ToDevice(data.getCol("high_diff").index({Slice(option.normalizeLen, None)}),device)),
    // _low_diff(util::tensor::ToDevice(data.getCol("low_diff").index({Slice(option.normalizeLen, None)}),device))
{
}

Model::~Model() {

}

std::vector<torch::Tensor> Model::trade(torch::Tensor &state, std::vector<torch::Tensor> &hs) {
    std::vector<torch::Tensor> params{state};
    for (auto &h : hs)
        params.push_back(h);

    torch::NoGradGuard no_grad;
    auto ret = _agent->forward(params);

    hs.clear();
    // 1 -> action
    // 2 -> value
    for (size_t i = 2; i < ret.size(); ++i)
        hs.push_back(ret[i]);

    return ret;
}

}