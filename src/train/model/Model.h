#ifndef _MODEL_H_
#define _MODEL_H_

#include <string>
#include <torch/torch.h>
#include "../Cache/ReplayBuffer.h"
#include "../../data/CSVTensor.h"
#include "../TrainOption.h"
#include "../../utils/TensorUtil.h"

namespace train{
    struct TrainOption;

    namespace agent {
        class Agent;
    }
}

namespace train::model {

class Model {
public:
    Model(const data::CSVTensor &data,
        train::agent::Agent *agent,
        const train::TrainOption &option,
        torch::optim::Optimizer *optimizer,
        torch::Device device);

    virtual ~Model();

    static Model *create(const data::CSVTensor &data,
        train::agent::Agent *agent,
        const train::TrainOption &option,
        torch::optim::Optimizer *optimizer,
        torch::Device device);

    virtual float train(torch::Device device) = 0;
    virtual float test(torch::Device device, std::vector<float> *actOut) = 0;

    std::vector<torch::Tensor> trade(torch::Tensor &state, std::vector<torch::Tensor> &hs);

protected:
    const data::CSVTensor &_data;
    train::agent::Agent *_agent;
    const TrainOption &_options;
    torch::optim::Optimizer *_optimizer;
    ReplayBuffer _buff;
    torch::Device _device;

    const torch::Tensor _diff;
};

class RNNModel : public Model {
public:
    RNNModel(const data::CSVTensor &data,
        train::agent::Agent *agent,
        const train::TrainOption &option,
        torch::optim::Optimizer *optimizer,
        torch::Device device)
        : Model(data, agent, option, optimizer, device)
        , _zscores(data.calcZscoresWithCols(option.cols, option.normalizeLen).to(device))
    {}

protected:
    const torch::Tensor _zscores;
};

class CNNModel : public Model {
public:
    CNNModel(const data::CSVTensor &data,
        train::agent::Agent *agent,
        const train::TrainOption &option,
        torch::optim::Optimizer *optimizer,
        torch::Device device)
        : Model(data, agent, option, optimizer, device)
        , _tzscores(data.calcTZscoresWithCols(option.cols, option.normalizeLen).to(device))
    {}

protected:
    const torch::Tensor _tzscores;
};

};

#endif // _MODEL_H_