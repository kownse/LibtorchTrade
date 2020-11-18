#ifndef _PPO_SAMPLER_H_
#define _PPO_SAMPLER_H_

#include "Model.h"
#include "../Cache/Batch.h"

namespace train::agent {
    class Agent;
}

namespace train::model {

class PPOSampler : public RNNModel {
public:
    PPOSampler(const data::CSVTensor &data,
        train::agent::Agent *agent,
        const train::TrainOption &option,
        torch::optim::Optimizer *optimizer,
        torch::Device device);
    ~PPOSampler();

    float train(torch::Device device) final;
    float test(torch::Device device, std::vector<float> *actOut) final;

private:
    void mini_batch(BatchTensor &bt);
};

};

#endif // _PPO_H_