#ifndef _ATTENTION_GRU_H_
#define _ATTENTION_GRU_H_

#include <torch/torch.h>
#include "Agent.h"
#include "Attention.h"

namespace train {
    struct TrainOption;
};

namespace train::agent {

class AttGru : public Agent {
public:
    AttGru(const train::TrainOption &option);
    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs);

private:
    train::agent::Attention attn;
    torch::nn::GRU gru;
    torch::nn::Linear fc_s_1;
    torch::nn::Linear fc_s_2;
    torch::nn::Linear fc_s_out;
    torch::nn::Linear fc_pg_1;
    torch::nn::Linear fc_pg_2;
    torch::nn::Linear fc_pg_out;
    torch::nn::BatchNorm1d norm_gru, norm_fc_1, norm_fc_2, norm_pg_1, norm_pg_2;
    float dropout;
};

};

#endif // _ATTENTION_GRU_H_