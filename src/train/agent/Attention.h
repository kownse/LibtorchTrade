#ifndef _ATTENTION_H_
#define _ATTENTION_H_

#include <torch/torch.h>

namespace train::agent {

class Attention : public torch::nn::Module {
public:
    Attention(long d_x, long d_h, float dp);
    ~Attention() = default;

    torch::Tensor forward(torch::Tensor input);

private:
    torch::nn::Linear w;
    float dropout;
};

};


#endif // _ATTENTION_H_