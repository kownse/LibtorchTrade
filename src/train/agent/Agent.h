#ifndef _AGENT_H_
#define _AGENT_H_

#include <torch/torch.h>
#include <string>
#include <vector>

namespace train {
    struct TrainOption;
};

namespace train::agent {

class Agent : public torch::nn::Module {
public:
    Agent(size_t hxs) : _hx_size(hxs) {}
    static Agent* create(const TrainOption &option);

    virtual std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x) = 0;

    void saveToFile(std::string path);
    void loadFromFile(std::string path);

    size_t hxSize() { return _hx_size; }

protected:
    size_t _hx_size = 1;
};
};

#endif // _AGENT_H_