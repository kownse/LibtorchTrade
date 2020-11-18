#ifndef _REPLAY_BUFFER_H_
#define _REPLAY_BUFFER_H_

#include <torch/torch.h>
#include <deque>
#include <vector>
#include "Batch.h"

class ReplayBuffer {
public:
    ReplayBuffer(unsigned maxSize, size_t hsize) : _hxs(hsize, std::deque<torch::Tensor>()), _maxSize(maxSize) {}
    ~ReplayBuffer() = default;

    void push_back(torch::Tensor &state,
                torch::Tensor &diff,
                torch::Tensor &act,
                torch::Tensor &ret,
                std::vector<torch::Tensor> &hx);
    
    size_t size() const {
        return _states.size();
    }

    size_t maxSize() const {
        return _maxSize;
    }

    BatchTensor toTensor(torch::Device dev) const;

private:
    std::vector< std::deque<torch::Tensor> > _hxs;
    std::deque< torch::Tensor > _states;
    std::deque< torch::Tensor > _acts;
    std::deque< torch::Tensor > _rets;
    std::deque< torch::Tensor > _diffs;
    unsigned _maxSize = 2412;

    friend class Sampler;
};

#endif // REPLAY_BUFFER