#ifndef _BATCH_H_
#define _BATCH_H_

#include <torch/torch.h>
#include <vector>

struct BatchTensor {
    std::vector< torch::Tensor > _hxs;
    torch::Tensor _states;
    torch::Tensor _acts;
    torch::Tensor _rets;
    torch::Tensor _diffs;
};

#endif // _BATCH_H_