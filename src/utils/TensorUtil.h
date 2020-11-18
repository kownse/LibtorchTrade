#ifndef _TENSOR_UTIL_H_
#define _TENSOR_UTIL_H_

#include <torch/torch.h>
#include <vector>

using torch::indexing::Slice;
using torch::indexing::None;

namespace util::tensor {

inline torch::Tensor selectCols(const torch::Device device, const torch::Tensor &src, std::vector<long> cols)
{
    auto idx = torch::tensor(cols);
    return src.index({Slice(), idx});
}

template<typename T>
inline T ToDevice(const T &src, const torch::Device device) {
    return src.to(device);
}

torch::Tensor calcZscores(const torch::Device device, const torch::Tensor &src, long normalizeLen);

}; // util::tensor

#endif // _TENSOR_UTIL_H_