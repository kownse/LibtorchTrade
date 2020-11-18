#include "TensorUtil.h"

namespace util::tensor {

torch::Tensor calcZscores(const torch::Device device, const torch::Tensor &src, long normalizeLen) {
    auto srcSize = src.sizes();
    auto zscores = torch::zeros({srcSize[0] - normalizeLen, srcSize[1]});
  
    for (long i = normalizeLen; i < srcSize[0]; ++i) {
        auto sub = src.index({Slice(i - normalizeLen, i), Slice()});
        auto zscore = (sub.index({None, -1, Slice()}) - sub.mean({0}, true)) / (sub.std({0}, true, true) + 1e-5);
        zscores.index({i - normalizeLen, Slice()}).copy_(zscore.index({0}));
    }
    return zscores;
}

}; // util::tensor