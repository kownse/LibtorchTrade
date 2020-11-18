#include "Attention.h"

using torch::indexing::Slice;
using torch::indexing::None;

namespace train::agent {

Attention::Attention(long d_x, long d_h, float dp)
    : w(d_x, d_h),
    dropout(dp)
{
    register_module("w", w);
}

torch::Tensor Attention::forward(torch::Tensor input) {
    auto v_t = torch::relu(w->forward(input));
    auto e_t = torch::matmul(v_t, torch::transpose(v_t, 2, 1));
    e_t = torch::softmax(e_t, {-1});
    return torch::matmul(e_t, input);
}


};