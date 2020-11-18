#include "ReplayBuffer.h"
#include "../../utils/TensorUtil.h"

void ReplayBuffer::push_back(torch::Tensor &state,
                            torch::Tensor &diff,
                            torch::Tensor &act,
                            torch::Tensor &ret,
                            std::vector<torch::Tensor> &hx)
{
    _states.push_back(state);
    _diffs.push_back(diff);
    _acts.push_back(act);
    _rets.push_back(ret);
    for (size_t i = 0; i < hx.size(); ++i) {
        _hxs[i].push_back(hx[i]);
    }

    if (_states.size() > _maxSize) {
        // std::cout << "pop_back from replay buffer" << std::endl;
        _states.pop_front();
        _diffs.pop_front();
        _acts.pop_front();
        _rets.pop_front();

        for (auto &h : _hxs)
            h.pop_front();
    }
}

BatchTensor ReplayBuffer::toTensor(torch::Device dev) const
{
    std::vector< torch::Tensor > hxs;
    for (auto &hx : _hxs) {
        hxs.emplace_back(util::tensor::ToDevice(torch::stack(std::vector<torch::Tensor>(hx.begin(), hx.end())), dev));
    }

    return {
        std::move(hxs),
        util::tensor::ToDevice(torch::stack(std::vector<torch::Tensor>(_states.begin(), _states.end())), dev),
        util::tensor::ToDevice(torch::stack(std::vector<torch::Tensor>(_acts.begin(), _acts.end())), dev),
        util::tensor::ToDevice(torch::stack(std::vector<torch::Tensor>(_rets.begin(), _rets.end())), dev),
        util::tensor::ToDevice(torch::stack(std::vector<torch::Tensor>(_diffs.begin(), _diffs.end())), dev)
    };
}