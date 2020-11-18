#ifndef _SAMPLER_H_
#define _SAMPLER_H_

#include "ReplayBuffer.h"
#include "Batch.h"
#include "../TrainOption.h"

class Sampler {

public:
    virtual ~Sampler() = default;

    virtual BatchTensor sample();

    bool hasMore() {
        return _cur_idx < _options.trainLen;
    }

    virtual void resetIdx() = 0;
    void setCnn(bool cnn) { _cnn = cnn; }

    static Sampler *create(const train::TrainOption &option, const ReplayBuffer &buff, torch::Device dev);

protected:
    Sampler(const train::TrainOption &option, const ReplayBuffer &buff, torch::Device dev);

    const ReplayBuffer &_buff;
    const train::TrainOption &_options;
    torch::Device _device;

    BatchTensor _bufTensor;
    torch::Tensor _idx;
    long _cur_idx = 0;
    bool _cnn = false;
};

#endif // _SAMPLER_H_