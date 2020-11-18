#ifndef _TRAINER_H_
#define _TRAINER_H_

#include <torch/torch.h>
#include <memory>
#include "TrainOption.h"

class TimerCounter;
namespace data {
    class CSVTensor;
};

namespace train::model {
    class Model;
};

namespace train::agent {
    class Agent;
};

namespace train {

void feature_add_rnn(train::TrainOption &base, data::CSVTensor &assetData, unsigned threads, unsigned batchMultiplier);
void feature_add_cnn(train::TrainOption &base, data::CSVTensor &assetData, unsigned threads, unsigned batchMultiplier, std::vector< std::vector<unsigned> > &params);

class Trainer {
public:
    Trainer(const data::CSVTensor &data, TrainOption &option, torch::Device device);
    ~Trainer();

    float train(TimerCounter *tc);
    float backtest(std::string *path, std::vector<float> *actOut, TimerCounter *tc);

private:
    torch::optim::Optimizer *createOptimizer(const TrainOption &option, train::agent::Agent *agent);

    const data::CSVTensor &_data;
    TrainOption &_options;
    torch::Device _device;
};

};

#endif // _TRAINER_H_