#include "Trainer.h"
#include "../data/CSVTensor.h"
#include "agent/Agent.h"
#include "model/Model.h"
#include "../utils/TimerCounter.h"
#include "../utils/Compiler.h"
#include <chrono>
#include <string>
#include <thread>
using namespace std::chrono;

#include "../utils/Global.h"
#include "../utils/ctpl_stl.h"

#include "../utils/Logger.h"
static Logger gLogger("Trainer");

using torch::indexing::Slice;
using torch::indexing::None;

namespace train {

static float trainFunc(data::CSVTensor *assetData, train::TrainOption *option, TimerCounter *tc) {
  train::Trainer trainer(*assetData, *option, g_device);
  return trainer.train(tc);
}

static std::tuple<float, long> pickParamAndTrain(int id, data::CSVTensor *assetData, train::TrainOption *option, TimerCounter *tc) {
  return std::make_tuple(trainFunc(assetData, option, tc), option->diff_col);
}

static bool feature_add_internal(train::TrainOption &base, data::CSVTensor &assetData, unsigned threads, unsigned batchMultiplier) {
  std::vector<train::TrainOption> params;
  TimerCounter tc(threads, 1);
  float best_score = 0.0f;

  while (true) {
    // gLogger.debug() << "base.cols.size=" << base.cols.size() << std::endl;
    std::set<long> seen(base.cols.begin(), base.cols.end());
    params.clear();
    params.reserve(assetData.numCol());

    for (long i = 0; i < assetData.numCol(); ++i) {
      if (seen.count(i))
        continue;

      train::TrainOption newconf = base;
      newconf.cols.push_back(i);
      newconf.diff_col = i;
      newconf.repeat = 1;
      for (int i = 0; i < base.repeat; ++i) {
        params.push_back(newconf);
      }
    }

    tc.setTotalRound(params.size());
    pickParamAndTrain(0, &assetData, &params[0], &tc);
    return false;

    long best_col = -1;
    float best_score_round = 0.0f;
    long best_col_round = -1;
    size_t idx = 0;
    // put the cumputation into batches to avoid huge memory usage
    const size_t BATCH_SIZE = threads * batchMultiplier;
    const size_t FINISH_CNT = params.size();
    while (idx < FINISH_CNT) {
      ctpl::thread_pool p(threads);
      std::vector<std::future<std::tuple<float, long> > > results;
      for (size_t cnt = 0; idx < params.size() && cnt < BATCH_SIZE; ++idx, ++cnt) {
        results.push_back(p.push(pickParamAndTrain, &assetData, &params[idx], &tc));
      }

      for (auto &r : results) {
        std::tuple<float, long> score_col = r.get();
        float score = std::get<0>(score_col);
        long col = std::get<1>(score_col);

        if (score > best_score_round) {
          best_score_round = score;
          best_col_round = col;
        }
      }
    }

    if (best_score_round > best_score) {
      best_col = best_col_round;
      if (best_score_round > best_score)
        best_score = best_score_round;
    }
    // gLogger.debug() << std::cout << "best_col=" << best_col << std::endl;
    if (best_col < 0)
      break;

    base.cols.push_back(best_col);
  }

  return true;
}

void feature_add_rnn(train::TrainOption &base, data::CSVTensor &assetData, unsigned threads, unsigned batchMultiplier)
{
  gLogger.info() << "feature_add_rnn start" << std::endl;
  auto init_cols = base.cols;
  long init_rnnBase = base.rnnBase;
  // long init_leanerBase = base.linearBase;

  while (base.linearBase <= 32) {
    base.rnnBase = init_rnnBase;
    base.cols = init_cols;

    while (base.rnnBase <= 128) {
      
      if (!feature_add_internal(base, assetData, threads, batchMultiplier))
        return;

      base.rnnBase *= 2;
    }

    base.linearBase *= 2;
  }
  gLogger.info() << "feature_add_rnn end" << std::endl;
}

void feature_add_cnn(train::TrainOption &base, data::CSVTensor &assetData,
                    unsigned threads, unsigned batchMultiplier,
                    std::vector< std::vector<unsigned> > &params)
{
    gLogger.info() << "feature_add_cnn start" << std::endl;
    for (auto &param : params) {
        base.rnnBase = param[0];
        base.cnn_kernel_size = param[1];
        base.cnn_stride = param[2];
        base.maxpool_size = param[3];
        base.maxpool_stride = param[4];
        base.cnn_outsize = param[5];

        feature_add_internal(base, assetData, threads, batchMultiplier);
    }
    gLogger.info() << "feature_add_cnn end" << std::endl;
}

Trainer::Trainer(const data::CSVTensor &data, TrainOption &option, torch::Device device)
    : _data(data)
    , _options(option)
    , _device(device)
{
}

Trainer::~Trainer() {

}

#define INIT_MODEL   \
std::unique_ptr<train::agent::Agent> _agent(agent::Agent::create(_options));   \
std::unique_ptr<torch::optim::Optimizer> _opt(createOptimizer(_options, _agent.get()));  \
std::unique_ptr<model::Model> _model(model::Model::create(_data, _agent.get(), _options, _opt.get(), _device));   \
_agent->to(_device);

torch::optim::Optimizer *Trainer::createOptimizer(const TrainOption &option, train::agent::Agent *agent) {
    if (option.optType == "Adam") {
        return new torch::optim::Adam(agent->parameters(), torch::optim::AdamOptions(option.learnRate));
    } else if (option.optType == "SGD") {
        return new torch::optim::SGD(agent->parameters(), torch::optim::SGDOptions(option.learnRate));
    }
    return nullptr;
}

float Trainer::train(TimerCounter *tc) {
    float max_test_reward_tot = 0.0f;

    for (unsigned round = 0; round < _options.repeat; ++round) {

        float last_train_reward = 0.0f;
        unsigned train_stuck_cnt = 0;
        float max_test_reward = 0.0f;
        unsigned patient = 0;

        INIT_MODEL;

        uint64_t round_start = duration_cast< milliseconds >(
                system_clock::now().time_since_epoch()
            ).count();

        unsigned epoch_cnt = 0;
        unsigned best_epoch = -1;
        for (; epoch_cnt < _options.maxEpoch; ++epoch_cnt) {
            uint64_t start = duration_cast< milliseconds >(
                system_clock::now().time_since_epoch()
            ).count();

            float train_reward = _model->train(_device);
            float test_reward = _model->test(_device, nullptr);

            gLogger.debug() <<  epoch_cnt << " train_reward=" << train_reward << " test_reward=" << test_reward << std::endl;
            if (unlikely(train_reward != last_train_reward)) {
                last_train_reward = train_reward;
                train_stuck_cnt = 0;
            }
            else {
                if (unlikely(++train_stuck_cnt >= 5 )) {
                    gLogger.warn() << " train stucked " << std::endl;
                    break;
                }
            }

            if (unlikely((test_reward > _options.saveCumprod) && (test_reward > max_test_reward * 0.98f)))
            {
                auto tag = _options.genTag();
                tag << "_prod" << test_reward;
                tag << "_epo" << epoch_cnt;
                const auto p1 = std::chrono::system_clock::now();
                tag << "_" << std::chrono::duration_cast<std::chrono::seconds>(
                    p1.time_since_epoch()).count();
                tag << "_" << _options.sampler;
                tag << ".pt";
                std::string path = "../models_train/" + _options.traintype + "/" + _options.interval + "/" + tag.str();
                gLogger.info() << " save model: epoch=" << epoch_cnt << " path=" << path << std::endl;
                _agent->saveToFile(path);
            }

            if (unlikely(test_reward > max_test_reward)) {
                best_epoch = epoch_cnt;
                max_test_reward = test_reward;
                patient = 0;

                max_test_reward_tot = max_test_reward_tot < test_reward ? test_reward : max_test_reward_tot;
            }
            else if (unlikely(++patient > _options.patient)) {
                break;
            }

            tc->epochEnd(start);
        }
        auto round_elapsed = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count() - round_start;
        gLogger.info() << _options 
                      << " max_test_reward=" << max_test_reward
                      << " best_epoch=" << best_epoch
                      << " elapsed=" << round_elapsed
                      << " duration_epoch=" << round_elapsed / epoch_cnt << std::endl;
        // gLogger.info(); tc->roundEnd();
    }
    
    return max_test_reward_tot;
}

float Trainer::backtest(std::string *path, std::vector<float> *actOut, TimerCounter *tc) {
    INIT_MODEL;

    try {
      _agent->loadFromFile(*path);
      return _model->test(_device, actOut);
    } catch (...) {
      std::cerr << "error in back test path=" << *path;
    }
    return 0.0f;
}

};
