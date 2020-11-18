#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include "utils/FileUtil.h"
#include "data/CSVTensor.h"
#include "utils/TensorUtil.h"
#include "train/Trainer.h"
#include "train/TrainOption.h"
#include "utils/ctpl_stl.h"
#include "utils/TimerCounter.h"

#include <filesystem>
namespace fs = std::filesystem;

#include <dirent.h>

#include <thread>
#include <mutex>
#include <condition_variable>
#include "utils/Global.h"

using torch::indexing::Slice;
using torch::indexing::None;

void feature_add()
{
  const char *path = "../data/btc.csv";
  data::CSVTensor assetData(path, g_device);

  train::TrainOption base({"AttGru", "PPOSampler", "Adam", "1d", "btc", "roulette",
    {8,9,104,114,122,129},   // cols
    32,   // rnn_base
    1,    // rnn_layer
    32,   // linear_base
    30,   // normalize_len,
    0.2,  // dropout
    5e-4, // lr
    2412, // train_len,
    256,  // batch_size
    600,  // maxEpoch
    200,  // patient,
    10,   // repeat,
    2.5f, // save threshold
    0,    // dummy diff_col
    0.0f, // dummy score
    5,    // cnn_kernel_size
    2,    // cnn_stride
    32,   // cnn_outsize
    3,    // maxpool_size
    1,    //maxpool_stride
    });

  feature_add_rnn(base, assetData, 1, 5);
}

void run_score(std::string root, std::vector<std::string> &paths,
  const char *csv_path = "../output/df_actions.csv",
  const char *data_path = "../data/btc.csv",
  const char *trainLenStr = "2412",
  const char *score_path = "../output/df_scores.csv")
{
  data::CSVTensor assetData(data_path, g_device);
  std::vector<std::vector<float> > acts;
  std::vector<float> cumprods;
  std::vector<std::string> paths_accept;
  std::vector<std::string> paths_all;
  long normalizeLen = 30;
  unsigned trainLen = atoi(trainLenStr) - normalizeLen;
  unsigned threads = 1;
  TimerCounter tc(threads, 10);
  tc.setTotalRound(paths.size());

  unsigned drop_cnt = 0;
  for (auto &path : paths) {

    train::TrainOption option;
    option.fromTag(path.c_str());
    option.trainLen = trainLen;
    assert(option.normalizeLen == normalizeLen);

    train::Trainer trainer(assetData, option, g_device);
    std::string fullpath = root + path;
    std::vector<float> act;
    float test_reward = trainer.backtest(&fullpath, &act, &tc);

    paths_all.push_back(path);
    cumprods.push_back(test_reward);

    if (true || test_reward > 3.55f) {
      paths_accept.push_back(path);
      acts.push_back(act);
    } else {
      ++drop_cnt;
    }

    tc.epochRoundEnd();
  }

  std::cout << "num_path=" << cumprods.size() << " drop_cnt=" << drop_cnt << std::endl;
  std::ofstream df_actions;
  df_actions.open (csv_path);
  df_actions << "timestamp,diff,close";
  for (auto& path : paths_accept) {
    df_actions << ',' << path;
  }
  df_actions << std::endl;

  auto diffs = assetData.getCol("diff").index({Slice(normalizeLen + trainLen, None)});
  size_t model_cnt = acts.size();
  size_t test_length = acts.back().size();
  for (size_t i = 0; i < test_length; ++i) {
    float diff = diffs[i].item<float>();
    df_actions << i << "," << diff << "," << diff;
    for (size_t j = 0; j < model_cnt; ++j) {
      df_actions << "," << std::fixed << std::setprecision(5) << acts[j][i];
    }
    df_actions << std::endl;
  }
  df_actions.close();

  std::ofstream df_scores;
  std::cout << "score_path=" << score_path << std::endl;
  df_scores.open (score_path);
  df_scores << "path,score" << std::endl;
  for (size_t i = 0; i < paths_all.size(); ++i) {
    df_scores << paths_all[i] << "," << cumprods[i] << std::endl;
  }
  df_scores.close();
}

int main(int argc, char *argv[]) {

  // std::cout << std::fixed << std::setprecision(2);
  if (argc < 2) {
    std::cout << "usage: train feature_add|ensemble" << std::endl;
    return 0;
  }

  if (std::strcmp(argv[1], "feature_add") == 0) {
    feature_add();
  } else if (std::strcmp(argv[1], "back_test") == 0) {
    std::vector<std::string> paths;
    paths.push_back(argv[3]);
    run_score(argv[2], paths, argv[4], argv[5], argv[6]);
  }
    
  return 0;
}
