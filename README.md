# LibtorchTrade
RL BTC trader written in C++ using libtorch

## Guiding principles
* Fast
    * Written purely in C++.
    * Support multi-threading training.
    * Since the daily historical data is a small dataset(4MB), it can easily fit into the cache. So it is at least twice faster to train using CPU than GPUs.
    * A sample training log, it is just 20 seconds training.
    ```
    2020-11-18 22:13:36 +0000 CSVTensor INFO load data from ../data/btc.csv
    2020-11-18 22:13:37 +0000 Trainer INFO feature_add_rnn start
    2020-11-18 22:13:38 +0000 Trainer DEBUG 0 train_reward=12.9045 test_reward=1.62863
    ...
    2020-11-18 22:13:51 +0000 Trainer DEBUG 20 train_reward=204.805 test_reward=2.5557
    2020-11-18 22:13:51 +0000 Trainer INFO  save model: epoch=20 path=../models_train/btc/1d/PPOSampler_AttGru_1d_1rnn32_linear32_drop0.2_nlen30_Adam_8-9-104-114-122-129-0_prod2.5557_epo20_1605737631_roulette.pt
    ```
* Effective
    
    

## Build
* You need to install [libtorch C++](https://pytorch.org/cppdocs/) first, then run the commands below:
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=<path to libtorch> ..
make
```

## Training
* A simple heuristic feature selection algorithm is implemented. To train with this algorithm:
```
train feature_add
```
