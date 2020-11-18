cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(train)
find_package(Torch REQUIRED)
add_compile_options(-std=c++17)
add_compile_options(-Wl,--whole-archive)
add_compile_options(-static -lrt -pthread -lpthread)
add_compile_options(-march=native)
add_compile_options(-mtune=native)
add_compile_options(-mavx2)
add_compile_options(-O2)
add_compile_options(-funroll-loops)
add_compile_options(-ftree-vectorizer-verbose=1)
add_compile_options(-ftree-vectorize)
#add_compile_options(-g)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCES
    src/main.cpp
    src/utils/FileUtil.cpp
    src/utils/TensorUtil.cpp
    src/utils/TimerCounter.cpp
    src/utils/CombinationUtil.cpp
    src/utils/FileWriter.cpp
    src/utils/Logger.cpp
    src/data/CSVTensor.cpp
    src/train/Trainer.cpp
    src/train/TrainOption.cpp
    src/train/agent/Agent.cpp
    src/train/agent/AttGru.cpp
    src/train/agent/Attention.cpp
    src/train/model/Model.cpp
    src/train/model/PPOSampler.cpp
    src/train/Cache/ReplayBuffer.cpp
    src/train/Cache/RandomSampler.cpp
    src/train/Cache/RouletteSampler.cpp
    src/train/Cache/Sampler.cpp
)

add_executable(train ${SOURCES})
target_link_libraries(train "${TORCH_LIBRARIES}")
set_property(TARGET train PROPERTY CXX_STANDARD 17)
