#include "TrainOption.h"
#include <iostream>
#include <sstream>

namespace train {

std::ostream& operator<<(std::ostream& os, const TrainOption& op)
{
    os << "model=" << op.modelType
       << " agent=" << op.agentType
       << " interval=" << op.interval
       << " rnnLayers=" << op.rnnLayers
       << " rnnBase=" << op.rnnBase
       << " linearBase=" << op.linearBase
       << " drop=" << op.drop
       << " nlen=" << op.normalizeLen
       << " opt=" << op.optType
       << " lr=" << op.learnRate;

    bool cnn = op.agentType.find("Cnn") != std::string::npos;
    if (cnn) {
        os << " cnn_kernel_size=" << op.cnn_kernel_size
           << " cnn_stride=" << op.cnn_stride
           << " maxpool_size=" << op.maxpool_size
           << " maxpool_stride" << op.maxpool_stride;
    }
    os << " cols=[" << op.stringCols() << "]";

    return os;
}

std::string TrainOption::stringCols() const
{
    std::stringstream ss;
    std::string delimiter = "";
    for(std::vector<long>::const_iterator it = cols.begin(); it != cols.end(); ++it) {
        ss << delimiter << *it;
        delimiter = "-";
    }
    return ss.str();
}

std::stringstream TrainOption::genTag() const {
    //ORPG_Torch_RNNBiFC_1gru32_fc32_nlen30_1-3-8-24_hs_best_epoch637_sum1.47_prod3.95_1595462679
    std::stringstream ss;
    ss << modelType << '_' << agentType << '_' << interval << '_' 
        << rnnLayers << "rnn" << rnnBase << '_'
        << "linear" << linearBase << '_'
        << "drop" << drop << '_'
        << "nlen" << normalizeLen << '_'
        << optType << '_';

    bool cnn = agentType.find("Cnn") != std::string::npos;
    if (cnn) {
        ss << "cnnk" << cnn_kernel_size << '_'
            << "cnns" << cnn_stride << '_'
            << "cnnout" << cnn_outsize << '_'
            << "mxs" << maxpool_size << '_'
            << "mxstrd" << maxpool_stride << '_';
    }

    ss << stringCols();

    return ss;
}

std::string getChunk(std::string &str, size_t &start, size_t &end)
{
    end = str.find('_', start);
    size_t len = end - start;
    size_t s = start;
    start = end + 1;
    return str.substr(s, len);
}

void TrainOption::fromTag(const char *tag)
{
    learnRate = 1e-3;
    std::string str(tag);
    size_t start = 0;
    size_t end;

    modelType = getChunk(str, start, end);
    // std::cout << modelType << std::endl;

    agentType = getChunk(str, start, end);
    // std::cout << agentType << std::endl;
    bool cnn = agentType.find("Cnn") != std::string::npos;

    interval = getChunk(str, start, end);
    // std::cout << interval << std::endl;

    std::string tmp = getChunk(str, start, end);
    sscanf(tmp.c_str(), "%ldrnn%ld", &rnnLayers, &rnnBase);

    tmp = getChunk(str, start, end);
    sscanf(tmp.c_str(), "linear%ld", &linearBase);

    tmp = getChunk(str, start, end);
    sscanf(tmp.c_str(), "drop%f", &drop);

    tmp = getChunk(str, start, end);
    sscanf(tmp.c_str(), "nlen%ld", &normalizeLen);

    optType = getChunk(str, start, end);

    if (cnn) {
        tmp = getChunk(str, start, end);
        sscanf(tmp.c_str(), "cnnk%u", &cnn_kernel_size);

        tmp = getChunk(str, start, end);
        sscanf(tmp.c_str(), "cnns%u", &cnn_stride);

        tmp = getChunk(str, start, end);
        sscanf(tmp.c_str(), "cnnout%u", &cnn_outsize);

        tmp = getChunk(str, start, end);
        sscanf(tmp.c_str(), "mxs%u", &maxpool_size);

        tmp = getChunk(str, start, end);
        sscanf(tmp.c_str(), "mxstrd%u", &maxpool_stride);
    }

    std::istringstream split(getChunk(str, start, end));
    std::string each;
    while(std::getline(split, each, '-')) {
        cols.push_back(std::stol(each));
    }

    tmp = getChunk(str, start, end);
    sscanf(tmp.c_str(), "prod%f", &cumprod);
}
};
