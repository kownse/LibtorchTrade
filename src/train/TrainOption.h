#ifndef _TRAIN_OPTION_H_
#define _TRAIN_OPTION_H_

#include <string>
#include <vector>
#include <ostream>

namespace train {

struct TrainOption {
    std::string agentType;
    std::string modelType;
    std::string optType;
    std::string interval;
    std::string traintype;
    std::string sampler = "random";
    
    std::vector<long> cols;
    long rnnBase;
    long rnnLayers;
    long linearBase;
    long normalizeLen;
    float drop;
    float learnRate;
    unsigned trainLen;
    unsigned batchSize;
    unsigned maxEpoch;
    unsigned patient;
    unsigned repeat;
    float saveCumprod;
    long diff_col;

    float cumprod;

    unsigned cnn_kernel_size;
    unsigned cnn_stride;
    unsigned cnn_outsize;
    unsigned maxpool_size;
    unsigned maxpool_stride;

    std::string stringCols() const;
    std::stringstream genTag() const;
    void fromTag(const char *tag);

};

std::ostream& operator<<(std::ostream& os, const TrainOption& op);

};

#endif // _TRAIN_OPTION_H_