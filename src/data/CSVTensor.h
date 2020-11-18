#ifndef _CSV_TENSOR_H_
#define _CSV_TENSOR_H_

#include <torch/torch.h>
#include <string>
#include <vector>
#include "../tsl/robin_map.h"

namespace data {

class CSVTensor {
public:
    CSVTensor(const char *path, torch::Device device);

    const torch::Tensor &getData() const { return _data; }

    long getColIdx(const char *col) const;
    torch::Tensor getCol(const char *col) const;
    torch::Tensor selectCols(std::vector<long> cols) const;
    torch::Tensor calcZscoresWithCols(std::vector<long> cols, long normaliseLen) const;
    torch::Tensor calcTZscoresWithCols(std::vector<long> cols, long normaliseLen) const;

    long numRow() const { return _numRow; }
    long numCol() const { return _numCol; }

private:
    void loadFromCSV(const char *path);
    torch::Tensor calcZscores(long normaliseLen) const;

    torch::Tensor _data;
    torch::Device _device;
    mutable tsl::robin_map<long, torch::Tensor > _mapZscores;
    mutable tsl::robin_map<long, torch::Tensor > _mapTZscores;
    tsl::robin_map<std::string, long> _mapColIdx;
    long _numRow;
    long _numCol;
};

}; // namespace data

#endif // _CSV_TENSOR_H_