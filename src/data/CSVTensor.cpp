#include "CSVTensor.h"
#include "../utils/FileUtil.h"
#include "../utils/TensorUtil.h"

#include "../utils/Logger.h"
static Logger gLogger("CSVTensor");

using torch::indexing::Slice;

namespace data {

CSVTensor::CSVTensor(const char *path, torch::Device device) : _device(device), _numRow(0), _numCol(0) {
    loadFromCSV(path);
}

void CSVTensor::loadFromCSV(const char *path) {
    gLogger.info() << "load data from " << path << std::endl;

    util::file::getNumRowAndCol(path, &_numRow, &_numCol);
    // skip header
    --_numRow;
    _data = torch::zeros({_numRow, _numCol});

    std::ifstream fi(path); 
    long row = 0;
    std::string line;

    // header
    std::getline(fi, line);
    util::file::TermsT terms = util::file::splitLine(line);
    for (size_t i = 0; i < terms.size(); ++i) {
        _mapColIdx[terms[i]] = i;
    }

    // data
    while (std::getline(fi, line)) {
        util::file::TermsT terms = util::file::splitLine(line);
        for (size_t i = 0; i < terms.size(); ++i) {
            _data[row][i] = std::stof(terms[i]);
        }

        ++row;
    }
    fi.close();
}

long CSVTensor::getColIdx(const char *col) const {
    auto it = _mapColIdx.find(col);
    if (it == _mapColIdx.end()) {
        return -1;
    }
    return it->second;
}

torch::Tensor CSVTensor::getCol(const char *col) const {
    auto idx = getColIdx(col);
    if (idx < 0)
        return torch::zeros({1});

    return _data.index({Slice(), idx});
}

torch::Tensor CSVTensor::selectCols(std::vector<long> cols) const {
    return util::tensor::selectCols(_device, _data, cols);
}

torch::Tensor CSVTensor::calcZscores(long normaliseLen) const {
    if (!_mapZscores.count(normaliseLen)) {
        _mapZscores[normaliseLen] = util::tensor::calcZscores(_device, _data, normaliseLen);
    }
    return _mapZscores[normaliseLen];
}

torch::Tensor CSVTensor::calcZscoresWithCols(std::vector<long> cols, long normaliseLen) const {
    return util::tensor::selectCols(_device, calcZscores(normaliseLen), cols);
}

torch::Tensor CSVTensor::calcTZscoresWithCols(std::vector<long> cols, long normaliseLen) const {
    if (!_mapTZscores.count(normaliseLen)) {
        auto zscores = calcZscores(normaliseLen);
        auto tzscores = zscores.transpose(0,1);

        std::vector<torch::Tensor> tmp;
        for (long i = normaliseLen; i < zscores.sizes()[0] - 1; ++i) {
            auto state = tzscores.index({None, Slice(), Slice(i - normaliseLen, i)});
            tmp.push_back(state);
        }
        _mapTZscores[normaliseLen] = torch::stack(tmp);
    }

    return _mapTZscores[normaliseLen].index({Slice(), Slice(), torch::tensor(cols)});
}

}; // namespace data