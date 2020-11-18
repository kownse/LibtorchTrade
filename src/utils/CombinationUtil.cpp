#include "CombinationUtil.h"
#include <iostream>

uint64_t calcNumComb(int n, int c) {
    uint64_t ret = 1;
    for (uint64_t i=1; i < c+1; ++i ) {
        ret *= (n - i + 1);
        ret /= i;
	}
    return ret;
}

uint64_t calcCombNum(int num_cols, int start, int end) {
    uint64_t ret = 0;
	int endloop = (num_cols < end ? num_cols : end) + 1;
    for (int i=start; i<endloop; ++i)
        ret += calcNumComb(num_cols, i);

	std::cout << "num_cols=" << num_cols << " start=" << start << " end=" << end << " : " << ret << std::endl;
    return ret;
}