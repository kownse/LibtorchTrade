#ifndef _CSV_HELPER_H_
#define _CSV_HELPER_H_

#include <cuchar>
#include <vector>
#include <string>

namespace util::file {

typedef std::vector<std::string> TermsT;

TermsT splitLine(const std::string &line);

void getNumRowAndCol(const char *path, long *numRow, long *numCol);

size_t getFileCnt(const char * root);
};


#endif //_CSV_HELPER_H_