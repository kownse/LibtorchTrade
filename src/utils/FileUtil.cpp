#include "FileUtil.h"
#include <fstream>
#include <algorithm>
#include <dirent.h>
#include <cstring>
#include <iostream>

namespace util::file {

TermsT splitLine(const std::string &line) {
    TermsT ret;
    size_t start = 0;
    size_t end = 0;
    do {
        end = line.find(",", start);
        ret.push_back(line.substr(start, end - start));
        start = end + 1;
    }
    while (end != std::string::npos);

    return ret;
}

void getNumRowAndCol(const char *path, long *numRow, long *numCol) {
    std::ifstream inFile(path); 
    *numRow = std::count(std::istreambuf_iterator<char>(inFile), 
            std::istreambuf_iterator<char>(), '\n');

    inFile.seekg(0, inFile.beg);
    std::string line;
    std::getline(inFile, line);
    *numCol = splitLine(line).size();
    inFile.close();
}

size_t getFileCnt(const char *root) {
    DIR *dir;
    struct dirent *ent;
    size_t cnt = 0;
    if ((dir = opendir (root)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            ++cnt;
        }
        closedir (dir);
    } else {
        std::cout << "could not open " << root;
    }

    return cnt;
}

}; // namespace fileutil 
