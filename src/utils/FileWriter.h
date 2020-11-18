#ifndef _FILE_WRITER_H_
#define _FILE_WRITER_H_

#include <cstdio>

#define WIRTE_BATCH_SIZE 128
#define LINE_SIZE 512
#define BUFF_SIZE (WIRTE_BATCH_SIZE * LINE_SIZE)

class FileWriter {
public:
    FileWriter(const char *path);
    ~FileWriter();

    void write(const char * format, ...);
private:
    void writeToFile();

    char _buf[BUFF_SIZE];
    size_t _pos;
    FILE *_fo;
};

#endif // _FILE_WRITER_H