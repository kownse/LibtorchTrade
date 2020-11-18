#include "FileWriter.h"
#include <stdarg.h>
#include <cstdio>
#include <cstring>

static const int WRITE_THREAHOLD = BUFF_SIZE - LINE_SIZE * 16;

FileWriter::FileWriter(const char *path)
    : _pos(0),
    _fo(fopen(path, "w+")) {
    memset(_buf, '\0', sizeof(char) * BUFF_SIZE);
}

FileWriter::~FileWriter() {
    writeToFile();
    fclose(_fo);
}

void FileWriter::write(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    _pos += vsprintf(_buf + _pos, fmt, args);
    va_end(args);

    if (_pos >= WRITE_THREAHOLD)
        writeToFile();
}

void FileWriter::writeToFile() {
	fprintf(_fo, "%s", _buf);
	memset(_buf, '\0', sizeof(char) * BUFF_SIZE);
	_pos = 0;
}