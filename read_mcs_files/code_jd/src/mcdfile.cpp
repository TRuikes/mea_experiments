#include <iostream>
#include "helpers.h"
#include "mcdfile.h"

#define DATASTART 287864
#define FRAMEBYTES 1028064

#define HEADER_TIME_START 840
#define HEADER_TIME_STOP 896
#define HEADER_TIME_PUSH 920

MCDFile::MCDFile() : filename("experiment.mcd") {}
void MCDFile::load(){ LOG("Load MCD"); }
const std::string& MCDFile::getFilename() const {
    return filename;
};
int MCDFile::getFilesize() const{
    std::fstream source;
    source.open(filename, std::fstream::binary | std::fstream::in);
    source.seekg(0,std::fstream::end);
    return source.tellg();
}

int MCDFile::getFramesCount() const{
    int size = getFilesize();
    int data = size - DATASTART;
    return data / FRAMEBYTES;
}

uint64_t MCDFile::getTimeStart(){
    return getFiletime(HEADER_TIME_START);
}
uint64_t MCDFile::getTimeStop(){
    return getFiletime(HEADER_TIME_STOP);
}
uint64_t MCDFile::getTimePush(){
    return getFiletime(HEADER_TIME_PUSH);
}

