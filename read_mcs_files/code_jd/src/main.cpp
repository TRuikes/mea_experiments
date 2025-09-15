#include <emscripten/bind.h>
#include <vector>
#include <fstream>
#include "helpers.h"
#include "mcdfile.h"
#include "electrode.h"

using namespace emscripten;

#define READDIGITAL 0
#define DATASTART 287864
#define FRAMEBYTES 1028064
#define FRAMEMARGINSTART 32
#define ELECTRODEBYTES 4000
#define FRAMEPOINTS ELECTRODEBYTES / 2
#define ELECTRODES_COUNT 256

MCDFile mcdfile;
std::vector<Electrode> electrodes(ELECTRODES_COUNT+1);

void initElectrodes(){
    LOG("Init Electrodes");
    for(int n=0; n<257; n++){
        electrodes[n].setN(n);
    }
}

void resetElectrodes(){
    LOG("Reset Electrodes");
    for(int n=0; n<257; n++){
        electrodes[n].clear();
    }
}

MCDFile initMCDFile(){
    LOG("Init MCD");
    mcdfile.load();
    initElectrodes();
    return mcdfile;
}

Electrode getElectrode(size_t n){
    return electrodes[n];
}

int getSize(size_t n){
    return electrodes[n].size();
}

DATATYPE get(size_t n, size_t index){
    return electrodes[n].get(index);
}

void setCutoffFrequency(double frequency){
    for(int n = 0; n <= ELECTRODES_COUNT; n++){
        electrodes[n].setCutoffFrequency(frequency);
    }
}

double getFiltered(size_t n, size_t index){
    return electrodes[n].getFiltered(index);
}

bool getSpikesorted(size_t n, size_t index){
    return electrodes[n].getSpikesorted(index);
}

bool getStim(size_t triggerElectrode, size_t index){
    return electrodes[triggerElectrode].get(index) > 32798; //30 (threshold) + 32768 (zero)
}

bool getStimFromCache(size_t n, size_t index){
    return electrodes[n].getStim(index);
}

bool getRastered(size_t n, size_t index){
    return electrodes[n].getRastered(index);
}

double getMAD(size_t n){
    return electrodes[n].getMAD();
}

void clearElectrodes(){
    for(int n = 0; n <= ELECTRODES_COUNT; n++){
        electrodes[n].clear();
    }
}

void loadFrames(int firstFrame, int size)
{
    resetElectrodes();

    std::fstream source;
    source.open("experiment.mcd", std::fstream::binary | std::fstream::in);
    if(!source) return;

    DATATYPE x;
    LOG("Load Frames [" << firstFrame << ";+" << size << "]");
    
    //Find end
    source.seekg(0,std::fstream::end);
    const int end = source.tellg();

    source.seekg(DATASTART);

    //Move to frame beginning
    source.seekg(firstFrame * FRAMEBYTES,std::fstream::cur);

    for(int frame = firstFrame; frame < firstFrame + size; frame++){
        //Skip digital
        source.seekg(FRAMEMARGINSTART+ELECTRODEBYTES,std::fstream::cur);
        //Move to analog data
        source.seekg(FRAMEMARGINSTART,std::fstream::cur);


        for(int i = 0; i < FRAMEPOINTS; i++){
            for(int n = 1; n <= ELECTRODES_COUNT; n++){
                source.read((char*)&x,2);
                electrodes[n].push(x);
            }
        }

        //Calculate MAD
        if(!electrodes[1].MADDone){
            for(int n = 1; n < 257; n++){
                electrodes[n].computeMAD();
            }
        }
    }
    source.close();
}

bool rastered = false;

void loadRaster(){
    int size = electrodes[1].size();
    for(int i = 0; i < size; i++){
        for(int n = 1; n <= ELECTRODES_COUNT; n++){
            electrodes[n].pushRaster(getSpikesorted(n,i),getStim(128,i));
        }
    }
}

void setRastered(bool value){
    rastered = value;
}

bool isRastered(){
    return rastered;
}

int getRasteredSize(){
    return electrodes[1].getRasteredSize();
}

void resetRastered(){
    for(int n = 1; n <= ELECTRODES_COUNT; n++){
        electrodes[n].clearRaster();
    }
    setRastered(false);
}

void setMADThreshold(double threshold){
    for(int n = 1; n <= ELECTRODES_COUNT; n++){
        electrodes[n].setMADThreshold(threshold);
    }
}

EMSCRIPTEN_BINDINGS(my_module) {
    class_<Electrode>("Electrode")
        .constructor()
        .property("n", &Electrode::getN)
        .function("get", &Electrode::get)
        .function("getFiltered", &Electrode::getFiltered)
        .function("getSpikesorted", &Electrode::getSpikesorted)
        .property("MAD",&Electrode::getMAD)
        .function("size", &Electrode::size)
        .function("clear", &Electrode::clear);
    class_<MCDFile>("MCDFile")
        .constructor()
        .property("filename",&MCDFile::getFilename)
        .property("filesize",&MCDFile::getFilesize)
        .property("framesCount",&MCDFile::getFramesCount)
        .function("getTimeStart",&MCDFile::getTimeStart)
        .function("getTimeStop",&MCDFile::getTimeStop)
        .function("getTimePush",&MCDFile::getTimePush);
    function("initMCDFile",&initMCDFile);
    function("getElectrode",&getElectrode);
    function("getSize",&getSize);
    function("get",&get);
    function("setCutoffFrequency",&setCutoffFrequency);
    function("getFiltered",&getFiltered);
    function("getSpikesorted",&getSpikesorted);
    function("getStim",&getStim);
    function("getStimFromCache",&getStimFromCache);
    function("clearElectrodes",&clearElectrodes);
    function("initElectrodes",&initElectrodes);
    function("resetElectrodes",&resetElectrodes);
    function("loadFrames",&loadFrames);
    function("loadRaster",&loadRaster);
    function("isRastered",&isRastered);
    function("getRastered",&getRastered);
    function("setRastered",&setRastered);
    function("getRasteredSize",&getRasteredSize);
    function("resetRastered",&resetRastered);
    function("setMADThreshold",&setMADThreshold);
    function("getMAD",&getMAD);
}