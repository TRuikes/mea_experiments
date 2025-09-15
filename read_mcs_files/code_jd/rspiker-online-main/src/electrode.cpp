#include "electrode.h"

#define FRAMESTART 287864
#define FRAMEBYTES 1028064
#define FRAMEMARGINSTART 16
#define FRAMEMARGINEND 16

Electrode::Electrode() :
MADDone(false),
MADThreshold(DEFAULT_MAD_THRESHOLD)
{}

int Electrode::getN() const {return n;}
void Electrode::setN(int newN){
    n = newN;
}

void Electrode::push(DATATYPE value){
    data.push_back(value);
    pushFiltered(value);
}
DATATYPE Electrode::get(size_t index){
    if(index < data.size()) return data[index];
    else return -1;
}
int Electrode::size(){
    return data.size();
}
void Electrode::clear(){
    data.clear();
    dataFiltered.clear();
}

void Electrode::setCutoffFrequency(double frequency){
    filter.setCutoffFrequency(frequency);
}

void Electrode::pushFiltered(DATATYPE value){
    dataFiltered.push_back(filter.process(static_cast<double>(value)));
}

double Electrode::getFiltered(size_t index){
    if(index < dataFiltered.size()) return dataFiltered[index];
    else return -1.0;
}

void Electrode::pushRaster(bool valueRaster, bool valueStimulation){
    raster.push_back(valueRaster);
    stimulation.push_back(valueStimulation);
}

bool Electrode::getRastered(size_t index){
    return raster[index];
}

bool Electrode::getStim(size_t index){
    return stimulation[index];
}

int Electrode::getRasteredSize(){
    return raster.size();
}

void Electrode::clearRaster(){
    raster.clear();
    stimulation.clear();
}

double Electrode::getMAD() const {
    return MAD;
}

void Electrode::computeMAD(){
    if(MADDone) return;
    std::vector<double> dataFilteredSorted = dataFiltered;
    MAD = calculateMAD(dataFilteredSorted);
    MADDone = true;
}

void Electrode::setMADThreshold(double threshold){
    MADThreshold = threshold;
}

bool Electrode::getSpikesorted(size_t index){
    if(index == 0) return false;
    // return (dataFiltered[index] > MAD * MADThreshold || dataFiltered[index] < -1 * MAD * MADThreshold) && (dataFiltered[index-1] < MAD * MADThreshold && dataFiltered[index-1] > -1 * MAD * MADThreshold);
    double threshold = MAD * MADThreshold;
    return (dataFiltered[index] < -1 * threshold) && (dataFiltered[index-1] > -1 * threshold);
}