#include <emscripten/bind.h>
#include <iostream>
#include <vector>
#include <fstream>

#define LOG(x) std::cout << x << std::endl

#define DATATYPE unsigned short int

#define DATASTART 287864
#define FRAMEMARGINSTART 32
#define FRAMEBYTES 1028064
#define ELECTRODEBYTES 4000
#define ELECTRODES_COUNT 256

using namespace emscripten;

std::vector<std::vector<DATATYPE>> electrodes(ELECTRODES_COUNT+1);

unsigned short int get(int electrode, int position){
    std::fstream source;
    source.open("experiment.mcd", std::fstream::binary | std::fstream::in);
    if(!source) return 0;

    int positionInFrame = position % 2000;
    int frame = position / 2000;

    //Move to data start
    source.seekg(DATASTART);

    //Move to frame
    source.seekg(frame*FRAMEBYTES,std::fstream::cur);
    //Skip digital
    source.seekg(FRAMEMARGINSTART+ELECTRODEBYTES,std::fstream::cur);
    //Move to analog data
    source.seekg(FRAMEMARGINSTART,std::fstream::cur);
    //Move to electrode
    source.seekg((electrode-1)*2,std::fstream::cur);    
    //Move to position
    source.seekg(positionInFrame*ELECTRODES_COUNT*2,std::fstream::cur);

    //Read
    unsigned short int x;
    source.read((char*)&x,2);
    
    source.close();
    return x;
}

void load(){
    std::fstream source;
    source.open("experiment.mcd", std::fstream::binary | std::fstream::in);
    if(!source) return;

    //Find end
    source.seekg(0,std::fstream::end);
    const int end = source.tellg();

    //Data size
    std::vector<DATATYPE> testVec;
    LOG("Test Vec Max Size :" << testVec.max_size());
    unsigned short int x;
    LOG("Point size" << sizeof(x));

    const int dataSizeWithExtra = end - DATASTART;
    const int framesCount = dataSizeWithExtra / FRAMEBYTES;
    const int dataSize = framesCount * FRAMEBYTES;

    LOG(framesCount << " frames (" << dataSize << ")");

    LOG("Bytes per electrode " << framesCount * ELECTRODEBYTES);
    LOG("Bytes for electrodes vec " << framesCount * ELECTRODEBYTES * (ELECTRODES_COUNT + 1));

    //Allocate memory
    // electrodes.resize((ELECTRODES_COUNT+1), std::vector<DATATYPE>(20 * FRAMEBYTES));

    LOG("Memory allocated");

    //Move to data start
    source.seekg(DATASTART);

    // int frame = 0;
    // while(DATASTART + (frame + 1) * FRAMEBYTES < end){
    for(int frame = 0; frame < 20; frame++){
        if(frame % 10 == 0) LOG(" secs");
        //Skip digital
        source.seekg(FRAMEMARGINSTART+ELECTRODEBYTES,std::fstream::cur);
        //Move to analog data
        source.seekg(FRAMEMARGINSTART,std::fstream::cur);
        for(int i = 0; i < 2000; i++){
            for(int n = 1; n <= ELECTRODES_COUNT; n++){
                source.read((char*)&x,2);
                electrodes[n].push_back(x);
            }
        }
        frame++;
    }

    source.close();
}

std::vector<DATATYPE> data;

void loadFileInMemory(){
    // std::fstream source;
    // source.open("experiment.mcd", std::fstream::binary | std::fstream::in);
    // if(!source) return;

    // //Find end
    // source.seekg(0,std::fstream::end);
    // const int end = source.tellg();

    std::ifstream file("experiment.mcd", std::ios::binary); // Open the binary file

    if (file) {
        std::vector<uint8_t> byteVector(std::istreambuf_iterator<char>(file), {});

        // Display the number of bytes read
        LOG("Bytes: " << byteVector.size());

        int size = byteVector.size();
        int size2 = size/2;

        LOG("2Bytes: " << size2);

        // data.resize(size2);

        // LOG("Data size: " << data.capacity());

        int p = DATASTART;

        while(p+FRAMEBYTES < size){
            //Skip Digital
            p += FRAMEMARGINSTART+ELECTRODEBYTES;
            p += FRAMEMARGINSTART;
            for(int dataPoint = 0; dataPoint < 2000; dataPoint++){
                for(int electrode = 1; electrode <= 256; electrode++){
                    // data.push_back((byteVector[p] | byteVector[p+1] << 8));
                    p+=2;
                }
            }
        }

        LOG("Data size " << data.size());

        file.close(); // Close the file when done
    } else {
        std::cerr << "Error opening the file!" << std::endl;
    }
}

DATATYPE getFromVec(int position){
    return data[position];
}


EMSCRIPTEN_BINDINGS(my_module) {
    function("get",&get);
    function("load",&load);
    function("getFromVec",&getFromVec);
    function("loadFileInMemory",&loadFileInMemory);
}