#include <vector>
#include "helpers.h"
#include "math.h"

class Electrode{
    public:
        Electrode();
        int getN() const;
        void setN(int newN);

        void push(DATATYPE value);
        DATATYPE get(size_t index);

        void setCutoffFrequency(double frequency);
        void pushFiltered(DATATYPE value);
        double getFiltered(size_t index);

        void pushRaster(bool valueRaster, bool valueStimulation);
        bool getRastered(size_t index);
        bool getStim(size_t index);
        int getRasteredSize();
        void clearRaster();

        bool MADDone;
        double getMAD() const;
        void computeMAD();
        void setMADThreshold(double threshold);
        bool getSpikesorted(size_t index);

        int size();

        void clear();


    private:
        int n;
        double MAD;
        double MADThreshold;
        std::vector<DATATYPE> data;
        HighPassFilter filter;
        std::vector<double> dataFiltered;
        std::vector<bool> raster;
        std::vector<bool> stimulation;
};