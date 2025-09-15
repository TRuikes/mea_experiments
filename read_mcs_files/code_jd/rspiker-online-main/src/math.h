#include <vector>
#include <algorithm>

class HighPassFilter{
    public:
        HighPassFilter();
        void setCutoffFrequency(double frequency);
        double getSampleRate() const;
        double getTimeConstant() const;
        double process(double input);
    
    private:
        double cutoffFrequency;
        double sampleRate;
        double timeConstant;
        double deltaT;
        double alpha;
        double previousInput;
        double previousOutput;
        double previousInput2;
        double previousOutput2;
        double previousInput3;
        double previousOutput3;

};

double calculateMedian(std::vector<double>& values);
double calculateMAD(std::vector<double>& values);