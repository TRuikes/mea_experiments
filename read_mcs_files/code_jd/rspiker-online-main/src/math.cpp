#include "helpers.h"
#include "math.h"

HighPassFilter::HighPassFilter()
{
    cutoffFrequency = CUTOFFFREQUENCY;
    sampleRate = static_cast<double>(SAMPLERATE);
    timeConstant = 1.0 / (2 * 3.14159 * cutoffFrequency);
    deltaT = 1 / sampleRate;
    alpha = timeConstant / (timeConstant + deltaT);
    previousInput = 0.0;
    previousInput2 = 0.0;
    previousInput3 = 0.0;
    previousOutput = 0.0;
    previousOutput2 = 0.0;
    previousOutput3 = 0.0;
}

void HighPassFilter::setCutoffFrequency(double frequency){
    cutoffFrequency = frequency;
    timeConstant = 1.0 / (2 * 3.14159 * cutoffFrequency);
    alpha = timeConstant / (timeConstant + deltaT);
}

double HighPassFilter::getSampleRate() const{
    return sampleRate;
}

double HighPassFilter::getTimeConstant() const{
    return timeConstant;
}

double HighPassFilter::process(double input){
    // double output = alpha * alpha * (previousOutput2 + input - 2 * previousInput + previousInput2);
    double output = alpha * alpha * alpha * (previousOutput3 + input - 3 * previousInput + 3 * previousInput2 - previousInput3);
    
    previousOutput = output;
    previousOutput2 = previousOutput;
    previousOutput3 = previousOutput2;

    previousInput = input;
    previousInput2 = previousInput;
    previousInput3 = previousInput2;

    return output;
}

double calculateMedian(std::vector<double>& values) {
    size_t size = values.size();
    std::sort(values.begin(), values.end());

    if (size % 2 == 0) {
        return (values[size / 2 - 1] + values[size / 2]) / 2.0;
    } else {
        return values[size / 2];
    }
}

double calculateMAD(std::vector<double>& values) {
    double median = calculateMedian(values);

    // Calculate absolute deviations from the median
    std::vector<double> absoluteDeviations;
    absoluteDeviations.reserve(values.size());
    for (double val : values) {
        absoluteDeviations.push_back(std::abs(val - median));
    }

    // Calculate the median of absolute deviations
    return calculateMedian(absoluteDeviations);
}