#pragma once


#include <vector>

class baseCnnLayer
{
public:

    virtual std::vector<std::vector<std::vector<double>>> forward(std::vector<std::vector<std::vector<double>>> _ghostsTensor) = 0;
    virtual std::vector<std::vector<std::vector<double>>> backward(std::vector<std::vector<std::vector<double>>> _inputGrad, double _alpha) = 0;
    virtual std::vector<int> returnConfig() = 0;

    virtual ~baseCnnLayer() = default;


    //std::vector<std::vector<std::vector<double>>> inputGhosts;
};
