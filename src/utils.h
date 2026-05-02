#pragma once

#include <cmath>
#include <vector>
#include <string>

enum class ActivationsType
{
    RELU,
    SIGMOID,
    TANH,
    LINEAR,
    LeakyRELU,
    SOFTMAX
    
};

enum class Color
{
    RED,
    YELLOW,
    GREEN,
    BLUE,
    PURPLE,
    WHITE
};

namespace NN_Utils
{
    double functionActivation(double _value, ActivationsType _AT);

    double functionActivationDerivative(double _value, double _activated, ActivationsType _AT);

    void applySoftmax(std::vector<double>& _inputData);

    std::vector<double> computePolicyGradient(const std::vector<double> _probs, int _action);
}

namespace Math_Utils
{

}

namespace Ñool_Print
{
    std::string colorText(std::string _input, Color _color);
}