#include "utils.h"

namespace NN_Utils
{
    double functionActivation(double _value, ActivationsType _AT)
    {
        switch (_AT)
        {
        case ActivationsType::RELU:
            return (_value > 0) ? _value : 0;
        case ActivationsType::SIGMOID:
            return 1.0 / (1.0 + std::exp(-_value));
        case ActivationsType::TANH:
            return std::tanh(_value);
        case ActivationsType::LINEAR:
            return _value;
        case ActivationsType::LeakyRELU:
            return (_value > 0) ? _value : _value * 0.001;
        default:
            return _value;
        }
    }

    double functionActivationDerivative(double _value, double _activated, ActivationsType _AT)
    {
        switch (_AT)
        {
        case ActivationsType::RELU:
            return (_value >= 0) ? 1 : 0;
        case ActivationsType::SIGMOID:
            return _activated * (1 - _activated);
        case ActivationsType::TANH:
            return 1 - _activated * _activated;
        case ActivationsType::LINEAR:
            return 1;
        case ActivationsType::LeakyRELU:
            return (_value > 0) ? 1 : 0.001;
        default:
            return 1;
        }
    }

    void applySoftmax(std::vector<double>& _inputData)
    {
        double maxValue = *std::max_element(_inputData.begin(), _inputData.end());
        double sum = 0;

        const double zeroSaver = 1e-10;

        for (int i = 0; i < _inputData.size(); i++)
        {
            _inputData[i] = std::exp(_inputData[i] - maxValue);
            sum += _inputData[i];
        }

        for (int i = 0; i < _inputData.size(); i++)
        {
            _inputData[i] = _inputData[i] / (sum + zeroSaver);
        }
    }

    std::vector<double> computePolicyGradient(const std::vector<double> _probs, int _action)
    {
        std::vector<double> resultGradient(_probs.size(), 0.0);

        for (int i = 0; i < _probs.size(); i++)
        {
            if (i == _action)
            {
                resultGradient[i] = 1.0 - _probs[i];
            }
            else
            {
                resultGradient[i] = -_probs[i];
            }
        }

        return resultGradient;
    }
}

namespace Ñool_Print
{
    std::string colorText(std::string _input, Color _color)
    {
        std::string colorUnicode = "";

        switch (_color)
        {
        case Color::RED:
            colorUnicode = "\u001b[31m";
            break;
        case Color::YELLOW:
            colorUnicode = "\u001b[33m";
            break;
        case Color::GREEN:
            colorUnicode = "\u001b[32m";
            break;
        case Color::BLUE:
            colorUnicode = "\u001b[34m";
            break;
        case Color::PURPLE:
            colorUnicode = "\u001b[35m";
            break;
        case Color::WHITE:
            colorUnicode = "\u001b[0m";
            break;
        default:
            colorUnicode = "";
            break;
        }


        return colorUnicode + _input + "\u001b[0m";
    }

}