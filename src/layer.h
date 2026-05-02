#pragma once

#include "Tensor.h"
#include "utils.h"
#include <vector>

enum LayerType
{
    DENSE,
    CONV_TWO_D,
    FLATTEN,
    FUNCTION_ACTIVATION
};

class Layer
{
public:

    virtual Tensor forward(Tensor _input) = 0;
    virtual Tensor backward(Tensor _input) = 0;
    virtual void update() = 0;

    virtual Tensor returnNotActivated() = 0;
    virtual Tensor returnActivated() = 0;
    virtual Tensor returnSaveInput() = 0;

    virtual LayerType returnLayerType() = 0;
    virtual ActivationsType returnFunctionActivation() = 0;
    virtual std::vector<int> returnConfig() = 0;




    virtual void printDebug()
    {
        std::cout << Ñool_Print::colorText("[LAYER_DEBUG]", Color::YELLOW) << "\n";
        std::cout << "[LAYER TYPE] -> ";
        switch (La_type)
        {
        case DENSE:
            std::cout << Ñool_Print::colorText("DENSE", Color::GREEN) << "\n";
            break;
        case CONV_TWO_D:
            std::cout << Ñool_Print::colorText("CONV_TWO_D", Color::GREEN) << "\n";
            break;
        case FLATTEN:
            std::cout << Ñool_Print::colorText("FLATTEN", Color::GREEN) << "\n";
            break;
        default:
            break;
        }

        std::cout << "[Activation Function] -> ";
        switch (AT_type)
        {
        case ActivationsType::RELU:
            std::cout << Ñool_Print::colorText("RELU", Color::GREEN) << "\n";
            break;
        case ActivationsType::SIGMOID:
            std::cout << Ñool_Print::colorText("SIGMOID", Color::GREEN) << "\n";
            break;
        case ActivationsType::TANH:
            std::cout << Ñool_Print::colorText("TANH", Color::GREEN) << "\n";
            break;
        case ActivationsType::LINEAR:
            std::cout << Ñool_Print::colorText("LINEAR", Color::GREEN) << "\n";
            break;
        case ActivationsType::LeakyRELU:
            std::cout << Ñool_Print::colorText("LeakyRELU", Color::GREEN) << "\n";
            break;
        case ActivationsType::SOFTMAX:
            std::cout << Ñool_Print::colorText("SOFTMAX", Color::GREEN) << "\n";
            break;
        default:
            break;
        }

        std::cout << "\nConfig -> \n";

        std::vector<int> conf = returnConfig();

        std::cout << "INPUT SIZE -> " << std::to_string(conf[0]) << "\n";
        std::cout << "OUTPUT SIZE -> " << std::to_string(conf[1]) << "\n";
        std::cout << "COUNT WEIGHTS -> " << std::to_string(conf[0] * conf[1]) << "\n\n\n";

    }

    virtual void setAlpha(double _alpha) = 0;

    virtual ~Layer() = default;

    LayerType La_type;
    ActivationsType AT_type;
};