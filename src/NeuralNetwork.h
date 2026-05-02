#pragma once

#include <vector>
#include <random>
#include <iostream>


#include "Layer.h"
#include "Tensor.h"
#include "utils.h"

enum class LossFunType
{
    MSE,
    MAE,
    HuberLoss,
    CrossEntropy
};

class NeuralNetwork
{
public:
    ~NeuralNetwork()
    {
        for (int i = 0; i < layers.size(); i++)
        {
            delete layers[i];
        }

        std::cout << layers.size() << "\n";
    }

    /*NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;*/

    void init(double _alpha);

    void add(Layer* _layer);
    Tensor forward(Tensor _data);
    void SLtrain(Tensor _data, Tensor _want);
    void RLtrain(Tensor _state, int _action_, double _reward);

    double getError()
    {
        return totalError;
    }

private:
    std::vector<Layer*> layers;

    double totalError;
    double alpha;
    double beta = 0.001;

    bool adam_used = false;
    bool impulse_used = false;
    bool gradClipping_used = false;

};

