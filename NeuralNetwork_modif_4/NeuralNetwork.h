#pragma once
#pragma once

#include <vector>
#include <iostream>

#include "Layer.h"
#include "Neuron.h"

enum ActivationsType
{
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    LINEAR,
    SOFTMAX
};

enum Mode
{
    SL,
    RL
};


class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<int> _Layers, double _alpha, ActivationsType _hiddenActivation, ActivationsType _outputActivation) :
        //set inputs

        LayerNeuronSizes(_Layers),
        inputSize(_Layers.size() >= 2 ? _Layers[0] : throw std::invalid_argument("error")),
        outputSize(_Layers.back()),
        alpha(_alpha),
        hiddenActivation(_hiddenActivation),
        outputActivation(_outputActivation),
        countLayers(_Layers.size())

    {
        //initialization of parameters and vectors
        initializeNeuralNet();
    }

    //Main functions
    void RLtrain(std::vector<double> _inputState, int _action, double _reward);
    void SLtrain(std::vector<double> _inputData, std::vector<double> _want);

    std::vector<double> forward(std::vector<double> _inputData);

    //User-defined functions
    bool tryLoadWeight();
    void trySaveWeight();
    void reInit(std::vector<int> _Layers, double _alpha, ActivationsType _hiddenActivation, ActivationsType _outputActivation); //If necessary, can completely change the neural network.
    
    //Auxiliary functions

    //Debug Functions
    void printWeight(); //Outputs all internal information about the neural network including weights, bias neurons, and neuron values
    void printConfig(); //Outputs information about the neural network

    //Auxiliary variables
    double totalError = 1.0;
private:
    void initializeNeuralNet();
    void applySoftmax(std::vector<double>& _inputData);
    std::vector<double> computePolicyGradient(const std::vector<double> _probs, int _action);
    
    double functionActivation(double _input, ActivationsType _AT);
    double functionActivationDerivative(double _input, double _activatedInput, ActivationsType _AT);

    std::vector<Layer> layers; //All layers
    std::vector<int> LayerNeuronSizes;

    int inputSize; //Number of input neurons
    int outputSize; //Number of output neurons
    int countLayers; //Total number of layer
    double alpha; //Learning rate


    ActivationsType hiddenActivation;
    ActivationsType outputActivation;


};
