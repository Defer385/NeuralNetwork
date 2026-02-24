#pragma once

#include <vector>
#include <iostream>
#include <optional>
#include <memory>

#include "Layer.h"
#include "Neuron.h"
#include "baseCnnLayer.h"

enum ActivationsType
{
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    LINEAR,
    SOFTMAX
};

enum NNArchMode
{
    NORMAL,
    CNN
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
    void CNN_Train(std::vector<double> _inputGrad);
    void CNN_With_FCN_SL_train(std::vector<std::vector<std::vector<double>>> _inputData, std::vector<double> _want);

    std::vector<double> forward(std::vector<double> _inputData);
    std::vector<double> CnnForward(std::vector<std::vector<std::vector<double>>> _inputData);
    std::vector<double> MultForward(std::vector<std::vector<std::vector<double>>> _inputData);

    void init(std::vector<int> _Layers, double _alpha, ActivationsType _hiddenActivation, ActivationsType _outputActivation);
    
    static double functionActivation(double _input, ActivationsType _AT);
    static double functionActivationDerivative(double _input, double _activatedInput, ActivationsType _AT);

    //User-defined functions
    bool tryLoadWeight();
    void trySaveWeight();
    void reInit(std::vector<int> _Layers, double _alpha, ActivationsType _hiddenActivation, ActivationsType _outputActivation); //If necessary, can completely change the neural network.

    //Auxiliary functions
    void checkСompatibility();

    //Debug Functions
    void printWeight(); //Outputs all internal information about the neural network including weights, bias neurons, and neuron values
    void printConfig(); //Outputs information about the neural network

    //Auxiliary variables
    double totalError = 1.0;
    std::vector<std::unique_ptr<baseCnnLayer>> CnnLayers;
private:
    //mains
    void initializeNeuralNet();
    void initializeNeuralNetCnnPart();
    void applySoftmax(std::vector<double>& _inputData);
    std::vector<double> computePolicyGradient(const std::vector<double> _probs, int _action);


    std::vector<Layer> layers; //All layers
    std::vector<int> LayerNeuronSizes;
    std::vector<std::vector<int>> CnnLayersNeuronSizes;

    int inputSize; //Number of input neurons
    int outputSize; //Number of output neurons
    int countLayers; //Total number of layer
    int countCnnLayers;
    double alpha; //Learning rate
    //helped

    ActivationsType hiddenActivation;
    ActivationsType outputActivation;

    NNArchMode mode;


};
