#include "NeuralNetwork.h"


#include <cmath>
#include <vector>
#include <iostream>
#include <format>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>

//activations 

double NeuralNetwork::functionActivation(double _input, ActivationsType _AT)
{
	switch (_AT)
	{
	case SIGMOID:
		return 1.0 / (1.0 + std::exp(-_input));
	case TANH:
		return std::tanh(_input);
	case RELU:
		return (_input > 0) ? _input : 0;
	case LEAKY_RELU:
		return (_input > 0) ? _input : _input * 0.01;
	case LINEAR:
		return _input;
	default:
		return _input;
	}
}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

double NeuralNetwork::functionActivationDerivative(double _input, double _activatedInput, ActivationsType _AT)
{
	//three arguments are needed because in some derivatives you need to 
	//take either a value or an already activated value or all at once

	switch (_AT)
	{
	case SIGMOID:
		return _activatedInput * (1 - _activatedInput);
	case TANH:
		return 1 - _activatedInput * _activatedInput;
	case RELU:
		return (_input > 0) ? 1 : 0;
	case LEAKY_RELU:
		return (_input > 0) ? 1 : 0.01;
	case LINEAR:
		return 1;
	default:
		return 1;
	}
}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

void NeuralNetwork::applySoftmax(std::vector<double>& _inputData)
{
	//I have no freaking idea what's going on here
	//But it's necessary for... stabilization of the output values (if I understand correctly) ? 


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

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

void NeuralNetwork::checkСompatibility()
{
	countCnnLayers = CnnLayers.size();
	//ConvLayer(int _h, int _w, int _countFilters, int _countInputGhosts, int _coreSize, int _padding)
	for (int l = 0; l < countCnnLayers - 1; l++)
	{
		std::vector<int> currentConfig = CnnLayers[l]->returnConfig();
		std::vector<int> nextConfig = CnnLayers[l + 1]->returnConfig();


		// h and w
		if (currentConfig[0] + 2 * currentConfig[5] - currentConfig[4] + 1 != nextConfig[0])
		{
			throw std::invalid_argument("Ghost H incompatibility");
		}
		if (currentConfig[1] + 2 * currentConfig[5] - currentConfig[4] + 1 != nextConfig[1])
		{
			throw std::invalid_argument("Ghost W incompatibility");
		}
		//count ghosts
		if (currentConfig[2] != nextConfig[3])
		{
			throw std::invalid_argument("Incorrect count filters or input ghost");
		}

	}


}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

void NeuralNetwork::initializeNeuralNet()
{
	//it requires resizing the cubic vector of weights and generating random weights

	//generator random numbers

	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_real_distribution<double> distr(-0.3, 0.3);


	//init layers
	layers.resize(countLayers);

	//layers[0].neurons[0].neuronWeights[0]
	//layers[0].bias[0];

	for (int lr = 0; lr < countLayers; lr++)
	{
		layers[lr].neurons.resize(LayerNeuronSizes[lr]);

		layers[lr].bias.resize(LayerNeuronSizes[lr]);

		for (int n = 0; n < LayerNeuronSizes[lr]; n++)
		{
			if (lr > 0)
			{
				layers[lr].bias[n] = distr(generator);

				layers[lr].neurons[n].neuronWeights.resize(LayerNeuronSizes[lr - 1]);

				for (int weightIndex = 0; weightIndex < LayerNeuronSizes[lr - 1]; weightIndex++)
				{
					layers[lr].neurons[n].neuronWeights[weightIndex] = distr(generator);
				}
			}
			else
			{
				layers[lr].neurons[n].neuronWeights.resize(1);
				layers[lr].neurons[n].neuronWeights[0] = -1;
			}
		}
	}


	//=====================================================================
	//============================DEBUG====================================
	//=====================================================================
	std::cout << "Initializing random weights\n";

	/*if (DEBUG_MODE)
	{
		printWeight();
	}*/
	std::cout << "\nSuccessfully\n";
}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================


void NeuralNetwork::reInit(std::vector<int> _Layers, double _alpha, ActivationsType _hiddenActivation, ActivationsType _outputActivation)
{
	LayerNeuronSizes = _Layers;
	inputSize = _Layers.size() >= 2 ? _Layers[0] : throw std::invalid_argument("error");
	outputSize = _Layers.back();
	countLayers = _Layers.size();
	alpha = _alpha;
	hiddenActivation = _hiddenActivation;
	outputActivation = _outputActivation;
	initializeNeuralNet();
}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

std::vector<double> NeuralNetwork::forward(std::vector<double> _inputData)
{
	if (_inputData.size() != LayerNeuronSizes[0])
	{
		throw std::invalid_argument("discrepancy between the output data and the first layer");
	}


	for (int i = 0; i < inputSize; i++)
	{
		layers[0].neurons[i].value = _inputData[i];
	}

	for (int l = 1; l < countLayers - 1; l++)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			double sum = layers[l].bias[n];

			for (int w = 0; w < LayerNeuronSizes[l - 1]; w++)
			{
				sum += layers[l].neurons[n].neuronWeights[w] * layers[l - 1].neurons[w].value;
			}
			layers[l].neurons[n].value = functionActivation(sum, hiddenActivation);
		}
	}

	std::vector<double> result;


	if (outputActivation == SOFTMAX)
	{
		for (int n = 0; n < LayerNeuronSizes[countLayers - 1]; n++)
		{
			double sum = layers[countLayers - 1].bias[n];

			for (int w = 0; w < LayerNeuronSizes[countLayers - 2]; w++)
			{
				sum += layers[countLayers - 1].neurons[n].neuronWeights[w] * layers[countLayers - 2].neurons[w].value;
			}

			result.push_back(sum);
		}
		applySoftmax(result);
	}

	else
	{
		for (int n = 0; n < LayerNeuronSizes[countLayers - 1]; n++)
		{
			double sum = layers[countLayers - 1].bias[n];

			for (int w = 0; w < LayerNeuronSizes[countLayers - 2]; w++)
			{
				sum += layers[countLayers - 1].neurons[n].neuronWeights[w] * layers[countLayers - 2].neurons[w].value;
			}

			layers[countLayers - 1].neurons[n].value = functionActivation(sum, outputActivation);
			result.push_back(functionActivation(sum, outputActivation));
			//std::cout << "\n\n" << sum << "sum" << "\n\n";
			//std::cout << "\n\n" << functionActivation(sum, outputActivation) << "act" << "\n\n";
		}

	}

	return result;
}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

std::vector<double> NeuralNetwork::CnnForward(std::vector<std::vector<std::vector<double>>> _inputData)
{
	std::vector<std::vector<std::vector<double>>> layerResult;
	
	for (int l = 0; l < countCnnLayers; l++)
	{

		if (l == 0)
		{
			layerResult = CnnLayers[l]->forward(_inputData);
		}
		else
		{
			layerResult = CnnLayers[l]->forward(layerResult);
		}
	}

	//it's really cursed...
	std::vector<double> result(layerResult.size() * layerResult.back().size() * layerResult.back().back().size());
	//.back().back().back().back().back().back().back().back().back().back().back().back().back().back().back()

	std::vector<int> lastConfig = CnnLayers.back()->returnConfig();
	//{ hGhostSize , wGhostSize, countFilters, countInputGhosts, filterSize, padding};

	int totalIndex = 0;

	for (int l = 0; l < lastConfig[2]; l++)
	{
		for (int h = 0; h < lastConfig[0] + lastConfig[5] * 2 - lastConfig[4] + 1; h++)
		{
			for (int w = 0; w < lastConfig[1] + lastConfig[5] * 2 - lastConfig[4] + 1; w++)
			{
				result[totalIndex] = layerResult[l][h][w];
				totalIndex++;
			}
		}
	}

	return result;
	
}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

std::vector<double> NeuralNetwork::MultForward(std::vector<std::vector<std::vector<double>>> _inputData)
{
	return forward(CnnForward(_inputData));
}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

void NeuralNetwork::RLtrain(std::vector<double> _inputState, int _action, double _reward)
{
	//just forward with saves values

	std::vector<std::vector<double>> layerActivations(countLayers); //save the values of the neurons after applying the activation function
	std::vector<std::vector<double>> layerNotActivatedSum(countLayers); //I save the values of the neurons BEFORE applying the activation function (or just the "sum")

	for (int i = 0; i < countLayers; i++)
	{
		layerActivations[i].resize(LayerNeuronSizes[i], 0.0);
		layerNotActivatedSum[i].resize(LayerNeuronSizes[i], 0.0);
	}

	for (int i = 0; i < inputSize; i++)
	{
		layers[0].neurons[i].value = _inputState[i];
		layerActivations[0][i] = _inputState[i];
		layerNotActivatedSum[0][i] = _inputState[i];
	}

	for (int l = 1; l < countLayers - 1; l++)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			double sum = layers[l].bias[n];

			for (int w = 0; w < LayerNeuronSizes[l - 1]; w++)
			{
				//std::cout << std::format("Layer -> {}   Neuron -> {}   Weight Index -> {}\n", l, n, w);

				sum += layers[l].neurons[n].neuronWeights[w] * layers[l - 1].neurons[w].value;
			}

			//std::cout << std::format("layer -> {} neuron -> {} before activate -> {} after activate -> {}", l, n, sum, layers[l].neurons[n].value);

			layerNotActivatedSum[l][n] = sum;
			layers[l].neurons[n].value = functionActivation(sum, hiddenActivation);
			layerActivations[l][n] = layers[l].neurons[n].value;


			//std::cout << "\n\n" << sum << "\n\n";
		}
	}

	std::vector<double> activateResult;
	std::vector<double> notActivateResult;

	for (int n = 0; n < LayerNeuronSizes[countLayers - 1]; n++)
	{
		double sum = layers[countLayers - 1].bias[n];

		for (int w = 0; w < LayerNeuronSizes[countLayers - 2]; w++)
		{
			sum += layers[countLayers - 1].neurons[n].neuronWeights[w] * layers[countLayers - 2].neurons[w].value;
		}

		layerNotActivatedSum[countLayers - 1][n] = sum;

		if (outputActivation == SOFTMAX)
		{
			layers[countLayers - 1].neurons[n].value = sum;
		}
		else
		{
			layers[countLayers - 1].neurons[n].value = functionActivation(sum, outputActivation);
		}

		layerActivations[countLayers - 1][n] = layers[countLayers - 1].neurons[n].value;
	}

	if (outputActivation == SOFTMAX)
	{
		applySoftmax(layerActivations[countLayers - 1]);
		for (int n = 0; n < LayerNeuronSizes[countLayers - 1]; n++)
		{
			layers[countLayers - 1].neurons[n].value = layerActivations[countLayers - 1][n];
		}
	}



	std::vector<double> policyGradient = computePolicyGradient(layerActivations[countLayers - 1], _action);

	for (int i = 0; i < policyGradient.size(); i++)
	{
		policyGradient[i] *= _reward;
	}

	std::vector<double> outputGradient(outputSize);

	for (int i = 0; i < outputSize; i++)
	{
		if (outputActivation == SOFTMAX)
		{
			outputGradient[i] = policyGradient[i];
		}
		else
		{
			outputGradient[i] = policyGradient[i] * functionActivationDerivative(layerNotActivatedSum[countLayers - 1][i], layerActivations[countLayers - 1][i], outputActivation);
		}
	}


	std::vector<std::vector<double>> gradients(countLayers);

	for (int i = 0; i < countLayers; i++)
	{
		gradients[i].resize(LayerNeuronSizes[i], 0.0);
	}

	for (int i = 0; i < outputSize; i++)
	{
		gradients[countLayers - 1][i] = outputGradient[i];
	}

	for (int l = countLayers - 2; l >= 0; l--)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			double Error = 0.0;

			for (int next = 0; next < LayerNeuronSizes[l + 1]; next++)
			{
				Error += gradients[l + 1][next] * layers[l + 1].neurons[next].neuronWeights[n];
			}

			gradients[l][n] = Error * functionActivationDerivative(layerNotActivatedSum[l][n], layers[l].neurons[n].value, hiddenActivation);
		}
	}


	double maxGradient = 1.0;

	for (int l = 0; l < countLayers; l++)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			if (gradients[l][n] > maxGradient)
			{
				gradients[l][n] = maxGradient;
			}
			else if (gradients[l][n] < -maxGradient)
			{
				gradients[l][n] = -maxGradient;
			}

			if (std::isnan(gradients[l][n]))
			{
				gradients[l][n] = 0.0;
			}
		}
	}


	//after calculating the errors, you can calculate the change in weights.

	for (int l = 1; l < countLayers; l++)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			for (int w = 0; w < LayerNeuronSizes[l - 1]; w++)
			{
				double gradient = gradients[l][n] * layers[l - 1].neurons[w].value;

				layers[l].neurons[n].neuronWeights[w] += alpha * gradient;
			}

			layers[l].bias[n] += alpha * gradients[l][n];
		}
	}

}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

void NeuralNetwork::SLtrain(std::vector<double> _inputData, std::vector<double> _want)
{
	//just forward with saves values

	std::vector<std::vector<double>> layerActivations(countLayers); //save the values of the neurons after applying the activation function
	std::vector<std::vector<double>> layerNotActivatedSum(countLayers); //I save the values of the neurons BEFORE applying the activation function (or just the "sum")

	for (int i = 0; i < countLayers; i++)
	{
		layerActivations[i].resize(LayerNeuronSizes[i], 0.0);
		layerNotActivatedSum[i].resize(LayerNeuronSizes[i], 0.0);
	}

	for (int i = 0; i < inputSize; i++)
	{
		layers[0].neurons[i].value = _inputData[i];
		layerActivations[0][i] = _inputData[i];
		layerNotActivatedSum[0][i] = _inputData[i];
	}

	for (int l = 1; l < countLayers - 1; l++)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			double sum = layers[l].bias[n];

			for (int w = 0; w < LayerNeuronSizes[l - 1]; w++)
			{
				//std::cout << std::format("Layer -> {}   Neuron -> {}   Weight Index -> {}\n", l, n, w);

				sum += layers[l].neurons[n].neuronWeights[w] * layers[l - 1].neurons[w].value;
			}

			//std::cout << std::format("layer -> {} neuron -> {} before activate -> {} after activate -> {}", l, n, sum, layers[l].neurons[n].value);

			layerNotActivatedSum[l][n] = sum;
			layers[l].neurons[n].value = functionActivation(sum, hiddenActivation);
			layerActivations[l][n] = layers[l].neurons[n].value;


			//std::cout << "\n\n" << sum << "\n\n";
		}
	}

	std::vector<double> activateResult;
	std::vector<double> notActivateResult;

	for (int n = 0; n < LayerNeuronSizes[countLayers - 1]; n++)
	{
		double sum = layers[countLayers - 1].bias[n];

		for (int w = 0; w < LayerNeuronSizes[countLayers - 2]; w++)
		{
			sum += layers[countLayers - 1].neurons[n].neuronWeights[w] * layers[countLayers - 2].neurons[w].value;
		}

		layerNotActivatedSum[countLayers - 1][n] = sum;

		if (outputActivation == SOFTMAX)
		{
			layers[countLayers - 1].neurons[n].value = sum;
		}
		else
		{
			layers[countLayers - 1].neurons[n].value = functionActivation(sum, outputActivation);
		}

		layerActivations[countLayers - 1][n] = layers[countLayers - 1].neurons[n].value;
	}

	if (outputActivation == SOFTMAX)
	{
		applySoftmax(layerActivations[countLayers - 1]);
		for (int n = 0; n < LayerNeuronSizes[countLayers - 1]; n++)
		{
			layers[countLayers - 1].neurons[n].value = layerActivations[countLayers - 1][n];
		}
	}

	//calculates errors

	//calculate output error

	std::vector<double> outputError(outputSize);

	for (int i = 0; i < outputSize; i++)
	{
		double Error = layers[countLayers - 1].neurons[i].value - _want[i];
		if (outputActivation == SOFTMAX)
		{
			outputError[i] = Error;
		}
		else
		{
			outputError[i] = Error * functionActivationDerivative(layerNotActivatedSum[countLayers - 1][i], layers[countLayers - 1].neurons[i].value, outputActivation);
		}
	}


	//calculate hidden errors


	std::vector<std::vector<double>> ERRORS(countLayers);

	for (int i = 0; i < countLayers; i++)
	{
		ERRORS[i].resize(LayerNeuronSizes[i], 0.0);
	}

	for (int i = 0; i < outputSize; i++)
	{
		ERRORS[countLayers - 1][i] = outputError[i];
	}

	for (int l = countLayers - 2; l >= 1; l--)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			double Error = 0.0;

			for (int next = 0; next < LayerNeuronSizes[l + 1]; next++)
			{
				Error += ERRORS[l + 1][next] * layers[l + 1].neurons[next].neuronWeights[n];
			}

			ERRORS[l][n] = Error * functionActivationDerivative(layerNotActivatedSum[l][n], layers[l].neurons[n].value, hiddenActivation);
		}
	}

	//after calculating the errors calculate the change in weights

	for (int l = 1; l < countLayers; l++)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			for (int w = 0; w < LayerNeuronSizes[l - 1]; w++)
			{
				double gradient = ERRORS[l][n] * layers[l - 1].neurons[w].value;

				layers[l].neurons[n].neuronWeights[w] -= alpha * gradient;
			}

			layers[l].bias[n] -= alpha * ERRORS[l][n];
		}
	}


	//calculate the total error for viewing statistics and optimizing training

	totalError = 0;

	for (int i = 0; i < outputSize; i++)
	{
		if (outputActivation == SOFTMAX)
		{
			totalError += -_want[i] * std::log(layers[countLayers - 1].neurons[i].value + 1e-10);
		}
		else
		{
			totalError += 0.5 * std::pow(_want[i] - layers[countLayers - 1].neurons[i].value, 2);
		}
	}

}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

void NeuralNetwork::CNN_With_FCN_SL_train(std::vector<std::vector<std::vector<double>>> _inputData, std::vector<double> _want)
{
	std::vector<double> getCnnResult = CnnForward(_inputData);

	std::vector<std::vector<double>> layerActivations(countLayers); //save the values of the neurons after applying the activation function
	std::vector<std::vector<double>> layerNotActivatedSum(countLayers); //I save the values of the neurons BEFORE applying the activation function (or just the "sum")

	for (int i = 0; i < countLayers; i++)
	{
		layerActivations[i].resize(LayerNeuronSizes[i], 0.0);
		layerNotActivatedSum[i].resize(LayerNeuronSizes[i], 0.0);
	}

	for (int i = 0; i < inputSize; i++)
	{
		layers[0].neurons[i].value = getCnnResult[i];
		layerActivations[0][i] = getCnnResult[i];
		layerNotActivatedSum[0][i] = getCnnResult[i];
	}

	for (int l = 1; l < countLayers - 1; l++)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			double sum = layers[l].bias[n];

			for (int w = 0; w < LayerNeuronSizes[l - 1]; w++)
			{
				//std::cout << std::format("Layer -> {}   Neuron -> {}   Weight Index -> {}\n", l, n, w);

				sum += layers[l].neurons[n].neuronWeights[w] * layers[l - 1].neurons[w].value;
			}

			//std::cout << std::format("layer -> {} neuron -> {} before activate -> {} after activate -> {}", l, n, sum, layers[l].neurons[n].value);

			layerNotActivatedSum[l][n] = sum;
			layers[l].neurons[n].value = functionActivation(sum, hiddenActivation);
			layerActivations[l][n] = layers[l].neurons[n].value;


			//std::cout << "\n\n" << sum << "\n\n";
		}
	}

	std::vector<double> activateResult;
	std::vector<double> notActivateResult;

	for (int n = 0; n < LayerNeuronSizes[countLayers - 1]; n++)
	{
		double sum = layers[countLayers - 1].bias[n];

		for (int w = 0; w < LayerNeuronSizes[countLayers - 2]; w++)
		{
			sum += layers[countLayers - 1].neurons[n].neuronWeights[w] * layers[countLayers - 2].neurons[w].value;
		}

		layerNotActivatedSum[countLayers - 1][n] = sum;

		if (outputActivation == SOFTMAX)
		{
			layers[countLayers - 1].neurons[n].value = sum;
		}
		else
		{
			layers[countLayers - 1].neurons[n].value = functionActivation(sum, outputActivation);
		}

		layerActivations[countLayers - 1][n] = layers[countLayers - 1].neurons[n].value;
	}

	if (outputActivation == SOFTMAX)
	{
		applySoftmax(layerActivations[countLayers - 1]);
		for (int n = 0; n < LayerNeuronSizes[countLayers - 1]; n++)
		{
			layers[countLayers - 1].neurons[n].value = layerActivations[countLayers - 1][n];
		}
	}

	//calculates errors

	//calculate output error

	std::vector<double> outputError(outputSize);

	for (int i = 0; i < outputSize; i++)
	{
		double Error = layers[countLayers - 1].neurons[i].value - _want[i];
		if (outputActivation == SOFTMAX)
		{
			outputError[i] = Error;
		}
		else
		{
			outputError[i] = Error * functionActivationDerivative(layerNotActivatedSum[countLayers - 1][i], layers[countLayers - 1].neurons[i].value, outputActivation);
		}
	}


	//calculate hidden errors


	std::vector<std::vector<double>> ERRORS(countLayers);

	for (int i = 0; i < countLayers; i++)
	{
		ERRORS[i].resize(LayerNeuronSizes[i], 0.0);
	}

	for (int i = 0; i < outputSize; i++)
	{
		ERRORS[countLayers - 1][i] = outputError[i];
	}

	for (int l = countLayers - 2; l >= 1; l--)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			double Error = 0.0;

			for (int next = 0; next < LayerNeuronSizes[l + 1]; next++)
			{
				Error += ERRORS[l + 1][next] * layers[l + 1].neurons[next].neuronWeights[n];
			}

			ERRORS[l][n] = Error * functionActivationDerivative(layerNotActivatedSum[l][n], layers[l].neurons[n].value, hiddenActivation);
		}
	}

	//after calculating the errors calculate the change in weights

	std::vector<double> gradToCnn(LayerNeuronSizes[0]);
 
	for (int l = 1; l < countLayers; l++)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			if (l == 1)
			{
				for (int w = 0; w < LayerNeuronSizes[l - 1]; w++)
				{
					gradToCnn[w] += ERRORS[l][n] * layers[l].neurons[n].neuronWeights[w];
				}
			}

			for (int w = 0; w < LayerNeuronSizes[l - 1]; w++)
			{
				double gradient = ERRORS[l][n] * layers[l - 1].neurons[w].value;

				layers[l].neurons[n].neuronWeights[w] -= alpha * gradient;
			}

			layers[l].bias[n] -= alpha * ERRORS[l][n];
		}
	}

	totalError = 0;

	for (int i = 0; i < outputSize; i++)
	{
		if (outputActivation == SOFTMAX)
		{
			totalError += -_want[i] * std::log(layers[countLayers - 1].neurons[i].value + 1e-10);
		}
		else
		{
			totalError += 0.5 * std::pow(_want[i] - layers[countLayers - 1].neurons[i].value, 2);
		}
	}

	std::cout << "\n" << totalError << "\n";
	
	//{ hGhostSize , wGhostSize, countFilters, countInputGhosts, filterSize, padding};

	std::vector<int> retrunBackConfig = CnnLayers.back()->returnConfig();

	int countInputGhosts = retrunBackConfig[3];
	int outputH = retrunBackConfig[0] + 2 * retrunBackConfig[5] - retrunBackConfig[4] + 1;
	int outputW = retrunBackConfig[1] + 2 * retrunBackConfig[5] - retrunBackConfig[4] + 1;

	std::vector<std::vector<std::vector<double>>> gradTensorToCnn(countInputGhosts, std::vector<std::vector<double>>(outputH, std::vector<double>(outputW)));

	int index = 0;

	for (int l = 0; l < countInputGhosts; l++)
	{
		for (int h = 0; h < outputH; h++)
		{
			for (int w = 0; w < outputW; w++)
			{
				gradTensorToCnn[l][h][w] = gradToCnn[index];
				index++;
			}
		}
	}

}


//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

std::vector<double> NeuralNetwork::computePolicyGradient(const std::vector<double> _probs, int _action)
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

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

void NeuralNetwork::trySaveWeight()
{
	std::vector<double> numbers;

	for (int l = 1; l < countLayers; l++)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			for (int w = 0; w < LayerNeuronSizes[l - 1]; w++)
			{
				numbers.push_back(layers[l].neurons[n].neuronWeights[w]);
			}
		}
	}

	std::string fileName = "model_[-";

	for (int i = 0; i < countLayers; i++)
	{
		fileName += std::to_string(LayerNeuronSizes[i]) + "-";
	}

	fileName += ".bin";

	std::ofstream file(fileName, std::ios::binary);

	int size = numbers.size();

	file.write(reinterpret_cast<const char*>(&size), sizeof(size));
	file.write(reinterpret_cast<const char*>(numbers.data()), size * sizeof(double));
	file.close();
}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
//====================================================================================================================================================================================================

bool NeuralNetwork::tryLoadWeight()
{
	std::string fileName = "model_[-";

	for (int i = 0; i < countLayers; i++)
	{
		fileName += std::to_string(LayerNeuronSizes[i]) + "-";
	}

	fileName += ".bin";

	std::ifstream file(fileName, std::ios::binary);

	if (!file)
	{
		std::cout << "Cannot open file\n";
		std::cout << fileName;
		file.close();
		return false;
	}

	size_t size;
	file.read(reinterpret_cast<char*>(&size), sizeof(size));

	std::vector<double> numbers(size);

	file.read(reinterpret_cast<char*>(numbers.data()), size * sizeof(double));

	file.close();

	int index = 0;

	for (int l = 1; l < countLayers; l++)
	{
		for (int n = 0; n < LayerNeuronSizes[l]; n++)
		{
			for (int w = 0; w < LayerNeuronSizes[l - 1]; w++)
			{
				try
				{
					layers[l].neurons[n].neuronWeights[w] = numbers[index];
					index++;
				}
				catch (...)
				{
					std::cout << "\n\n" << l << "\n" << n << "\n" << w << "\n" << index << "\n";
					throw std::runtime_error("Weight set error");

				}
			}
		}
	}


	return true;
}

//====================================================================================================================================================================================================
//====================================================================================================================================================================================================
