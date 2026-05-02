#pragma once
#include "Layer.h"
#include "Tensor.h"
#include "utils.h"
#include <random>

class Dense : public Layer
{
public:

    Dense(int _input, int _output, ActivationsType _AT) :
        weights({ _output, _input }),
        bias({ _output }),
        saveInput({ _input }),
        saveActivated({ _output }),
        saveNotActivated({ _output }),
        gradBeforeAt({ _output }),
        gradWeight({ _output, _input}),
        gradBias({ _output })
    {
        La_type = DENSE;
        AT_type = _AT;
        inputSize = _input;
        outputSize = _output;


        std::random_device device;
        std::mt19937 generator(device());
        std::uniform_real_distribution<double> distr(-0.3, 0.3);


        for (int n = 0; n < outputSize; n++)
        {
            for (int w = 0; w < inputSize; w++)
            {
                weights.writeTwoD(distr(generator), n, w);
            }

            bias.writeOneD(distr(generator), n);
        }

    }


    Tensor forward(Tensor _input) override
    {
        Tensor result({outputSize});

        saveInput = _input;

        for (int n = 0; n < outputSize; n++)
        {
            double sum = 0.0;

            for (int w = 0; w < inputSize; w++)
            {
                sum += _input.readOneD(w) * weights.readTwoD(n, w);
            }
            
            sum += bias.readOneD(n);
            saveNotActivated.writeOneD(sum, n);

            double actSum = NN_Utils::functionActivation(sum, AT_type);

            result.writeOneD(actSum, n);
            saveActivated.writeOneD(actSum, n);
        }

        if (AT_type == ActivationsType::SOFTMAX)
        {
            NN_Utils::applySoftmax(result.dat());
        }
        

        return result;

    }
    Tensor backward(Tensor _input) override
    {
        //calculate gradient before activation

        gradBeforeAt.clear();
        gradBias.clear();
        gradWeight.clear();

        for (int n = 0; n < outputSize; n++)
        {
            if (AT_type == ActivationsType::SOFTMAX)
            {
                gradBeforeAt.writeOneD(_input.readOneD(n), n);
            }
            else
            {
                gradBeforeAt.writeOneD(_input.readOneD(n) * NN_Utils::functionActivationDerivative(saveNotActivated.readOneD(n),saveActivated.readOneD(n), AT_type), n);
            }
        }

        //calculate grad to weights and bias

        for (int n = 0; n < outputSize; n++)
        {
            gradBias.writeOneD(gradBeforeAt.readOneD(n), n);

            for (int w = 0; w < inputSize; w++)
            {
                double d = saveInput.readOneD(w) * gradBeforeAt.readOneD(n);

                gradWeight.writeTwoD(d, n, w);
            }
        }

        //calculate to prev (next in backpropagation) layer grad

        Tensor gradNext({ inputSize });

        for (int i = 0; i < inputSize; i++)
        {
            double sum = 0.0;
            for (int n = 0; n < outputSize; n++)
            {
                sum += gradBeforeAt.readOneD(n) * weights.readTwoD(n, i);
            }

            gradNext.writeOneD(sum, i);
        }

        //calculate the weight and bias changes
        

        return gradNext;
    }

    void update() override
    {
        for (int n = 0; n < outputSize; n++)
        {
            double biasDelta = alpha * gradBias.readOneD(n);

            bias.writeOneD(bias.readOneD(n) - biasDelta, n);

            for (int w = 0; w < inputSize; w++)
            {
                double delta = gradWeight.readTwoD(n, w) * alpha;
                weights.writeTwoD(weights.readTwoD(n, w) - delta, n, w);
                //std::cout << "WEIGHTS -> " << weights.readTwoD(n, w) << "\n";

            }

            //std::cout << "BIAS -> " << bias.readOneD(n) << "\n";
        }
    }

    LayerType returnLayerType() override
    {
        return La_type;
    }
    ActivationsType returnFunctionActivation() override
    {
        return AT_type;
    }

    std::vector<int> returnConfig() override
    {
        return {inputSize, outputSize};
    }

    Tensor returnNotActivated() override
    {
        return saveNotActivated;
    }
    Tensor returnActivated() override
    {
        return saveActivated;
    }
    Tensor returnSaveInput() override
    {
        return saveInput;
    }

    void setAlpha(double _alpha) override
    {
        alpha = _alpha;
    }

    //~Dense();

    

private:
    //main par
    Tensor weights;
    Tensor bias;

    int inputSize;
    int outputSize;
    double alpha;

    //to train
    Tensor saveInput;
    Tensor saveActivated;
    Tensor saveNotActivated;

    Tensor gradBeforeAt;
    Tensor gradWeight;
    Tensor gradBias;
};
