#include "NeuralNetwork.h"


void NeuralNetwork::init(double _alpha)
{
    std::cout << Ñool_Print::colorText("Initialization...", Color::YELLOW);
    std::cout << "\n\n";

    /*if (adam_used)
    {
        std::cout << Ñool_Print::colorText("ADAM USED",Color::GREEN);
    }
    else
    {
        std::cout << Ñool_Print::colorText("[WARMING] -> A D A M is not used", Color::YELLOW);
    }
    std::cout << "\n";
    if (adam_used)
    {
        std::cout << Ñool_Print::colorText("INPULSE USED", Color::GREEN);
    }
    else
    {
        std::cout << Ñool_Print::colorText("[WARMING] -> impulse is not used", Color::YELLOW);
    }
    std::cout << "\n";
    if (adam_used)
    {
        std::cout << Ñool_Print::colorText("GRAD CLIPPING USED", Color::GREEN);
    }
    else
    {
        std::cout << Ñool_Print::colorText("[WARMING] -> Grad clipping is not used",Color::YELLOW);
    }
    std::cout << "\n\n\n";*/


    if (layers.size() == 0)
    {
        std::cout << Ñool_Print::colorText("[CRITICAL ERROR]", Color::RED);
        std::invalid_argument("No layers");
    }

    std::vector<int> refConfig = layers[0]->returnConfig();

    for (int i = 1; i < layers.size(); i++)
    {
        std::vector<int> nowConfig = layers[i]->returnConfig();

        if (refConfig[1] != nowConfig[0])
        {
            std::cout << Ñool_Print::colorText("[CRITICAL ERROR]", Color::RED);
            std::invalid_argument("Incorrect layer sizes");
        }
    }

    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->setAlpha(_alpha);
    }

    std::cout << Ñool_Print::colorText("Successfully", Color::GREEN);
    std::cout << "\n\n";

    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->printDebug();
    }

    std::cout << "\n\n";

    
}

void NeuralNetwork::add(Layer* _layer)
{
    layers.push_back(_layer);
}

Tensor NeuralNetwork::forward(Tensor _data)
{
    Tensor preResult = _data;

    for (int l = 0; l < layers.size(); l++)
    {
        if (l == 0)
        {
            preResult = layers[l]->forward(_data);
        }
        else
        {
            preResult = layers[l]->forward(preResult);
        }
    }

    return preResult;
}

void NeuralNetwork::SLtrain(Tensor _data, Tensor _want)
{
    Tensor result = forward(_data);

    int outputSize = layers.back()->returnConfig()[1];
    Tensor gradWant({ outputSize });

    double totalError = 0;

    for (int i = 0; i < outputSize; i++)
    {
        double output = result.readOneD(i);
        double tar = _want.readOneD(i);
        double grad = 0;

        if (layers.back()->returnFunctionActivation() == ActivationsType::SOFTMAX)
        {
            grad = output - tar;
            totalError += -tar * std::log(output + 1e-10);
        }
        else
        {
            double err = output - tar;
            double notActiv = layers.back()->returnNotActivated().readOneD(i);
            double der = NN_Utils::functionActivationDerivative(notActiv, output, layers.back()->returnFunctionActivation());
            grad = err * der;
            totalError += 0.5 * std::pow(err,2);
        }

        gradWant.writeOneD(grad, i);
        
    }

    NeuralNetwork::totalError = totalError;

    Tensor gradInTime = gradWant;
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        gradInTime = layers[i]->backward(gradInTime);
    }

    for (int i = layers.size() - 1; i >= 0; i--)
    {
        layers[i]->update();
    }
    
}

void NeuralNetwork::RLtrain(Tensor _state, int _action, double _reward)
{
    int outputSize = layers.back()->returnConfig()[1];
    double totalError = 0;
    
    Tensor result = forward(_state);

    //std::cout << result.readOneD(0) << "  " << result.readOneD(1) << "\n";
    
    Tensor gradWant({ outputSize });
    
    gradWant.insertVector(NN_Utils::computePolicyGradient(layers.back()->returnActivated().dat(), _action));

    for (int i = 0; i < outputSize; i++)
    {
        double grad = gradWant.readOneD(i);

        grad *= - _reward;

        gradWant.writeOneD(grad, i);
    }

    NeuralNetwork::totalError = totalError;

    Tensor gradInTime = gradWant;
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        gradInTime = layers[i]->backward(gradInTime);
    }

    for (int i = layers.size() - 1; i >= 0; i--)
    {
        layers[i]->update();
    }
}