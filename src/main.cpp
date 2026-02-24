#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <format>
#include <assert.h>
#include <memory>

#include "NeuralNetwork.h"
#include "ConvLayer.h"

#ifndef TRAIN_DIR
#define TRAIN_DIR "train"
#endif

int generateRandomNumber(int _min, int _max)
{
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> distr(_min, _max);

    return distr(generator);
}

int main()
{
    NeuralNetwork nn({ 6272,256,10 }, 0.0005, RELU, SOFTMAX);
    //{ hGhostSize , wGhostSize, countFilters, countInputGhosts, filterSize, padding};
    nn.CnnLayers.push_back(std::make_unique<ConvLayer>(28, 28, 4, 1, 3, 1));

    nn.CnnLayers.push_back(std::make_unique<ConvLayer>(28, 28, 8, 4, 3, 1));

    nn.checkСompatibility(); //NECESSARILY to CNN

    int fileNumber = 0;

    std::string folder = TRAIN_DIR "/MNIST/digits_zero_one/";
    std::string fileName = "mnist_";
    
    std::cout << "Use folder to train -> " << folder + fileName << "\n";
    std::cin.ignore();

    while (fileNumber < 50000)
    {
        if (fileNumber < 49950)
        {
            std::string fullPath = folder + fileName + std::to_string(generateRandomNumber(0,19000)) + ".txt";

            

            // Открываем файл
            std::ifstream file(fullPath);

            if (!file.is_open()) {
                std::cerr << "file open error: " << fullPath << std::endl;
                return 1;
            }

            std::vector<double> want;

            std::string firstLine;
            if (std::getline(file, firstLine))
            {
                std::stringstream ss(firstLine);

                for (int i = 0; i < 10; i++)
                {
                    int digit;
                    if (!(ss >> digit))
                    {
                        std::cerr << "not 10" << std::endl;
                        return 1;
                    }
                    want.push_back(digit);
                }
            }
            else
            {
                std::cerr << "empty file" << std::endl;
                return 1;
            }

            std::vector<std::vector<double>> matrix(28, std::vector<double>(28));

            for (int i = 0; i < 28; i++)
            {
                std::string line;
                if (!std::getline(file, line))
                {
                    std::cerr << "no 28x28" << std::endl;
                    return 1;
                }

                std::stringstream ss(line);
                for (int j = 0; j < 28; j++)
                {
                    if (!(ss >> matrix[i][j]))
                    {
                        std::cout << fullPath << "\n";
                        std::cerr << "error in " << i + 2 << " no numbers" << std::endl;

                        return 1;
                    }
                }
            }

            std::vector<std::vector<std::vector<double>>> tensor(1, matrix);

            nn.CNN_With_FCN_SL_train(tensor, want);

            std::cout << "\n" << "filenumber ->" << fileNumber << "\n";

            fileNumber++;

        }
        else
        {
            std::string folder = "train/MNIST/digits_zero_one/";
            std::string fileName = "mnist_";

            std::string fullPath = folder + fileName + std::to_string(generateRandomNumber(19001, 19999)) + ".txt";

            std::ifstream file(fullPath);

            if (!file.is_open()) {
                std::cerr << "file open error: " << fullPath << std::endl;
                return 1;
            }

            std::vector<double> want;


            std::string firstLine;
            if (std::getline(file, firstLine))
            {
                std::stringstream ss(firstLine);

                for (int i = 0; i < 10; i++)
                {
                    int digit;
                    if (!(ss >> digit))
                    {
                        std::cerr << "not 10" << std::endl;
                        return 1;
                    }
                    want.push_back(digit);
                }
            }
            else
            {
                std::cerr << "empty file" << std::endl;
                return 1;
            }

            std::vector<std::vector<double>> matrix(28, std::vector<double>(28));

            for (int i = 0; i < 28; i++)
            {
                std::string line;
                if (!std::getline(file, line))
                {
                    std::cerr << "no 28x28" << std::endl;
                    return 1;
                }

                std::stringstream ss(line);
                for (int j = 0; j < 28; j++)
                {
                    if (!(ss >> matrix[i][j]))
                    {
                        std::cout << fullPath << "\n";
                        std::cerr << "error in " << i + 2 << " no numbers" << std::endl;

                        return 1;
                    }
                }
            }

            std::vector<std::vector<std::vector<double>>> tensor(1, matrix);

            std::vector<double> getResult = nn.MultForward(tensor);

            for (int h = 0; h < 28; h++)
            {
                for (int w = 0; w < 28; w++)
                {
                    if (matrix[h][w] == 0)
                    {
                        std::cout << " ";
                    }
                    else
                    {
                        std::cout << "#";
                    }
                }
                std::cout << "\n";
            }

            std::cout << "result\n";

            for (int i = 0; i < 10; i++)
            {
                std::cout << i << " " << getResult[i] * 100.0 << "%\n";
            }

            

            fileNumber++;
        }
    }
    
    

    return 0;

}