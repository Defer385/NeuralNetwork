#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <format>
#include <assert.h>


#include "NeuralNetwork.h"

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
    std::string folder = TRAIN_DIR "/MNIST/digits_zero_one/";
    std::cout << "Using train folder: " << folder << "\n";

    std::vector<double> data = { 0,0,0,0,0,0,0 ,0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0 };
    std::vector<double> want = { 0,0,0 };

    NeuralNetwork nn({ 784, 256,128,64,10 }, 0.0005, LEAKY_RELU, SOFTMAX);

    int countIter = 0;

    double secondToTrain = 0;

    if (true)
    {

        std::cin.ignore();
        clock_t startClock = clock();
        countIter = 0;
        while (countIter < 19000)
        {

            data.clear();
            want.clear();

            std::string filename = folder + "mnist_" + std::to_string(generateRandomNumber(1, 18999)) + ".txt";

            std::ifstream file(filename);
            if (!file.is_open())
            {
                std::cerr << "file open error: " << filename << std::endl;
                return 1;
            }

            std::string first_line;
            if (std::getline(file, first_line))
            {
                std::istringstream iss(first_line);
                int value;

                while (iss >> value)
                {
                    want.push_back(value);
                }

            }
            else
            {
                std::cerr << "error" << std::endl;
                return 1;
            }

            std::string line;
            int row_count = 0;
            while (std::getline(file, line))
            {
                if (line.empty()) continue;

                std::istringstream iss(line);
                double pixel;

                while (iss >> pixel)
                {
                    data.push_back(pixel);
                }

                row_count++;
            }

            //for (int i = 0; i < data.size(); i++)
            //{
            //    std::cout << "\n  --  " << data[i] << "\n";
            //}

            //std::cout << "\n\n" << data.size();


            //for (int i = 0; i < want.size(); i++)
            //{
            //    std::cout << "\n  --  " << want[i] << "\n";
            //}

            //std::cout << "\n\n" << want.size();
            nn.SLtrain(data, want);
            if (false)
            {
                std::cout << "\n\nError -> " << nn.totalError << "\n" << "Count iter. -> " << countIter << "\n" << "train file -> " << filename << "\n";
            }
            else
            {
                //Wait animation
                static std::string waitingSymbol = "|";
                if (countIter % 2 == 0)
                {
                    if (waitingSymbol == "|")
                    {
                        waitingSymbol = "/";
                    }
                    else if (waitingSymbol == "/")
                    {
                        waitingSymbol = "-";
                    }
                    else if (waitingSymbol == "-")
                    {
                        waitingSymbol = "\\";
                    }
                    else if (waitingSymbol == "\\")
                    {
                        waitingSymbol = "|";
                    }
                    std::cout << "\033c";
                    std::cout << "Training..." << waitingSymbol << "PLease wait..." << waitingSymbol << "[" << nn.totalError << "]\n";
                }
            }
            //std::vector<double> result = nn.forward(data);

            //for (int i = 0; i < 10; i++)
            //{
            //    std::cout << i << "----" << result[i] << "\n\n";
            //}


            //std::cin >> countIter;


            countIter++;
        }
        clock_t endClock = clock();
        secondToTrain = static_cast<double>(difftime(endClock, startClock) / 1000.0);

    }

    std::cout << "\nend\n" << "0\n";


    if (countIter != 0)
    {
        std::cout << "\n\nNumber of iterations: "<< countIter << "\nTraining time : " << secondToTrain << "sec. \n\n\n";
    }

    for (int k = 0; k < 100; k++)
    {

        data.clear();
        want.clear();

        std::string filename = folder + "mnist_" + std::to_string(generateRandomNumber(18001, 19999)) + ".txt";

        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "file open error: " << filename << std::endl;
            return 1;
        }

        std::string first_line;
        if (std::getline(file, first_line))
        {
            std::istringstream iss(first_line);
            int value;

            while (iss >> value)
            {
                want.push_back(value);
            }

        }
        else
        {
            std::cerr << "error" << std::endl;
            return 1;
        }

        std::string line;
        int row_count = 0;

        while (std::getline(file, line))
        {
            if (line.empty()) continue;

            std::istringstream iss(line);
            double pixel;

            while (iss >> pixel)
            {
                data.push_back(pixel);
            }

            row_count++;
        }

        std::cout << "     \n\n" << "File to test -> " << filename << "        \n\n" << "File data -> \n";


        std::vector<double> result = nn.forward(data);


        for (int i = 0; i < data.size(); i++)
        {
            if (i % 28 == 0)
            {
                std::cout << "\n";
            }

            if (data[i] > 0)
            {
                std::cout << '#';
            }
            else
            {
                std::cout << ' ';
            }
        }

        std::cout << "\n\n";


        for (int i = 0; i < 10; i++)
        {
            std::cout << i << "----" << result[i] * 100 << "%" << "\n";
        }
    }

    nn.trySaveWeight();

}