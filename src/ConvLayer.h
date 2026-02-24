#pragma once
#include <vector>
#include <random>
#include <iostream>


#include "baseCnnLayer.h"

class ConvLayer : public baseCnnLayer
{
public:

    ConvLayer(int _h, int _w, int _countFilters, int _countInputGhosts, int _coreSize, int _padding)
    {
        hGhostSize = _h;
        wGhostSize = _w;
        countFilters = _countFilters;
        padding = _padding;
        countInputGhosts = _countInputGhosts;
        filterSize = _coreSize;

        real_H_GhostSize = hGhostSize + 2 * padding;
        real_W_GhostSize = wGhostSize + 2 * padding;

        
        std::random_device device;
        std::mt19937 generator(device());
        std::uniform_real_distribution<double> distr(-0.3, 0.3);

        //init values tensor

        activatedOutputValues.resize(countFilters);
        notActivatedOutputValues.resize(countFilters);
        for (int l = 0; l < countFilters; l++)
        {
            activatedOutputValues[l].resize(real_H_GhostSize - filterSize + 1);
            notActivatedOutputValues[l].resize(real_H_GhostSize - filterSize + 1);
            for (int h = 0; h < real_H_GhostSize - filterSize + 1; h++)
            {
                activatedOutputValues[l][h].resize(real_W_GhostSize - filterSize + 1);
                notActivatedOutputValues[l][h].resize(real_W_GhostSize - filterSize + 1);
            }
        }


        saveInput.resize(countInputGhosts);
        for (int l = 0; l < countInputGhosts; l++)
        {
            saveInput[l].resize(hGhostSize);
            for (int h = 0; h < hGhostSize; h++)
            {
                saveInput[l][h].resize(wGhostSize);
            }
        }

        //init filters tensor


        filters.resize(countFilters);
        for (int l = 0; l < countFilters; l++)
        {
            filters[l].resize(countInputGhosts);
            for (int g = 0; g < countInputGhosts; g++)
            {
                filters[l][g].resize(filterSize);
                for (int h = 0; h < filterSize; h++)
                {
                    filters[l][g][h].resize(filterSize);
                    for (int w = 0; w < filterSize; w++)
                    {
                        filters[l][g][h][w] = distr(generator);
                    }
                }
            }
        }


        //output values

        std::cout << "\n\n========================================================\n";
        std::cout << "======================GHOSTS============================\n";
        std::cout << "========================================================\n\n";


        for (int l = 0; l < countInputGhosts; l++)
        {
            for (int h = 0; h < activatedOutputValues[l].size(); h++)
            {
                for (int w = 0; w < activatedOutputValues[l][h].size(); w++)
                {
                    std::cout << activatedOutputValues[l][h][w] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n\n";
        }


        //output filters values
        std::cout << "\n\n========================================================\n";
        std::cout << "======================FILTERS===========================\n";
        std::cout << "========================================================\n\n";

        for (int l = 0; l < countFilters; l++)
        {
            for (int g = 0; g < countInputGhosts; g++)
            {
                for (int h = 0; h < filterSize; h++)
                {
                    for (int w = 0; w < filterSize; w++)
                    {
                        std::cout << filters[l][g][h][w] << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n\n";
            }
        }

        std::cout << "\n\n========================================================\n";
        std::cout << "======================CONFIG============================\n";
        std::cout << "========================================================\n\n";
        
        std::cout << "ghost h -> " << hGhostSize << "\n";
        std::cout << "ghost w -> " << wGhostSize << "\n";
        std::cout << "read ghost h (with padding) -> " << real_H_GhostSize << "\n";
        std::cout << "read ghost w (with padding) -> " << real_W_GhostSize << "\n";
        std::cout << "count filters -> " << countFilters << "\n";
        std::cout << "count input ghosts -> "  << countInputGhosts << "\n";
        std::cout << "size of filters -> " << filterSize << "\n";
        std::cout << "padding -> " << padding << "\n";
    }


    std::vector<std::vector<std::vector<double>>> forward(std::vector<std::vector<std::vector<double>>> _ghostsTensor) override
    {
        //save inputs to barckward

        for (int l = 0; l < countInputGhosts; l++)
        {
            assignValuesVector(saveInput[l], _ghostsTensor[l], hGhostSize, wGhostSize);
        }


        //forward
        for (int l = 0; l < countFilters; l++)
        {

            std::vector<std::vector<double>> outputGhost(real_H_GhostSize - filterSize + 1, std::vector<double>(real_W_GhostSize - filterSize + 1, 0));
            for (int gt = 0; gt < countInputGhosts; gt++)
            {
                sumMatrices(outputGhost, applyFilter(applyPadding(_ghostsTensor[gt], padding, hGhostSize, wGhostSize), filters[l][gt]), real_H_GhostSize - filterSize + 1, real_W_GhostSize - filterSize + 1);
            }

            assignValuesVector(notActivatedOutputValues[l], outputGhost, real_H_GhostSize - filterSize + 1, real_W_GhostSize - filterSize + 1);
            applyRELU(outputGhost, real_H_GhostSize - filterSize + 1, real_W_GhostSize - filterSize + 1);
            assignValuesVector(activatedOutputValues[l], outputGhost, real_H_GhostSize - filterSize + 1, real_W_GhostSize - filterSize + 1);
        }

        //std::cout << "\n\n========================================================\n";
        //std::cout << "======================NOT_ACTIVATED=====================\n";
        //std::cout << "========================================================\n\n";

        //for (int l = 0; l < countFilters; l++)
        //{
        //    for (int h = 0; h < notActivatedOutputValues[l].size(); h++)
        //    {
        //        for (int w = 0; w < notActivatedOutputValues[l][h].size(); w++)
        //        {
        //            std::cout << notActivatedOutputValues[l][h][w] << " ";
        //        }
        //        std::cout << "\n";
        //    }
        //    std::cout << "\n\n";
        //}

        //std::cout << "\n\n========================================================\n";
        //std::cout << "======================ACTIVATED=========================\n";
        //std::cout << "========================================================\n\n";

        //for (int l = 0; l < countFilters; l++)
        //{
        //    for (int h = 0; h < activatedOutputValues[l].size(); h++)
        //    {
        //        for (int w = 0; w < activatedOutputValues[l][h].size(); w++)
        //        {
        //            std::cout << activatedOutputValues[l][h][w] << " ";
        //        }
        //        std::cout << "\n";
        //    }
        //    std::cout << "\n\n";
        //}

        return activatedOutputValues;
    }
    std::vector<std::vector<std::vector<double>>> backward(std::vector<std::vector<std::vector<double>>> _inputGrad, double _alpha) override
    {
        int hGhostOut = real_H_GhostSize - filterSize + 1;
        int wGhostOut = real_W_GhostSize - filterSize + 1;

        std::vector<std::vector<std::vector<double>>> gradBeforeActiv(countFilters, std::vector<std::vector<double>>(hGhostOut, std::vector<double>(wGhostOut, 0)));

        for (int f = 0; f < countFilters; f++)
        {
            for (int h = 0; h < hGhostOut; h++)
            {
                for (int w = 0; w < wGhostOut; w++)
                {
                    //derivative
                    gradBeforeActiv[f][h][w] = notActivatedOutputValues[f][h][w] > 0 ? _inputGrad[f][h][w] : 0.0;
                }
            }
        }


        //grad to update filters
        std::vector<std::vector<std::vector<std::vector<double>>>> gradForFilters = filters; //uuhhh...
        //gradForFilters.clear(); //uuhhh...

        for (int filter = 0; filter < countFilters; filter++)
        {
            for (int inputGhost = 0; inputGhost < countInputGhosts; inputGhost++)
            {
                for (int filtH = 0; filtH < filterSize; filtH++)
                {
                    for (int filtW = 0; filtW < filterSize; filtW++)
                    {
                        double gradTotalSum = 0;

                        for (int gOutH = 0; gOutH < hGhostOut; gOutH++)
                        {
                            for (int gOutW = 0; gOutW < wGhostOut; gOutW++)
                            {

                                //consider padding
                                int hPad = gOutH + filtH - padding;
                                int wPad = gOutW + filtW - padding;

                                //check if have gone beyond the limit of padding
                                if (hPad >= 0 && hPad < hGhostSize && wPad >= 0 && wPad < wGhostSize)
                                {
                                    gradTotalSum += saveInput[inputGhost][hPad][wPad] * gradBeforeActiv[filter][gOutH][gOutW];
                                }
                            }
                        }

                        gradForFilters[filter][inputGhost][filtH][filtW] = gradTotalSum;
                    }
                }
            }
        }


        std::vector<std::vector<std::vector<double>>> gradOutput(countInputGhosts, std::vector<std::vector<double>>(hGhostSize, std::vector<double>(wGhostSize, 0)));

        for (int ghost = 0; ghost < countInputGhosts; ghost++)
        {
            for (int inputH = 0; inputH < hGhostSize; inputH++)
            {
                for (int inputW = 0; inputW < wGhostSize; inputW++)
                {
                    double gradTotalSum = 0;


                    for (int filter = 0; filter < countFilters; filter++)
                    {
                        for (int gOutH = 0; gOutH < hGhostOut; gOutH++)
                        {
                            for (int gOutW = 0; gOutW < wGhostOut; gOutW++)
                            {
                                for (int filtH = 0; filtH < filterSize; filtH++)
                                {
                                    for (int filtW = 0; filtW < filterSize; filtW++)
                                    {
                                        //check if have gone beyond the limit of padding
                                        if (inputH == gOutH + filtH - padding && inputW == gOutW + filtW - padding)
                                        {
                                            gradTotalSum += filters[filter][ghost][filtH][filtW] * gradBeforeActiv[filter][gOutH][gOutW];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    gradOutput[ghost][inputH][inputW] = gradTotalSum;
                }
            }
        }

        for (int filter = 0; filter < countFilters; filter++)
        {
            for (int ghost = 0; ghost < countInputGhosts; ghost++)
            {
                for (int filterH = 0; filterH < filterSize; filterH++)
                {
                    for (int filterW = 0; filterW < filterSize; filterW++)
                    {
                        filters[filter][ghost][filterH][filterW] -= _alpha * gradForFilters[filter][ghost][filterH][filterW];
                    }
                }
            }
        }

        return gradOutput;

    }


    std::vector<std::vector<double>> applyFilter(std::vector<std::vector<double>> _ghost, std::vector<std::vector<double>> _filter)
    {
        std::vector<std::vector<double>> result;

        result.resize(real_H_GhostSize - filterSize + 1);

        for (int i = 0; i < real_H_GhostSize - filterSize + 1; i++)
        {
            result[i].resize(real_W_GhostSize - filterSize + 1);
        }

        for (int h = 0; h < real_H_GhostSize - filterSize + 1; h++)
        {
            for (int w = 0; w < real_W_GhostSize - filterSize + 1; w++)
            {
                double sum = 0;

                for (int i = 0; i < filterSize; i++)
                {
                    for (int k = 0; k < filterSize; k++)
                    {
                        //std::cout << "h " << h << " w " << w << " i " << i << " k " << k << " sum hi " << h + i << " sum wk " << w + k << "\n";
                        sum += _ghost[h + i][w + k] * _filter[i][k];
                    }
                }

                result[h][w] = sum;
            }
        }

        return result;
    }


    void sumMatrices(std::vector<std::vector<double>>& _main, std::vector<std::vector<double>> _plus, int _h, int _w)
    {
        for (int h = 0; h < _h; h++)
        {
            for (int w = 0; w < _w; w++)
            {
                _main[h][w] += _plus[h][w];
            }
        }
    }

    void assignValuesVector(std::vector<std::vector<double>>& _main, std::vector<std::vector<double>> _assign, int _h, int _w)
    {
        for (int h = 0; h < _h; h++)
        {
            for (int w = 0; w < _w; w++)
            {
                _main[h][w] = _assign[h][w];
            }
        }
    }

    std::vector<std::vector<double>> applyPadding(std::vector<std::vector<double>> _matrix, int _padding, int _h, int _w)
    {
        std::vector<std::vector<double>> result(_h + 2 * _padding, std::vector<double>(_w + 2 * _padding, 0));

        for (int h = _padding; h < _h + _padding; h++)
        {
            for (int w = _padding; w < _w + _padding; w++)
            {
                result[h][w] = _matrix[h - _padding][w - _padding];
            }
        }

        return result;
    }

    void applyRELU(std::vector<std::vector<double>>& _main, int _h, int _w)
    {
        for (int h = 0; h < _h; h++)
        {
            for (int w = 0; w < _w; w++)
            {
                _main[h][w] = NeuralNetwork::functionActivation(_main[h][w], RELU);
            }
        }
    }

    std::vector<std::vector<std::vector<double>>> return_nAOV()
    {
        return notActivatedOutputValues;
    }

    std::vector<std::vector<std::vector<double>>> return_AOV()
    {
        return activatedOutputValues;
    }


    //ConvLayer(int _h, int _w, int _countFilters, int _countInputGhosts, int _coreSize, int _padding)
    std::vector<int> returnConfig() override
    {
        std::vector<int> a = { hGhostSize , wGhostSize, countFilters, countInputGhosts, filterSize, padding};
        return a;
    }
    


private:

    int hGhostSize;
    int wGhostSize;
    int real_H_GhostSize;
    int real_W_GhostSize;
    int countFilters;
    int countInputGhosts;
    int filterSize;
    int padding;
 


    std::vector<std::vector<std::vector<std::vector<double>>>> filters; //or cores or kernels
    std::vector<std::vector<std::vector<double>>> saveInput;
    std::vector<std::vector<std::vector<double>>> notActivatedOutputValues; //extra data to train
    std::vector<std::vector<std::vector<double>>> activatedOutputValues; //extra data to train and forward


    //just auxiliary function for "forward" func


};
