#pragma once
#include <vector>
#include <random>


//#define TENSOR_SAVE_MODE




class Tensor
{
public:

    explicit Tensor(std::vector<int> _shape) : shape(_shape)
    {   
        /*if (_shape.size() <= 0 || _shape.size() > 4)
        {
            std::invalid_argument("measurement dimension error");
        }*/

        int Size = 1;

        for (int i = 0; i < _shape.size(); i++)
        {
            Size *= shape[i];
        }

        data.resize(Size);
        totalSize = Size;
    }

    //READ

    double readOneD(int i)
    {
        #ifdef TENSOR_SAVE_MODE
        if (i > totalSize)
        {
            throw std::runtime_error("An attempt to go abroad");
        }
        #endif


        return data[i];
    }
    
    double readTwoD(int i, int k)
    {
        #ifdef TENSOR_SAVE_MODE
        if (i * shape[1] + k > totalSize)
        {
            throw std::runtime_error("An attempt to go abroad");
        }
        #endif


        return data[i * shape[1] + k];
    }
    
    double readThreeD(int i, int k, int j)
    {
        #ifdef TENSOR_SAVE_MODE
        if (i * shape[1] * shape[2] + k * shape[2] + j > totalSize)
        {
            throw std::runtime_error("An attempt to go abroad");
        }
        #endif
        return data[i * shape[1] * shape[2] + k * shape[2] + j];
    }
    
    double readFourD(int i, int k, int j, int m)
    {
        #ifdef TENSOR_SAVE_MODE
        if (i * shape[1] * shape[2] * shape[3] + k * shape[2] * shape[3] + j * shape[3] + m > totalSize)
        {
            throw std::runtime_error("An attempt to go abroad");
        }
        #endif

        return data[i * shape[1] * shape[2] * shape[3] + k * shape[2] * shape[3] + j * shape[3] + m];
    }
    //
    //WRITE
    //
    void writeOneD(double _value, int i)
    {
        #ifdef TENSOR_SAVE_MODE
        if (i > totalSize)
        {
            throw std::runtime_error("An attempt to go abroad");
        }
        #endif

        data[i] = _value;
    }
    
    void writeTwoD(double _value, int i, int k)
    {
        #ifdef TENSOR_SAVE_MODE
        if (i * shape[1] + k > totalSize)
        {
            throw std::runtime_error("An attempt to go abroad");
        }
        #endif

        data[i * shape[1] + k] = _value;
    }
    
    void writeThreeD(double _value, int i, int k, int j)
    {
        #ifdef TENSOR_SAVE_MODE
        if (i * shape[1] * shape[2] + k * shape[2] + j > totalSize)
        {
            throw std::runtime_error("An attempt to go abroad");
        }
        #endif

        data[i * shape[1] * shape[2] + k * shape[2] + j] = _value;
    }
    
    void writeFourD(double _value, int i, int k, int j, int m)
    {
        #ifdef TENSOR_SAVE_MODE
        if (i * shape[1] * shape[2] * shape[3] + k * shape[2] * shape[3] + j * shape[3] + m > totalSize)
        {
            throw std::runtime_error("An attempt to go abroad");
        }
        #endif

        data[i * shape[1] * shape[2] * shape[3] + k * shape[2] * shape[3] + j * shape[3] + m] = _value;
    }

    //EXTRA

    void reshape(std::vector<int> _shape)
    {
        shape = _shape;

        int Size = 1;

        for (int i = 0; i < _shape.size(); i++)
        {
            Size *= shape[i];
        }

        data.resize(Size);
        totalSize = Size;
    }

    void fillRandom(double _min, double _max)
    {
        std::random_device device;
        std::mt19937 generator(device());
        std::uniform_real_distribution<double> distr(_min, _max);

        for (int i = 0; i < totalSize; i++)
        {
            data[i] = distr(generator);
        }
    }

    void insertVector(std::vector<double> _input)
    {
        #ifdef TENSOR_SAVE_MODE
        if (_input.size() != data.size())
        {
            throw std::runtime_error("Wrong size inserted vector");
        }
        #endif 

        data = std::move(_input);

    }

    std::vector<double>& returnData()
    {
        return data;
    }

    std::vector<int> returnConfig()
    {   
        return shape;
    }

    int size()
    {
        return totalSize;
    }

    std::vector<double>& dat()
    {
        return data;
    }

    void clear()
    {
        std::fill(data.begin(), data.end(), 0);
    }

private:

    std::vector<double> data;
    std::vector<int> shape;
    int totalSize;
};
