#include "NeuralNetwork.h"
#include "Dense.h"

int main()
{
    std::srand(std::time(NULL));
    
    NeuralNetwork net;

    net.add(new Dense(2,2,ActivationsType::SIGMOID));
    net.add(new Dense(2, 1, ActivationsType::SIGMOID));
    net.init(0.1);

    const int ITER_BEFORE_OUTPUT = 5000;
    int iter = 0;

    while (true)
    {
        double one = std::rand() % 2;
        double two = std::rand() % 2;

        double tar = 0;

        if ((one == 1) != (two == 1))
        {
            tar = 1;
        }

        Tensor data({ 2 });
        Tensor want({ 1 });

        data.insertVector({ one,two });
        want.insertVector({ tar });

        double pred = net.forward(data).readOneD(0);
        net.SLtrain(data, want);

        if (iter == ITER_BEFORE_OUTPUT)
        {
            iter = 0;

            std::cout <<
                "ONE - >" << one <<
                "\nTWO -> " << two <<
                "\nReal XOR -> " << tar <<
                "\nNet prediction -> " << pred <<
                "\nIts true? " << (std::abs(pred - tar) < 0.1 ? "YEEEEEEEEEEEEE\n\n" : "NOOOOOOOOOOOOOOOOOOO\n\n");
        }
        
        iter++;
    }

}




