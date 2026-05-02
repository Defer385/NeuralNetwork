#pragma once

#include "NeuralNetwork.h"
#include <algorithm>

//struct  Status
//{
//    std::vector<double> state;
//    std::vector<double> nextState;
//    int action;
//    double reward;
//    bool done;
//};


class Agent
{
public:

    Agent(NeuralNetwork& _actor,NeuralNetwork& _critic, double _gamma) : actor(_actor), critic(_critic)
    {
        gamma = _gamma;
    }

    int argmax(Tensor _state)
    {
        Tensor result = actor.forward(_state);

        return std::distance(result.dat().begin(), std::max_element(result.dat().begin(), result.dat().end()));
    }

    int act(std::vector<double> _state)
    {
	Tensor state({ (int)_state.size() });

        Tensor netResult = actor.forward(state);

        double r = (double)rand() / RAND_MAX;

        double cumulative = 0.0;

        for (int i = 0; i < netResult.size(); i++)
        {
            cumulative += netResult.readOneD(i);

            if (r <= cumulative)
                return i;
        }

        return netResult.size() - 1;
    }

    //void store(std::vector<double> _state, std::vector<double> _next_state, int _action, double _reward)
    //{
    //    Status thisStatus;
    //    thisStatus.state = _state;
    //    thisStatus.nextState = _next_state;
    //    thisStatus.action = _action;
    //    thisStatus.reward = _reward;

    //    episodes.push_back(thisStatus);
    //}

    void train(std::vector<double> _state, std::vector<double> _next_state, int action, double reward, bool done)
    {
        int stateSize = _state.size();

        Tensor state({ stateSize });
        Tensor nextState({ stateSize });

        state.insertVector(_state);
        nextState.insertVector(_next_state);

        double V = critic.forward(state).readOneD(0);

        double V_next = 0.0;

        if (!done)
        {
            V_next = critic.forward(nextState).readOneD(0);
        }

        double target = done ? reward : reward + gamma * V_next;

        double avg = target - V;

        Tensor targetTensor({ 1 });
        targetTensor.insertVector({ target });

        critic.SLtrain(state, targetTensor);

        actor.RLtrain(state, action, avg);

        //std::cout << "AVG -> " << avg << "\r";
    }


private:

NeuralNetwork& actor;
NeuralNetwork& critic;
//std::vector<Status> episodes;
double gamma;

};