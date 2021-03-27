/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#pragma once

#include <vector>
#include <evox/Perceptron.h>


class Layer {
public:
    Layer(int num_inputs, int num_neurons);

    std::vector<double> &feed(const std::vector<double> &inputs);

    virtual std::vector<double> &backpropagate(const std::vector<double> &deltas);

    void train(double learning_rate);

    void addNeuron();

    void addInput();

    int numNeurons();

    ~Layer();

protected:
    int num_inputs;
    std::vector<Perceptron *> neurons;
    std::vector<double> last_inputs;
    std::vector<double> outputs;
    std::vector<double> deltas;
};
