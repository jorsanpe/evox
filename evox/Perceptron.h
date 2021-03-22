/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#pragma once

#include <vector>

class Perceptron {
public:
    Perceptron(int num_inputs, int neuron_id);

    double feed(const std::vector<double> &inputs);

    std::vector<double> &backpropagate(const double error);

    void train(double learning_rate, const std::vector<double>& inputs);

    void addInputs(int n);

    void removeInputs(int n);

private:
    int id;
    int num_inputs;
    std::vector<double> weights;
    double bias;
    double activation;
    double delta;
    std::vector<double> weighed_deltas;
    double leaky_term;

    double relu(double value);

    double reluDerivative(double value);
};

