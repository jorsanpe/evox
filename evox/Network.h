/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#pragma once

#include <vector>
#include <evox/Layer.h>


class Network {
public:
    Network(std::vector<Layer *> &layers);

    std::vector<double> feed(std::vector<double> inputs);

    void train(const std::vector<double>& expected);

private:
    std::vector<Layer *> layers;
    std::vector<double> predicted;
    double learning_rate;

    void backpropagate(const std::vector<double>& errors);

    std::vector<double> errors(std::vector<double> expected);
};
