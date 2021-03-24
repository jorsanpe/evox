/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#pragma once

#include <list>
#include <vector>
#include <evox/Layer.h>


struct Sample {
    std::vector<double> inputs;
    std::vector<double> expected_outputs;
};


class Network {
public:
    double in_sample_error;
    double in_sample_error_improvement_rate;
    double out_of_sample_error;
    double out_of_sample_error_improvement_rate;

    Network(std::vector<Layer *> &layers);

    std::vector<double> feed(std::vector<double> inputs);

    void train(const std::vector<double>& expected);

    void reflect();

    void evolve();

private:
    std::vector<Layer *> layers;
    std::vector<double> last_input;
    std::vector<double> last_prediction;
    std::vector<double> max_output_values;
    double learning_rate;
    std::list<Sample> sample_memory;

    void backpropagate(const std::vector<double>& errors);

    std::vector<double> errors(std::vector<double> expected);

    double mse();

    void storeSample(const std::vector<double> &expected);

    void updateInSampleError();

    void updateOutOfSampleError(const std::vector<double> &output_error);
};
