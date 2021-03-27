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
    double in_sample_error_improvement_rate{};
    double out_of_sample_error;
    double out_of_sample_error_improvement_rate{};

    Network(const std::vector<Layer *> &layers);

    std::vector<double> feed(std::vector<double> inputs);

    void train(const std::vector<double>& expected);

    void reflect();

private:
    std::vector<Layer *> layers;
    std::vector<double> last_input;
    std::vector<double> last_prediction;
    std::vector<double> max_output_values;
    std::list<Sample> sample_memory;
    double learning_rate;
    double target_error;
    double stagnation_rate;
    bool reflecting;
    int reflection_iterations;
    int memory_size;

    void backpropagate(const std::vector<double>& errors);

    std::vector<double> errors(std::vector<double> expected);

    double averageInSampleError();

    double smooth(double previous, double next);

    void storeSample(const std::vector<double> &expected);

    void updateInSampleError();

    void updateOutOfSampleError(const std::vector<double> &output_error);

    bool stoppedLearning();

    bool modelTooSimple();

    bool modelTooComplex();

    void evolve();

    void simplify();
};

