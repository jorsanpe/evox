/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <iostream>
#include <evox/random.h>
#include <evox/Network.h>
#include <cmath>


using namespace std;

const int DEFAULT_MEMORY_SIZE = 1000;
const int DEFAULT_REFLECTION_ITERATIONS = 100;
const double DEFAULT_MISTAKE_DEVIATION = 0.05;
const double DEFAULT_LEARNING_RATE = 0.01;
const double DEFAULT_TARGET_ERROR = 1;
const double DEFAULT_STAGNATION_RATE = 0.5;


Network::Network(const vector<Layer *> &layers)
{
    // Meta-learning: these parameters could be learned as well. Also, they could be dynamic
    this->layers = layers;
    this->learning_rate = DEFAULT_LEARNING_RATE;
    this->in_sample_error = 1;
    this->out_of_sample_error = 1;
    this->target_error = DEFAULT_TARGET_ERROR;
    this->memory_size = DEFAULT_MEMORY_SIZE;
    this->reflection_iterations = DEFAULT_REFLECTION_ITERATIONS;
    this->reflecting = false;
    this->stagnation_rate = DEFAULT_STAGNATION_RATE;
    this->best_in_sample_error = 1000000;
    max_output_values.resize(layers.back()->numNeurons(), 1);
}


vector<double> Network::feed(const vector<double>& inputs)
{
    last_input = inputs;
    last_prediction = inputs;
    for (auto layer: layers) {
        last_prediction = layer->feed(last_prediction);
    }

    return last_prediction;
}


void Network::train(const vector<double>& expected)
{
    vector<double> output_error;

    for (int i=0; i<expected.size(); ++i) {
        if (expected[i] > max_output_values[i]) {
            max_output_values[i] = expected[i];
        }
    }

    output_error = errors(expected);

    if (!reflecting) {
        updateOutOfSampleError(output_error);
        if (last_out_of_sample_error > DEFAULT_MISTAKE_DEVIATION) {
            storeSample(expected);
        }
    }

    backpropagate(output_error);

    for (auto layer: layers) {
        layer->train(learning_rate);
    }
}

void Network::storeSample(const vector<double> &expected)
{
    sample_memory.push_back(
        Sample{last_input, expected}
    );
    if (sample_memory.size() > memory_size) {
        sample_memory.pop_front();
    }
}


void Network::backpropagate(const vector<double>& errors)
{
    vector<double> weighed_deltas = errors;

    for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer ) {
        weighed_deltas = (*layer)->backpropagate(weighed_deltas);
    }
}


vector<double> Network::errors(vector<double> expected)
{
    vector<double> diff(last_prediction.size());

    for (int i=0; i < last_prediction.size(); ++i) {
        diff[i] = last_prediction[i] - expected[i];
    }

    return diff;
}


void Network::updateOutOfSampleError(const vector<double> &output_error)
{
    double next_value = 0;

    for (int i=0; i<output_error.size(); ++i) {
        next_value += pow(output_error[i], 2); //abs(output_error[i] / max_output_values[i]);
    }
    last_out_of_sample_error = next_value;

    next_value = smooth(out_of_sample_error, next_value);

    out_of_sample_error_improvement_rate = (out_of_sample_error - next_value) / out_of_sample_error;
    out_of_sample_error = next_value;
}


void Network::reflect()
{
    double mse;
    vector<double> predicted;

    reflecting = true;
    in_sample_error_improvement_rate = 1;
    for (int i=0; i<reflection_iterations&&!stoppedLearning(); ++i) {
        mse = 0;

        for (const auto &sample: sample_memory) {
            predicted = feed(sample.inputs);
            train(sample.expected_outputs);

            for (int j=0; j<predicted.size(); ++j) {
                mse += (predicted[j] - sample.expected_outputs[j]) * (predicted[j] - sample.expected_outputs[j]);
            }
        }

        updateInSampleError(mse);
    }

    reflecting = false;

    cout << "" << in_sample_error;
    cout << ", ( " << in_sample_error_improvement_rate << " )";
    cout << ", " << out_of_sample_error;
    cout << ", ( " << out_of_sample_error_improvement_rate << " )" << endl;

    if (best_in_sample_error < (in_sample_error+stagnation_rate) && modelTooSimple()) {
        evolve();
    }

    if (in_sample_error < best_in_sample_error) {
        best_in_sample_error = in_sample_error;
    }
}


bool Network::stoppedLearning()
{
    return in_sample_error_improvement_rate >= 0 && in_sample_error_improvement_rate < stagnation_rate;
}


void Network::updateInSampleError(double mse)
{
    in_sample_error_improvement_rate = in_sample_error - mse;
    in_sample_error = smooth(in_sample_error, mse);
}


double Network::meanSquareError()
{
    double average_error=0;
    vector<double> predicted;

    for (auto sample: sample_memory) {
        predicted = feed(sample.inputs);
        for (int i=0; i<predicted.size(); ++i) {
            average_error += pow(predicted[i] - sample.expected_outputs[i], 2);
        }
    }

    return average_error / sample_memory.size();
}


bool Network::modelTooSimple()
{
    return in_sample_error > target_error;
}


bool Network::modelTooComplex()
{
    return in_sample_error < target_error && out_of_sample_error > 10 * in_sample_error;
}


void Network::simplify()
{
    cout << "simplifying" << endl;
}


void Network::evolve()
{
    int index;
    auto layer = layers.begin();

    index = (int)random(0, layers.size()-1);
    advance(layer, index);

    cout
        << "stagnant (" << in_sample_error_improvement_rate << " at " << stagnation_rate << "),"
        << " add neuron to layer " << index << " " << " (" << (*layer)->numNeurons() << ")" << endl;

    (*layer)->addNeuron();
    (*next(layer))->addInput();
}


double Network::smooth(double previous, double next)
{
    return previous * (1-learning_rate) + next * learning_rate;
}
