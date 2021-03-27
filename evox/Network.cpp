/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <algorithm>
#include <iostream>
#include <utility>
#include <evox/random.h>
#include <evox/Network.h>


using namespace std;

const int DEFAULT_MEMORY_SIZE = 1000;
const int DEFAULT_REFLECTION_ITERATIONS = 10;
const double DEFAULT_LEARNING_RATE = 0.01;
const double DEFAULT_TARGET_ERROR = 0.005;
const double DEFAULT_STAGNATION_RATE = 0.001;


Network::Network(const vector<Layer *> &layers)
{
    this->layers = layers;
    this->learning_rate = DEFAULT_LEARNING_RATE;
    this->in_sample_error = 1;
    this->out_of_sample_error = 1;
    this->target_error = DEFAULT_TARGET_ERROR;
    this->memory_size = DEFAULT_MEMORY_SIZE;
    this->reflection_iterations = DEFAULT_REFLECTION_ITERATIONS;
    this->reflecting = false;
    this->stagnation_rate = DEFAULT_STAGNATION_RATE;
    max_output_values.resize(layers.back()->numNeurons(), 1);
}


vector<double> Network::feed(vector<double> inputs)
{
    last_input = inputs;

    last_prediction = move(inputs);
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

    if (!reflecting) {
        storeSample(expected);
    }

    output_error = errors(expected);

    updateOutOfSampleError(output_error);

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
    double next_value = smooth(out_of_sample_error, abs(output_error[0] / max_output_values[0]));

    out_of_sample_error_improvement_rate = (out_of_sample_error - next_value) / out_of_sample_error;
    out_of_sample_error = next_value;
}


void Network::reflect()
{
    reflecting = true;
    for (int i=0; i<reflection_iterations; ++i) {
        for (const auto &sample: sample_memory) {
            feed(sample.inputs);
            train(sample.expected_outputs);
        }
        updateInSampleError();
    }
    reflecting = false;

    cout << "in_sample_error: " << in_sample_error << endl;
    cout << "in_sample_error_improvement_rate: " << in_sample_error_improvement_rate << endl;
    cout << "out_of_sample_error: " << out_of_sample_error << endl;
    cout << "out_of_sample_error_improvement_rate: " << out_of_sample_error_improvement_rate << endl;

    if (stoppedLearning()) {
        if (modelTooSimple()) {
            evolve();
        }
        if (modelTooComplex()) {
            simplify();
        }
    }
}


void Network::updateInSampleError()
{
    double next_value = averageInSampleError();

    in_sample_error_improvement_rate = smooth(
            in_sample_error_improvement_rate,
            max(0.0, (in_sample_error - next_value)));
    in_sample_error = smooth(in_sample_error, next_value);
}


double Network::averageInSampleError()
{
    double average_error=0;

    for (auto sample: sample_memory) {
        average_error += abs(feed(sample.inputs)[0] - sample.expected_outputs[0]) / max_output_values[0];
    }

    return average_error / sample_memory.size();
}


bool Network::stoppedLearning()
{
    return in_sample_error_improvement_rate < stagnation_rate;
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

    cout << "evolving, add neuron to layer " << index << " " << (*layer) << endl;

    (*layer)->addNeuron();
    (*next(layer))->addInput();
}


double Network::smooth(double previous, double next)
{
    return previous * (1-learning_rate) + next * learning_rate;
}
