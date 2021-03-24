/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <utility>
#include <evox/Network.h>


using namespace std;

const double DEFAULT_LEARNING_RATE = 0.01;
const double MEMORY_SIZE = 1000;
const double TARGET_ERROR = 0.001;


Network::Network(vector<Layer *> &layers)
{
    this->layers = layers;
    this->learning_rate = DEFAULT_LEARNING_RATE;
    this->in_sample_error = 1;
    this->out_of_sample_error = 1;
    max_output_values.resize(layers.back()->numNeurons());
    fill(max_output_values.begin(), max_output_values.end(), 1);
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

    storeSample(expected);

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
    if (sample_memory.size() > MEMORY_SIZE) {
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


void Network::reflect()
{
    updateInSampleError();
}


double Network::mse()
{
    double max=0, mse=0;

    for (auto sample: sample_memory) {
        if (sample.expected_outputs[0] > max) {
            max = sample.expected_outputs[0];
        }
    }

    for (auto sample: sample_memory) {
        mse += abs(feed(sample.inputs)[0] - sample.expected_outputs[0]) / max;
    }

    return mse / sample_memory.size();
}


void Network::updateOutOfSampleError(const vector<double> &output_error)
{
    double next_value = out_of_sample_error * (1 - learning_rate) + learning_rate * abs(output_error[0] / max_output_values[0]);

    out_of_sample_error_improvement_rate = (out_of_sample_error - next_value) / out_of_sample_error;
    out_of_sample_error = next_value;
}


void Network::updateInSampleError()
{
    double next_value = mse();

    in_sample_error_improvement_rate = (in_sample_error - next_value) / in_sample_error;
    in_sample_error = next_value;
}


void Network::evolve()
{
}
