/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <utility>
#include <evox/Network.h>


using namespace std;

const double DEFAULT_LEARNING_RATE = 0.01;


Network::Network(vector<Layer *> &layers)
{
    this->layers = layers;
    this->learning_rate = DEFAULT_LEARNING_RATE;
}


vector<double> Network::feed(vector<double> inputs)
{
    predicted = move(inputs);
    for (auto layer: layers) {
        predicted = layer->feed(predicted);
    }

    return predicted;
}


void Network::backpropagate(const vector<double>& errors)
{
    vector<double> weighed_deltas = errors;

    for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer ) {
        weighed_deltas = (*layer)->backpropagate(weighed_deltas);
    }
}


void Network::train(const vector<double>& expected)
{
    backpropagate(errors(expected));

    for (auto layer: layers) {
        layer->train(learning_rate);
    }
}


vector<double> Network::errors(vector<double> expected)
{
    vector<double> diff(predicted.size());

    for (int i=0; i<predicted.size(); ++i) {
        diff[i] = predicted[i] - expected[i];
    }

    return diff;
}
