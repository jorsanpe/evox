/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <evox/Layer.h>

using namespace std;

Layer::Layer(int num_inputs, int num_neurons)
{
    for (int i=0; i<num_neurons; ++i) {
        neurons.push_back(new Perceptron(num_inputs, i+1));
    }
    deltas.resize(num_inputs);
    outputs.resize(num_neurons);
}


Layer::~Layer()
{
    for (auto neuron: neurons) {
        delete neuron;
    }
}


vector<double> &Layer::feed(const vector<double> &inputs)
{
    this->last_inputs = inputs;

    for (int i=0; i<neurons.size(); ++i) {
        outputs[i] = neurons[i]->feed(inputs);
    }

    return outputs;
}


vector<double> &Layer::backpropagate(const vector<double> &wdeltas)
{
    vector<double> backpropagation;

    fill(deltas.begin(), deltas.end(), 0);

    for (int i=0; i<neurons.size(); ++i) {
        backpropagation = neurons[i]->backpropagate(wdeltas[i]);
        for (int j=0; j<backpropagation.size(); ++j) {
            deltas[j] += backpropagation[j];
        }
    }

    return deltas;
}


void Layer::train(double learning_rate)
{
    for (auto neuron: neurons) {
        neuron->train(learning_rate, this->last_inputs);
    }
}
