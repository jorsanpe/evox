/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <iostream>
#include <evox/Layer.h>

using namespace std;

Layer::Layer(int num_inputs, int num_neurons)
{
    this->num_inputs = num_inputs;
    for (int i=0; i<num_neurons; ++i) {
        neurons.push_back(new Perceptron(num_inputs, i+1));
    }
    deltas.resize(num_inputs);
    outputs.resize(neurons.size());
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


int Layer::numNeurons()
{
    return neurons.size();
}


void Layer::addNeuron()
{
    neurons.push_back(new Perceptron(num_inputs, outputs.size()+1));
    outputs.resize(neurons.size());
}


void Layer::addInput()
{
    num_inputs += 1;
    for (auto neuron: neurons) {
        neuron->addInputs(1);
    }
    deltas.resize(num_inputs);
}
