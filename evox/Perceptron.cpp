/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <evox/random.h>
#include <evox/Perceptron.h>

constexpr int MIN_WEIGHT = -1;
constexpr int MAX_WEIGHT = 1;
using namespace std;


Perceptron::Perceptron(int num_inputs, int neuron_id)
{
    this->num_inputs = num_inputs;
    weights.resize(num_inputs);
    weighed_deltas.resize(num_inputs);
    for (int i=0; i<num_inputs; ++i) {
        weights[i] = random(MIN_WEIGHT, MAX_WEIGHT);
    }
    bias = random(MIN_WEIGHT, MAX_WEIGHT);
    activation = 0.0;
    id = neuron_id;
    leaky_term = 0.01;
}


double Perceptron::feed(const std::vector<double> &inputs)
{
    double z=bias;

    for (int input=0; input<inputs.size(); ++input) {
        z += inputs[input] * weights[input];
    }
    activation = relu(z);

    return activation;
}


std::vector<double> &Perceptron::backpropagate(const double error)
{
    delta = error * reluDerivative(activation);
    for (int i=0; i<weights.size(); ++i) {
        weighed_deltas[i] = weights[i] * delta;
    }
    return this->weighed_deltas;
}


void Perceptron::train(double learning_rate, const vector<double> &inputs)
{
    for (int i=0; i<weights.size(); ++i) {
        weights[i] -= delta * inputs[i] * learning_rate;
    }
    bias -= learning_rate * delta;
}


double Perceptron::reluDerivative(double value)
{
    return value > 0 ? 1 : leaky_term;
}


double Perceptron::relu(double value)
{
    return value < 0 ? value*leaky_term : value;
}


void Perceptron::addInputs(int n)
{
    for (int i=0; i<n; ++i) {
        weights.push_back(random(-0.1, 0.1));
    }
    this->num_inputs += n;
}


void Perceptron::removeInputs(int n)
{
    this->num_inputs -= n;
    weights.resize(this->num_inputs);
}
