/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <iostream>
#include <evox/random.h>
#include <evox/Perceptron.h>

constexpr double MIN_WEIGHT = -0.5;
constexpr double MAX_WEIGHT = 0.5;
constexpr double LEAKY_TERM = 0.01;
using namespace std;


Perceptron::Perceptron(int num_inputs, int neuron_id)
{
    this->num_inputs = num_inputs;
    for (int i=0; i<num_inputs; ++i) {
        weights.push_back(random(MIN_WEIGHT, MAX_WEIGHT));
    }
    bias = random(MIN_WEIGHT, MAX_WEIGHT);
    weighed_deltas.resize(num_inputs);
    derivatives.resize(num_inputs);
    activation = 0.0;
    id = neuron_id;
    leaky_term = LEAKY_TERM;
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
        derivatives[i] = delta * inputs[i];
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
    this->num_inputs += n;
    for (int i=0; i<n; ++i) {
        if (delta > 0) {
            weights.push_back(random(-0.5, 0));
        } else {
            weights.push_back(random(0, 0.5));
        }
    }
    weighed_deltas.resize(num_inputs);
}


void Perceptron::removeInputs(int n)
{
    this->num_inputs -= n;
    weights.resize(this->num_inputs);
}
