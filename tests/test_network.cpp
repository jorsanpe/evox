/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <helpers/cpputest.h>

#include <cmath>
#include <iostream>
#include <evox/random.h>
#include <evox/Network.h>


using namespace std;
const double LEARNING_RATE = 0.01;
const double SAMPLES = 500;
const double EPOCHS = 500;


TEST_GROUP(Evox)
{
};


double cost(std::vector<double> predicted, std::vector<double> expected)
{
    double sum = 0;

    for (int i=0; i<predicted.size(); ++i) {
        sum += pow((expected[i] - predicted[i]), 2);
    }

    return sum / predicted.size();
}


TEST_WITH_MOCK(Evox, test_network)
{
    vector<Layer *> layers = {
        new Layer(1, 25),
        new Layer(25, 25),
        new Layer(25, 1)
    };
    Network network(layers);
    vector<vector<double>> inputs;
    vector<vector<double>> expected_outputs;
    std::vector<double> outputs;

    for (int i=0; i<SAMPLES; ++i) {
        double x = random(0, 2*3.1415926535);
        inputs.push_back(vector<double>{x});
        expected_outputs.push_back(vector<double>{sin(x)+1});
    }

    cout << "Learning a sine function ..." << endl;
    for (int epoch=0; epoch<500; epoch++) {
        for (int i=0; i<inputs.size(); ++i) {
            network.feed(inputs[i]);
            network.train(expected_outputs[i]);
        }
    }

    cout << "Generating predictions" << endl;
    for (int i=0; i<inputs.size(); ++i) {
        double x = random(0, 2*3.1415926535);
        outputs = network.feed(vector<double>{x});
        cout << x << "," << outputs[0] << endl;
    }

    for (auto layer: layers) {
        delete layer;
    }
}
