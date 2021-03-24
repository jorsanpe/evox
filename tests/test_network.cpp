/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <helpers/cpputest.h>

#include <cmath>
#include <iostream>
#include <evox/random.h>
#include <evox/Network.h>


using namespace std;
const double SAMPLES = 500;
const double EPOCHS = 1000;
const double PI = 3.1415926535897932;


TEST_GROUP(Network)
{
};


TEST_WITH_MOCK(Network, test_offline_learning)
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
        double x = random(0, 2*PI);
        inputs.push_back(vector<double>{x});
        expected_outputs.push_back(vector<double>{sin(x)+1});
    }

    cout << "Learning a sine function ..." << endl;
    for (int epoch=0; epoch<EPOCHS; epoch++) {
        for (int i=0; i<inputs.size(); ++i) {
            network.feed(inputs[i]);
            network.train(expected_outputs[i]);
        }
    }

    cout << "Generating predictions" << endl;
    for (int i=0; i<inputs.size(); ++i) {
        double x = random(0, 2*PI);
        outputs = network.feed(vector<double>{x});
        cout << x << "," << outputs[0] << endl;
    }

    for (auto layer: layers) {
        delete layer;
    }
}


TEST_WITH_MOCK(Network, test_online_learning)
{
    vector<Layer *> layers = {
        new Layer(1, 3),
        new Layer(3, 3),
        new Layer(3, 1)
    };
    Network network(layers);
    vector<vector<double>> inputs;
    vector<vector<double>> expected_outputs;
    std::vector<double> outputs;

    for (int i=0; i<SAMPLES; ++i) {
        double x = random(0, 2*PI);
        inputs.push_back(vector<double>{x});
        expected_outputs.push_back(vector<double>{sin(x)+1});
    }

    cout << "Evolving a network to learn a sine function ..." << endl;
    for (int epoch=0; epoch<EPOCHS; epoch++) {
        for (int i=0; i<inputs.size(); ++i) {
            network.feed(inputs[i]);
            network.train(expected_outputs[i]);
        }
        network.reflect();

        cout
            << network.in_sample_error << "   "
            << network.out_of_sample_error
            << endl;
    }

    cout << "Generating predictions" << endl;
    for (int i=0; i<inputs.size(); ++i) {
        double x = random(0, 2*PI);
        outputs = network.feed(vector<double>{x});
        cout << x << "," << outputs[0] << endl;
    }

    for (auto layer: layers) {
        delete layer;
    }
}
