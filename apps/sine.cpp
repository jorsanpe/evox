/*
 * Automatically created with CPM
 */
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <evox/Network.h>
#include <evox/random.h>

using namespace std;

static const double BATCH_SIZE = 1000;
static const double PI = 3.1415926535897932;


double realOutput(double x)
{
    return sin(x) + sin(x*5)/2 + sin(x*10)/2 + 4;
}


double realOutputWithNoise(double x)
{
    return realOutput(x) + random(-0.2, 0.2);
}


void generatePredictions(Network &network, int epoch)
{
    vector<double> outputs;
    ofstream outfile;
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << epoch;

    outfile.open("files/epoch" + ss.str() + ".csv");

    outfile << "Input,Expected,Predicted,Error" << endl;
    for (int i=0; i<BATCH_SIZE; ++i) {
        double x = random(0, 2*PI);
        outputs = network.feed(vector<double>{x});
        outfile << x << "," << realOutput(x) << "," << outputs[0] << "," << outputs[0]-realOutput(x) << endl;
    }

    outfile.close();
}


void sine()
{
    vector<Layer *> layers = {
            new Layer(2, 2),
            new Layer(2, 2),
            new Layer(2, 1)
    };
    Network network(layers);

    generatePredictions(network, 0);

    for (int batch=1; batch<500; ++batch) {
        for (int sample=0; sample<BATCH_SIZE; sample++) {
            double x = random(0, 2*PI);
            vector<double> input = {x};
            vector<double> expected = {realOutputWithNoise(x)};
            network.feed(input);
            network.train(expected);
        }
        cout << "batch " << batch << endl;
        generatePredictions(network, batch);
        network.reflect();
    }
}
