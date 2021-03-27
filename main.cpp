/*
 * Automatically created with CPM
 */
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
    return (sin(x) + 1) / 2;
}


void generatePredictions(Network &network, int epoch)
{
    vector<double> outputs;
    ofstream outfile;
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << epoch;

    outfile.open("files/epoch" + ss.str() + ".csv");

    outfile << "Input,Predicted,Expected" << endl;
    for (int i=0; i<BATCH_SIZE; ++i) {
        double x = random(0, 2*PI);
        outputs = network.feed(vector<double>{x});
        outfile << x << "," << outputs[0] << "," << realOutput(x) << endl;
    }

    outfile.close();
}


int main() 
{
    Network network(vector<Layer *> {
        new Layer(1, 2),
        new Layer(2, 2),
        new Layer(2, 2),
        new Layer(2, 2),
        new Layer(2, 1)
    });

    generatePredictions(network, 0);

    for (int epoch=1; epoch<1000; ++epoch) {
        for (int sample=0; sample<BATCH_SIZE; sample++) {
            double x = random(0, 2*PI);
            network.feed(vector<double>{x});
            network.train(vector<double>{realOutput(x)});
        }
        cout << "epoch " << epoch << endl;
        generatePredictions(network, epoch);
        network.reflect();
    }

    return 0;
}
