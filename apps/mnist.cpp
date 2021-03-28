/*
 * Automatically created with CPM
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <evox/Network.h>
#include <evox/MnistReader.h>

using namespace std;

static const int BATCH_SIZE = 1000;


void mnist() {
    Network network(vector<Layer *>{
            new Layer(28 * 28, 2),
            new Layer(2, 2),
            new Layer(2, 2),
            new Layer(2, 2),
            new Layer(2, 10)
    });
    MnistReader reader("mnist/train.csv");
    reader.nextBatch(1);
    auto inputs = reader.nextBatch(42000);
    auto expected = reader.outputs();

    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (int batch=0; batch<42; ++batch) {
            for (int i = 0; i < BATCH_SIZE; i++) {
                network.feed(inputs[i + BATCH_SIZE * batch]);
                network.train(expected[i + BATCH_SIZE * batch]);
            }

            cout << "batch " << batch << endl;
            network.reflect();
        }
    }
}
