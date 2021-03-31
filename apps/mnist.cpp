/*
 * Automatically created with CPM
 */
#include <iostream>
#include <evox/Network.h>
#include <evox/MnistReader.h>

using namespace std;

static const int BATCH_SIZE = 1000;


static int outputToNumber(const vector<double>& output)
{
    double max = 0.0;
    int number = 0;

    for (int i=0; i<output.size(); ++i) {
        if (output[i] > max) {
            max = output[i];
            number = i;
        }
    }

    return number;
}


static void trainNetwork(Network &network)
{
    MnistReader train_reader("mnist/train.csv");
    train_reader.nextBatch(1);
    auto inputs = train_reader.nextBatch(42000);
    auto expected = train_reader.outputs();

    for (int batch = 0; batch < 42; ++batch) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            network.feed(inputs[i + BATCH_SIZE * batch]);
            network.train(expected[i + BATCH_SIZE * batch]);
        }
        network.reflect();
        cout
            << "batch " << batch
            << " (" << network.in_sample_error << "," << network.out_of_sample_error << ")"
            << endl;
    }
}


static void testNetwork(Network &network)
{
    int correct=0;
    MnistReader test_reader("mnist/train.csv");
    test_reader.nextBatch(1);
    auto inputs = test_reader.nextBatch(42000);
    auto expected = test_reader.outputs();

    cout << "Testing network" << endl;

    for (int i=0; i<inputs.size(); ++i) {
        auto prediction = network.feed(inputs[i]);
        if (outputToNumber(prediction) == outputToNumber(expected[i])) {
            correct++;
        }
    }

    cout << "got " << correct << " from 42000 (" << (double)correct/42000 << ")" << endl;
}


static void generateSubmission(Network &network)
{
    ofstream outfile;
    MnistReader test_reader("mnist/test.csv");
    test_reader.nextBatch(1);
    auto inputs = test_reader.nextBatch(28000);
    auto expected = test_reader.outputs();

    outfile.open("mnist/submission.csv");
    outfile << "ImageId,Label" << endl;

    for (int i=0; i<inputs.size(); ++i) {
        auto prediction = network.feed(inputs[i]);

        outfile << i+1 << "," << outputToNumber(prediction) << endl;
    }

    outfile.close();
}


void mnist()
{
    Network network(vector<Layer *>{
            new Layer(28 * 28, 10),
            new Layer(10, 10),
            new Layer(10, 10),
            new Layer(10, 10),
            new Layer(10, 10)
    });

    trainNetwork(network);

    testNetwork(network);

    generateSubmission(network);
}
