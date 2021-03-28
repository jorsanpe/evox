/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <iostream>
#include <iosfwd>
#include <sstream>
#include <evox/MnistReader.h>

using namespace std;


MnistReader::MnistReader(std::string filename)
{
    this->filename = filename;
    fin.open(filename, ios::in);
}


std::vector<std::vector<double>> &MnistReader::nextBatch(int num_samples)
{
    vector<string> row;
    vector<double> input, output;
    string line, word;

    inputs.clear();
    expected.clear();

    for (int i=0; i<num_samples; ++i) {
        row.clear();
        input.clear();
        output.clear();

        getline(fin, line);
        stringstream s(line);
        while (getline(s, word, ',')) {
            row.push_back(word);
        }

        if (row.size() == 0) {
            continue;
        }

        for (int j=1; j<row.size(); ++j) {
            input.push_back(atof(row[j].c_str())/256);
        }

        int cls = atoi(row.front().c_str());
        for (int j=0; j<cls; ++j) {
            output.push_back(0.0);
        }
        output.push_back(1.0);
        for (int j=cls+1; j<10; ++j) {
            output.push_back(0.0);
        }

        inputs.push_back(input);
        expected.push_back(output);
    }

    return inputs;
}


std::vector<std::vector<double>> &MnistReader::outputs()
{
    return expected;
}
