/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <evox/Sample.h>


class MnistReader {
public:
    MnistReader(std::string filename);

    std::vector<std::vector<double>> &nextBatch(int num_samples);

    std::vector<std::vector<double>> &outputs();

private:
    std::string filename;
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> expected;
    std::fstream fin;
};