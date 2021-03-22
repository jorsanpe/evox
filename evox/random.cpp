/*
 * Copyright (c) 2020. Jordi Sánchez
 */
#include <ctime>
#include <random>
#include <cstdlib>


double first(double min, double max);
static double (*nextRandom)(double, double) = first;
static std::default_random_engine generator;
static std::uniform_real_distribution<double> distribution(0,1);

double next(double min, double max)
{
    double random_number = distribution(generator);
    return random_number * (max - min) + min;
}


double first(double min, double max)
{
    generator.seed(time(nullptr));
    nextRandom = next;
    return nextRandom(min, max);
}


double random(double min, double max)
{
    return nextRandom(min, max);
}
