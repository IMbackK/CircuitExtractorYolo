#include "randomgen.h"
#include <assert.h>
#include <cstddef>
#include <random>
#include <limits>

static std::default_random_engine randomEngine;
static std::uniform_real_distribution<double> dist(0, 1);
static std::uniform_int_distribution<size_t> distSt(1, std::numeric_limits<uint64_t>::max());

double rd::rand(double max)
{
	return dist(randomEngine)*max;
}

uint64_t rd::uid()
{
	return distSt(randomEngine);
}

void rd::init()
{
	std::random_device randomDevice;
	randomEngine.seed(randomDevice());
}
