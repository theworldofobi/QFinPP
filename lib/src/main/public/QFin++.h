/*
 * QFin++ - Quantitative Finance C++ Library
 * A comprehensive library for option pricing, stochastic processes, 
 * and Monte Carlo simulations
 */

#ifndef QFIN_H
#define QFIN_H

#ifdef _WIN32
#define QFIN_EXPORT_FUNC __declspec(dllexport)
#else
#define QFIN_EXPORT_FUNC
#endif

// Core library includes
#include <string>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <optional>
#include <memory>
#include <tuple>
#include <stdexcept>

// Include all library headers
#include "Stochastics.h"
#include "Options.h"
#include "Simulations.h"

// QFin++ namespace for library components
namespace QFin
{
    // Legacy Greeter class
    class Greeter
    {
    public:
        std::string QFIN_EXPORT_FUNC greeting();
    };
}

#endif // QFIN_H
