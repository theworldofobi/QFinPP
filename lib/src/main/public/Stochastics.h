#ifndef QFIN_STOCHASTICS_H
#define QFIN_STOCHASTICS_H

#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <memory>
#include <tuple>
#include <stdexcept>

namespace QFin
{
    // Helper functions for normal distribution
    namespace
    {
        // Standard normal CDF
        double norm_cdf(double x)
        {
            return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        }
        
        // Standard normal PDF
        double norm_pdf(double x)
        {
            return 1.0 / std::sqrt(2.0 * M_PI) * std::exp(-0.5 * x * x);
        }
    }

    // Abstract class framework for a stochastic process
    class StochasticModel
    {
    public:
        StochasticModel(const std::vector<double>& params) : params_(params) {}
        virtual ~StochasticModel() = default;
        
        // Risk neutral pricing kernel
        virtual double vanilla_pricing(double F0, double X, double T, const std::string& op_type = "CALL") = 0;
        
        // Calibration is conducted via an inputted vol surface and the vanilla pricing function
        virtual void calibrate(double impl_vol, double T, const std::string& op_type = "CALL")
        {
            // Default implementation - can be overridden
        }
        
        // Simulation is conducted after calibration of params to price exotics
        virtual std::vector<std::vector<double>> simulate(double F0, int n, double dt, double T) = 0;

    protected:
        std::vector<double> params_;
    };

    // Closed form vanilla euro option pricing
    class ArithmeticBrownianMotion : public StochasticModel
    {
    public:
        ArithmeticBrownianMotion(const std::vector<double>& params) : StochasticModel(params) {}
        
        // Closed form vanilla euro option pricing
        double vanilla_pricing(double F0, double X, double T, const std::string& op_type = "CALL") override
        {
            // Return closed-form Bachelier call
            if (op_type == "CALL")
            {
                double sigma = params_[0];
                double d = (F0 - X) / (sigma * std::sqrt(T));
                return (F0 - X) * norm_cdf(d) + sigma * std::sqrt(T) * norm_pdf(d);
            }
            // Use call-put parity for put price
            else if (op_type == "PUT")
            {
                return vanilla_pricing(F0, X, T, "CALL") - F0 + X;
            }
            else
            {
                throw std::invalid_argument("Option type must be CALL/PUT");
            }
        }
        
        // Simulating paths of arithmetic Brownian motion
        std::vector<std::vector<double>> simulate(double F0, int n, double dt, double T) override
        {
            std::vector<std::vector<double>> paths;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dist(0.0, 1.0);
            
            double sigma = params_[0];
            
            for (int i = 0; i < n; ++i)
            {
                // n simulations
                double ttm = T;
                std::vector<double> path;
                path.push_back(F0);
                
                // While the step is greater than zero diffuse the process
                while (ttm - dt > 0)
                {
                    double random_value = dist(gen);
                    path.push_back(path.back() + sigma * random_value * std::sqrt(dt));
                    ttm -= dt;
                }
                
                // Final time increment
                if (dt > 0)
                {
                    double random_value = dist(gen);
                    path.push_back(path.back() + sigma * random_value * std::sqrt(dt));
                }
                
                // Append the path
                paths.push_back(path);
            }
            
            // Store paths and path characteristics
            path_characteristics_ = std::make_tuple(paths, n, dt, T);
            
            // Return paths and path characteristics
            return paths;
        }
        
        // Calibration cannot be conducted due to flat volatility surface
        // (inherited default implementation from base class)

    private:
        std::tuple<std::vector<std::vector<double>>, int, double, double> path_characteristics_;
    };
} // namespace QFin

#endif // QFIN_STOCHASTICS_H
