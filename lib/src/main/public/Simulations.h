#ifndef QFIN_SIMULATIONS_H
#define QFIN_SIMULATIONS_H

#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <optional>

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
    
    // Inverse CDF (PPF) using approximation
    double norm_ppf(double p)
    {
        // Beasley-Springer-Moro algorithm for inverse normal CDF
        if (p <= 0.0 || p >= 1.0)
        {
            return 0.0; // Handle edge cases
        }
        
        const double a0 = 2.50662823884;
        const double a1 = -18.61500062529;
        const double a2 = 41.39119773534;
        const double a3 = -25.44106049637;
        const double b1 = -8.47351093090;
        const double b2 = 23.08336743743;
        const double b3 = -21.06224101826;
        const double b4 = 3.13082909833;
        const double c0 = 0.3374754822726147;
        const double c1 = 0.9761690190917186;
        const double c2 = 0.1607979714918209;
        const double c3 = 0.0276438810333863;
        const double c4 = 0.0038405729373609;
        const double c5 = 0.0003951896511919;
        const double c6 = 0.0000321767881768;
        const double c7 = 0.0000002888167364;
        const double c8 = 0.0000003960315187;
        
        double y = p - 0.5;
        double r;
        
        if (std::abs(y) < 0.42)
        {
            r = y * y;
            r = y * (((a3 * r + a2) * r + a1) * r + a0) / 
                ((((b4 * r + b3) * r + b2) * r + b1) * r + 1.0);
        }
        else
        {
            r = p;
            if (y > 0.0)
            {
                r = 1.0 - p;
            }
            r = std::log(-std::log(r));
            r = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * (c7 + r * c8)))))));
            if (y < 0.0)
            {
                r = -r;
            }
        }
        return r;
        }
    }

    class GeometricBrownianMotion
    {
    public:
    GeometricBrownianMotion(double S, double mu, double sigma, double dt, double T)
        : simulated_path_(simulate_path(S, mu, sigma, dt, T)) {}
    
    const std::vector<double>& simulated_path() const { return simulated_path_; }

private:
    std::vector<double> simulate_path(double S, double mu, double sigma, double dt, double T)
    {
        std::vector<double> prices;
        double prev_price = S;
        double step = 0.0;
        double current_dt = dt;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 1.0);
        
        while (step < T)
        {
            double random_value = dist(gen);
            double ds = prev_price * mu * current_dt + prev_price * sigma * random_value * std::sqrt(current_dt);
            prev_price = prev_price + ds;
            prices.push_back(prev_price);
            
            if (step + current_dt > T)
            {
                current_dt = T - step;
            }
            else
            {
                step += current_dt;
            }
        }
        return prices;
    }
    
    std::vector<double> simulated_path_;
};

class StochasticVarianceModel
{
public:
    StochasticVarianceModel(double S, double mu, double r, double div, double alpha, 
                           double beta, double rho, double vol_var, double inst_var, 
                           double dt, double T)
        : simulated_path_(simulate_path(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T)) {}
    
    const std::vector<double>& simulated_path() const { return simulated_path_; }

private:
    std::vector<double> simulate_path(double S, double mu, double r, double div, 
                                      double alpha, double beta, double rho, 
                                      double vol_var, double inst_var, double dt, double T)
    {
        std::vector<double> prices;
        double price_now = S;
        double inst_var_now = inst_var;
        double prev_inst_var = inst_var_now;
        double step = 0.0;
        double current_dt = dt;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        
        while (step < T)
        {
            double u1 = uniform(gen);
            double u2 = uniform(gen);
            double e1 = norm_ppf(u1);
            double e2 = e1 * rho + std::sqrt(1.0 - (rho * rho)) * norm_ppf(u2);
            
            price_now = price_now + (r - div) * price_now * current_dt 
                       + price_now * std::sqrt(prev_inst_var * current_dt) * e1;
            prev_inst_var = inst_var_now;
            inst_var_now = prev_inst_var + alpha * (beta - prev_inst_var) * current_dt 
                          + vol_var * std::sqrt(prev_inst_var * current_dt) * e2;
            
            // Avoid negative cases and floor variance at zero
            if (inst_var_now <= 0.0000001)
            {
                inst_var_now = 0.0000001;
            }
            
            prices.push_back(price_now);
            
            if (step + current_dt > T)
            {
                current_dt = T - step;
            }
            else
            {
                step += current_dt;
            }
        }
        return prices;
    }
    
    std::vector<double> simulated_path_;
};

class MonteCarloCall
{
public:
    MonteCarloCall(double strike, int n, double r, double S, double mu, double sigma, 
                   double dt, double T,
                   std::optional<double> alpha = std::nullopt,
                   std::optional<double> beta = std::nullopt,
                   std::optional<double> rho = std::nullopt,
                   std::optional<double> div = std::nullopt,
                   std::optional<double> vol_var = std::nullopt)
        : price_(alpha.has_value() 
                 ? simulate_price_svm(strike, n, S, mu, r, div.value(), alpha.value(), 
                                     beta.value(), rho.value(), vol_var.value(), 
                                     std::sqrt(sigma), dt, T)
                 : simulate_price_gbm(strike, n, r, S, mu, sigma, dt, T)) {}
    
    double price() const { return price_; }

private:
    double simulate_price_gbm(double strike, int n, double r, double S, double mu, 
                               double sigma, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            GeometricBrownianMotion gbm(S, mu, sigma, dt, T);
            const auto& path = gbm.simulated_path();
            if (!path.empty() && path.back() >= strike)
            {
                payouts.push_back((path.back() - strike) * std::exp(-r * T));
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double simulate_price_svm(double strike, int n, double S, double mu, double r, 
                              double div, double alpha, double beta, double rho, 
                              double vol_var, double inst_var, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            StochasticVarianceModel svm(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T);
            const auto& path = svm.simulated_path();
            if (!path.empty() && path.back() >= strike)
            {
                payouts.push_back((path.back() - strike) * std::exp(-r * T));
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double price_;
};

class MonteCarloPut
{
public:
    MonteCarloPut(double strike, int n, double r, double S, double mu, double sigma, 
                  double dt, double T,
                  std::optional<double> alpha = std::nullopt,
                  std::optional<double> beta = std::nullopt,
                  std::optional<double> rho = std::nullopt,
                  std::optional<double> div = std::nullopt,
                  std::optional<double> vol_var = std::nullopt)
        : price_(alpha.has_value() 
                 ? simulate_price_svm(strike, n, S, mu, r, div.value(), alpha.value(), 
                                     beta.value(), rho.value(), vol_var.value(), 
                                     std::sqrt(sigma), dt, T)
                 : simulate_price_gbm(strike, n, r, S, mu, sigma, dt, T)) {}
    
    double price() const { return price_; }

private:
    double simulate_price_gbm(double strike, int n, double r, double S, double mu, 
                               double sigma, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            GeometricBrownianMotion gbm(S, mu, sigma, dt, T);
            const auto& path = gbm.simulated_path();
            if (!path.empty() && path.back() <= strike)
            {
                payouts.push_back((strike - path.back()) * std::exp(-r * T));
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double simulate_price_svm(double strike, int n, double S, double mu, double r, 
                              double div, double alpha, double beta, double rho, 
                              double vol_var, double inst_var, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            StochasticVarianceModel svm(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T);
            const auto& path = svm.simulated_path();
            if (!path.empty() && path.back() <= strike)
            {
                payouts.push_back((strike - path.back()) * std::exp(-r * T));
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double price_;
};

class MonteCarloBinaryCall
{
public:
    MonteCarloBinaryCall(double strike, int n, double payout, double r, double S, 
                         double mu, double sigma, double dt, double T,
                         std::optional<double> alpha = std::nullopt,
                         std::optional<double> beta = std::nullopt,
                         std::optional<double> rho = std::nullopt,
                         std::optional<double> div = std::nullopt,
                         std::optional<double> vol_var = std::nullopt)
        : price_(alpha.has_value() 
                 ? simulate_price_svm(strike, n, payout, S, mu, r, div.value(), alpha.value(), 
                                     beta.value(), rho.value(), vol_var.value(), 
                                     std::sqrt(sigma), dt, T)
                 : simulate_price_gbm(strike, n, payout, r, S, mu, sigma, dt, T)) {}
    
    double price() const { return price_; }

private:
    double simulate_price_gbm(double strike, int n, double payout, double r, double S, 
                               double mu, double sigma, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            GeometricBrownianMotion gbm(S, mu, sigma, dt, T);
            const auto& path = gbm.simulated_path();
            if (!path.empty() && path.back() >= strike)
            {
                payouts.push_back(payout * std::exp(-r * T));
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double simulate_price_svm(double strike, int n, double payout, double S, double mu, 
                              double r, double div, double alpha, double beta, double rho, 
                              double vol_var, double inst_var, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            StochasticVarianceModel svm(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T);
            const auto& path = svm.simulated_path();
            if (!path.empty() && path.back() >= strike)
            {
                payouts.push_back(payout * std::exp(-r * T));
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double price_;
};

class MonteCarloBinaryPut
{
public:
    MonteCarloBinaryPut(double strike, int n, double payout, double r, double S, 
                        double mu, double sigma, double dt, double T,
                        std::optional<double> alpha = std::nullopt,
                        std::optional<double> beta = std::nullopt,
                        std::optional<double> rho = std::nullopt,
                        std::optional<double> div = std::nullopt,
                        std::optional<double> vol_var = std::nullopt)
        : price_(alpha.has_value() 
                 ? simulate_price_svm(strike, n, payout, S, mu, r, div.value(), alpha.value(), 
                                     beta.value(), rho.value(), vol_var.value(), 
                                     std::sqrt(sigma), dt, T)
                 : simulate_price_gbm(strike, n, payout, r, S, mu, sigma, dt, T)) {}
    
    double price() const { return price_; }

private:
    double simulate_price_gbm(double strike, int n, double payout, double r, double S, 
                               double mu, double sigma, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            GeometricBrownianMotion gbm(S, mu, sigma, dt, T);
            const auto& path = gbm.simulated_path();
            if (!path.empty() && path.back() <= strike)
            {
                payouts.push_back(payout * std::exp(-r * T));
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double simulate_price_svm(double strike, int n, double payout, double S, double mu, 
                              double r, double div, double alpha, double beta, double rho, 
                              double vol_var, double inst_var, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            StochasticVarianceModel svm(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T);
            const auto& path = svm.simulated_path();
            if (!path.empty() && path.back() <= strike)
            {
                payouts.push_back(payout * std::exp(-r * T));
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double price_;
};

class MonteCarloBarrierCall
{
public:
    MonteCarloBarrierCall(double strike, int n, double barrier, double r, double S, 
                          double mu, double sigma, double dt, double T,
                          bool up = true, bool out = true,
                          std::optional<double> alpha = std::nullopt,
                          std::optional<double> beta = std::nullopt,
                          std::optional<double> rho = std::nullopt,
                          std::optional<double> div = std::nullopt,
                          std::optional<double> vol_var = std::nullopt)
        : price_(alpha.has_value() 
                 ? simulate_price_svm(strike, n, barrier, up, out, S, mu, r, div.value(), 
                                     alpha.value(), beta.value(), rho.value(), vol_var.value(), 
                                     std::sqrt(sigma), dt, T)
                 : simulate_price_gbm(strike, n, barrier, up, out, r, S, mu, sigma, dt, T)) {}
    
    double price() const { return price_; }

private:
    double simulate_price_gbm(double strike, int n, double barrier, bool up, bool out, 
                               double r, double S, double mu, double sigma, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            bool barrier_triggered = false;
            GeometricBrownianMotion gbm(S, mu, sigma, dt, T);
            const auto& path = gbm.simulated_path();
            
            for (double price : path)
            {
                if (up)
                {
                    if (price >= barrier)
                    {
                        barrier_triggered = true;
                        break;
                    }
                }
            else
            {
                    if (price <= barrier)
                    {
                        barrier_triggered = true;
                        break;
                    }
                }
            }
            
            if (out && !barrier_triggered)
            {
                if (!path.empty() && path.back() >= strike)
            {
                    payouts.push_back((path.back() - strike) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else if (!out && barrier_triggered)
            {
                if (!path.empty() && path.back() >= strike)
            {
                    payouts.push_back((path.back() - strike) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double simulate_price_svm(double strike, int n, double barrier, bool up, bool out, 
                              double S, double mu, double r, double div, double alpha, 
                              double beta, double rho, double vol_var, double inst_var, 
                              double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            bool barrier_triggered = false;
            StochasticVarianceModel svm(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T);
            const auto& path = svm.simulated_path();
            
            for (double price : path)
            {
                if (up)
                {
                    if (price >= barrier)
                    {
                        barrier_triggered = true;
                        break;
                    }
                }
            else
            {
                    if (price <= barrier)
                    {
                        barrier_triggered = true;
                        break;
                    }
                }
            }
            
            if (out && !barrier_triggered)
            {
                if (!path.empty() && path.back() >= strike)
            {
                    payouts.push_back((path.back() - strike) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else if (!out && barrier_triggered)
            {
                if (!path.empty() && path.back() >= strike)
            {
                    payouts.push_back((path.back() - strike) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double price_;
};

class MonteCarloBarrierPut
{
public:
    MonteCarloBarrierPut(double strike, int n, double barrier, double r, double S, 
                         double mu, double sigma, double dt, double T,
                         bool up = true, bool out = true,
                         std::optional<double> alpha = std::nullopt,
                         std::optional<double> beta = std::nullopt,
                         std::optional<double> rho = std::nullopt,
                         std::optional<double> div = std::nullopt,
                         std::optional<double> vol_var = std::nullopt)
        : price_(alpha.has_value() 
                 ? simulate_price_svm(strike, n, barrier, up, out, S, mu, r, div.value(), 
                                     alpha.value(), beta.value(), rho.value(), vol_var.value(), 
                                     std::sqrt(sigma), dt, T)
                 : simulate_price_gbm(strike, n, barrier, up, out, r, S, mu, sigma, dt, T)) {}
    
    double price() const { return price_; }

private:
    double simulate_price_gbm(double strike, int n, double barrier, bool up, bool out, 
                               double r, double S, double mu, double sigma, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            bool barrier_triggered = false;
            GeometricBrownianMotion gbm(S, mu, sigma, dt, T);
            const auto& path = gbm.simulated_path();
            
            for (double price : path)
            {
                if (up)
                {
                    if (price >= barrier)
                    {
                        barrier_triggered = true;
                        break;
                    }
                }
            else
            {
                    if (price <= barrier)
                    {
                        barrier_triggered = true;
                        break;
                    }
                }
            }
            
            if (out && !barrier_triggered)
            {
                if (!path.empty() && path.back() <= strike)
            {
                    payouts.push_back((strike - path.back()) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else if (!out && barrier_triggered)
            {
                if (!path.empty() && path.back() <= strike)
            {
                    payouts.push_back((strike - path.back()) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double simulate_price_svm(double strike, int n, double barrier, bool up, bool out, 
                              double S, double mu, double r, double div, double alpha, 
                              double beta, double rho, double vol_var, double inst_var, 
                              double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            bool barrier_triggered = false;
            StochasticVarianceModel svm(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T);
            const auto& path = svm.simulated_path();
            
            for (double price : path)
            {
                if (up)
                {
                    if (price >= barrier)
                    {
                        barrier_triggered = true;
                        break;
                    }
                }
            else
            {
                    if (price <= barrier)
                    {
                        barrier_triggered = true;
                        break;
                    }
                }
            }
            
            if (out && !barrier_triggered)
            {
                if (!path.empty() && path.back() <= strike)
            {
                    payouts.push_back((strike - path.back()) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else if (!out && barrier_triggered)
            {
                if (!path.empty() && path.back() <= strike)
            {
                    payouts.push_back((strike - path.back()) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double price_;
};

class MonteCarloAsianCall
{
public:
    MonteCarloAsianCall(double strike, int n, double r, double S, double mu, double sigma, 
                        double dt, double T,
                        std::optional<double> alpha = std::nullopt,
                        std::optional<double> beta = std::nullopt,
                        std::optional<double> rho = std::nullopt,
                        std::optional<double> div = std::nullopt,
                        std::optional<double> vol_var = std::nullopt)
        : price_(alpha.has_value() 
                 ? simulate_price_svm(strike, n, S, mu, r, div.value(), alpha.value(), 
                                     beta.value(), rho.value(), vol_var.value(), 
                                     std::sqrt(sigma), dt, T)
                 : simulate_price_gbm(strike, n, r, S, mu, sigma, dt, T)) {}
    
    double price() const { return price_; }

private:
    double simulate_price_gbm(double strike, int n, double r, double S, double mu, 
                               double sigma, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            GeometricBrownianMotion gbm(S, mu, sigma, dt, T);
            const auto& path = gbm.simulated_path();
            if (!path.empty())
            {
                double avg_price = std::accumulate(path.begin(), path.end(), 0.0) / path.size();
                if (avg_price >= strike)
                {
                    payouts.push_back((avg_price - strike) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double simulate_price_svm(double strike, int n, double S, double mu, double r, 
                              double div, double alpha, double beta, double rho, 
                              double vol_var, double inst_var, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            StochasticVarianceModel svm(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T);
            const auto& path = svm.simulated_path();
            if (!path.empty())
            {
                double avg_price = std::accumulate(path.begin(), path.end(), 0.0) / path.size();
                if (avg_price >= strike)
                {
                    payouts.push_back((avg_price - strike) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double price_;
};

class MonteCarloAsianPut
{
public:
    MonteCarloAsianPut(double strike, int n, double r, double S, double mu, double sigma, 
                       double dt, double T,
                       std::optional<double> alpha = std::nullopt,
                       std::optional<double> beta = std::nullopt,
                       std::optional<double> rho = std::nullopt,
                       std::optional<double> div = std::nullopt,
                       std::optional<double> vol_var = std::nullopt)
        : price_(alpha.has_value() 
                 ? simulate_price_svm(strike, n, S, mu, r, div.value(), alpha.value(), 
                                     beta.value(), rho.value(), vol_var.value(), 
                                     std::sqrt(sigma), dt, T)
                 : simulate_price_gbm(strike, n, r, S, mu, sigma, dt, T)) {}
    
    double price() const { return price_; }

private:
    double simulate_price_gbm(double strike, int n, double r, double S, double mu, 
                               double sigma, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            GeometricBrownianMotion gbm(S, mu, sigma, dt, T);
            const auto& path = gbm.simulated_path();
            if (!path.empty())
            {
                double avg_price = std::accumulate(path.begin(), path.end(), 0.0) / path.size();
                if (avg_price <= strike)
                {
                    payouts.push_back((strike - avg_price) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double simulate_price_svm(double strike, int n, double S, double mu, double r, 
                              double div, double alpha, double beta, double rho, 
                              double vol_var, double inst_var, double dt, double T)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            StochasticVarianceModel svm(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T);
            const auto& path = svm.simulated_path();
            if (!path.empty())
            {
                double avg_price = std::accumulate(path.begin(), path.end(), 0.0) / path.size();
                if (avg_price <= strike)
                {
                    payouts.push_back((strike - avg_price) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double price_;
};

class MonteCarloExtendibleCall
{
public:
    MonteCarloExtendibleCall(double strike, int n, double r, double S, double mu, 
                             double sigma, double dt, double T, double extension,
                             std::optional<double> alpha = std::nullopt,
                             std::optional<double> beta = std::nullopt,
                             std::optional<double> rho = std::nullopt,
                             std::optional<double> div = std::nullopt,
                             std::optional<double> vol_var = std::nullopt)
        : price_(alpha.has_value() 
                 ? simulate_price_svm(strike, n, S, mu, r, div.value(), alpha.value(), 
                                     beta.value(), rho.value(), vol_var.value(), 
                                     std::sqrt(sigma), dt, T, extension)
                 : simulate_price_gbm(strike, n, r, S, mu, sigma, dt, T, extension)) {}
    
    double price() const { return price_; }

private:
    double simulate_price_gbm(double strike, int n, double r, double S, double mu, 
                               double sigma, double dt, double T, double extension)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            GeometricBrownianMotion gbm(S, mu, sigma, dt, T);
            const auto& path = gbm.simulated_path();
            if (!path.empty() && path.back() >= strike)
            {
                payouts.push_back((path.back() - strike) * std::exp(-r * T));
            }
            else if (!path.empty())
            {
                // Continue the simulation
                GeometricBrownianMotion gbm2(path.back(), mu, sigma, dt, extension);
                const auto& path2 = gbm2.simulated_path();
                if (!path2.empty() && path2.back() >= strike)
                {
                    payouts.push_back((path2.back() - strike) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double simulate_price_svm(double strike, int n, double S, double mu, double r, 
                              double div, double alpha, double beta, double rho, 
                              double vol_var, double inst_var, double dt, double T, 
                              double extension)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            StochasticVarianceModel svm(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T);
            const auto& path = svm.simulated_path();
            if (!path.empty() && path.back() >= strike)
            {
                payouts.push_back((path.back() - strike) * std::exp(-r * T));
            }
            else if (!path.empty())
            {
                // Continue the simulation
                StochasticVarianceModel svm2(path.back(), mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, extension);
                const auto& path2 = svm2.simulated_path();
                if (!path2.empty() && path2.back() >= strike)
                {
                    payouts.push_back((path2.back() - strike) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double price_;
};

class MonteCarloExtendiblePut
{
public:
    MonteCarloExtendiblePut(double strike, int n, double r, double S, double mu, 
                            double sigma, double dt, double T, double extension,
                            std::optional<double> alpha = std::nullopt,
                            std::optional<double> beta = std::nullopt,
                            std::optional<double> rho = std::nullopt,
                            std::optional<double> div = std::nullopt,
                            std::optional<double> vol_var = std::nullopt)
        : price_(alpha.has_value() 
                 ? simulate_price_svm(strike, n, S, mu, r, div.value(), alpha.value(), 
                                     beta.value(), rho.value(), vol_var.value(), 
                                     std::sqrt(sigma), dt, T, extension)
                 : simulate_price_gbm(strike, n, r, S, mu, sigma, dt, T, extension)) {}
    
    double price() const { return price_; }

private:
    double simulate_price_gbm(double strike, int n, double r, double S, double mu, 
                               double sigma, double dt, double T, double extension)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            GeometricBrownianMotion gbm(S, mu, sigma, dt, T);
            const auto& path = gbm.simulated_path();
            if (!path.empty() && path.back() <= strike)
            {
                payouts.push_back((strike - path.back()) * std::exp(-r * T));
            }
            else if (!path.empty())
            {
                // Continue the simulation
                GeometricBrownianMotion gbm2(path.back(), mu, sigma, dt, extension);
                const auto& path2 = gbm2.simulated_path();
                if (!path2.empty() && path2.back() <= strike)
                {
                    payouts.push_back((strike - path2.back()) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double simulate_price_svm(double strike, int n, double S, double mu, double r, 
                              double div, double alpha, double beta, double rho, 
                              double vol_var, double inst_var, double dt, double T, 
                              double extension)
    {
        std::vector<double> payouts;
        for (int i = 0; i < n; ++i)
        {
            StochasticVarianceModel svm(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T);
            const auto& path = svm.simulated_path();
            if (!path.empty() && path.back() <= strike)
            {
                payouts.push_back((strike - path.back()) * std::exp(-r * T));
            }
            else if (!path.empty())
            {
                // Continue the simulation
                StochasticVarianceModel svm2(path.back(), mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, extension);
                const auto& path2 = svm2.simulated_path();
                if (!path2.empty() && path2.back() <= strike)
                {
                    payouts.push_back((strike - path2.back()) * std::exp(-r * T));
                }
            else
            {
                    payouts.push_back(0.0);
                }
            }
            else
            {
                payouts.push_back(0.0);
            }
        }
        return std::accumulate(payouts.begin(), payouts.end(), 0.0) / n;
    }
    
    double price_;
};

} // namespace QFin

#endif // QFIN_SIMULATIONS_H

