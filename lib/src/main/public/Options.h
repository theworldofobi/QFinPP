#ifndef QFIN_OPTIONS_H
#define QFIN_OPTIONS_H

#include <cmath>

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

    class BlackScholesCall
    {
    public:
    BlackScholesCall(double asset_price, double asset_volatility, double strike_price,
                     double time_to_expiration, double risk_free_rate)
        : asset_price_(asset_price)
        , asset_volatility_(asset_volatility)
        , strike_price_(strike_price)
        , time_to_expiration_(time_to_expiration)
        , risk_free_rate_(risk_free_rate)
        , price_(call_price(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
        , delta_(call_delta(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
        , gamma_(call_gamma(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
        , vega_(call_vega(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
        , theta_(call_theta(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)) 
        {}

    ~BlackScholesCall() = default;

    double call_delta(double asset_price, double asset_volatility, 
                      double strike_price, double time_to_expiration, double risk_free_rate)
    {
        double b = std::exp(-risk_free_rate * time_to_expiration);
        double x1 = std::log(asset_price / strike_price) + 0.5 
        * (asset_volatility * asset_volatility) * time_to_expiration;
        x1 /= asset_volatility * std::sqrt(time_to_expiration);
        double z1 = norm_cdf(x1);
        return z1;
    }
    
    double call_gamma(double asset_price, double asset_volatility, 
                      double strike_price, double time_to_expiration, 
                      double risk_free_rate)
    {
        double b = std::exp(-risk_free_rate * time_to_expiration);
        double x1 = std::log(asset_price / strike_price) + 0.5 
        * (asset_volatility * asset_volatility) * time_to_expiration;
        x1 /= asset_volatility * std::sqrt(time_to_expiration);
        double z1 = norm_cdf(x1);
        double z2 = z1 / (asset_price * asset_volatility 
            * std::sqrt(time_to_expiration));
        return z2;
    }
    
    double call_vega(double asset_price, double asset_volatility, 
                     double strike_price, double time_to_expiration, double risk_free_rate)
    {
        double b = std::exp(-risk_free_rate * time_to_expiration);
        double x1 = std::log(asset_price / strike_price) + 0.5 
        * (asset_volatility * asset_volatility) * time_to_expiration;
        x1 /= asset_volatility * std::sqrt(time_to_expiration);
        double z1 = norm_pdf(x1);
        double z2 = asset_price * z1 * std::sqrt(time_to_expiration);
        return z2;
    }

    double call_theta(double asset_price, double asset_volatility, 
                      double strike_price, double time_to_expiration, double risk_free_rate)
    {
        double x1 = std::log(asset_price / strike_price) 
                    + 0.5 * (asset_volatility * asset_volatility) * time_to_expiration;
        x1 /= asset_volatility * std::sqrt(time_to_expiration);
        double n1 = -((asset_price * asset_volatility * norm_pdf(x1)) / (2 * std::sqrt(time_to_expiration)));
        double n2 = -(risk_free_rate * strike_price 
                     * std::exp(-risk_free_rate * time_to_expiration) 
                     * norm_cdf((x1 - (asset_volatility * std::sqrt(time_to_expiration)))));
        return (n1 + n2);
    }

    double call_price(double asset_price, double asset_volatility, double strike_price,
                      double time_to_expiration, double risk_free_rate)
    {
        double b = std::exp(-risk_free_rate * time_to_expiration);
        double x1 = std::log(asset_price / strike_price) 
                    + 0.5 * (asset_volatility * asset_volatility) * time_to_expiration;
        x1 /= asset_volatility * std::sqrt(time_to_expiration);
        double z1 = norm_cdf(x1);
        z1 = z1 * asset_price;
        double x2 = std::log(asset_price / strike_price) 
                    - 0.5 * (asset_volatility * asset_volatility) * time_to_expiration;
        x2 /= asset_volatility * std::sqrt(time_to_expiration);
        double z2 = norm_cdf(x2);
        z2 = b * strike_price * z2;
        return z1 - z2;
    }

    // Getters for member variables
    double asset_price() const { return asset_price_; }
    double asset_volatility() const { return asset_volatility_; }
    double strike_price() const { return strike_price_; }
    double time_to_expiration() const { return time_to_expiration_; }
    double risk_free_rate() const { return risk_free_rate_; }
    double price() const { return price_; }
    double delta() const { return delta_; }
    double gamma() const { return gamma_; }
    double vega() const { return vega_; }
    double theta() const { return theta_; }

private:
    double asset_price_;
    double asset_volatility_;
    double strike_price_;
    double time_to_expiration_;
    double risk_free_rate_;
    double price_;
    double delta_;
    double gamma_;
    double vega_;
    double theta_;
    };

    class BlackScholesPut
    {
    public:
    BlackScholesPut(double asset_price, double asset_volatility, double strike_price,
                    double time_to_expiration, double risk_free_rate)
        : asset_price_(asset_price)
        , asset_volatility_(asset_volatility)
        , strike_price_(strike_price)
        , time_to_expiration_(time_to_expiration)
        , risk_free_rate_(risk_free_rate)
        , price_(put_price(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
        , delta_(put_delta(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
        , gamma_(put_gamma(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
        , vega_(put_vega(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
        , theta_(put_theta(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate)) 
        {}

    ~BlackScholesPut() = default;

    double put_delta(double asset_price, double asset_volatility, double strike_price,
                     double time_to_expiration, double risk_free_rate)
    {
        double b = std::exp(-risk_free_rate * time_to_expiration);
        double x1 = std::log(asset_price / strike_price) + 0.5 
        * (asset_volatility * asset_volatility) * time_to_expiration;
        x1 /= asset_volatility * std::sqrt(time_to_expiration);
        double z1 = norm_cdf(x1);
        return z1 - 1;
    }

    double put_gamma(double asset_price, double asset_volatility, double strike_price,
                     double time_to_expiration, double risk_free_rate)
    {
        double b = std::exp(-risk_free_rate * time_to_expiration);
        double x1 = std::log(asset_price / strike_price) + 0.5 
        * (asset_volatility * asset_volatility) * time_to_expiration;
        x1 /= asset_volatility * std::sqrt(time_to_expiration);
        double z1 = norm_cdf(x1);
        double z2 = z1 / (asset_price * asset_volatility 
            * std::sqrt(time_to_expiration));
        return z2;
    }

    double put_vega(double asset_price, double asset_volatility, double strike_price,
                    double time_to_expiration, double risk_free_rate)
    {
        double b = std::exp(-risk_free_rate * time_to_expiration);
        double x1 = std::log(asset_price / strike_price) + 0.5 
        * (asset_volatility * asset_volatility) * time_to_expiration;
        x1 /= asset_volatility * std::sqrt(time_to_expiration);
        double z1 = norm_pdf(x1);
        double z2 = asset_price * z1 * std::sqrt(time_to_expiration);
        return z2;
    }

    double put_theta(double asset_price, double asset_volatility, double strike_price,
                     double time_to_expiration, double risk_free_rate)
    {
        double x1 = std::log(asset_price / strike_price) + 0.5 
        * (asset_volatility * asset_volatility) * time_to_expiration;
        x1 /= asset_volatility * std::sqrt(time_to_expiration);
        double n1 = -((asset_price * asset_volatility * norm_pdf(x1)) 
        / (2 * std::sqrt(time_to_expiration)));
        double n2 = risk_free_rate * strike_price 
                    * std::exp(-risk_free_rate * time_to_expiration)
                    * norm_cdf(-(x1 - (asset_volatility * std::sqrt(time_to_expiration))));
        return (n1 + n2);
    }

    double put_price(double asset_price, double asset_volatility, double strike_price,
                     double time_to_expiration, double risk_free_rate)
    {
        double b = std::exp(-risk_free_rate * time_to_expiration);
        double x1 = std::log(asset_price / strike_price) + 0.5 
        * (asset_volatility * asset_volatility) * time_to_expiration;
        x1 /= asset_volatility * std::sqrt(time_to_expiration);
        double z1 = norm_cdf(x1);
        z1 = b * strike_price * z1;
        double x2 = std::log(asset_price / strike_price) - 0.5 
        * (asset_volatility * asset_volatility) * time_to_expiration;
        x2 /= asset_volatility * std::sqrt(time_to_expiration);
        double z2 = norm_cdf(x2);
        z2 = asset_price * z2;
        return z1 - z2;
    }

    // Getters for member variables
    double asset_price() const { return asset_price_; }
    double asset_volatility() const { return asset_volatility_; }
    double strike_price() const { return strike_price_; }
    double time_to_expiration() const { return time_to_expiration_; }
    double risk_free_rate() const { return risk_free_rate_; }
    double price() const { return price_; }
    double delta() const { return delta_; }
    double gamma() const { return gamma_; }
    double vega() const { return vega_; }
    double theta() const { return theta_; }

private:
    double asset_price_;
    double asset_volatility_;
    double strike_price_;
    double time_to_expiration_;
    double risk_free_rate_;
    double price_;
    double delta_;
    double gamma_;
    double vega_;
        double theta_;
    };
} // namespace QFin

#endif // QFIN_OPTIONS_H
