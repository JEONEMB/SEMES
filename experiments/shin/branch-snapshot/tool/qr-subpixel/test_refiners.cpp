#include "qr_refinement.hpp"
#include <iostream>
#include <random>
using namespace subpixel;
double jacError(const Objective &fun, const cv::Mat &p)
{
    cv::Mat r, j;
    fun(p, r, j);
    double error = 0;
    for (int k = 0; k < p.rows; ++k)
    {
        auto pp = p.clone(), pm = p.clone();
        pp.at<double>(k) += 1e-6;
        pm.at<double>(k) -= 1e-6;
        cv::Mat rp, rm, z;
        fun(pp, rp, z);
        fun(pm, rm, z);
        error = std::max(error, cv::norm((rp - rm) / 2e-6 - j.col(k), cv::NORM_INF));
    }
    return error;
}
int main()
{
    double modelError = 0, fourierError = 0, identityError = 0;
    std::mt19937 random(19);
    std::uniform_real_distribution<double> u(-1, 1);
    for (int repetition = 0; repetition < 64; ++repetition)
    {
        Patch samples;
        for (int y = -4; y <= 4; ++y)
        {
            for (int x = -4; x <= 4; ++x)
            {
                samples.push_back({x + .23, y - .17, .5 + .4 * u(random)});
            }
        }
        double a1 = .2 * u(random), a2 = pi / 2 + .2 * u(random),
               logs = -1.2 + 1.8 * (u(random) + 1) / 2;
        cv::Mat p =
            (cv::Mat_<double>(7, 1) << .3 * u(random), .3 * u(random), a1, a2, logs, .3, .5);
        modelError = std::max(modelError, jacError(modelObjective(samples), p));
        fourierError =
            std::max(fourierError, jacError(fourierObjective(samples, 1.8, repetition % 2),
                                            (cv::Mat_<double>(2, 1) << .13, -.11)));
        double value, d[5];
        corner(0, 0, a1, a2, logs, value, d);
        identityError =
            std::max(identityError, std::abs(value - 2 / pi * std::asin(std::cos(a1 - a2))));
        double x = 2 * u(random), y = 2 * u(random);
        corner(x, y, 0, pi / 2, logs, value, d);
        identityError = std::max(
            identityError, std::abs(value - std::erf(x / (std::sqrt(2.) * std::exp(logs))) *
                                                std::erf(y / (std::sqrt(2.) * std::exp(logs)))));
    }
    auto f = [](const cv::Mat &p, cv::Mat &r, cv::Mat &j)
    {
        r = p.clone();
        r.at<double>(4) += 2;
        j = cv::Mat::eye(7, 7, CV_64F);
    };
    cv::Mat lo(7, 1, CV_64F, cv::Scalar(-1)), hi(7, 1, CV_64F, cv::Scalar(1));
    auto bounded = optimize(cv::Mat::zeros(7, 1, CV_64F), f, lo, hi);
    std::cout << "model_jacobian_max=" << modelError << " fourier_jacobian_max=" << fourierError
              << " convolution_identity_max=" << identityError
              << " constrained_converged=" << bounded.converged
              << " nuisance_boundary_valid=" << bounded.valid << '\n';
    return modelError < 1e-6 && fourierError < 1e-6 && identityError < 1e-8 && bounded.converged &&
                   bounded.valid
               ? 0
               : 1;
}
