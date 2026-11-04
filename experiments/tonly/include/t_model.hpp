#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <string>
#include <vector>
namespace t_only {
constexpr double tick = 0.01; // command-label units, physical degree not certified
using Features = std::array<double,22>;
struct Sample { std::string path; int group; double target; Features features; };
struct Config { int representation=0; int degree=1; double alpha=0; int loss=0; int rank=0; };
struct Model { Config config; cv::Mat mean,scale,coefficient; double intercept=0; double minimum_t=.01,maximum_t=2.; bool requires_y_fixed_at_reference=false; };
std::vector<double> representation(const Features&, int kind);
Model fit(const std::vector<Sample>&, const std::vector<int>&, Config);
double predict(const Model&, const Features&);
Config select(const std::vector<Sample>&, const std::vector<int>&, int mode);
void save(const Model&, const std::string&);
Model load(const std::string&);
}
