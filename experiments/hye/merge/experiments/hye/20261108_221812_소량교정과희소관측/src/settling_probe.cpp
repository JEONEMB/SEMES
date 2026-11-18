#include "cache.hpp"

#include <array>
#include <chrono>
#include <iostream>
#include <random>

namespace {
constexpr double drift = 0.03;
constexpr double frequency = 2.8;
constexpr double amplitude = 2.0;
constexpr double zeta = 0.009;
const double pi = std::acos(-1.0);

std::vector<double> noiseSample(const std::filesystem::path& path, size_t& count) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Missing empirical repeat residuals");
    std::map<std::string, std::vector<double>> groups;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::replace(line.begin(), line.end(), '\t', ',');
        const auto row = a0::split_csv(line);
        groups[row.at(1)].push_back(std::stod(row.at(5)));
    }
    count = groups.size();
    std::vector<double> residuals;
    for (const auto& [group, values] : groups) {
        if (values.size() != 3) throw std::runtime_error("Expected three samples per group");
        for (double v : values) residuals.push_back(v - a0::mean(values));
    }
    if (residuals.empty()) throw std::runtime_error("No noise samples");
    return residuals;
}

std::pair<double, bool> fitDrift(const std::vector<double>& data, double dt) {
    const double mean = a0::mean(data);
    const double midpoint = 0.5 * (data.size() - 1) * dt;
    double numerator = 0, denominator = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        const double x = i * dt - midpoint;
        numerator += x * (data[i] - mean);
        denominator += x * x;
    }
    const double estimate = numerator / denominator;
    double squared = 0;
    for (size_t i = 0; i < data.size(); ++i)
        squared += std::pow(data[i] - mean - estimate * (i * dt - midpoint), 2);
    const double se = std::sqrt(squared / (data.size() - 2) / denominator);
    return {estimate, std::abs(estimate) > 1.96 * se};
}

int bestAllan(const std::vector<double>& data) {
    std::vector<double> cumulative{0};
    for (double value : data) cumulative.push_back(cumulative.back() + value);
    int best = 1;
    double minimum = std::numeric_limits<double>::infinity();
    for (size_t k = 1; k <= data.size() / 4; ++k) {
        double sum = 0;
        for (size_t i = 0; i + 2*k <= data.size(); ++i)
            sum += std::pow((cumulative[i + 2*k] - 2*cumulative[i + k] + cumulative[i]) / k, 2);
        const double variance = sum / (2*(data.size() - 2*k + 1));
        if (variance < minimum) {
            minimum = variance;
            best = static_cast<int>(k);
        }
    }
    return best;
}

std::array<double, 3> fitDecay(const std::vector<double>& data, double dt) {
    cv::Mat response(static_cast<int>(data.size()), 1, CV_64F);
    std::copy(data.begin(), data.end(), response.ptr<double>());
    cv::Mat x(static_cast<int>(data.size()), 3, CV_64F);
    double least = std::numeric_limits<double>::infinity();
    std::array<double, 3> best{};
    for (int fi = 40; fi <= 72; ++fi) {
        const double f = 0.05 * fi, omega = 2*pi*f;
        for (int di = 0; di <= 30; ++di) {
            const double damping = 0.001 * di;
            for (int i = 0; i < x.rows; ++i) {
                const double t = i * dt, decay = std::exp(-damping * omega * t);
                x.at<double>(i, 0) = 1.0;
                x.at<double>(i, 1) = decay * std::cos(omega * t);
                x.at<double>(i, 2) = decay * std::sin(omega * t);
            }
            cv::Mat b;
            if (!cv::solve(x, response, b, cv::DECOMP_QR)) continue;
            const double loss = cv::norm(response - x * b, cv::NORM_L2SQR);
            if (loss < least) {
                least = loss;
                best = {f, damping, std::hypot(b.at<double>(1), b.at<double>(2))};
            }
        }
    }
    if (!std::isfinite(least)) throw std::runtime_error("Decay grid fit failed");
    return best;
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("Usage: semes_settling_probe TRIPLET_TSV OUTPUT_TSV");
    size_t observedTriplets = 0;
    const auto noise = noiseSample(std::filesystem::u8path(argv[1]), observedTriplets);
    const double variance = std::inner_product(noise.begin(), noise.end(), noise.begin(), 0.0) / noise.size();
    const auto path = std::filesystem::u8path(argv[2]);
    std::ofstream out(path);
    std::ofstream curve(path.parent_path() / "settling_window_curve.tsv");
    if (!out || !curve) throw std::runtime_error("Cannot write simulation results");
    out << "n\tdt_s\treplicates\tmedian_drift_relative_error\tdrift_detected\tnull_false_positive\t"
           "median_allan_window_drift\tfrequency_status\tmedian_freq_relative_error\t"
           "median_zeta_relative_error\tmedian_amplitude_relative_error\tpopulation_allan_window\tmedian_trial_ms\n";
    curve << "n\tdt_s\twindow\tmedian_sd_of_noise_averages\tmedian_sd_of_drift_averages\n";
    std::mt19937 gen(20261108u);
    for (int n : {50, 100, 200}) {
        for (double dt : {0.1, 0.5, 1.0}) {
            std::vector<double> driftError, allan, frequencyError, dampingError, amplitudeError, timing;
            std::vector<std::vector<double>> nullCurves(n/4 + 1), driftCurves(n/4 + 1);
            int bestPopulation = 1;
            double leastVariance = std::numeric_limits<double>::infinity();
            for (int k = 1; k <= n/4; ++k) {
                const double expected = variance/k + 0.5*std::pow(drift*dt*k, 2);
                if (expected < leastVariance) {
                    leastVariance = expected;
                    bestPopulation = k;
                }
            }
            int detects = 0, falsePositive = 0;
            const bool alias = frequency >= 0.5/dt;
            for (int repeat = 0; repeat < 30; ++repeat) {
                std::vector<double> nullSignal, driftSignal, dampedSignal;
                for (int i = 0; i < n; ++i) {
                    const double t = i*dt, omega = 2*pi*frequency;
                    nullSignal.push_back(noise[gen()%noise.size()]);
                    driftSignal.push_back(drift*t + noise[gen()%noise.size()]);
                    dampedSignal.push_back(amplitude*std::exp(-zeta*omega*t)*std::cos(omega*t) +
                                           noise[gen()%noise.size()]);
                }
                const auto start = std::chrono::steady_clock::now();
                for (int k = 1; k <= n/4; ++k) {
                    const auto sd = [k](const std::vector<double>& signal) {
                        std::vector<double> averages;
                        double sum = std::accumulate(signal.begin(), signal.begin()+k, 0.0);
                        averages.push_back(sum/k);
                        for (size_t i = k; i < signal.size(); ++i) {
                            sum += signal[i] - signal[i-k];
                            averages.push_back(sum/k);
                        }
                        return a0::sd(averages);
                    };
                    nullCurves[k].push_back(sd(nullSignal));
                    driftCurves[k].push_back(sd(driftSignal));
                }
                const auto [estimated, detected] = fitDrift(driftSignal, dt);
                driftError.push_back(std::abs(estimated - drift)/drift);
                detects += detected;
                falsePositive += fitDrift(nullSignal, dt).second;
                allan.push_back(bestAllan(driftSignal));
                if (!alias) {
                    const auto fit = fitDecay(dampedSignal, dt);
                    frequencyError.push_back(std::abs(fit[0]-frequency)/frequency);
                    dampingError.push_back(std::abs(fit[1]-zeta)/zeta);
                    amplitudeError.push_back(std::abs(fit[2]-amplitude)/amplitude);
                }
                timing.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now()-start).count());
            }
            for (int k = 1; k <= n/4; ++k)
                curve << n << '\t' << dt << '\t' << k << '\t' << a0::median(nullCurves[k])
                      << '\t' << a0::median(driftCurves[k]) << '\n';
            out << n << '\t' << dt << "\t30\t" << a0::median(driftError) << '\t' << detects << '\t'
                << falsePositive << '\t' << a0::median(allan) << '\t' << (alias ? "alias" : "fitted") << '\t'
                << a0::median(frequencyError) << '\t' << a0::median(dampingError) << '\t'
                << a0::median(amplitudeError) << '\t' << bestPopulation << '\t' << a0::median(timing) << '\n';
        }
    }
    std::cout << observedTriplets << " observed triplets: no recorded spacing; no temporal fit. "
              << "At three samples there are no overlapping Allan comparisons above a one-frame window.\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
