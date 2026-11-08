#include "cache.hpp"

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <utility>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

std::vector<std::string> tabFields(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> fields;
    size_t begin = 0;
    for (size_t end = line.find('\t'); end != std::string::npos; end = line.find('\t', begin)) {
        fields.push_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
    fields.push_back(line.substr(begin));
    return fields;
}

struct TripletStats {
    std::vector<double> adjacent, ranges, absResiduals;
    double sse = 0;

    void add(const std::array<double, 3>& sample) {
        const double mean = (sample[0] + sample[1] + sample[2]) / 3;
        adjacent.push_back(std::abs(sample[1] - sample[0]));
        adjacent.push_back(std::abs(sample[2] - sample[1]));
        ranges.push_back(*std::max_element(sample.begin(), sample.end()) -
                         *std::min_element(sample.begin(), sample.end()));
        for (double x : sample) {
            const double residual = x - mean;
            sse += residual * residual;
            absResiduals.push_back(std::abs(residual));
        }
    }

    void write(std::ostream& out) const {
        const double sd = std::sqrt(sse / (2 * ranges.size()));
        out << ranges.size() << '\t' << sd << '\t' << 6 * sd << '\t'
            << a0::median(adjacent) << '\t' << a0::quantile(absResiduals, 0.95)
            << '\t' << a0::quantile(ranges, 0.95) << '\t' << *std::max_element(ranges.begin(), ranges.end());
    }
};

struct Frame {
    int repeat;
    double y, t;
    std::array<double, 80> values{};
};

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("Usage: repeat_floor REPO OUTPUT");
    const fs::path root = argv[1], out = argv[2];
    const fs::path hye = root / "experiments/hye";
    const auto cache = a0::load(hye / fs::u8path(u8"20261102_124309_A0_분할측정고정_계층분할과점측정캐시/결과/cache"));
    std::map<std::string, int> repeatNumbers;
    for (const auto& shot : cache.shots) repeatNumbers.emplace("data/clean/" + shot.set + "/" + shot.rel, shot.repeat);
    const fs::path v1 = hye / fs::u8path(u8"20261104_123000_v1v2_LSJ_5fold평가");
    std::ifstream features(hye / "20261104_173000_pin_random5fold/results_with_bars/features.tsv");
    std::ifstream predicted(v1 / "results/predictions.tsv");
    std::ifstream paths(v1 / "input.tsv");
    if (!features || !predicted || !paths) throw std::runtime_error("Missing recorded observations");
    std::string line;
    std::vector<std::string> v1Paths;
    std::getline(paths, line);
    while (std::getline(paths, line)) {
        auto path = tabFields(line).at(1);
        std::replace(path.begin(), path.end(), '\\', '/');
        const auto offset = path.find("data/clean/");
        v1Paths.push_back(path.substr(offset));
    }
    std::map<std::string, std::array<double, 2>> predictions;
    std::getline(predicted, line);
    while (std::getline(predicted, line)) {
        const auto row = tabFields(line);
        if (row.at(0) != "joint_v1_retrained_GLS" || !row.at(10).empty()) continue;
        predictions.emplace(v1Paths.at(std::stoi(row.at(4))),
                            std::array<double, 2>{std::stod(row.at(7)), std::stod(row.at(8))});
    }

    std::map<std::pair<std::string, std::string>, std::vector<Frame>> groups;
    std::getline(features, line);
    while (std::getline(features, line)) {
        const auto row = tabFields(line);
        if (row.at(1) == "20261014" || std::abs(std::stod(row.at(6))) > 1 || !row.at(14).empty()) continue;
        const std::string path = row.at(2);
        Frame frame{repeatNumbers.at(path), std::stod(row.at(5)), std::stod(row.at(6))};
        std::string values = row.at(8);
        std::replace(values.begin(), values.end(), ';', ',');
        const auto observed = a0::split_csv(values);
        if (observed.size() != 102) throw std::runtime_error("Unexpected number of joint components");
        for (int point = 0; point < 27; ++point) frame.values[point] = std::stod(observed[2 * point + 1]);
        for (int block = 0; block < 48; ++block) frame.values[27 + block] = std::stod(observed[54 + block]);
        const double qr = std::accumulate(frame.values.begin(), frame.values.begin() + 27, 0.0) / 27;
        const double bars = std::accumulate(frame.values.begin() + 27, frame.values.begin() + 75, 0.0) / 48;
        frame.values[75] = qr - bars;
        frame.values[76] = predictions.at(path)[0];
        frame.values[77] = predictions.at(path)[1];
        frame.values[78] = qr;
        frame.values[79] = bars;
        groups[{row.at(1), row.at(4)}].push_back(frame);
    }

    const std::vector<std::string> channels = {"QR_point_dy_px", "bar_block_dy_px",
        "QR_minus_bars_px", "joint_Y_mm", "joint_T_command_unit", "QR_mean_dy_px", "bar_mean_dy_px"};
    std::map<std::string, std::vector<TripletStats>> byDate;
    std::vector<double> timing;
    using Errors = std::array<std::vector<double>, 2>;
    std::map<std::pair<std::string, int>, Errors> averageErrors;
    std::ofstream table(out / "repeat_stats.tsv");
    std::ofstream means(out / "triplet_frames.tsv");
    if (!table || !means) throw std::runtime_error("Cannot write output");
    means << "date\tgroup\trepeat\tQR_mean_dy_px\tbar_mean_dy_px\tQR_minus_bars_px\tjoint_Y\tjoint_T\n";
    std::ofstream exclusions(out / "non_triplet_groups.tsv");
    if (!exclusions) throw std::runtime_error("Cannot write exclusions");
    exclusions << "date\tgroup\timages\n";
    int count = 0;
    for (auto& [key, frames] : groups) {
        if (frames.size() != 3) {
            exclusions << key.first << '\t' << key.second << '\t' << frames.size() << '\n';
            continue;
        }
        ++count;
        std::sort(frames.begin(), frames.end(), [](const Frame& a, const Frame& b) { return a.repeat < b.repeat; });
        if (frames[0].repeat != 1 || frames[1].repeat != 2 || frames[2].repeat != 3)
            throw std::runtime_error("Unexpected repeat IDs");
        const auto& [date, group] = key;
        auto& day = byDate.try_emplace(date, channels.size()).first->second;
        auto& all = byDate.try_emplace("all", channels.size()).first->second;
        for (const auto& frame : frames)
            means << date << '\t' << group << '\t' << frame.repeat << '\t' << frame.values[78]
                  << '\t' << frame.values[79] << '\t' << frame.values[75] << '\t'
                  << frame.values[76] << '\t' << frame.values[77] << '\n';
        for (int mask = 1; mask < 8; ++mask) {
            int n = 0;
            double y = 0, t = 0;
            for (int i = 0; i < 3; ++i) {
                if (!(mask & (1 << i))) continue;
                ++n;
                y += frames[i].values[76] - frames[i].y;
                t += frames[i].values[77] - frames[i].t;
            }
            for (const std::string& bucket : {date, std::string("all")}) {
                auto& errors = averageErrors[{bucket, n}];
                errors[0].push_back(y / n);
                errors[1].push_back(t / n);
            }
        }
        const auto start = Clock::now();
        for (int i = 0; i < 80; ++i) {
            const int channel = i < 27 ? 0 : i < 75 ? 1 : i - 73;
            std::array<double, 3> sample{frames[0].values[i], frames[1].values[i], frames[2].values[i]};
            day[channel].add(sample);
            all[channel].add(sample);
        }
        timing.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count() / 3);
    }
    if (!count) throw std::runtime_error("No complete triplets");
    table << "date\tchannel\ttriplets_or_feature_triplets\tSD\t6SD\tmedian_abs_adjacent"
             "\tp95_abs_group_residual\tp95_group_range\tmax_group_range\n";
    table << std::setprecision(10);
    for (const auto& [date, stats] : byDate)
        for (size_t channel = 0; channel < channels.size(); ++channel) {
            table << date << '\t' << channels[channel] << '\t';
            stats[channel].write(table);
            table << '\n';
        }
    std::ofstream average(out / "averaging_predictions.tsv");
    if (!average) throw std::runtime_error("Cannot write averaging results");
    average << "date\timages_per_average\tcombinations\tSD_Y\tSD_T\tmax_abs_Y\tmax_abs_T\tpass_Y\tpass_T\tpass_both\tSD_Y_vs_sqrtN\tSD_T_vs_sqrtN\n";
    average << std::setprecision(10);
    for (const auto& [key, errors] : averageErrors) {
        const auto& [date, n] = key;
        const auto& baseline = averageErrors.at({date, 1});
        int py = 0, pt = 0, both = 0;
        double my = 0, mt = 0;
        for (size_t i = 0; i < errors[0].size(); ++i) {
            const double y = std::abs(errors[0][i]), t = std::abs(errors[1][i]);
            my = std::max(my, y);
            mt = std::max(mt, t);
            py += y <= 0.045 + 1e-12;
            pt += t <= 0.01 + 1e-12;
            both += y <= 0.045 + 1e-12 && t <= 0.01 + 1e-12;
        }
        const double sy = a0::sd(errors[0]), st = a0::sd(errors[1]);
        average << date << '\t' << n << '\t' << errors[0].size() << '\t' << sy << '\t' << st
                << '\t' << my << '\t' << mt << '\t' << py << '\t' << pt << '\t' << both
                << '\t' << sy * std::sqrt(n) / a0::sd(baseline[0])
                << '\t' << st * std::sqrt(n) / a0::sd(baseline[1]) << '\n';
    }
    std::cout << "Triplets " << count << "; aggregation ms/image median " << a0::median(timing)
              << "; p95 " << a0::quantile(timing, 0.95) << '\n';
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
