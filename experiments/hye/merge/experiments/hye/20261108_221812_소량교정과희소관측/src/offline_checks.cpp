#include "cache.hpp"

#include <array>
#include <chrono>
#include <random>
#include <iomanip>
#include <iostream>

namespace {
using Path = std::filesystem::path;
struct Error {
    int id;
    std::string date;
    double y, t, ey, et;
};

std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> result;
    size_t begin = 0, end;
    while ((end = line.find('\t', begin)) != std::string::npos) {
        result.push_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
    result.push_back(line.substr(begin));
    return result;
}

std::vector<Error> loadErrors(const Path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Missing held-out predictions");
    std::string line;
    std::getline(in, line);
    std::vector<Error> data;
    while (std::getline(in, line)) {
        const auto row = fields(line);
        if (row.at(1) != "1014_to_date" || row.at(2) != "local") continue;
        const double y = std::stod(row.at(6)), t = std::stod(row.at(7));
        data.push_back({std::stoi(row.at(4)), row.at(0), y, t,
                        std::stod(row.at(8)) - y, std::stod(row.at(9)) - t});
    }
    if (data.size() != 291) throw std::runtime_error("Expected 57 + 234 held-out images");
    return data;
}

double normalized(const Error& row) {
    return std::max(std::abs(row.ey) / 0.045, std::abs(row.et) / 0.01);
}

void intervalChecks(const a0::Cache& cache, const std::vector<Error>& data, const Path& output) {
    std::ofstream out(output / "cross_date_intervals.tsv");
    out << "cal_date\ttest_date\talpha\tcal_groups\tq\tY_halfwidth\tT_halfwidth"
           "\ttest_n\tvalid\tcovered\tstop_candidates\tfalse_stops\n";
    for (const std::string date : {"20261020", "20261029"}) {
        std::map<int, double> worst;
        std::string testDate;
        for (const auto& row : data) {
            if (row.date != date) testDate = row.date;
            else if (std::isfinite(row.ey + row.et))
                worst[cache.shots.at(row.id).group] = std::max(worst[cache.shots.at(row.id).group], normalized(row));
        }
        std::vector<double> scores;
        for (const auto& [group, score] : worst) scores.push_back(score);
        std::sort(scores.begin(), scores.end());
        for (double alpha : {0.05, 0.10, 0.20}) {
            const size_t rank = static_cast<size_t>(std::ceil((scores.size() + 1) * (1 - alpha)));
            const double q = rank <= scores.size() ? scores.at(rank - 1) : std::numeric_limits<double>::infinity();
            int n = 0, valid = 0, covered = 0, stopped = 0, falseStop = 0;
            for (const auto& row : data) if (row.date == testDate) {
                ++n;
                if (!std::isfinite(row.ey + row.et)) continue;
                ++valid;
                covered += normalized(row) <= q;
                const bool stop = std::abs(row.y + row.ey) + 0.045*q <= 0.045
                               && std::abs(row.t + row.et) + 0.01*q <= 0.01;
                stopped += stop;
                falseStop += stop && (std::abs(row.y) > 0.045 || std::abs(row.t) > 0.01);
            }
            out << date << '\t' << testDate << '\t' << alpha << '\t' << scores.size() << '\t' << q
                << '\t' << 0.045*q << '\t' << 0.01*q << '\t' << n << '\t' << valid << '\t'
                << covered << '\t' << stopped << '\t' << falseStop << '\n';
        }
    }
}

void rangeChecks(const a0::Cache& cache, const Path& output) {
    std::ofstream out(output / "range_gate.tsv");
    out << "gate\tthreshold\ttest_in_range\tin_range_flagged\tout_of_range\tout_of_range_flagged\n";
    for (const std::string gate : {"valid536", "min_ncc", "valid536_or_min_ncc"}) {
        std::vector<double> scores;
        const auto ncc = [](const a0::Shot& s) { return *std::min_element(s.ncc, s.ncc + 3); };
        for (const auto& shot : cache.shots)
            if (shot.set == "clean_20261014" && shot.condition != "center") scores.push_back(ncc(shot));
        const double q = a0::quantile(scores, 0.05);
        int n[2] = {}, flagged[2] = {};
        for (const auto& shot : cache.shots) {
            if (shot.set == "clean_20261014" || shot.condition == "center") continue;
            const bool outside = std::abs(shot.t) > 1.0 || std::abs(shot.y) > 2.0;
            const bool failCount = shot.valid_selected != 536;
            const bool failScore = ncc(shot) < q;
            const bool fail = gate == "valid536" ? failCount : gate == "min_ncc" ? failScore : failCount || failScore;
            ++n[outside];
            flagged[outside] += fail;
        }
        out << gate << '\t' << (gate == "valid536" ? 536 : q)
            << '\t' << n[0] << '\t' << flagged[0] << '\t' << n[1] << '\t' << flagged[1] << '\n';
    }
}

struct Response {
    double command, dx, readMs;
};
using ResponseGroup = std::vector<Response>;

cv::Mat fitResponse(const std::vector<ResponseGroup>& groups, const std::vector<size_t>& ids, int degree) {
    cv::Mat x, z;
    for (const auto id : ids) for (const auto& row : groups[id]) {
        cv::Mat design(1, degree + 1, CV_64F);
        design.at<double>(0) = 1;
        design.at<double>(1) = row.command;
        if (degree == 2) design.at<double>(2) = row.command * row.command;
        x.push_back(design);
        z.push_back(row.dx);
    }
    cv::Mat coefficients;
    cv::solve(x, z, coefficients, cv::DECOMP_SVD);
    return coefficients;
}

void responseChecks(const a0::Cache& cache, const Path& output) {
    std::ofstream out(output / "command_response.tsv"), bands(output / "command_response_bins.tsv");
    out << "date\taxis\tfixed_other_axis\tphotos\tposes\tslope\tslope_lo95\tslope_hi95\t"
           "linear_rms_px\tlinear_max_px\tquadratic_rms_px\tAIC_linear_minus_quadratic\t"
           "cached_feature_ms_median\tcached_feature_ms_p95\n";
    bands << "date\taxis\tcommand_bin_lower\tcommand_bin_upper\tphotos\tmean_linear_residual_px\n";
    for (const std::string date : {"20261014", "20261020", "20261029"}) {
        for (int axis : {0, 1}) {
            const double fixed = date == "20261029" && axis == 1 ? 0.05 : 0.0;
            std::map<int, ResponseGroup> matched;
            for (const auto& shot : cache.shots) {
                if (shot.set != "clean_" + date || shot.condition == "center" ||
                    std::abs((axis == 0 ? shot.t : shot.y) - fixed) > 1e-9) continue;
                const auto begin = std::chrono::steady_clock::now();
                double sum = 0;
                int valid = 0;
                for (int i = 0; i < a0::kPoints; ++i) {
                    if (!cache.points[i].selected || !cache.v(shot.idx)[i]) continue;
                    sum += cache.dz(shot.idx, i).x;
                    ++valid;
                }
                const auto elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                if (valid != 536) continue;
                matched[shot.group].push_back({axis == 0 ? shot.y : shot.t, sum / valid, elapsed});
            }
            if (matched.size() < 3) continue;
            std::vector<ResponseGroup> groups;
            for (auto& entry : matched) groups.push_back(std::move(entry.second));
            std::vector<size_t> ids(groups.size());
            std::iota(ids.begin(), ids.end(), 0);
            const cv::Mat linear = fitResponse(groups, ids, 1), quadratic = fitResponse(groups, ids, 2);
            std::vector<double> bootstrap, time;
            std::mt19937 generator(20261108);
            for (int iteration = 0; iteration < 1000; ++iteration) {
                auto sample = ids;
                for (auto& id : sample) id = generator() % groups.size();
                bootstrap.push_back(fitResponse(groups, sample, 1).at<double>(1));
            }
            const auto [low, high] = std::minmax_element(groups.begin(), groups.end(),
                [](const ResponseGroup& a, const ResponseGroup& b) { return a.front().command < b.front().command; });
            const double lo = low->front().command, hi = high->front().command;
            constexpr int binCount = 8;
            double squaredLinear = 0, squaredQuadratic = 0, maxLinear = 0;
            std::array<double, binCount> binSum{};
            std::array<int, binCount> binN{};
            for (const auto& group : groups) for (const auto& row : group) {
                const double x = row.command;
                const double e1 = row.dx - linear.at<double>(0) - linear.at<double>(1)*x;
                const double e2 = row.dx - quadratic.at<double>(0) - quadratic.at<double>(1)*x - quadratic.at<double>(2)*x*x;
                squaredLinear += e1*e1;
                squaredQuadratic += e2*e2;
                maxLinear = std::max(maxLinear, std::abs(e1));
                time.push_back(row.readMs);
                const int bin = std::min(binCount - 1, static_cast<int>((x-lo)/(hi-lo)*binCount));
                binSum[bin] += e1;
                ++binN[bin];
            }
            const char* name = axis == 0 ? "Y" : "T";
            const size_t n = time.size();
            out << date << '\t' << name << '\t' << fixed << '\t' << n << '\t' << groups.size()
                << '\t' << linear.at<double>(1) << '\t' << a0::quantile(bootstrap, 0.025) << '\t' << a0::quantile(bootstrap, 0.975)
                << '\t' << std::sqrt(squaredLinear/n) << '\t' << maxLinear << '\t' << std::sqrt(squaredQuadratic/n)
                << '\t' << n*std::log(squaredLinear/squaredQuadratic)-2.0
                << '\t' << a0::median(time) << '\t' << a0::quantile(time, 0.95) << '\n';
            for (int bin = 0; bin < binCount; ++bin) if (binN[bin])
                bands << date << '\t' << name << '\t' << lo+(hi-lo)*bin/binCount << '\t'
                      << lo+(hi-lo)*(bin+1)/binCount << '\t' << binN[bin] << '\t' << binSum[bin]/binN[bin] << '\n';
        }
    }
}

bool within(double y, double t, double margin = 1.0) {
    return std::abs(y) <= 0.045 * margin && std::abs(t) <= 0.01 * margin;
}
struct Simulation {
    int moves = 0;
    bool stop = false, pass = false, stalled = false;
};

Simulation simulate(const Error& row, const std::string& order, double gain, double guard, bool disturbed) {
    // Sensitivity test only: fixed held-out bias, ideal or explicitly assumed motion.
    double y = row.y, t = row.t;
    double prevY = 0, prevT = 0;
    bool enteredY = false;
    Simulation result;
    if (!std::isfinite(row.ey + row.et)) return result;
    for (int turn = 0; turn <= 20; ++turn) {
        const double yhat = y + row.ey, that = t + row.et;
        if (within(yhat, that, guard)) {
            result.stop = true;
            result.pass = within(y, t);
            break;
        }
        if (turn == 20) break;
        double moveY = -std::clamp(gain * yhat, -0.5, 0.5);
        double moveT = -std::clamp(gain * that, -0.25, 0.25);
        if (order != "joint") {
            const double cutoff = guard * 0.01;
            const bool doT = !enteredY ? std::abs(that) > cutoff
                                      : std::abs(that) > (order == "T_then_Y" ? cutoff : 1.5 * cutoff);
            if (doT) moveY = 0;
            else {
                enteredY = true;
                moveT = 0;
                if (std::abs(yhat) <= guard * 0.045) {
                    moveY = 0;
                    moveT = -std::clamp(gain * that, -0.25, 0.25);
                }
            }
        }
        if (disturbed) {
            moveY = std::round(moveY / 0.05) * 0.05;
            moveT = std::round(moveT / 0.01) * 0.01;
        }
        if (moveY == 0 && moveT == 0) {
            result.stalled = true;
            break;
        }
        if (disturbed) {
            const double effectiveY = moveY * 0.9 - (prevY * moveY < 0 ? std::copysign(0.015, moveY) : 0.0);
            const double effectiveT = moveT * 0.9 - (prevT * moveT < 0 ? std::copysign(0.003, moveT) : 0.0);
            y += effectiveY + 0.08 * effectiveT;
            t += effectiveT + 0.01 * effectiveY;
        } else {
            y += moveY;
            t += moveT;
        }
        if (moveY != 0) prevY = moveY;
        if (moveT != 0) prevT = moveT;
        ++result.moves;
    }
    return result;
}

void motionChecks(const std::vector<Error>& data, const Path& output) {
    std::ofstream out(output / "motion_sensitivity.tsv");
    out << "world\tpolicy\tgain\tguard\tn\tpass\tfalse_stop\tstalled\tunusable_input\tmean_moves\n";
    for (const bool disturbed : {false, true})
        for (const std::string policy : {"joint", "T_then_Y", "hysteresis"})
            for (double gain : {0.5, 0.8, 1.0})
                for (double guard : {0.5, 0.75, 1.0}) {
                    int passes = 0, falseStops = 0, stalled = 0, missing = 0, moves = 0;
                    for (const auto& row : data) {
                        const auto run = simulate(row, policy, gain, guard, disturbed);
                        passes += run.pass;
                        falseStops += run.stop && !run.pass;
                        stalled += run.stalled;
                        missing += !std::isfinite(row.ey + row.et);
                        moves += run.moves;
                    }
                    out << (disturbed ? "assumed_disturbed" : "ideal") << '\t' << policy << '\t' << gain
                        << '\t' << guard << '\t' << data.size() << '\t' << passes << '\t' << falseStops
                        << '\t' << stalled << '\t' << missing << '\t' << double(moves) / data.size() << '\n';
                }
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 4) throw std::runtime_error("Usage: sparse_offline_checks A0_CACHE PREDICTIONS OUTPUT");
    const auto cache = a0::load(std::filesystem::u8path(argv[1]));
    const auto data = loadErrors(std::filesystem::u8path(argv[2]));
    const auto output = std::filesystem::u8path(argv[3]);
    std::filesystem::create_directories(output);
    intervalChecks(cache, data, output);
    rangeChecks(cache, output);
    responseChecks(cache, output);
    motionChecks(data, output);
    std::cout << "Checked range gates, six interval transfers and 54 assumed motion conditions.\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
