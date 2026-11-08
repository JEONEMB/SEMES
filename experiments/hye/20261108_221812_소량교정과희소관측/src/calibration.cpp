#include "calibration.hpp"
#include "cache.hpp"

#include <array>
#include <chrono>
#include <iomanip>
#include <random>
#include <set>
#include <tuple>

namespace semes_study {
namespace {
using a0::Cache;
using a0::Shot;
using Path = std::filesystem::path;
using Ids = std::vector<int>;
using Points = std::vector<int>;
using Groups = std::vector<Ids>;

struct Spec {
    std::string name;
    Points points, blocks;
    bool geometry;
};

struct Model {
    cv::Mat coefficients;  // feature = intercept + Y * slopeY + T * slopeT
    cv::Mat scale;
    cv::Mat inverse;
    int axis = -1;         // -1: both; 0 or 1: prescribed single-axis protocol
    double fixed = 0;
    double condition = 0;
};

struct Score {
    int n = 0, valid = 0, yPass = 0, tPass = 0, both = 0;
    double ySum = 0, tSum = 0, yMax = 0, tMax = 0;
};

Points spread(const Cache& cache, int count) {
    if (count == 536) {
        Points selected;
        for (int i = 0; i < a0::kPoints; ++i)
            if (cache.points[i].selected) selected.push_back(i);
        if (selected.size() != 536) throw std::runtime_error("Expected 536 eligible points");
        return selected;
    }
    Points result;
    for (int region = 0; region < 3; ++region) {
        Points pool;
        double cx = 0, cy = 0;
        for (int i = 0; i < a0::kPoints; ++i) {
            const auto& p = cache.points[i];
            if (p.selected && p.region == region) {
                pool.push_back(i);
                cx += p.x;
                cy += p.y;
            }
        }
        if (pool.size() < static_cast<size_t>(count / 3)) throw std::runtime_error("Point region too small");
        cx /= pool.size();
        cy /= pool.size();
        Points chosen;
        const auto distance = [&](int i, double x, double y) {
            return std::hypot(cache.points[i].x - x, cache.points[i].y - y);
        };
        while (static_cast<int>(chosen.size()) < count / 3) {
            const auto score = [&](int i) {
                if (chosen.empty()) return -distance(i, cx, cy);
                double nearest = std::numeric_limits<double>::infinity();
                for (int other : chosen) nearest = std::min(nearest, distance(i, cache.points[other].x, cache.points[other].y));
                return nearest;
            };
            const auto best = std::max_element(pool.begin(), pool.end(), [&](int a, int b) { return score(a) < score(b); });
            chosen.push_back(*best);
            pool.erase(best);
        }
        result.insert(result.end(), chosen.begin(), chosen.end());
    }
    return result;
}

Points lineBlocks(int count) {
    Points blocks;
    for (int side = 0; side < 2; ++side)
        for (int i = 0; i < count / 2; ++i)
            blocks.push_back(24 * side + static_cast<int>(std::lround(23.0 * i / (count / 2 - 1))));
    return blocks;
}

double slope(const std::vector<double>& x, const std::vector<double>& y) {
    const double meanX = a0::mean(x), meanY = a0::mean(y);
    double cov = 0, var = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        cov += (x[i] - meanX) * (y[i] - meanY);
        var += (x[i] - meanX) * (x[i] - meanX);
    }
    if (var <= 0) throw std::runtime_error("Degenerate observation geometry");
    return cov / var;
}

cv::Mat observe(const Cache& cache, const Shot& shot, const Spec& spec) {
    std::vector<double> dx, dy, x, y;
    for (int p : spec.points) {
        if (!cache.v(shot.idx)[p]) return {};
        const auto d = cache.dz(shot.idx, p);
        if (!std::isfinite(d.x) || !std::isfinite(d.y)) return {};
        dx.push_back(d.x);
        dy.push_back(d.y);
        x.push_back(cache.points[p].x / 100.0);
        y.push_back(cache.points[p].y / 100.0);
    }
    std::vector<double> features{a0::mean(dx)};
    if (spec.geometry) {
        features.push_back(slope(x, dx));
        features.push_back(slope(x, dy));
        features.push_back(slope(y, dx));
    }
    if (!spec.blocks.empty()) {
        std::array<std::vector<double>, 2> abscissa, shift;
        for (int b : spec.blocks) {
            if (!cache.line_ok(shot.idx)[b] || !std::isfinite(cache.line(shot.idx)[2*b])) return {};
            abscissa[b / 24].push_back(b % 24);
            shift[b / 24].push_back(cache.line(shot.idx)[2*b]);
        }
        features.push_back(slope(abscissa[0], shift[0]) - slope(abscissa[1], shift[1]));
    }
    cv::Mat row(1, static_cast<int>(features.size()), CV_64F);
    std::copy(features.begin(), features.end(), row.ptr<double>());
    return row;
}

cv::Mat design(const Shot& shot) {
    return (cv::Mat_<double>(1, 3) << 1.0, shot.y, shot.t);
}

int matrixRank(const cv::Mat& matrix, double tolerance = 1e-8) {
    if (matrix.empty()) return 0;
    cv::Mat singular;
    cv::SVD::compute(matrix, singular);
    int rank = 0;
    for (int i = 0; i < singular.rows; ++i)
        rank += singular.at<double>(i) > tolerance * singular.at<double>(0);
    return rank;
}

void pack(const Cache& cache, const std::vector<cv::Mat>& features, const Ids& ids, cv::Mat& x, cv::Mat& z) {
    for (int id : ids) {
        if (features.at(id).empty()) continue;
        x.push_back(design(cache.shots.at(id)));
        z.push_back(features.at(id));
    }
}

cv::Mat residualScales(const cv::Mat& residual) {
    cv::Mat scale(1, residual.cols, CV_64F);
    for (int i = 0; i < residual.cols; ++i) {
        const double rms = cv::norm(residual.col(i)) / std::sqrt(residual.rows);
        scale.at<double>(i) = std::max(rms, 1e-5);
    }
    return scale;
}

Model finalize(cv::Mat coefficients, cv::Mat scale, int axis = -1, double fixed = 0) {
    Model model;
    if (coefficients.empty()) return model;
    cv::Mat sensitivity = axis < 0 ? coefficients.rowRange(1, 3).clone() : coefficients.row(axis + 1).clone();
    for (int i = 0; i < sensitivity.cols; ++i) sensitivity.col(i) /= scale.at<double>(i);
    if (matrixRank(sensitivity) < sensitivity.rows) return model;
    cv::Mat singular;
    cv::SVD::compute(sensitivity, singular);
    model.condition = singular.at<double>(0) / singular.at<double>(singular.rows - 1);
    if (!std::isfinite(model.condition) || model.condition > 1e5) return {};
    cv::Mat inverse;
    cv::invert(sensitivity.t(), inverse, cv::DECOMP_SVD);
    model.inverse = inverse.t();
    model.coefficients = std::move(coefficients);
    model.scale = std::move(scale);
    model.axis = axis;
    model.fixed = fixed;
    return model;
}

Model fit(const cv::Mat& x, const cv::Mat& z, const std::string& method,
          const Model& prior, int axis = -1, double fixed = 0) {
    if (method == "source_only") return prior;
    if (x.empty()) return {};
    cv::Mat coef, scale;
    if (method == "local" || method == "unweighted" || method == "local_axis") {
        const cv::Mat independent = axis < 0 ? x : cv::Mat();
        cv::Mat singleAxis;
        if (axis >= 0) cv::hconcat(x.col(0), x.col(axis + 1), singleAxis);
        const auto& predictors = axis < 0 ? independent : singleAxis;
        if (matrixRank(predictors) < predictors.cols) return {};
        cv::Mat small;
        if (predictors.rows != z.rows) throw std::runtime_error("Unexpected fit shapes " + std::to_string(predictors.rows) + "x" + std::to_string(predictors.cols) + " and " + std::to_string(z.rows) + "x" + std::to_string(z.cols));
        cv::solve(predictors, z, small, cv::DECOMP_SVD);
        if (axis < 0) coef = small;
        else {
            coef = cv::Mat::zeros(3, z.cols, CV_64F);
            small.row(0).copyTo(coef.row(0));
            small.row(1).copyTo(coef.row(axis + 1));
        }
        scale = method == "unweighted" ? cv::Mat::ones(1, z.cols, CV_64F) : residualScales(z - predictors * small);
    } else {
        if (prior.coefficients.empty()) return {};
        coef = prior.coefficients.clone();
        scale = prior.scale;
        cv::Mat residual = z - x * coef;
        if (method == "offset") {
            cv::Mat mean;
            cv::reduce(residual, mean, 0, cv::REDUCE_AVG);
            coef.row(0) += mean;
        } else {
            // Fixed penalty on changes to the two sensitivities. Intercept is free.
            cv::Mat penalty = cv::Mat::zeros(2, 3, CV_64F);
            penalty.at<double>(0, 1) = 1.0;
            penalty.at<double>(1, 2) = 1.0;
            cv::Mat regularizedX, regularizedZ, delta;
            cv::vconcat(x, penalty, regularizedX);
            cv::vconcat(residual, cv::Mat::zeros(2, residual.cols, CV_64F), regularizedZ);
            cv::solve(regularizedX, regularizedZ, delta, cv::DECOMP_SVD);
            coef += delta;
        }
    }
    return finalize(coef, scale, axis, fixed);
}

cv::Vec2d predict(const Model& model, const cv::Mat& features) {
    cv::Mat input = features - model.coefficients.row(0);
    for (int i = 0; i < input.cols; ++i) input.at<double>(i) /= model.scale.at<double>(i);
    const cv::Mat value = input * model.inverse;
    if (model.axis == 0) return {value.at<double>(0), model.fixed};
    if (model.axis == 1) return {model.fixed, value.at<double>(0)};
    return {value.at<double>(0), value.at<double>(1)};
}

Ids choose(const Cache& cache, const Groups& candidates, int count, bool balanced) {
    Ids pool, order;
    for (const auto& group : candidates) pool.push_back(group.front());
    while (!pool.empty() && static_cast<int>(order.size()) < count) {
        auto best = pool.begin();
        if (balanced) {
            const auto value = [&](int id) {
                const Shot& s = cache.shots[id];
                if (order.empty()) return -std::hypot(s.y / 2, s.t);
                double distance = std::numeric_limits<double>::infinity();
                for (int other : order) {
                    const Shot& a = cache.shots[other];
                    distance = std::min(distance, std::hypot((s.y-a.y)/2, s.t-a.t));
                }
                return distance;
            };
            best = std::max_element(pool.begin(), pool.end(), [&](int a, int b) { return value(a) < value(b); });
        }
        order.push_back(*best);
        pool.erase(best);
    }
    return order;
}

int rankFor(const Cache& cache, const Ids& ids) {
    cv::Mat x;
    for (int id : ids) x.push_back(design(cache.shots[id]));
    return std::max(0, matrixRank(x) - 1);
}

void experiment(const Cache& cache, const std::vector<cv::Mat>& features, const Spec& spec,
                const std::string& date, const std::string& protocol, int seed, const std::string& layout,
                const Ids& calibration, const Ids& test, const Model& prior, const std::string& method,
                std::ostream& results, std::ostream& predictions, int axis = -1, double fixed = 0) {
    cv::Mat x, z;
    pack(cache, features, calibration, x, z);
    Model model;
    try {
        model = fit(x, z, method, prior, axis, fixed);
    } catch (const cv::Exception& error) {
        throw std::runtime_error(spec.name + " " + date + " " + protocol + " " + method +
            " (" + std::to_string(x.rows) + " design, " + std::to_string(z.rows) +
            " observations): " + error.what());
    }
    Score score;
    for (int id : test) {
        ++score.n;
        const Shot& shot = cache.shots[id];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        cv::Vec2d prediction{nan, nan};
        if (!model.coefficients.empty() && !features[id].empty()) prediction = predict(model, features[id]);
        if (std::isfinite(prediction[0]) && std::isfinite(prediction[1])) {
            ++score.valid;
            const double y = std::abs(prediction[0] - shot.y), t = std::abs(prediction[1] - shot.t);
            score.yPass += y <= 0.045 + 1e-12;
            score.tPass += t <= 0.01 + 1e-12;
            score.both += y <= 0.045 + 1e-12 && t <= 0.01 + 1e-12;
            score.ySum += y;
            score.tSum += t;
            score.yMax = std::max(score.yMax, y);
            score.tMax = std::max(score.tMax, t);
        }
        if ((seed == 1 && layout == "diverse" && spec.name == "qr27_bar6") ||
            (seed == 0 && protocol == "1014_to_date" && spec.name == "qr9_bar48"))
            predictions << date << '\t' << protocol << '\t' << method << '\t' << calibration.size() << '\t'
                        << id << '\t' << shot.rel << '\t' << shot.y << '\t' << shot.t
                        << '\t' << prediction[0] << '\t' << prediction[1] << '\n';
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    results << date << '\t' << protocol << '\t' << seed << '\t' << layout << '\t' << spec.name << '\t'
            << method << '\t' << calibration.size() << '\t' << x.rows << '\t' << rankFor(cache, calibration)
            << '\t' << score.n << '\t' << score.valid << '\t' << score.yPass << '\t' << score.tPass
            << '\t' << score.both << '\t' << (score.valid ? score.ySum / score.valid : nan)
            << '\t' << (score.valid ? score.tSum / score.valid : nan)
            << '\t' << (score.valid ? score.yMax : nan) << '\t' << (score.valid ? score.tMax : nan)
            << '\t' << (model.coefficients.empty() ? nan : model.condition) << '\n';
}
} // namespace

void run(const Path& directory, const Path& output) {
    const Cache cache = a0::load(directory);
    if (cache.n() != 575) throw std::runtime_error("Unexpected A0 image count");
    std::filesystem::create_directories(output);
    std::ofstream results(output / "runs.tsv"), splits(output / "splits.tsv"), predictions(output / "predictions.tsv"),
        layouts(output / "observation_layouts.tsv"), coverage(output / "observation_coverage.tsv");
    if (!results || !splits || !predictions || !layouts || !coverage) throw std::runtime_error("Cannot write results");
    results << std::setprecision(12)
            << "date\tprotocol\tseed\tlayout\tfeatures\tmethod\tcal_photos\tvalid_cal\tlocal_rank\ttest\tvalid\tpass_y\tpass_t\tpass_both\tmae_y\tmae_t\tmax_y\tmax_t\tcondition\n";
    splits << "date\tprotocol\tseed\trole\tlayout\torder\tid\tgroup\tY\tT\n";
    predictions << "date\tprotocol\tmethod\tcal_photos\tid\tpath\tY\tT\tpred_y\tpred_t\n" << std::setprecision(12);
    layouts << "features\ttype\tindex\n";
    coverage << "features\tdate\tvalid\ttotal\n";

    const Points p9 = spread(cache, 9), p27 = spread(cache, 27), p81 = spread(cache, 81), p536 = spread(cache, 536);
    std::vector<Spec> specs{
        {"qr9_mean", p9, {}, false}, {"qr27_geometry", p27, {}, true},
        {"qr9_bar6", p9, lineBlocks(6), false}, {"qr27_bar6", p27, lineBlocks(6), false},
        {"qr81_bar12", p81, lineBlocks(12), false}, {"qr536_bar48", p536, lineBlocks(48), false},
        {"qr27_geometry_bar6", p27, lineBlocks(6), true}};
    for (int n : {12, 24, 48}) {
        specs.push_back({"qr9_bar" + std::to_string(n), p9, lineBlocks(n), false});
        specs.push_back({"qr27_bar" + std::to_string(n), p27, lineBlocks(n), false});
        if (n != 48) specs.push_back({"qr536_bar" + std::to_string(n), p536, lineBlocks(n), false});
    }
    specs.push_back({"qr536_bar6", p536, lineBlocks(6), false});
    for (const Spec& spec : specs) {
        for (int id : spec.points) layouts << spec.name << "\tQR\t" << id << '\n';
        for (int id : spec.blocks) layouts << spec.name << "\tbar\t" << id << '\n';
        std::vector<cv::Mat> features;
        for (const Shot& shot : cache.shots) {
            if (shot.idx != static_cast<int>(features.size())) throw std::runtime_error("Cache order mismatch");
            features.push_back(observe(cache, shot, spec));
        }
        for (const std::string date : {"20261014", "20261020", "20261029"}) {
            Ids source, allTarget, source1014, centers;
            std::map<int, Ids> targetGroups;
            for (const Shot& shot : cache.shots) {
                if (shot.condition == "center") {
                    if (shot.set == "clean_" + date) centers.push_back(shot.idx);
                    continue;
                }
                if (shot.set == "clean_" + date) {
                    targetGroups[shot.group].push_back(shot.idx);
                    allTarget.push_back(shot.idx);
                } else {
                    source.push_back(shot.idx);
                    if (shot.set == "clean_20261014") source1014.push_back(shot.idx);
                }
            }
            coverage << spec.name << '\t' << date << '\t'
                     << std::count_if(allTarget.begin(), allTarget.end(), [&](int id) { return !features[id].empty(); })
                     << '\t' << allTarget.size() << '\n';
            if (date == "20261014") {
                // Rehearse a prespecified sparse installation capture on available axis data.
                Ids selected{centers.at(0)};
                for (const auto& [y, t] : std::vector<std::pair<double, double>>{
                        {-0.5, 0}, {0.5, 0}, {0, -0.25}, {0, 0.25},
                        {-1, 0}, {1, 0}, {0, -0.5}, {0, 0.5}}) {
                    const auto match = std::find_if(allTarget.begin(), allTarget.end(), [&](int id) {
                        const auto& shot = cache.shots[id];
                        return std::abs(shot.y - y) < 1e-9 && std::abs(shot.t - t) < 1e-9;
                    });
                    if (match == allTarget.end()) throw std::runtime_error("Missing prescribed axial calibration pose");
                    selected.push_back(*match);
                }
                for (int budget : {5, 9}) {
                    const Ids cal(selected.begin(), selected.begin() + budget);
                    for (const std::string region : {"all_axes", "within_calibration_range"}) {
                        Ids test;
                        for (int id : allTarget) {
                            if (std::find(selected.begin(), selected.end(), id) != selected.end()) continue;
                            const Shot& shot = cache.shots[id];
                            if (region == "within_calibration_range" &&
                                (std::abs(shot.y) > (budget == 5 ? 0.5 : 1) ||
                                 std::abs(shot.t) > (budget == 5 ? 0.25 : 0.5))) continue;
                            test.push_back(id);
                        }
                        experiment(cache, features, spec, date, "axial_capture_" + region, 0,
                                   "prespecified_cross", cal, test, {}, "local", results, predictions);
                    }
                }
            }
            if (!source1014.empty()) {
                for (const std::string method : {"local", "unweighted"})
                    experiment(cache, features, spec, date, "1014_to_date", 0, "all_source", source1014,
                               allTarget, {}, method, results, predictions);
            }
            cv::Mat x, z;
            pack(cache, features, source, x, z);
            const Model prior = fit(x, z, "local", {});
            if (!centers.empty()) {
                experiment(cache, features, spec, date, "center_reference", 0, "center_only",
                           {centers.front()}, allTarget, prior, "offset", results, predictions);
                if (centers.size() > 1)
                    experiment(cache, features, spec, date, "center_reference", 0, "center_only",
                               centers, allTarget, prior, "offset", results, predictions);
                experiment(cache, features, spec, date, "center_reference", 0, "center_only",
                           {}, allTarget, prior, "source_only", results, predictions);
            }
            for (const std::string protocol : {"joint", "Y_T0", "T_Yfixed"}) {
                const int axis = protocol == "Y_T0" ? 0 : protocol == "T_Yfixed" ? 1 : -1;
                const double fixed = axis == 1 && date == "20261029" ? 0.05 : 0.0;
                Groups groups;
                for (const auto& [key, group] : targetGroups) {
                    const Shot& shot = cache.shots[group.front()];
                    if (axis == 0 && std::abs(shot.t - fixed) > 1e-9) continue;
                    if (axis == 1 && std::abs(shot.y - fixed) > 1e-9) continue;
                    groups.push_back(group);
                }
                if (groups.size() < 5) continue;
                for (int seed = 1; seed <= 5; ++seed) {
                    Groups shuffled = groups;
                    std::mt19937 generator(20261108u + static_cast<unsigned>(seed));
                    // Fixed Fisher-Yates draw, reproducible across standard libraries.
                    for (size_t i = shuffled.size() - 1; i > 0; --i)
                        std::swap(shuffled[i], shuffled[generator() % (i + 1)]);
                    const size_t poolSize = std::min<size_t>(32, shuffled.size() / 2);
                    const Groups pool(shuffled.begin(), shuffled.begin() + poolSize);
                    Ids test;
                    for (size_t i = poolSize; i < shuffled.size(); ++i)
                        test.insert(test.end(), shuffled[i].begin(), shuffled[i].end());
                    for (const std::string layout : {"diverse", "random"}) {
                        const Ids order = choose(cache, pool, static_cast<int>(poolSize), layout == "diverse");
                        if (spec.name == specs.front().name) {
                            if (layout == "diverse") for (int id : test)
                                splits << date << '\t' << protocol << '\t' << seed << "\ttest\t-\t-\t"
                                       << id << '\t' << cache.shots[id].group << '\t' << cache.shots[id].y << '\t' << cache.shots[id].t << '\n';
                            for (size_t j = 0; j < order.size(); ++j) {
                                const Shot& shot = cache.shots[order[j]];
                                splits << date << '\t' << protocol << '\t' << seed << "\tcal\t" << layout << '\t'
                                       << j + 1 << '\t' << order[j] << '\t' << shot.group << '\t' << shot.y << '\t' << shot.t << '\n';
                            }
                        }
                        if (axis < 0 && layout == "diverse")
                            experiment(cache, features, spec, date, protocol, seed, layout, {}, test, prior,
                                       "source_only", results, predictions);
                        for (int budget : {1, 2, 3, 5, 8, 12, 20, 32}) {
                            if (budget > static_cast<int>(order.size())) continue;
                            const Ids cal(order.begin(), order.begin() + budget);
                            if (axis >= 0) {
                                experiment(cache, features, spec, date, protocol, seed, layout, cal, test, {},
                                           "local_axis", results, predictions, axis, fixed);
                            } else {
                                for (const std::string method : {"local", "offset", "transfer"})
                                    experiment(cache, features, spec, date, protocol, seed, layout, cal, test,
                                               prior, method, results, predictions);
                            }
                        }
                    }
                }
            }
        }
    }
}

namespace {
const char* residualBand(double t) {
    const double magnitude = std::abs(t);
    if (magnitude < 1e-9) return "0";
    if (magnitude <= 0.01 + 1e-9) return "(0,0.01]";
    if (magnitude <= 0.03 + 1e-9) return "(0.01,0.03]";
    if (magnitude <= 0.1 + 1e-9) return "(0.03,0.10]";
    if (magnitude <= 1 + 1e-9) return "(0.10,1.00]";
    return ">1.00";
}

struct RouteScore {
    int test = 0, valid = 0, yPass = 0, tPass = 0, both = 0;
    int usedSingle = 0, singleYMiss = 0, singleTOutside = 0;
    double maxY = 0, maxT = 0;
};

void tally(RouteScore& score, const Shot& shot, const cv::Vec2d& estimate, bool usedSingle) {
    ++score.test;
    if (!std::isfinite(estimate[0]) || !std::isfinite(estimate[1])) return;
    ++score.valid;
    const double ey = std::abs(shot.y - estimate[0]), et = std::abs(shot.t - estimate[1]);
    const bool yPass = ey <= 0.045 + 1e-12, tPass = et <= 0.01 + 1e-12;
    score.yPass += yPass;
    score.tPass += tPass;
    score.both += yPass && tPass;
    score.usedSingle += usedSingle;
    score.singleYMiss += usedSingle && !yPass;
    score.singleTOutside += usedSingle && std::abs(shot.t) > 0.01 + 1e-12;
    score.maxY = std::max(score.maxY, ey);
    score.maxT = std::max(score.maxT, et);
}

cv::Vec2d estimate(const Model& model, const cv::Mat& observation) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (model.coefficients.empty() || observation.empty()) return {nan, nan};
    return predict(model, observation);
}

Ids findPrescribed(const Cache& cache, const std::vector<std::pair<double, double>>& coordinates) {
    Ids result;
    for (const auto& [y, t] : coordinates) {
        const auto found = std::find_if(cache.shots.begin(), cache.shots.end(), [&](const Shot& shot) {
            return shot.set == "clean_20261014" && std::abs(shot.y-y) < 1e-9 && std::abs(shot.t-t) < 1e-9;
        });
        if (found == cache.shots.end()) throw std::runtime_error("Missing prescribed calibration pose");
        result.push_back(found->idx);
    }
    return result;
}
} // namespace

void runResidualT(const Path& directory, const Path& output) {
    const Cache cache = a0::load(directory);
    if (cache.n() != 575) throw std::runtime_error("Unexpected A0 image count");
    std::filesystem::create_directories(output);
    std::ofstream scores(output / "residual_t.tsv"), shots(output / "per_shot.tsv"), training(output / "calibration.tsv");
    if (!scores || !shots || !training) throw std::runtime_error("Cannot create residual-T output");

    const Points points = spread(cache, 9);
    const Spec singleSpec{"qr9_mean", points, {}, false}, jointSpec{"qr9_bar48", points, lineBlocks(48), false};
    std::vector<cv::Mat> singleObs, jointObs;
    for (const Shot& shot : cache.shots) {
        singleObs.push_back(observe(cache, shot, singleSpec));
        jointObs.push_back(observe(cache, shot, jointSpec));
    }
    const Ids yAnchors = findPrescribed(cache, {{0, 0}, {-0.5, 0}, {0.5, 0}, {-1, 0}, {1, 0}});
    const Ids tAnchors = findPrescribed(cache, {{0, -0.25}, {0, 0.25}, {0, -0.50}, {0, 0.50}});
    const std::set<int> reservedY(yAnchors.begin(), yAnchors.end()), reservedT(tAnchors.begin(), tAnchors.end());

    training << "setup\tmodel\tid\tdate\tgroup\tY\tT\n";
    shots << "setup\tid\tdate\tgroup\tY\tT\tY_single\tY_joint\tT_joint\tgate003\tgate005\tgate010\n"
          << std::setprecision(15);
    scores << "setup\tdate\tabs_true_T_band\troute\ttest\tvalid\tpass_Y\tpass_T\tpass_both"
              "\tused_single\tsingle_Y_miss\tsingle_true_abs_T_gt_0.01\tmax_Y\tmax_T\n"
           << std::setprecision(12);
    const std::array<double, 3> gates{0.003, 0.005, 0.010};
    const std::array<std::string, 3> gateNames{"guard_0.003", "guard_0.005", "guard_0.010"};
    for (const std::string setup : {"small", "wide", "full_cross_date"}) {
        Ids yTrain, jointTrain;
        if (setup == "full_cross_date") {
            for (const auto& shot : cache.shots) if (shot.set == "clean_20261014") {
                if (std::abs(shot.t) < 1e-9) yTrain.push_back(shot.idx);
                jointTrain.push_back(shot.idx);
            }
        } else {
            const int yCount = setup == "small" ? 3 : 5;
            const int tCount = setup == "small" ? 2 : 4;
            yTrain.assign(yAnchors.begin(), yAnchors.begin() + yCount);
            jointTrain = yTrain;
            jointTrain.insert(jointTrain.end(), tAnchors.begin(), tAnchors.begin() + tCount);
        }
        const auto build = [&](const std::string& name, const Ids& ids, const auto& obs, bool single) {
            cv::Mat x, z;
            pack(cache, obs, ids, x, z);
            const Model model = fit(x, z, single ? "local_axis" : "local", {}, single ? 0 : -1);
            if (model.coefficients.empty() || x.rows != static_cast<int>(ids.size()))
                throw std::runtime_error("Calibration failure for " + setup + "/" + name);
            for (const int id : ids) {
                const Shot& shot = cache.shots[id];
                if (single && std::abs(shot.t) > 1e-9) throw std::runtime_error("Single-axis training leaked nonzero T");
                training << setup << '\t' << name << '\t' << id << '\t' << shot.set.substr(6) << '\t'
                         << shot.group << '\t' << shot.y << '\t' << shot.t << '\n';
            }
            return model;
        };
        const Model single = build("single_y", yTrain, singleObs, true);
        const Model joint = build("joint", jointTrain, jointObs, false);
        std::map<std::tuple<std::string, std::string, std::string>, RouteScore> by;
        for (const auto& shot : cache.shots) {
            if (shot.condition == "center" || reservedY.count(shot.idx) || reservedT.count(shot.idx)) continue;
            if (setup == "full_cross_date" && shot.set == "clean_20261014") continue;
            const std::string date = shot.set.substr(6);
            const auto a = estimate(single, singleObs[shot.idx]);
            const auto b = estimate(joint, jointObs[shot.idx]);
            const auto singleFitsGate = [&](double g) {
                return std::isfinite(a[0]) && std::isfinite(b[0]) && std::isfinite(b[1]) && std::abs(b[1]) <= g;
            };
            const auto add = [&](const std::string& route, const cv::Vec2d& result, bool usedSingle) {
                for (const std::string bucket : {std::string("all"), std::string(residualBand(shot.t))})
                    tally(by[{date, bucket, route}], shot, result, usedSingle);
            };
            add("single_y_assume_T0", a, true);
            add("joint", b, false);
            for (size_t i = 0; i < gates.size(); ++i) {
                const bool singleOK = singleFitsGate(gates[i]);
                // Keep the measured joint T estimate when the single-axis path supplies Y.
                add(gateNames[i], singleOK ? cv::Vec2d(a[0], b[1]) : b, singleOK);
            }
            shots << setup << '\t' << shot.idx << '\t' << date << '\t' << shot.group << '\t'
                  << shot.y << '\t' << shot.t << '\t' << a[0] << '\t' << b[0] << '\t' << b[1];
            for (const double g : gates) shots << '\t' << singleFitsGate(g);
            shots << '\n';
        }
        for (const auto& [key, value] : by) {
            const auto& [date, bin, route] = key;
            const double nan = std::numeric_limits<double>::quiet_NaN();
            scores << setup << '\t' << date << '\t' << bin << '\t' << route << '\t' << value.test
                   << '\t' << value.valid << '\t' << value.yPass << '\t' << value.tPass << '\t' << value.both
                   << '\t' << value.usedSingle << '\t' << value.singleYMiss << '\t' << value.singleTOutside
                   << '\t' << (value.valid ? value.maxY : nan) << '\t' << (value.valid ? value.maxT : nan) << '\n';
        }
    }
}

} // namespace semes_study
