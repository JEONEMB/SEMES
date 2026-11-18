#include "alignment.hpp"
#include "qr_refinement.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
using namespace subpixel;

void writeFit(std::ostream &out, int frame, const std::string &payload, int id,
              const std::string &method, cv::Point2d initial, Fit f, double ms)
{
    out << frame << ',' << payload << ',' << id << ',' << method << ',' << initial.x + f.p.x << ','
        << initial.y + f.p.y << ',' << f.valid << ',' << f.converged << ',' << f.boundary << ','
        << f.rms << ',' << f.condition << ',' << f.sigma << ',' << f.iterations << ',' << ms << ','
        << f.boundaryMask << '\n';
}
std::vector<std::pair<std::string, Fit>> refineAll(const cv::Mat &gray, cv::Point2d init, double a1,
                                                   double a2, double pitch)
{
    std::vector<std::pair<std::string, Fit>> fits;
    for (int radius : {3, 4, 5})
    {
        fits.push_back({"model_r" + std::to_string(radius),
                        modelFit(patch(gray, init, radius), a1, a2, pitch * .45)});
    }
    for (double alpha : {1.2, 1.8, 2.4})
    {
        for (bool demean : {false, true})
        {
            int r = 5;
            std::string name =
                "fourier_a" + std::to_string(int(alpha * 10)) + (demean ? "_centered" : "_raw");
            fits.push_back({name, fourierFit(patch(gray, init, r), alpha, demean, pitch * .45)});
        }
    }
    return fits;
}
int collect(const std::string &listpath, const std::string &output)
{
    std::ifstream list(listpath);
    std::ofstream pts(output + "/points.csv"), meta(output + "/frames.csv"),
        log(output + "/pipeline.log");
    std::ofstream candidates(output + "/candidates.csv");
    if (!list || !pts || !meta || !log || !candidates)
    {
        return 1;
    }
    pts << "frame,payload,id,method,x,y,valid,converged,boundary,objective_rms,condition,sigma,"
           "iterations,ms,boundary_mask\n"
        << std::setprecision(12);
    candidates
        << "frame,payload,id,model_x,model_y,initial_x,initial_y,baseline_accepted,pitch,a1,a2\n"
        << std::setprecision(12);
    meta << "frame,path,plate,decoded,candidates\n";
    auto original = std::cout.rdbuf(log.rdbuf());
    std::string path;
    int frame = 0;
    while (std::getline(list, path))
    {
        if (!path.empty() && path.back() == '\r')
        {
            path.pop_back();
        }
        cv::Mat im = cv::imread(path), gray;
        PlateDetectionResult d;
        AlignmentOptions options;
        options.visualize = false;
        // Explicitly preserve the pre-experiment baseline after integration.
        // The macro is enabled only when the production selector exists.
#ifdef HAS_QR_REFINEMENT_SELECTOR
        options.qrRefinement = QrRefinementMethod::CornerSubPix;
#endif
        alignment(im, options, &d);
        cv::cvtColor(im, gray, cv::COLOR_BGR2GRAY);
        int decoded = 0, count = 0;
        for (const auto &q : d.qrRois)
        {
            decoded += q.decoded;
            count += int(q.features.size());
            if (q.features.size() < 4)
            {
                continue;
            }
            std::vector<cv::Point2f> m, p;
            for (auto f : q.features)
            {
                m.push_back(f.model);
                p.push_back(f.initial);
            }
            cv::Mat h = cv::findHomography(m, p, 0);
            for (auto f : q.features)
            {
                std::vector<cv::Point2f> local{f.model, f.model + cv::Point2f(.01f, 0),
                                               f.model + cv::Point2f(0, .01f)},
                    proj;
                cv::perspectiveTransform(local, proj, h);
                auto dx = (proj[1] - proj[0]) * 100, dy = (proj[2] - proj[0]) * 100;
                // Normals to the two projected grid boundaries.
                double a1 = std::atan2(dy.y, dy.x) - pi / 2, a2 = std::atan2(dx.y, dx.x) + pi / 2;
                double pitch = (cv::norm(dx) + cv::norm(dy)) / 2;
                candidates << frame << ',' << q.payload << ',' << f.id << ',' << f.model.x << ','
                           << f.model.y << ',' << f.initial.x << ',' << f.initial.y << ','
                           << f.accepted << ',' << pitch << ',' << a1 << ',' << a2 << '\n';
                Fit base;
                base.p = cv::Point2d(f.refined) - cv::Point2d(f.initial);
                base.valid = f.accepted;
                base.converged = true;
                writeFit(pts, frame, q.payload, f.id, "cornerSubPix", f.initial, base, 0);
                // All algorithms receive the same projected initial location.
                for (int radius : {3, 4, 5})
                {
                    auto start = std::chrono::steady_clock::now();
                    auto fit = modelFit(patch(gray, f.initial, radius), a1, a2, pitch * .45);
                    double ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
                    writeFit(pts, frame, q.payload, f.id, "model_r" + std::to_string(radius),
                             f.initial, fit, ms);
                }
                for (double alpha : {1.2, 1.8, 2.4})
                {
                    for (bool demean : {false, true})
                    {
                        auto start = std::chrono::steady_clock::now();
                        auto fit =
                            fourierFit(patch(gray, f.initial, 5), alpha, demean, pitch * .45);
                        double ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - start)
                                        .count();
                        writeFit(pts, frame, q.payload, f.id,
                                 "fourier_a" + std::to_string(int(alpha * 10)) +
                                     (demean ? "_centered" : "_raw"),
                                 f.initial, fit, ms);
                    }
                }
            }
        }
        meta << frame << ',' << path << ',' << d.found << ',' << decoded << ',' << count << '\n';
        if (frame % 10 == 0)
        {
            pts.flush();
            candidates.flush();
            std::cerr << "frames=" << frame + 1 << '\n';
        }
        ++frame;
    }
    std::cout.rdbuf(original);
    return 0;
}
double checkJacobian(Objective fun, cv::Mat p)
{
    cv::Mat r, j;
    fun(p, r, j);
    double err = 0;
    for (int k = 0; k < p.rows; ++k)
    {
        auto pp = p.clone(), pm = p.clone();
        pp.at<double>(k) += 1e-6;
        pm.at<double>(k) -= 1e-6;
        cv::Mat rp, rm, z;
        fun(pp, rp, z);
        fun(pm, rm, z);
        cv::Mat fd = (rp - rm) / 2e-6;
        err = std::max(err, cv::norm(fd - j.col(k), cv::NORM_INF));
    }
    return err;
}
int synthetic(const std::string &output)
{
    std::ofstream out(output + "/synthetic.csv");
    out << "case,blur,angle,texture,method,error,valid\n" << std::setprecision(12);
    std::mt19937 rng(507);
    std::uniform_real_distribution<double> uniform(-.45, .45);
    std::normal_distribution<double> normal(0, .006);
    int id = 0;
    double maxModel = 0, maxFourier = 0;
    for (double sigma : {.45, .8, 1.3})
    {
        for (double angle : {75., 90., 105.})
        {
            for (bool texture : {false, true})
            {
                for (int rep = 0; rep < 8; ++rep)
                {
                    // Independent renderer: high-resolution binary image, discrete Gaussian
                    // convolution, then integer point sampling; no analytic fitted model.
                    constexpr int S = 32, N = 33;
                    double tx = uniform(rng), ty = uniform(rng), a1 = .04,
                           a2 = a1 + angle * pi / 180;
                    cv::Mat high(N * S, N * S, CV_32F);
                    for (int y = 0; y < high.rows; ++y)
                    {
                        for (int x = 0; x < high.cols; ++x)
                        {
                            double u = double(x) / S - 16 - tx, v = double(y) / S - 16 - ty;
                            high.at<float>(y, x) = ((u * std::cos(a1) + v * std::sin(a1)) *
                                                        (u * std::cos(a2) + v * std::sin(a2)) >=
                                                    0)
                                                       ? 1.f
                                                       : -1.f;
                        }
                    }
                    cv::GaussianBlur(high, high, {}, sigma * S, sigma * S, cv::BORDER_REPLICATE);
                    cv::Mat image(N, N, CV_8U);
                    for (int y = 0; y < N; ++y)
                    {
                        for (int x = 0; x < N; ++x)
                        {
                            double tex = texture ? .025 * std::sin(.8 * x + .3 * y) +
                                                       .012 * std::cos(1.2 * y)
                                                 : 0;
                            double val = .5 + .3 * high.at<float>(y * S, x * S) + normal(rng) + tex;
                            image.at<uchar>(y, x) = cv::saturate_cast<uchar>(255 * val);
                        }
                    }
                    cv::Point2d init(16 + .23, 16 - .17), truth(16 + tx, 16 + ty);
                    auto fits = refineAll(image, init, a1, a2, 8);
                    std::vector<cv::Point2f> bp{cv::Point2f(init)};
                    cv::cornerSubPix(image, bp, {2, 2}, {-1, -1},
                                     {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 40, .001});
                    Fit base;
                    base.p = cv::Point2d(bp[0]) - init;
                    base.valid = true;
                    fits.push_back({"cornerSubPix", base});
                    for (int radius : {3, 4, 5})
                    {
                        bp = {cv::Point2f(init)};
                        cv::cornerSubPix(
                            image, bp, {radius, radius}, {-1, -1},
                            {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 40, .001});
                        base.p = cv::Point2d(bp[0]) - init;
                        fits.push_back({"cornerSubPix_r" + std::to_string(radius), base});
                    }
                    for (auto z : fits)
                    {
                        out << id << ',' << sigma << ',' << angle << ',' << texture << ','
                            << z.first << ',' << cv::norm(init + z.second.p - truth) << ','
                            << z.second.valid << '\n';
                    }
                    if (id == 0)
                    {
                        auto v = patch(image, init, 4);
                        cv::Mat p = (cv::Mat_<double>(7, 1) << .13, -.11, a1 + .03, a2 - .02,
                                     std::log(.9), .3, .5);
                        maxModel = checkJacobian(modelObjective(v), p);
                        maxFourier = checkJacobian(fourierObjective(v, 1.8, false),
                                                   (cv::Mat_<double>(2, 1) << .13, -.11));
                    }
                    ++id;
                }
            }
        }
    }
    std::ofstream test(output + "/numerics.json");
    test << std::setprecision(12) << "{\"model_max_jacobian_error\":" << maxModel
         << ",\"fourier_max_jacobian_error\":" << maxFourier << ",\"synthetic_cases\":" << id
         << "}\n";
    std::cerr << "Jacobian errors: " << maxModel << ", " << maxFourier << '\n';
    return maxModel < 1e-6 && maxFourier < 1e-6 ? 0 : 3;
}
int main(int argc, char **argv)
{
    cv::setNumThreads(1);
    if (argc == 3 && std::string(argv[1]) == "synthetic")
    {
        return synthetic(argv[2]);
    }
    if (argc == 4 && std::string(argv[1]) == "collect")
    {
        return collect(argv[2], argv[3]);
    }
    std::cerr << "compare synthetic output-dir | collect images.txt output-dir\n";
    return 1;
}
