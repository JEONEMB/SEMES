#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>
#include <chrono>
std::vector<std::string> columns(std::string line)
{
    std::vector<std::string> out;
    std::stringstream stream(line);
    std::string s;
    while (std::getline(stream, s, ','))
    {
        out.push_back(s);
    }
    return out;
}
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        return 1;
    }
    cv::setNumThreads(1);
    std::string dir = argv[1];
    std::ifstream meta(dir + "/frames.csv"), cand(dir + "/candidates.csv");
    std::ofstream out(dir + "/baseline-extra.csv");
    if (!meta || !cand || !out)
    {
        return 1;
    }
    out << "frame,payload,id,method,x,y,valid,converged,boundary,objective_rms,condition,sigma,"
           "iterations,ms\n"
        << std::setprecision(12);
    std::map<int, std::string> paths;
    std::string line;
    std::getline(meta, line);
    while (std::getline(meta, line))
    {
        auto c = columns(line);
        paths[std::stoi(c[0])] = c[1];
    }
    int frame = -1;
    cv::Mat gray;
    std::getline(cand, line);
    while (std::getline(cand, line))
    {
        auto c = columns(line);
        int f = std::stoi(c[0]);
        if (f != frame)
        {
            cv::cvtColor(cv::imread(paths[f]), gray, cv::COLOR_BGR2GRAY);
            frame = f;
        }
        cv::Point2f initial(std::stof(c[5]), std::stof(c[6]));
        double bound = std::stod(c[8]) * .45;
        for (int radius : {3, 4, 5})
        {
            std::vector<cv::Point2f> p{initial};
            auto start = std::chrono::steady_clock::now();
            cv::cornerSubPix(gray, p, {radius, radius}, {-1, -1},
                             {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 40, .001});
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            bool ok =
                std::isfinite(p[0].x) && std::isfinite(p[0].y) && cv::norm(p[0] - initial) < bound;
            out << f << ',' << c[1] << ',' << c[2] << ",cornerSubPix_r" << radius << ',' << p[0].x
                << ',' << p[0].y << ',' << ok << ",1,0,0,0,0,0," << ms << '\n';
        }
    }
}
