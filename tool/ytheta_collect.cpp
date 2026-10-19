#include "direct_registration.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
int main(int argc, char **argv)
{
    if (argc != 5) { std::cerr << "Usage: ytheta_collect reference.bmp list.txt output.csv affine|homography\n"; return 1; }
    const std::string method = argv[4];
    if (method != "affine" && method != "homography") return 1;
    cv::setNumThreads(1);
    cv::Mat reference = cv::imread(argv[1]);
    if (reference.empty()) return 1;
    AlignmentOptions options; options.visualize = false;
    PlateDetectionResult rd; alignment(reference, options, &rd);
    std::vector<cv::Point2f> grid;
    // Fixed virtual points distributed inside each reference QR; not independent measurements.
    for (const auto &qr : rd.qrRois)
    {
        if (!qr.featuresValid) return 2;
        for (int row = 1; row <= 3; ++row) for (int col = 1; col <= 5; ++col)
        {
            float x = col / 6.f, y = row / 4.f;
            grid.push_back((1-y)*((1-x)*qr.corners[0]+x*qr.corners[1])+
                           y*((1-x)*qr.corners[3]+x*qr.corners[2]));
        }
    }
    std::ifstream list(argv[2]); std::ofstream out(argv[3]);
    if (!list || !out) return 1;
    out << "path,status,ecc,initial_residual,id,u0,v0,u,v\n" << std::setprecision(12);
    std::string path;
    while (std::getline(list, path))
    {
        try
        {
            cv::Mat current = cv::imread(path); PlateDetectionResult cd;
            alignment(current, options, &cd);
            auto r = registerQrPixels(reference, current, rd, cd,
                method == "affine" ? cv::MOTION_AFFINE : cv::MOTION_HOMOGRAPHY);
            if (!r.valid)
            {
                out << '"' << path << "\"," << r.status << ',' << r.correlation << ',' << r.initialResidual << ",-1,0,0,0,0\n";
                continue;
            }
            std::vector<cv::Point2f> q;
            cv::perspectiveTransform(grid, q, r.referenceToCurrent);
            for (size_t i = 0; i < grid.size(); ++i)
                out << '"' << path << "\",ok," << r.correlation << ',' << r.initialResidual << ',' << i << ','
                    << grid[i].x << ',' << grid[i].y << ',' << q[i].x << ',' << q[i].y << '\n';
        }
        catch (const cv::Exception &) { out << '"' << path << "\",pipeline_failed,0,0,-1,0,0,0,0\n"; }
    }
    return out.good() ? 0 : 1;
}
