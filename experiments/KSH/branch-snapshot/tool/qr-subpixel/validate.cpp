#include "alignment.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        return 1;
    }
    cv::setNumThreads(1);
    std::string dir = argv[2], method = argv[3];
    AlignmentOptions options;
    options.visualize = false;
    if (method == "model")
    {
        options.qrRefinement = QrRefinementMethod::ImageFormation;
    }
    else if (method == "corner")
    {
        options.qrRefinement = QrRefinementMethod::CornerSubPix;
    }
    else if (method != "fourier")
    {
        return 1;
    }
    std::ifstream list(argv[1]);
    std::ofstream out(dir + "/points.csv"), meta(dir + "/frames.csv"), log(dir + "/pipeline.log"),
        qrmeta(dir + "/qr-status.csv");
    if (!list || !out || !meta || !log || !qrmeta)
    {
        return 1;
    }
    out << "frame,payload,id,method,x,y,valid,converged,boundary,objective_rms,condition,sigma,"
           "iterations,ms\n"
        << std::setprecision(12);
    meta << "frame,path,plate,decoded,candidates,valid_features\n";
    qrmeta << "frame,payload,status,accepted,candidates\n";
    auto original = std::cout.rdbuf(log.rdbuf());
    std::string path;
    int frame = 0;
    while (std::getline(list, path))
    {
        if (!path.empty() && path.back() == '\r')
        {
            path.pop_back();
        }
        int decoded = 0, count = 0, valid = 0;
        PlateDetectionResult d;
        cv::Mat image = cv::imread(path);
        alignment(image, options, &d);
        for (auto &q : d.qrRois)
        {
            decoded += q.decoded;
            count += int(q.features.size());
            valid += q.featuresValid;
            int accepted = 0;
            for (auto f : q.features)
            {
                accepted += f.accepted;
                out << frame << ',' << q.payload << ',' << f.id << ",operational_" << method << ','
                    << f.refined.x << ',' << f.refined.y << ',' << f.accepted << ','
                    << f.refinementConverged << ',' << (f.refinementBoundaryMask != 0) << ','
                    << f.refinementResidual << ',' << f.refinementCondition << ",0,0,0\n";
            }
            qrmeta << frame << ',' << q.payload << ',' << q.featureStatus << ',' << accepted << ','
                   << q.features.size() << '\n';
        }
        meta << frame << ',' << path << ',' << d.found << ',' << decoded << ',' << count << ','
             << valid << '\n';
        if (frame % 20 == 0)
        {
            std::cerr << "frames=" << frame + 1 << '\n';
        }
        ++frame;
    }
    std::cout.rdbuf(original);
    return 0;
}
