#include "alignment.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
// Input preparation and calls only. The library never sends motion commands.
int wmain(int argc,wchar_t** argv)
{
    try {
        if (argc < 2) {
            std::cerr << "Usage: main IMAGE [--assets DIRECTORY] [--gui] [--output PREFIX] "
                         "[--detect-only] [--refinement corner|model|fourier]\n";
            return 2;
        }
        AlignmentOptions options;
        options.visualize = false;
        bool debugOutput = false;
        for (int i=2;i<argc;i++) {
            std::wstring argument=argv[i];
            if (argument==L"--assets" && i+1<argc) options.poseAssetsDirectory=std::filesystem::path(argv[++i]).u8string();
            else if (argument==L"--gui") options.visualize=true;
            else if (argument==L"--no-gui") options.visualize=false;
            else if (argument==L"--detect-only") { options.estimatePose=false; debugOutput=true; }
            else if (argument==L"--output" && i+1<argc) { options.outputPrefix=std::filesystem::path(argv[++i]).u8string();debugOutput=true; }
            else if (argument==L"--refinement" && i+1<argc) {
                std::wstring method=argv[++i];debugOutput=true;
                if (method==L"corner") options.qrRefinement=QrRefinementMethod::CornerSubPix;
                else if (method==L"model") options.qrRefinement=QrRefinementMethod::ImageFormation;
                else if (method==L"fourier") options.qrRefinement=QrRefinementMethod::Fourier;
                else throw std::invalid_argument("Unknown debug refinement method");
            } else throw std::invalid_argument("Invalid or incomplete option");
        }
        std::ifstream input(std::filesystem::path(argv[1]),std::ios::binary);
        if (!input) throw std::runtime_error("Cannot read image");
        std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input),{}};
        cv::Mat image=cv::imdecode(bytes,cv::IMREAD_GRAYSCALE);
        if (image.empty()) throw std::runtime_error("Cannot decode image");
        cv::setNumThreads(1);
        PlateDetectionResult detection;
        auto pose=alignment(image,options,debugOutput?&detection:nullptr);
        if (options.estimatePose) {
            std::cout<<std::setprecision(17)<<"{\"y_mm\":"<<pose.y<<",\"yaw_deg\":"<<pose.t
                     <<",\"method\":\"lanczos3\"}\n";
        } else {
            std::cout<<"{\"detection_only\":true,\"plate_found\":"
                     <<(detection.found?"true":"false")<<"}\n";
            if(!detection.found)return 3;
        }
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
