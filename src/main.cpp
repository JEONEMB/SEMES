#include <iostream>
#include <string>
#include "alignment.hpp"
int main(int argc, char** argv) {
    if(argc<2) {
        std::cerr << "Usage: main <image_path> [--output <prefix>] [--no-gui]\n";
        return 1;
    }
    AlignmentOptions options;
    for(int i=2;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--no-gui") options.visualize=false;
        else if(arg=="--output" && i+1<argc) options.outputPrefix=argv[++i];
        else { std::cerr<<"Invalid or incomplete option: "<<arg<<"\n";return 1; }
    }
    try {
        cv::Mat image=cv::imread(argv[1],cv::IMREAD_COLOR);
        if(image.empty()) {std::cerr<<"Cannot read image: "<<argv[1]<<"\n";return 1;}
        PlateDetectionResult detection;
        alignment(image,options,&detection);
        if(!detection.found) {std::cerr<<"QR plate not found\n";return 2;}
        std::cout<<"Plate detected. Boundary score: "<<detection.score<<"\n";
        for(const auto& point:detection.corners) std::cout<<point<<"\n";
        std::cout<<"Y/theta alignment errors are not estimated yet.\n";
        return 0;
    } catch(const cv::Exception& e) {
        std::cerr<<e.what()<<"\n";return 1;
    }
}
