#include "sequential_alignment.hpp"
#include <chrono>
#include <iostream>

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}

int main(int argc, char** argv) {
    try {
        if(argc!=3) throw std::invalid_argument("Expected UTF-8 manifest and output filenames");
        cv::setNumThreads(2);
        cv::FileStorage input(semes::readTextUtf8(argv[1]),cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
        const std::string refPath=input["reference"], tcPath=input["theta_config"], ycPath=input["y_config"];
        const auto initStart=Clock::now();
        const auto tc=semes::ThetaConfig::load(tcPath); const auto yc=semes::YConfig::load(ycPath);
        const auto reference=semes::readGrayUtf8(refPath);
        const semes::SequentialAligner aligner(reference,tc,yc);
        const double initMs=elapsed(initStart);
        const auto warmup=aligner.estimate(reference);
        if(!warmup.aligned) throw std::runtime_error("Reference self-check failed; benchmark not started");
        cv::FileStorage output(".json",cv::FileStorage::WRITE|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
        output << "schema_version" << 1 << "opencv_version" << CV_VERSION << "opencv_threads" << 2
               << "initialization_ms" << initMs << "warmup_reference_excluded_from_timing" << 1
               << "mode" << "one_independent_estimate_per_real_image_no_physical_moves" << "results" << "[";
        int count=0;
        const auto cases=input["cases"];
        for(auto it=cases.begin();it!=cases.end();++it) {
            const auto node=*it;
            const std::string id=node["id"], path=node["path"];
            output << "{" << "id" << id;
            double loadMs=0, estimateMs=0;
            bool loaded=false;
            Clock::time_point start=Clock::now();
            try {
                const auto image=semes::readGrayUtf8(path);
                loadMs=elapsed(start); loaded=true; start=Clock::now();
                const auto result=aligner.estimate(image);
                estimateMs=elapsed(start);
                output << "measurement_valid" << 1 << "decision" << "{";
                semes::writeAlignmentDecision(output,result); output << "}";
            } catch(const semes::AlignmentError& e) {
                if(loaded) estimateMs=elapsed(start); else loadMs=elapsed(start);
                output << "measurement_valid" << 0 << "error_code" << semes::errorCodeName(e.code()) << "error" << e.what();
            } catch(const std::exception& e) {
                if(loaded) estimateMs=elapsed(start); else loadMs=elapsed(start);
                output << "measurement_valid" << 0 << "error_code" << "measurement_failed" << "error" << e.what();
            }
            output << "load_ms" << loadMs << "estimate_ms" << estimateMs << "total_ms" << loadMs+estimateMs << "}";
            if(++count%25==0) std::cout << "Processed " << count << "/" << cases.size() << std::endl;
        }
        output << "]";
        semes::writeTextUtf8(argv[2],output.releaseAndGetString());
        std::cout << "Finished " << count << " independent real images" << std::endl;
        return 0; // Per-image failures are measurements and remain in the report.
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
