#include "alignment.hpp"
#include "joint_reference.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <limits>
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b-a).count(); }
static cv::Mat read_image(const std::string& path) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto length = f.tellg();
    if (length <= 0) return {};
    std::vector<unsigned char> bytes(static_cast<size_t>(length));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) return {};
    return bytes.empty() ? cv::Mat() : cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
}
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: evaluate_photos MODEL PHOTOS_TSV OUTPUT_DIRECTORY ROUNDS");
        const auto out = std::filesystem::u8path(argv[3]);
        if (std::filesystem::exists(out / "measurements.tsv")) throw std::runtime_error("Output exists");
        const int rounds = std::stoi(argv[4]);
        if (rounds < 1) throw std::runtime_error("ROUNDS must be positive");
        std::ifstream input(std::filesystem::u8path(argv[2]));
        if (!input) throw std::runtime_error("Cannot read photos list");
        std::vector<std::pair<std::string,std::string>> photos;
        std::string line;
        while (std::getline(input,line)) {
            if (!line.empty() && line.back()=='\r') line.pop_back();
            const auto tab=line.find('\t');
            if (tab==std::string::npos) throw std::runtime_error("Invalid photos row");
            photos.push_back({line.substr(0,tab),line.substr(tab+1)});
        }
        if (photos.empty()) throw std::runtime_error("Empty photos list");
        cv::setNumThreads(1);
        auto start=Clock::now(); initialize_joint(argv[1]); const double init_ms=ms(start,Clock::now());
        set_joint_health_check(false);
        auto warm=read_image(photos.front().second);
        for (int i=0;i<10;++i) joint(warm);
        std::filesystem::create_directories(out);
        std::ofstream result(out/"measurements.tsv",std::ios::binary);
        result<<std::setprecision(17)<<"round\tid\ty\tt\tvalid\twarning\toutside\thealth\tread_ms\tmeasure_ms\ttotal_ms\tmessage\n";
        std::ofstream meta(out/"run.txt",std::ios::binary);
        meta<<std::setprecision(17)<<"images "<<photos.size()<<"\nrounds "<<rounds<<"\nthreads "<<cv::getNumThreads()<<"\ninit_ms "<<init_ms
            <<"\nwarmup 10\nreader binary_block_read_imdecode\nmethod "<<joint_qr_method()<<"\nmodel "<<joint_model_id()<<"\nzero default_absolute\n";
        meta.flush();
        for (int round=1;round<=rounds;++round) {
            size_t index=0;
            for (const auto& photo:photos) {
                auto a=Clock::now(),b=a,c=a;
                double y=std::numeric_limits<double>::quiet_NaN(),t=y,health=y;
                bool valid=false,warning=false,outside=false;
                std::string message; bool loaded=false;
                try {
                    auto image=read_image(photo.second); b=Clock::now(); loaded=true;
                    if(image.empty()) throw std::runtime_error("Cannot read image");
                    const auto value=joint(image); c=Clock::now();
                    const auto d=alignment_diagnostics(); y=value.y;t=value.t;health=d.model_cost;
                    valid=d.measurement_valid&&std::isfinite(y)&&std::isfinite(t);
                    warning=d.normalized_cost_margin>1;outside=d.outside_training_range;
                    if(warning) message=d.message;
                } catch(const std::exception& e) { c=Clock::now();if(!loaded)b=c;message=e.what(); }
                for(auto& ch:message) if(ch=='\t'||ch=='\r'||ch=='\n')ch=' ';
                result<<round<<'\t'<<photo.first<<'\t'<<y<<'\t'<<t<<'\t'<<valid<<'\t'<<warning<<'\t'<<outside<<'\t'<<health<<'\t'
                      <<ms(a,b)<<'\t'<<ms(b,c)<<'\t'<<ms(a,c)<<'\t'<<message<<'\n';
                if(++index%200==0) std::cout<<"round "<<round<<" processed "<<index<<"/"<<photos.size()<<std::endl;
            }
            result.flush();
        }
        if(!result||!meta)throw std::runtime_error("Output write failed");
        std::cout<<"Completed "<<photos.size()<<" images x "<<rounds<<" rounds"<<std::endl;
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
