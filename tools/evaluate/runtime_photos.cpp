#include "alignment.hpp"
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
// ms: 구간 시간을 밀리초로 계산
//   입력: 시작과 끝 시각
//   출력: 경과 시간
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b-a).count(); }
// read_image: UTF-8 경로의 사진 읽기
//   입력: 사진 경로
//   출력: 회색 Mat, 읽기 실패 시 빈 Mat
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
// main: 정답 없이 사진 전체를 진단 측정하고 시간 기록
//   입력: 모델, id와 경로 TSV, 새 출력 폴더, 반복 수
//   출력: 사진별 관측값과 실행 시간, 보정은 허용하지 않음
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
        auto start=Clock::now(); const auto init=alignment::initialize(argv[1]); const double init_ms=ms(start,Clock::now());
        if(init.status==alignment::Status::Failed)throw std::runtime_error(init.message);
        auto warm=read_image(photos.front().second);
        for (int i=0;i<10;++i) alignment::measure_diagnostic(warm);
        std::filesystem::create_directories(out);
        std::ofstream result(out/"measurements.tsv",std::ios::binary);
        result<<std::setprecision(17)<<"round\tid\ty\tt\tvalid\twarning\toutside\thealth\tread_ms\tmeasure_ms\ttotal_ms\tqr\tscrew1\tscrew2\tscrew3\tscrew4\tscrews_used\tcode\tcorrection_allowed\tmessage\n";
        std::ofstream meta(out/"run.txt",std::ios::binary);
        meta<<std::setprecision(17)<<"images "<<photos.size()<<"\nrounds "<<rounds<<"\nthreads "<<cv::getNumThreads()<<"\ninit_ms "<<init_ms
            <<"\nwarmup 10\nreader binary_block_read_imdecode\nmethod "<<"wide_ncc"<<"\nmodel "<<argv[1]<<"\nzero diagnostic_absolute_no_correction\n";
        meta.flush();
        for (int round=1;round<=rounds;++round) {
            size_t index=0;
            for (const auto& photo:photos) {
                auto a=Clock::now(),b=a,c=a;
                double y=std::numeric_limits<double>::quiet_NaN(),t=y,health=y;
                bool valid=false,warning=false,outside=false;
                std::string message,combination,code; bool loaded=false,correction=false;
                double features[5]={y,y,y,y,y};
                try {
                    auto image=read_image(photo.second); b=Clock::now(); loaded=true;
                    if(image.empty()) throw std::runtime_error("Cannot read image");
                    const auto value=alignment::measure_diagnostic(image); c=Clock::now();
                    y=value.y_mm;t=value.t_deg;health=value.check_value;
                    valid=value.status!=alignment::Status::Failed&&std::isfinite(y)&&std::isfinite(t);
                    warning=value.status==alignment::Status::Warning;
                    for(const auto& notice:value.notices)if(notice.code=="W_OUT_OF_RANGE")outside=true;
                    message=value.message;combination=value.screws_used;code=value.code;correction=value.correction_allowed;
                    features[0]=value.qr_shift_px;for(int k=0;k<4;++k)features[k+1]=value.screw_shift_px[k];
                } catch(const std::exception& e) { c=Clock::now();if(!loaded)b=c;message=e.what(); }
                for(auto& ch:message) if(ch=='\t'||ch=='\r'||ch=='\n')ch=' ';
                result<<round<<'\t'<<photo.first<<'\t'<<y<<'\t'<<t<<'\t'<<valid<<'\t'<<warning<<'\t'<<outside<<'\t'<<health<<'\t'
                      <<ms(a,b)<<'\t'<<ms(b,c)<<'\t'<<ms(a,c);
                for(double feature:features)result<<'\t'<<feature;
                result<<'\t'<<combination<<'\t'<<code<<'\t'<<correction<<'\t'<<message<<'\n';
                if(++index%200==0) std::cout<<"round "<<round<<" processed "<<index<<"/"<<photos.size()<<std::endl;
            }
            result.flush();
        }
        if(!result||!meta)throw std::runtime_error("Output write failed");
        std::cout<<"Completed "<<photos.size()<<" images x "<<rounds<<" rounds"<<std::endl;
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
