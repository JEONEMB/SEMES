#include "alignment.hpp"
#include "common/sha256.hpp"
#include "common/reference.hpp"
#include "steps/01_find_qr.hpp"
#include "steps/02_find_screws.hpp"
#include "common/coefficient_fit.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
namespace fs=std::filesystem;
// main: 기존 SSAFY 학습 사진으로 나사 조합별 계산식 생성
//   입력: 기준 모델 폴더, id/path/y/t TSV, 비어 있는 출력 폴더
//   출력: 16개 계산식, 11개 점검식, 사진별 관측값. 누락이나 검출 실패가 있으면 중단
int main(int argc,char**argv) {
 try {
    if(argc!=4)throw std::runtime_error("usage: train_combinations REFERENCE TRAINING_TSV NEW_MODEL");
    cv::setNumThreads(1);fs::path out=fs::u8path(argv[3]);
    if(fs::exists(out))throw std::runtime_error("Output already exists");
    alignment::Reference ref;std::string error;
    if(!alignment::load_reference(argv[1],ref,error))throw std::runtime_error(error);
    alignment::QrFinder qr(ref);alignment::ScrewFinder screws(ref);
    std::ifstream input(fs::u8path(argv[2]));if(!input)throw std::runtime_error("Cannot read training list");
    fs::create_directories(out/"coefficients");fs::create_directories(out/"checks");
    std::ofstream features(out/"training_features.tsv");features<<std::setprecision(17)<<"id\tqr\tscrew1\tscrew2\tscrew3\tscrew4\ty\tt\n";
    std::vector<tools::Values> values;std::vector<double> ys,ts;std::string line;std::getline(input,line);
    while(std::getline(input,line)) {
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        std::istringstream row(line);std::string id,path,y,t;
        if(!std::getline(row,id,'\t')||!std::getline(row,path,'\t')||!std::getline(row,y,'\t')||!std::getline(row,t,'\t'))throw std::runtime_error("Invalid training row");
        const auto image=alignment::read_photo(path);if(image.empty())throw std::runtime_error("Cannot read "+id);
        const auto q=qr.find_qr_shift(image);if(!q.found)throw std::runtime_error(id+": "+q.problem);
        const auto found=screws.find_screw_shifts(image,q.dx,q.dy,{true,true,true,true});
        tools::Values v{q.dx,0,0,0,0};
        for(int k=0;k<4;++k) {if(!found[k].found||found[k].partial)throw std::runtime_error(id+" screw "+std::to_string(k+1)+": "+found[k].problem);v[k+1]=found[k].dx;}
        const double ly=std::stod(y),lt=std::stod(t);if(!std::isfinite(ly)||!std::isfinite(lt))throw std::runtime_error("Invalid label");
        values.push_back(v);ys.push_back(ly);ts.push_back(lt);features<<id;for(double a:v)features<<'\t'<<a;features<<'\t'<<ly<<'\t'<<lt<<'\n';
        if(values.size()%200==0)std::cout<<"training observations "<<values.size()<<std::endl;
    }
    if(values.size()!=1959)throw std::runtime_error("Expected original 1959 training images");
    std::ofstream report(out/"fit_report.txt");report<<"SSAFY original L2 training images "<<values.size()<<"\nTraining reuse, not independent accuracy\n";
    for(const char* name:tools::kScrewSetNames) {
        const auto set=tools::fit_coefficient_set(name,values,ys,ts);const std::string file=std::string("screws_")+name+".tsv";
        tools::write_coefficient_file(out/"coefficients"/file,set,int(values.size()));
        if(tools::has_agreement_check(name))tools::write_agreement_file(out/"checks"/file,tools::fit_agreement_check(set,values),int(values.size()));
        int passed=0;for(size_t i=0;i<values.size();++i){const auto p=tools::predict(set,values[i]);passed+=tools::passes(p.y_mm-ys[i],p.t_deg-ts[i]);}
        report<<name<<'\t'<<passed<<'/'<<values.size()<<'\n';
    }
    fs::copy_file(fs::u8path(argv[1])/"center.bmp",out/"center.bmp");
    fs::copy_file(fs::u8path(argv[1])/"reference.json",out/"reference.json");
    if(!features||!report)throw std::runtime_error("Output write failed");
    std::ofstream manifest(out/"model_manifest.tsv");
    for(const auto& entry:fs::recursive_directory_iterator(out)) {
        if(!entry.is_regular_file())continue;
        const auto name=fs::relative(entry.path(),out).generic_string();
        if(name!="center.bmp"&&name!="reference.json"&&name.find("coefficients/")!=0&&name.find("checks/")!=0)continue;
        std::ifstream file(entry.path(),std::ios::binary);std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),{});
        manifest<<name<<'\t'<<final00_sha::sha256_hex(bytes)<<'\n';
    }
    if(!manifest)throw std::runtime_error("Manifest write failed");
    std::cout<<"Created all 16 coefficient sets and 11 checks"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
