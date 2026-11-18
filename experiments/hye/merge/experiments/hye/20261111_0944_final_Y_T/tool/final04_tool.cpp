#include "alignment.hpp"
#include "final04.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>
#include <map>
#include <stdexcept>
namespace fs=std::filesystem;
namespace {
std::vector<std::string> split(std::string s,char delimiter) {
    if(!s.empty() && s.back()=='\r') s.pop_back();
    std::stringstream in(s);std::string field;std::vector<std::string> out;
    while(std::getline(in,field,delimiter)) out.push_back(field);
    return out;
}
cv::Mat numbers(const std::string& s,int count) {
    const auto fields=split(s,';');if(int(fields.size())!=count)throw std::runtime_error("Bad measurement dimensions");
    cv::Mat row(1,count,CV_64F);
    for(int k=0;k<count;++k)row.at<double>(k)=std::stod(fields[k]);
    if(!cv::checkRange(row))throw std::runtime_error("Nonfinite measurements");return row;
}
int run(const std::vector<std::string>& args) {
    cv::setNumThreads(1);
    if(args.size()<2)throw std::invalid_argument("predict ASSETS IMAGE | repeat5 ASSETS IMAGE1 IMAGE2 IMAGE3 IMAGE4 IMAGE5 | train ASSETS FEATURES.tsv OUTPUT.yml | crossval ASSETS FEATURES.tsv FOLDS.tsv OUTPUT.tsv | audit ASSETS FEATURES.tsv IMAGE_ROOT OUTPUT.tsv [BEGIN END]");
    const auto assets=fs::u8path(args[1]);
    if(args[0]=="predict" && args.size()==3) {
        initialize_joint(args[1]);auto image=alignment_detail::read_gray(fs::u8path(args[2]));
        const auto p=joint(image);const auto d=alignment_diagnostics();
        std::cout<<std::setprecision(17)<<"{\"y\":"<<p.y<<",\"t\":"<<p.t<<",\"measurement_valid\":"<<(d.measurement_valid?"true":"false")
            <<",\"outside_training_range\":"<<(d.outside_training_range?"true":"false")<<",\"model\":\"huber_linear66\"}\n";return 0;
    }
    if(args[0]=="repeat5" && args.size()==7) {
        initialize_joint(args[1]);std::set<std::string> hashes;
        double miny=1e100,maxy=-1e100,mint=1e100,maxt=-1e100;
        for(int k=2;k<7;++k) {
            const auto path=fs::u8path(args[k]);
            if(!hashes.insert(alignment_detail::sha256(path)).second)
                throw std::invalid_argument("repeat5 requires five distinct photos; repeated file bytes are not repeated captures");
            auto image=alignment_detail::read_gray(path);const auto p=joint(image);
            miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);mint=std::min(mint,p.t);maxt=std::max(maxt,p.t);
        }
        std::cout<<std::setprecision(17)<<"{\"range_y\":"<<maxy-miny<<",\"range_t\":"<<maxt-mint
            <<",\"range_pass\":"<<((maxy-miny<=.045 && maxt-mint<=.01)?"true":"false")
            <<",\"note\":\"Caller must establish same-pose captures; range does not measure absolute accuracy.\"}\n";
        return 0;
    }
    const bool crossval=args[0]=="crossval" && args.size()==5;
    const bool train=args[0]=="train" && args.size()==4;
    const bool audit=args[0]=="audit" && (args.size()==5 || args.size()==7);
    if(!train && !audit && !crossval)throw std::invalid_argument("Invalid command/argument count");
    const auto output=fs::u8path(args[train?3:4]);
    if(fs::exists(output))throw std::runtime_error("Use a new output path; existing results are preserved");
    alignment_detail::Observer geometry(assets);const auto positions=geometry.block_centers();
    std::ifstream in(fs::u8path(args[2]));if(!in)throw std::runtime_error("Cannot read measurements");
    std::string line;std::getline(in,line);
    cv::Mat x,y;
    std::unique_ptr<final04::Estimator> estimator;
    final04::LinearModel frozen;std::ofstream out;
    const int begin=args.size()==7?std::stoi(args[5]):0,end=args.size()==7?std::stoi(args[6]):2147483647;
    if(audit) {
        estimator=std::make_unique<final04::Estimator>(args[1]);
        frozen.load((assets/"huber_linear66.yml").u8string());
        out.open(fs::u8path(args[4]));if(!out)throw std::runtime_error("Cannot write audit");
        out<<"index\tstatus\tY\tT\traw_max_delta\tpred_max_delta\tms\tqr_delta\tpin_delta\tmessage\n"<<std::setprecision(17);
    }
    int processed=0,failures=0;
    while(std::getline(in,line)) {
        const auto fields=split(line,'\t');if(fields.size()<10)throw std::runtime_error("Bad measurement TSV row");
        const int id=std::stoi(fields[0]);
        if(audit && (id<begin || id>=end))continue;
        const auto qr=numbers(fields[8],102),pins=numbers(fields[9],8);
        const auto features=final04::features(qr,pins,positions);
        if(train || crossval) {
            if(id!=x.rows)throw std::runtime_error("Training cache indices must be sequential");
            x.push_back(features);
            cv::Mat label=(cv::Mat_<double>(1,2)<<std::stod(fields[5]),std::stod(fields[6]));
            y.push_back(label);
        } else {
            try {
                const auto path=fs::u8path(args[3])/fs::u8path(fields[2]);
                if(alignment_detail::sha256(path)!=fields[3])throw std::runtime_error("Image SHA256 mismatch");
                auto image=alignment_detail::read_gray(path);const auto start=std::chrono::steady_clock::now();
                const auto r=estimator->predict(image);
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                cv::Mat raw;cv::hconcat(qr,pins,raw);
                const auto expected=frozen.predict(features);
                const double dr=cv::norm(raw,r.raw110,cv::NORM_INF);
                const double dp=std::max(std::abs(r.pose.y-expected.at<double>(0)),std::abs(r.pose.t-expected.at<double>(1)));
                const bool ok=dr<=1e-6 && dp<=1e-7;failures+=!ok;
                out<<id<<'\t'<<(ok?"PASS":"MISMATCH")<<'\t'<<r.pose.y<<'\t'<<r.pose.t<<'\t'<<dr<<'\t'<<dp<<'\t'<<ms<<'\t'<<cv::norm(raw.colRange(0,102),r.raw110.colRange(0,102),cv::NORM_INF)<<'\t'<<cv::norm(raw.colRange(102,110),r.raw110.colRange(102,110),cv::NORM_INF)<<"\t\n";
            } catch(const std::exception& e) {++failures;out<<id<<"\tERROR\t\t\t\t\t\t"<<e.what()<<'\n';}
            out.flush();
        }
        ++processed;
    }
    if(!processed)throw std::runtime_error("No input rows processed");
    if(train) {final04::LinearModel model;model.fit(x,y);model.save(args[3]);}
    if(crossval) {
        std::ifstream folds(fs::u8path(args[3]));if(!folds)throw std::runtime_error("Cannot read folds");
        std::getline(folds,line);std::map<int,std::map<int,int>> split_by_seed;
        while(std::getline(folds,line)) {auto v=split(line,'\t');if(v.size()!=3)throw std::runtime_error("Invalid fold row");
            const int seed=std::stoi(v[0]),fold=std::stoi(v[1]),id=std::stoi(v[2]);
            if(fold<1 || fold>5 || id<0 || id>=x.rows || !split_by_seed[seed].emplace(id,fold).second)
                throw std::runtime_error("Invalid or duplicated fold membership");}
        if(split_by_seed.empty())throw std::runtime_error("Empty folds");
        std::ofstream scores(output);if(!scores)throw std::runtime_error("Cannot write predictions");
        scores<<"seed\tfold\tindex\tY\tT\tpred_Y\tpred_T\tpass\n"<<std::setprecision(17);
        int n=0,pass=0;double maxy=0,maxt=0;
        for(const auto& seed:split_by_seed) {
            if(int(seed.second.size())!=x.rows)throw std::runtime_error("Incomplete seed");
            for(int fold=1;fold<=5;++fold) {
                cv::Mat trainx,trainy;std::vector<int> test;
                for(int i=0;i<x.rows;++i) {if(seed.second.at(i)==fold)test.push_back(i);else{trainx.push_back(x.row(i));trainy.push_back(y.row(i));}}
                if(test.empty())throw std::runtime_error("Empty test fold");
                final04::LinearModel model;model.fit(trainx,trainy);
                for(int i:test) {const auto p=model.predict(x.row(i));const double ey=std::abs(p.at<double>(0)-y.at<double>(i,0)),et=std::abs(p.at<double>(1)-y.at<double>(i,1));
                    const bool ok=ey<=.045 && et<=.01;++n;pass+=ok;maxy=std::max(maxy,ey);maxt=std::max(maxt,et);
                    scores<<seed.first<<'\t'<<fold<<'\t'<<i<<'\t'<<y.at<double>(i,0)<<'\t'<<y.at<double>(i,1)<<'\t'<<p.at<double>(0)<<'\t'<<p.at<double>(1)<<'\t'<<ok<<'\n';}
            }
        }
        std::cout<<std::setprecision(17)<<"DEVELOPMENT_CV n="<<n<<" pass="<<pass<<" max_Y="<<maxy<<" max_T="<<maxt<<"\n";
    }
    std::cout<<"rows="<<processed<<" failures="<<failures<<'\n';return failures?1:0;
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
#else
int main(int argc,char** argv) {
#endif
    try {std::vector<std::string> args;for(int k=1;k<argc;++k)args.push_back(fs::path(argv[k]).u8string());return run(args);}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
