#include "aggregation.hpp"
#include <iostream>
#include <fstream>
#include <set>
using namespace semes;
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try {
        if(argc!=5)throw std::runtime_error("aggregation_check corners280.csv tracks.npz saved-features-directory new-output");
        auto cache=npz(fs::u8path(argv[2]));
        auto& reference=cache.at("points");
        const auto& reference_ids=cache.at("ids");
        std::vector<std::string> names;
        for(size_t i=0; i<cache.at("names").count(); ++i)names.push_back(cache.at("names").text(i));
        cv::Mat points(630,2,CV_64F);
        std::map<int,int> ids;
        for(int i=0; i<630; ++i){
            points.at<double>(i,0)=reference.number(2*i);
            points.at<double>(i,1)=reference.number(2*i+1);
            ids[int(reference_ids.number(i))]=i;
        }
        std::ifstream input(fs::u8path(argv[1]));
        std::string line;
        std::getline(input,line);
        auto h=csvrow(line);
        if(h[0].size()>=3&&(unsigned char)h[0][0]==239)h[0]=h[0].substr(3);
        std::map<std::string,int> columns;
        for(int i=0; i<int(h.size()); ++i)columns[h[i]]=i;
        std::map<std::string,cv::Mat> xy,quality;
        std::map<std::string,std::set<int>> seen;
        while(std::getline(input,line)) {
            auto r=csvrow(line);
            int id=std::stoi(r.at(columns.at("id")));
            if(!ids.count(id))continue;
            int i=ids.at(id);
            auto name=r.at(columns.at("file"));
            if(!xy.count(name)){
                xy[name]=cv::Mat(630,2,CV_64F);
                quality[name]=cv::Mat(630,3,CV_64F);
            }
            if(!seen[name].insert(id).second)throw std::runtime_error("Duplicate observed ID");
            xy[name].at<double>(i,0)=std::stod(r.at(columns.at("x")));
            xy[name].at<double>(i,1)=std::stod(r.at(columns.at("y")));
            quality[name].at<double>(i,0)=std::stod(r.at(columns.at("rmse")));
            quality[name].at<double>(i,1)=std::stod(r.at(columns.at("window_stability_px")));
            quality[name].at<double>(i,2)=std::stod(r.at(columns.at("contrast")));
        }
        for(auto& name:names)if(seen.at(name).size()!=630)throw std::runtime_error("Missing fixed observations");
        fs::path baseline=fs::u8path(argv[3]),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        fs::create_directories(out);
        std::ofstream table(out/"comparisons.csv");
        table<<"measurement,feature,images,dimensions,max_abs_difference,rmse,passed\n";
        Json comparisons=Json::list();
        int n=0,failed=0;
        for(auto measurement:{
            "uniform","photo","stability","contrast","huber","soft_l1","cauchy"
        }) {
            for(auto feature:{
                "global2","global1","global3","local1","local2","quadrants","homography"
            }) {
                auto path=baseline/(std::string(measurement)+"_"+feature+".npy");
                if(!fs::exists(path))continue;
                auto expected=npy(bytes(path));
                if(expected.shape.size()!=2||expected.shape[0]!=names.size())throw std::runtime_error("Feature cache row mismatch");
                double max_difference=0,squared=0;
                for(size_t i=0; i<names.size(); ++i) {
                    cv::Mat d=xy.at(names[i])-xy.at("center.bmp");
                    auto x=aggregate_features(points,d,quality.at(names[i]),measurement,feature);
                    if(x.cols!=int(expected.shape[1]))throw std::runtime_error("Feature dimension mismatch");
                    for(int k=0; k<x.cols; ++k) {
                        double delta=x.at<double>(k)-expected.number(i*x.cols+k);
                        max_difference=std::max(max_difference,std::abs(delta));
                        squared+=delta*delta;
                    }
                }
                bool passed=max_difference<1e-7;
                failed+=!passed;
                ++n;
                Json c=Json::dict();
                c["measurement"]=measurement;
                c["feature"]=feature;
                c["n"]=int(names.size());
                c["dimensions"]=int(expected.shape[1]);
                c["max_abs_difference"]=max_difference;
                c["rmse"]=std::sqrt(squared/expected.count());
                c["passed"]=passed;
                comparisons.push(c);
                table<<std::setprecision(17)<<measurement<<','<<feature<<','<<names.size()<<','<<expected.shape[1]<<','<<max_difference<<','<<std::sqrt(squared/expected.count())<<','<<passed<<'\n';
                table.flush();
                std::cout<<measurement<<' '<<feature<<" max difference "<<max_difference<<std::endl;
            }
        }
        Json report=Json::dict();
        report["configurations_compared"]=n;
        report["failed"]=failed;
        report["comparisons"]=comparisons;
        report["created_kst"]=now();
        report["scope"]="Saved fixed 630 observations and quality, all 281 names including center; feature extraction only, no new inference or training";
        write(out/"checks.json",report.dump());
        return failed?2:0;
    }
    catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
