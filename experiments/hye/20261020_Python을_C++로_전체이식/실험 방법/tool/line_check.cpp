#include "io.hpp"
#include <iostream>
#include <algorithm>
using namespace semes;
int wmain(int argc,wchar_t** wide){
    Utf8Arguments args(argc,wide);char** argv=args.data();
    try{
        if(argc!=5)throw std::runtime_error("line_check reference-features.npz native-run-directory displacement-tolerance new-output");auto reference=npz(fs::u8path(argv[1]));fs::path native=fs::u8path(argv[2]),out=fs::u8path(argv[4]);double tolerance=std::stod(argv[3]);if(fs::exists(out)||!std::isfinite(tolerance)||tolerance<=0)throw std::runtime_error("New output and positive finite tolerance required");auto shape=reference.at("measured_shifts").shape;if(shape.size()!=2||shape[1]!=48)throw std::runtime_error("Expected complete Nx48 line reference");
        std::map<std::string,double> maxima;int validity_changes=0,nonfinite_changes=0,feature_outside=0,condition_changes=0;Json per_image=Json::list();
        for(size_t i=0;i<shape[0];++i){auto fresh=Json::parse(read(native/"images"/fs::u8path(std::to_string(i)+".json")));double shift_max=0;
            for(auto key:{"measured_shifts","rmse","correlation","gain","offset","contrast","X_upper","X_lower"}){
                auto& expected=reference.at(key);auto& actual=fresh.at(key);size_t width=expected.shape.at(1);if(actual.size()!=width)throw std::runtime_error("Line feature shape mismatch");
                for(size_t k=0;k<width;++k){double a=expected.number(i*width+k),b=actual.at(k).type==Json::Null?std::numeric_limits<double>::quiet_NaN():actual.at(k).num();if(!std::isfinite(a)||!std::isfinite(b)){nonfinite_changes+=std::isfinite(a)!=std::isfinite(b);continue;}double difference=std::abs(a-b);maxima[key]=std::max(maxima[key],difference);if(std::string(key)=="measured_shifts"||std::string(key)=="X_upper"||std::string(key)=="X_lower"){feature_outside+=difference>tolerance;shift_max=std::max(shift_max,difference);}}
            }
            for(size_t k=0;k<48;++k)validity_changes+=bool(reference.at("valid_blocks").number(i*48+k))!=bool(fresh.at("valid_blocks").at(k).num());for(auto key:{"success_upper","success_lower"})condition_changes+=bool(reference.at(key).number(i))!=bool(fresh.at(key).num());
            Json image=Json::dict();image["index"]=int(i);image["max_displacement_difference_px"]=shift_max;per_image.push(image);
        }
        Json check=Json::dict();check["created_kst"]=now();check["images"]=int(shape[0]);check["blocks_per_image"]=48;check["displacement_tolerance_px"]=tolerance;check["validity_changes"]=validity_changes;check["condition_success_changes"]=condition_changes;check["nonfinite_changes"]=nonfinite_changes;check["displacement_values_outside_tolerance"]=feature_outside;check["maximum_numeric_differences"]=Json::dict();for(auto& p:maxima)check["maximum_numeric_differences"][p.first]=p.second;check["reference_sha256"]=sha256(fs::u8path(argv[1]));check["passed"]=!validity_changes&&!condition_changes&&!nonfinite_changes&&!feature_outside;check["auxiliary_diagnostics_scope"]="All finite auxiliary differences are reported; only displacement and exact validity/success gates define this measurement parity check.";write(out/"checks.json",check.dump());write(out/"per_image.json",per_image.dump());std::cout<<check.dump()<<'\n';return check.at("passed").num()?0:2;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
