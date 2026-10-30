#include "io.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{
    if(argc!=5)throw std::runtime_error("direct_check reference-features.npz native-output new-output tolerance-px");fs::path input=fs::u8path(argv[2]),out=fs::u8path(argv[3]);if(fs::exists(out))throw std::runtime_error("New output required");double tolerance=std::stod(argv[4]);if(!std::isfinite(tolerance)||tolerance<0)throw std::runtime_error("Finite nonnegative tolerance required");auto reference=npz(fs::u8path(argv[1]));size_t count=reference.at("index").count();Json variants=Json::dict();bool passed=true;
    for(std::string name:{"affine_g3","homography_g3","affine_g5","homography_g5"}){
        const auto& d=reference.at("D_"+name);const auto& success=reference.at("success_"+name);const auto& valid=reference.at("qr_success_"+name);int finite_changes=0,success_changes=0,valid_changes=0,outside=0;double maximum=0;
        for(size_t i=0;i<count;++i){auto file=Json::parse(read(input/"images"/fs::u8path(std::to_string(i)+".json")));const auto& candidate=file.at("candidates").at(name);success_changes+=bool(candidate.at("success").number)!=bool(success.number(i));for(size_t k=0;k<3;++k)valid_changes+=bool(candidate.at("qr_success").at(k).number)!=bool(valid.number(i*3+k));for(size_t p=0;p<630;++p)for(size_t axis=0;axis<2;++axis){const auto& value=candidate.at("D").at(p).at(axis);double expected=d.number((i*630+p)*2+axis);bool finite=std::isfinite(expected);if(finite!=(value.type==Json::Number)){++finite_changes;continue;}if(finite){double delta=std::abs(expected-value.num());maximum=std::max(maximum,delta);outside+=delta>tolerance;}}}
        Json j=Json::dict();j["images"]=int(count);j["max_displacement_difference_px"]=maximum;j["success_changes"]=success_changes;j["qr_validity_changes"]=valid_changes;j["nonfinite_changes"]=finite_changes;j["outside_tolerance"]=outside;j["passed"]=!success_changes&&!valid_changes&&!finite_changes&&!outside;passed=passed&&j.at("passed").number;variants[name]=j;
    }
    Json result=Json::dict();result["tolerance_px"]=tolerance;result["candidates"]=variants;result["passed"]=passed;write(out/"checks.json",result.dump());std::cout<<result.dump()<<'\n';return passed?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
