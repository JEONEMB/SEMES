#include "io.hpp"
#include <cmath>
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try{
        if(argc!=4)throw std::runtime_error("historical_g1_compare original-analysis-v3 cpp-run new-output");
        fs::path reference=fs::u8path(argv[1]),actual=fs::u8path(argv[2]),out=fs::u8path(argv[3]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        auto run=Json::parse(read(actual/"checks.json"));
        bool pass=run["completed"].num()==281&&run["failures"].num()==0;
        Json report=Json::dict();
        report["created_kst"]=now();
        report["policy"]="Explicit migration comparison tolerances, selected after the first full run. These are not independently pre-registered scientific criteria or new accuracy metrics.";
        Json arrays=Json::list();
        for(auto name:{
            "g1_est.npy","g1_nuisance.npy","g1_cost.npy"
        }){
            auto a=npy(bytes(reference/name)),b=npy(bytes(actual/name));
            if(a.shape!=b.shape||a.shape.empty()||a.shape[0]!=281)throw std::runtime_error("G1 shape mismatch");
            Json item=Json::dict();
            item["file"]=name;
            item["reference_sha256"]=sha256(reference/name);
            item["actual_sha256"]=sha256(actual/name);
            double maximum=0;
            int failures=0;
            for(size_t i=0; i<a.count(); ++i){
                double x=a.number(i),y=b.number(i),difference=std::abs(x-y);
                double tolerance=std::string(name)=="g1_cost.npy"?1e-5:1e-4;
                if(std::string(name)=="g1_est.npy"&&i%2==1)tolerance=1e-3;
                if(!std::isfinite(x)||!std::isfinite(y)||difference>tolerance)++failures;
                if(std::isfinite(difference))maximum=std::max(maximum,difference);
            }
            item["values"]=double(a.count());
            item["max_abs_difference"]=maximum;
            item["failed_values"]=failures;
            arrays.push(item);
            pass=pass&&failures==0;
        }
        report["T_tolerance"]=1e-4;
        report["Y_tolerance"]=1e-3;
        report["nuisance_tolerance"]=1e-4;
        report["cost_tolerance"]=1e-5;
        report["arrays"]=arrays;
        report["passed"]=pass;
        fs::create_directories(out);
        write(out/"checks.json",report.dump());
        std::cout<<report.dump()<<'\n';
        return pass?0:2;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
