#include "common.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try {
        if(argc!=3)throw std::runtime_error("historical_theta_only qr_measurements.csv new-output");
        CsvTable data(fs::u8path(argv[1]));
        fs::path out=fs::u8path(argv[2]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        fs::create_directories(out);
        std::vector<size_t> test;
        double dot=0,squared=0;
        for(size_t i=0; i<data.rows.size(); ++i)if(data.text(i,"axis")=="T") {
            auto p=agent_history::positions(data,i);
            double mean=(p[0]+p[2]+p[4])/3,v=data.number(i,"value");
            if(int(std::nearbyint(std::abs(v)*100))%2){
                dot+=mean*v;
                squared+=mean*mean;
            }
            else test.push_back(i);
        }
        if(squared==0)throw std::runtime_error("Zero training motion");
        double slope=dot/squared;
        std::ofstream predictions(out/"theta_only_holdout.csv");
        predictions<<"file,value,qr_mean_shift_px,theta_pred,theta_abs_error,qr_y_equivalent_error_mm\n";
        std::array<int,3> count{
        };
        std::array<double,3> sum{
        },maximum{
        };
        for(auto i:test) {
            auto p=agent_history::positions(data,i);
            double mean=(p[0]+p[2]+p[4])/3,v=data.number(i,"value"),pred=slope*mean,error=std::abs(pred-v);
            predictions<<csv(data.text(i,"file"))<<','<<std::setprecision(17)<<v<<','<<mean<<','<<pred<<','<<error<<','<<error*80.06/14.25<<'\n';
            for(int k=0; k<3; ++k)if(k==0||std::abs(v)<=(k==1?.25:.10)){
                ++count[k];
                sum[k]+=error;
                maximum[k]=std::max(maximum[k],error);
            }
        }
        Json summary=Json::dict(),subsets=Json::list();
        for(int k=0; k<3; ++k){
            Json s=Json::dict();
            s["scope"]=k==0?"all":k==1?"abs(T)<=0.25":"abs(T)<=0.10";
            s["n"]=count[k];
            s["T_MAE"]=sum[k]/count[k];
            s["T_max"]=maximum[k];
            s["QR_Y_equivalent_MAE_mm"]=sum[k]/count[k]*80.06/14.25;
            subsets.push(s);
        }
        summary["slope"]=slope;
        summary["subsets"]=subsets;
        summary["condition"]="Historical experiment assumes Y is fixed; not an unconstrained Y/T estimator. QR Y-equivalent units do not define physical T travel.";
        summary["input_sha256"]=sha256(fs::u8path(argv[1]));
        summary["created_kst"]=now();
        write(out/"summary.json",summary.dump());
        std::cout<<summary.dump()<<'\n';
        return 0;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
