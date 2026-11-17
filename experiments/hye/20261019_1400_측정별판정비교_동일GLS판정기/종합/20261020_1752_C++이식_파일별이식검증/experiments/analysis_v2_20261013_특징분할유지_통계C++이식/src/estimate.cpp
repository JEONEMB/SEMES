#include "features.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try {
        if(argc!=3&&argc!=4)throw std::runtime_error("historical_v2_estimate original-workspace new-output [lambda]");
        fs::path root=fs::u8path(argv[1]),out=fs::u8path(argv[2]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        v2_history::Features data(root);
        bool found_center=data.joined_center_present;
        std::vector<v2_history::Observation> observations=data.ys;
        for(auto& row:data.ts)if(row.file!="center.bmp")observations.push_back(row);
        const auto& configurations=data.configurations;
        double regularization=argc==4?std::stod(argv[3]):1e-9;
        fs::create_directories(out);
        std::ofstream metrics(out/"estimate_result.csv"),predictions(out/"predictions.csv");
        metrics<<u8"버전,Y_RMS_um,Y_최대_um,Y_90um초과,Θ_RMS,작은틀어짐_Y_RMS_um\n";
        predictions<<"configuration,file,fold,true_T,true_Y,pred_T,pred_Y\n";
        const int n=int(observations.size());
        for(auto& config:configurations) {
            int dim=int(config.second.size());
            cv::Mat x(n,dim+1,CV_64F),target=cv::Mat::zeros(n,2,CV_64F),pred=cv::Mat::zeros(n,2,CV_64F);
            for(int col=0; col<dim; ++col) {
                double mean=0,variance=0;
                for(auto& r:observations)mean+=r.features.at(config.second[col]);
                mean/=n;
                for(auto& r:observations)variance+=std::pow(r.features.at(config.second[col])-mean,2);
                double scale=std::sqrt(variance/n);
                if(scale==0)throw std::runtime_error("Zero historical feature scale");
                for(int i=0; i<n; ++i)x.at<double>(i,col)=(observations[i].features.at(config.second[col])-mean)/scale;
            }
            for(int i=0; i<n; ++i){
                x.at<double>(i,dim)=1;
                char axis=observations[i].file[0];
                if(axis=='T')target.at<double>(i,0)=observations[i].value;
                if(axis=='Y')target.at<double>(i,1)=observations[i].value;
            }
            for(int fold=0; fold<2; ++fold) {
                cv::Mat a,b;
                for(int i=0; i<n; ++i)if(i%2!=fold){
                    a.push_back(x.row(i));
                    b.push_back(target.row(i));
                }
                cv::Mat gram=a.t()*a,rhs=a.t()*b,co;
                for(int j=0; j<dim; ++j)gram.at<double>(j,j)+=regularization*n;
                if(!cv::solve(gram,rhs,co,cv::DECOMP_LU))throw std::runtime_error("Historical ridge solve failed");
                for(int i=0; i<n; ++i)if(i%2==fold){
                    cv::Mat p=x.row(i)*co;
                    p.copyTo(pred.row(i));
                }
            }
            double sumY=0,sumT=0,maxY=0,smallY=0;
            int over=0,smallN=0;
            for(int i=0; i<n; ++i) {
                double ey=std::abs(pred.at<double>(i,1)-target.at<double>(i,1))*1000,et=std::abs(pred.at<double>(i,0)-target.at<double>(i,0));
                sumY+=ey*ey;
                sumT+=et*et;
                maxY=std::max(maxY,ey);
                over+=ey>90;
                if(std::abs(target.at<double>(i,1))<=.5&&std::abs(target.at<double>(i,0))<=.09){
                    smallY+=ey*ey;
                    ++smallN;
                }
                predictions<<csv(config.first)<<','<<csv(observations[i].file)<<','<<i%2<<','<<std::setprecision(17)<<target.at<double>(i,0)<<','<<target.at<double>(i,1)<<','<<pred.at<double>(i,0)<<','<<pred.at<double>(i,1)<<'\n';
            }
            metrics<<csv(config.first)<<','<<std::setprecision(17)<<std::sqrt(sumY/n)<<','<<maxY<<','<<over<<','<<std::sqrt(sumT/n)<<','<<std::sqrt(smallY/smallN)<<'\n';
        }
        Json audit=Json::dict();
        audit["n"]=n;
        audit["joined_center_present"]=found_center;
        audit["lambda"]=regularization;
        audit["preserved_historical_limitation"]="Original estimate.py standardizes using all joined rows before its alternating two-fold split. The saved finder table has no center row, so the original inner join excludes center. Reproduced only as a historical comparison; not the current training-only 4-fold evaluation.";
        audit["changed"]="Reconstructs the same joined feature table from source CSVs instead of loading a Python pickle. Outputs CSV/JSON only.";
        audit["created_kst"]=now();
        write(out/"audit.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return 0;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
