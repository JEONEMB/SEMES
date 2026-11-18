#include "common.hpp"
#include <iostream>
using namespace semes;
using namespace historical_v1;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try{
        if(argc!=3)throw std::runtime_error("historical_v1_grid original-workspace new-output");
        fs::path root=fs::u8path(argv[1]),out=fs::u8path(argv[2]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Data data(root);
        fs::create_directories(out);
        std::ofstream metrics(out/"repro_17.csv"),predictions(out/"predictions.csv");
        metrics<<u8"key,영역,특징점,알고리즘,T_평균,T_RMS,T_최대,Y_평균,Y_RMS,Y_최대,Y_005안,Y사진_T착각,T사진_Y착각\n";
        predictions<<"key,file,fold,true_T,true_Y,pred_T,pred_Y\n";
        Json records=Json::list();
        const std::vector<std::string> algorithms={
            u8"C1 이동량÷계수 (Y=0 가정)",u8"C2 선형 역회귀",u8"C3 릿지 역회귀",u8"C4 야코비안 최소제곱",u8"C5 야코비안 가중(GLS)"
        };
        int complete=0;
        for(auto& config:data.configurations)for(int algorithm=1; algorithm<=5; ++algorithm){
            cv::Mat estimate(data.x.rows,2,CV_64F);
            bool supported=true;
            Json alphas=Json::list();
            for(int fold=0; fold<2; ++fold){
                std::vector<int> train,test;
                for(int i=0; i<data.x.rows; ++i)(i%2==fold?train:test).push_back(i);
                double alpha=0;
                auto predicted=predict(config.values,data.x,train,test,algorithm,&alpha);
                if(predicted.empty()){
                    supported=false;
                    break;
                }
                alphas.push(alpha);
                for(int i=0; i<predicted.rows; ++i)predicted.row(i).copyTo(estimate.row(test[i]));
            }
            if(!supported)continue;
            std::string key=config.name.substr(0,config.name.find(' '))+"_C"+std::to_string(algorithm);
            double absT=0,sqT=0,maxT=0,absY=0,sqY=0,maxY=0,confusedT=0,confusedY=0;
            int pass=0,ny=0,nt=0;
            for(int i=0; i<data.x.rows; ++i){
                double et=estimate.at<double>(i,0)-data.x.at<double>(i,0),ey=estimate.at<double>(i,1)-data.x.at<double>(i,1);
                absT+=std::abs(et);
                sqT+=et*et;
                maxT=std::max(maxT,std::abs(et));
                absY+=std::abs(ey);
                sqY+=ey*ey;
                maxY=std::max(maxY,std::abs(ey));
                pass+=std::abs(ey)<=.05;
                if(data.x.at<double>(i,0)==0&&data.x.at<double>(i,1)!=0){
                    confusedT+=et*et;
                    ++ny;
                }
                if(data.x.at<double>(i,1)==0&&data.x.at<double>(i,0)!=0){
                    confusedY+=ey*ey;
                    ++nt;
                }
                predictions<<key<<','<<csv(data.names[i])<<','<<(1-i%2)<<','<<std::setprecision(17)<<data.x.at<double>(i,0)<<','<<data.x.at<double>(i,1)<<','<<estimate.at<double>(i,0)<<','<<estimate.at<double>(i,1)<<'\n';
            }
            std::vector<double> values={
                absT/data.x.rows,std::sqrt(sqT/data.x.rows),maxT,absY/data.x.rows,std::sqrt(sqY/data.x.rows),maxY,pass*100.0/data.x.rows,std::sqrt(confusedT/ny),std::sqrt(confusedY/nt)
            };
            Json record=Json::dict();
            record["key"]=key;
            record["dimensions"]=config.values.cols;
            record["ridge_alpha_by_training_parity"]=alphas;
            Json numbers=Json::list();
            for(auto value:values)numbers.push(value);
            record["metrics_in_csv_order"]=numbers;
            records.push(record);
            metrics<<key<<','<<csv(config.group)<<','<<csv(config.name)<<','<<csv(algorithms[algorithm-1]);
            for(auto value:values)metrics<<','<<std::fixed<<std::setprecision(4)<<std::nearbyint(value*1e4)/1e4;
            metrics<<'\n';
            ++complete;
            std::cout<<key<<" complete"<<std::endl;
        }
        write(out/"unrounded_metrics.json",records.dump());
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["rows"]=data.x.rows;
        audit["configurations"]=complete;
        audit["historical_tracks_sha256"]=sha256(root/"analysis_v3/v1_data/tracks.npz");
        audit["changed"]="Symbolic, non-executing C++ parser reads only original NumPy file metadata; no Python imports or pickle execution. OpenCV SVD/LU replace NumPy. grid.py and grid_repro.py share one calculation and use explicit paths. Historical full-data feature selection and alternating two-fold splits retained; current independent/OOF performance is not claimed.";
        write(out/"audit.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
