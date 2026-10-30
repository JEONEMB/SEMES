#include "gn_features.hpp"
#include <iostream>
using namespace semes;
using namespace historical_v1;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try{
        if(argc!=3)throw std::runtime_error("historical_v3_evaluate original-workspace new-output");
        fs::path root=fs::u8path(argv[1]),out=fs::u8path(argv[2]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Data data(root);
        CsvTable hom(root/"analysis_v2/homography_features.csv");
        std::map<std::string,size_t> hom_index;
        for(size_t i=0; i<hom.rows.size(); ++i)hom_index.emplace(hom.text(i,"file"),i);
        std::vector<std::string> fields;
        for(auto& field:hom.header)if(field!="file"&&field!="cc"&&field!="h20"&&field!="h21")fields.push_back(field);
        cv::Mat homography(data.x.rows,int(fields.size()),CV_64F);
        for(int i=0; i<data.x.rows; ++i)for(int j=0; j<homography.cols; ++j)homography.at<double>(i,j)=hom.number(hom_index.at(data.names[i]),fields[j]);
        data.configurations.insert(data.configurations.begin()+5,{
            u8"Q5b 판 전체 원근 변환, 귀퉁이+QR중심 (14개)",u8"QR만",homography
        });
        std::vector<std::pair<double,double>> t_curve;
        std::vector<double> mean_dx(data.x.rows),twin(data.x.rows,std::numeric_limits<double>::quiet_NaN());
        double min_y=std::numeric_limits<double>::infinity(),max_y=-min_y;
        for(int i=0; i<data.x.rows; ++i){
            mean_dx[i]=data.configurations[0].values.at<double>(i,0);
            double t=data.x.at<double>(i,0),y=data.x.at<double>(i,1);
            if(y==0)t_curve.emplace_back(mean_dx[i],t);
            if(t==0){
                min_y=std::min(min_y,mean_dx[i]);
                max_y=std::max(max_y,mean_dx[i]);
            }
        }
        std::sort(t_curve.begin(),t_curve.end());
        std::vector<bool> is_y(data.x.rows),in_range(data.x.rows);
        for(int i=0; i<data.x.rows; ++i){
            double t=data.x.at<double>(i,0),y=data.x.at<double>(i,1),m=mean_dx[i];
            is_y[i]=t==0&&y!=0;
            in_range[i]=y==0&&t!=0&&m>=min_y&&m<=max_y;
            if(in_range[i])twin[i]=0;
            if(is_y[i]){
                auto b=std::upper_bound(t_curve.begin(),t_curve.end(),std::make_pair(m,std::numeric_limits<double>::infinity()));
                if(b==t_curve.begin())twin[i]=b->second;
                else if(b==t_curve.end())twin[i]=t_curve.back().second;
                else{
                    auto a=b-1;
                    twin[i]=a->second+(m-a->first)/(b->first-a->first)*(b->second-a->second);
                }
            }
        }
        fs::create_directories(out);
        std::ofstream metrics(out/"v3_results.csv"),predictions(out/"predictions.csv");
        metrics<<u8"key,영역,특징점,알고리즘,T_RMS,T_최대,Y_평균,Y_RMS,Y_최대,Y_009안,Y_005안,Y불가,Θ불가,오판율,σΘ\n";
        predictions<<"key,kind,file,true_T,true_Y,pred_T,pred_Y,twin_T,can_distinguish\n";
        int count=0;
        const std::vector<std::string> algorithms={
            u8"C1 이동량÷계수 (Y=0 가정)",u8"C2 선형 역회귀",u8"C3 릿지 역회귀",u8"C4 야코비안 최소제곱",u8"C5 야코비안 가중(GLS)",u8"C6 가우스-뉴턴 (곡선 모델, 반복)"
        };
        auto record=[&](const std::string& key,const std::string& group,const std::string& feature,const std::string& algorithm,const cv::Mat& estimated,bool replay){
            double t_square=0,t_max=0,y_sum=0,y_square=0,y_max=0,mean_error=0,variance=0;
            int y90=0,y50=0;
            for(int i=0; i<data.x.rows; ++i){
                double et=estimated.at<double>(i,0)-data.x.at<double>(i,0),ey=estimated.at<double>(i,1)-data.x.at<double>(i,1);
                t_square+=et*et;
                t_max=std::max(t_max,std::abs(et));
                y_sum+=std::abs(ey);
                y_square+=ey*ey;
                y_max=std::max(y_max,std::abs(ey));
                y90+=std::abs(ey)<=.09;
                y50+=std::abs(ey)<=.05;
                mean_error+=et;
            }
            mean_error/=data.x.rows;
            for(int i=0; i<data.x.rows; ++i)variance+=std::pow(estimated.at<double>(i,0)-data.x.at<double>(i,0)-mean_error,2);
            double sd=std::sqrt(variance/data.x.rows);
            int unavailable_y=0,unavailable_t=0,fp=0;
            for(int i=0; i<data.x.rows; ++i){
                bool can=std::abs(estimated.at<double>(i,0)-twin[i])>3*sd;
                unavailable_y+=is_y[i]&&!can;
                unavailable_t+=in_range[i]&&!can;
                fp+=std::abs(estimated.at<double>(i,0)-data.x.at<double>(i,0))>3*sd;
                predictions<<key<<','<<(replay?"saved_G1_prediction_replay":"native_Cpp_feature_fit")<<','<<csv(data.names[i])<<','<<std::setprecision(17)<<data.x.at<double>(i,0)<<','<<data.x.at<double>(i,1)<<','<<estimated.at<double>(i,0)<<','<<estimated.at<double>(i,1)<<',';
                if(std::isfinite(twin[i]))predictions<<twin[i];
                predictions<<','<<can<<'\n';
            }
            metrics<<key<<','<<csv(group)<<','<<csv(feature)<<','<<csv(algorithm)<<','<<std::setprecision(17)<<std::sqrt(t_square/data.x.rows)<<','<<t_max<<','<<y_sum/data.x.rows<<','<<std::sqrt(y_square/data.x.rows)<<','<<y_max<<','<<y90*100.0/data.x.rows<<','<<y50*100.0/data.x.rows<<','<<unavailable_y<<','<<unavailable_t<<','<<fp/double(data.x.rows)<<','<<sd<<'\n';
            ++count;
            std::cout<<key<<" complete"<<std::endl;
        };
        for(auto& config:data.configurations)for(int algorithm=1; algorithm<=6; ++algorithm){
            cv::Mat result(data.x.size(),CV_64F);
            bool supported=true;
            for(int fold=0; fold<2; ++fold){
                std::vector<int> train,test;
                for(int i=0; i<data.x.rows; ++i)(i%2==fold?train:test).push_back(i);
                auto estimated=algorithm==6?gn_predict(config.values,data.x,train,test):predict(config.values,data.x,train,test,algorithm);
                if(estimated.empty()){
                    supported=false;
                    break;
                }
                for(int i=0; i<estimated.rows; ++i)estimated.row(i).copyTo(result.row(test[i]));
            }
            if(supported)record(config.name.substr(0,config.name.find(' '))+"_C"+std::to_string(algorithm),config.group,config.name,algorithms[algorithm-1],result,false);
        }
        auto g1=array(npy(bytes(root/"analysis_v3/g1_est.npy")));
        if(g1.size()!=data.x.size())throw std::runtime_error("G1 snapshot row coverage differs");
        record("G1",u8"QR만",u8"G1 QR 판 픽셀 전체 (특징점 없음)",u8"G1 픽셀 직접 가우스-뉴턴 (Θ, Y + 방해 3)",g1,true);
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["result_rows"]=count;
        audit["native_feature_fit_rows"]=count-1;
        audit["G1_saved_prediction_replay_rows"]=1;
        audit["G1_pixel_solver_ported"]=false;
        audit["limitation"]="Preserves historical alternating two-fold split and rule3 threshold calibrated from all OOF residuals. Same feature selections and development data; not independent performance. Last G1 row is explicitly saved-prediction replay, not a C++ pixel fit.";
        write(out/"audit.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
