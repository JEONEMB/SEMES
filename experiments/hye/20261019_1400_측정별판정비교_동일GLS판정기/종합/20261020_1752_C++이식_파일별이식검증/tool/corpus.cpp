#include "joint.hpp"
#include <iostream>
#include <fstream>
#include <regex>
#include <chrono>
#include <numeric>
using namespace semes;
static void save_points(const Engine& engine,const Measurement& m,const fs::path& path){
    std::ostringstream s;
    s<<std::setprecision(17)<<"id,region,x,y,dx,dy,valid,flags\n";
    for(int i=0; i<630; i++)s<<engine.corners()[i].id<<','<<engine.corners()[i].region<<','<<m.current.at<double>(i,0)<<','<<m.current.at<double>(i,1)<<','<<m.displacement.at<double>(i,0)<<','<<m.displacement.at<double>(i,1)<<','<<m.valid[i]<<','<<m.flags[i]<<'\n';
    write(path,s.str());
}
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc!=6)throw std::runtime_error("corpus assets fixed-center joint-model.npz data-root new-output");
        Engine engine(fs::u8path(argv[1]),fs::u8path(argv[2]));
        JointOrigins model(fs::u8path(argv[3]));
        fs::path data=fs::u8path(argv[4]),out=fs::u8path(argv[5]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        std::vector<fs::path> images;
        for(auto& e:fs::recursive_directory_iterator(data))if(e.is_regular_file()&&e.path().extension()==".bmp")images.push_back(e.path());
        std::sort(images.begin(),images.end());
        if(images.size()!=341)throw std::runtime_error("Expected 341 existing corpus images");
        fs::create_directories(out);
        std::ofstream rows(out/"predictions.csv"),failures(out/"failures.csv");
        std::string header="relative,scope,true_Y,true_T,pred_Y,pred_T,error_Y,error_T,both_within_009,center_id,search_radius,valid,seconds,status\n";
        rows<<header;
        failures<<header;
        std::regex coordinate("([YT])([+-][0-9]+(?:\\.[0-9]*)?)"),center("center[0-9]*",std::regex::icase);
        std::vector<double> times;
        int n=0,failed=0,wide=0;
        std::map<std::string,int> counts,passes;
        std::map<std::string,double> sumY,sumT,maxY,maxT;
        for(auto& path:images){
            std::string relative=path.lexically_relative(data).generic_u8string(),stem=path.stem().u8string();
            double y=0,t=0;
            bool iscenter=std::regex_match(stem,center);
            int parsed=0;
            for(std::sregex_iterator it(stem.begin(),stem.end(),coordinate),end; it!=end; ++it){
                if((*it)[1]=="Y")y=std::stod((*it)[2]);
                else t=std::stod((*it)[2]);
                ++parsed;
            }
            if(!iscenter&&!parsed)throw std::runtime_error("Unrecognized evaluation label: "+relative);
            std::string scope=iscenter?"center":y!=0&&t!=0?"mixed":"single";
            ++counts[scope];
            ++n;
            auto begin=std::chrono::steady_clock::now();
            std::ostringstream row;
            row<<std::setprecision(17)<<csv(relative)<<','<<scope<<','<<y<<','<<t<<',';
            try{
                auto image=loadimage(path);
                Measurement m;
                int radius=130;
                for(int pad:{
                    130,260,512
                }){
                    radius=pad;
                    try{
                        m=engine.measure(image,pad);
                        break;
                    }
                    catch(const std::exception& e){
                        if(std::string(e.what())!="QR NCC below 0.6"||pad==512)throw;
                    }
                }
                auto prediction=model.predict(engine.features(m.displacement,model.feature_name()));
                double py=prediction.at("Y").num(),pt=prediction.at("T").num(),ey=std::abs(py-y),et=std::abs(pt-t),seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                bool pass=ey<=.09+1e-12&&et<=.09+1e-12;
                passes[scope]+=pass;
                sumY[scope]+=ey;
                sumT[scope]+=et;
                maxY[scope]=std::max(maxY[scope],ey);
                maxT[scope]=std::max(maxT[scope],et);
                wide+=radius>130;
                times.push_back(seconds);
                int valid=std::accumulate(m.valid.begin(),m.valid.end(),0);
                row<<py<<','<<pt<<','<<ey<<','<<et<<','<<pass<<','<<csv(prediction.at("center_id").str())<<','<<radius<<','<<valid<<','<<seconds<<",ok\n";
                rows<<row.str();
                if(!pass)failures<<row.str();
                prediction["search_radius"]=radius;
                prediction["valid"]=valid;
                prediction["seconds"]=seconds;
                write(out/"images"/fs::u8path(relative+".json"),prediction.dump());
                save_points(engine,m,out/"measurements"/fs::u8path(relative+".csv"));
            }
            catch(const std::exception&e){
                ++failed;
                row<<",,,,,,,,"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<','<<csv(e.what())<<'\n';
                rows<<row.str();
                failures<<row.str();
            }
            if(n%20==0)std::cout<<"Measured "<<n<<"/341"<<std::endl;
            rows.flush();
        }
        Json summary=Json::dict(),subsets=Json::list();
        for(auto scope:{
            "single","mixed","center"
        }){
            Json s=Json::dict();
            s["scope"]=scope;
            s["n"]=counts[scope];
            s["both_within_009"]=passes[scope];
            s["Y_MAE_successful"]=sumY[scope]/counts[scope];
            s["T_MAE_successful"]=sumT[scope]/counts[scope];
            s["Y_max_successful"]=maxY[scope];
            s["T_max_successful"]=maxT[scope];
            subsets.push(s);
        }
        summary["subsets"]=subsets;
        summary["n"]=n;
        summary["inference_failures"]=failed;
        summary["expanded_search_images"]=wide;
        summary["protocol"]="frozen_model_resub341_fresh_cpp_image_measurement";
        summary["model_sha256"]=sha256(fs::u8path(argv[3]));
        summary["created_kst"]=now();
        if(!times.empty()){
            std::sort(times.begin(),times.end());
            summary["seconds_median"]=times[times.size()/2];
        }
        write(out/"summary.json",summary.dump());
        std::cout<<summary.dump()<<'\n';
        return failed?2:0;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
