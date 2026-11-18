#include "selection.hpp"
#include "wintext.hpp"
#include <tuple>
#include <iostream>
using namespace semes;
using namespace historical_v1;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try{
        if(argc!=5)throw std::runtime_error("historical_v1_stable original-workspace data-bmp-20261014 combos.json new-output");
        fs::path root=fs::u8path(argv[1]),images=fs::u8path(argv[2]),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Tracks data(root);
        auto combos=Json::parse(read(fs::u8path(argv[3]))),reference=Json::parse(read(root/"analysis_v3/v1_data/stable.json"));
        struct Region{
            std::string name;
            cv::Rect bounds;
        };
        std::vector<Region> regions={
            {
                u8"라벨",cv::Rect(1740,1195,320,165)
            },{
                u8"QR 데이터",cv::Rect(1560,975,690,185)
            },{
                u8"케이블",cv::Rect(1840,0,420,560)
            }
        };
        std::vector<int> ok,excluded,fit,selection,final;
        Json reasons=Json::dict();
        for(int id:indices(combos.at("good"))){
            double x=data.points.at<double>(id,0),y=data.points.at<double>(id,1);
            std::string reason;
            for(auto& region:regions)if(id!=0&&region.bounds.x<x&&x<region.bounds.x+region.bounds.width&&region.bounds.y<y&&y<region.bounds.y+region.bounds.height){
                reason=region.name;
                break;
            }
            if(reason.empty())ok.push_back(id);
            else{
                excluded.push_back(id);
                reasons[std::to_string(id)]=reason;
            }
        }
        for(int i=0; i<data.x.rows; ++i)(i%4<2?fit:i%4==2?selection:final).push_back(i);
        using Pair=std::tuple<double,int,int>;
        std::vector<Pair> pairs;
        for(size_t i=0; i<ok.size(); ++i)for(size_t j=i+1; j<ok.size(); ++j)pairs.emplace_back(evaluate(data,{
            ok[i],ok[j]
        },fit,selection).y,ok[i],ok[j]);
        std::sort(pairs.begin(),pairs.end());
        if(pairs.empty())throw std::runtime_error("No stable pairs");
        std::vector<int> best={
            std::get<1>(pairs[0]),std::get<2>(pairs[0])
        },chosen=best,remaining;
        for(int id:ok)if(std::find(best.begin(),best.end(),id)==best.end())remaining.push_back(id);
        fs::create_directories(out);
        std::ofstream rankings(out/"pairs.csv"),greedy(out/"greedy.csv");
        rankings<<"rank,id1,id2,selection_Y_RMS,final_T_RMS,final_Y_RMS,condition\n";
        greedy<<"points,added_id,final_T_RMS,final_Y_RMS\n";
        for(size_t i=0; i<pairs.size(); ++i){
            auto& pair=pairs[i];
            auto score=evaluate(data,{
                std::get<1>(pair),std::get<2>(pair)
            },fit,final);
            rankings<<i+1<<','<<std::get<1>(pair)<<','<<std::get<2>(pair)<<','<<std::setprecision(17)<<std::get<0>(pair)<<','<<score.t<<','<<score.y<<','<<score.condition<<'\n';
        }
        for(int step=0; step<10; ++step){
            size_t best_index=0;
            double minimum=std::numeric_limits<double>::infinity();
            for(size_t i=0; i<remaining.size(); ++i){
                auto candidate=chosen;
                candidate.push_back(remaining[i]);
                double score=evaluate(data,candidate,fit,selection).y;
                if(score<minimum){
                    minimum=score;
                    best_index=i;
                }
            }
            int added=remaining[best_index];
            chosen.push_back(added);
            remaining.erase(remaining.begin()+best_index);
            auto result=evaluate(data,chosen,fit,final);
            greedy<<chosen.size()<<','<<added<<','<<std::setprecision(17)<<result.t<<','<<result.y<<'\n';
        }
        Json result=Json::dict();
        result["ok"]=ids_json(ok);
        result["excl"]=reasons;
        result["best"]=ids_json(best);
        write(out/"stable.json",result.dump());
        auto center=cv::imdecode(bytes(images/"center.bmp"),cv::IMREAD_COLOR);
        if(center.empty())throw std::runtime_error("Center decode failed");
        for(auto& region:regions){
            auto r=region.bounds;
            cv::rectangle(center,r,cv::Scalar(60,60,255),8);
            wintext(center,u8"제외: "+region.name,cv::Point(r.x+10,region.name==u8"케이블"?570:r.y+r.height+8),56,RGB(255,60,60),true);
        }
        for(int id:ok)cv::circle(center,cv::Point(cvRound(data.points.at<double>(id,0)),cvRound(data.points.at<double>(id,1))),16,cv::Scalar(255,200,60),6);
        for(int id:excluded){
            cv::Point p(cvRound(data.points.at<double>(id,0)),cvRound(data.points.at<double>(id,1)));
            cv::line(center,p-cv::Point(14,14),p+cv::Point(14,14),cv::Scalar(60,60,255),6);
            cv::line(center,p+cv::Point(-14,14),p+cv::Point(14,-14),cv::Scalar(60,60,255),6);
        }
        for(int id:best){
            cv::Point p(cvRound(data.points.at<double>(id,0)),cvRound(data.points.at<double>(id,1)));
            cv::circle(center,p,34,cv::Scalar(0,230,255),12);
            wintext(center,"#"+std::to_string(id),p+cv::Point(44,-40),72,RGB(255,230,0),true);
        }
        for(auto size:{
            cv::Size(1920,1080),cv::Size(960,540)
        }){
            cv::Mat resized;
            cv::resize(center,resized,size,0,0,cv::INTER_LANCZOS4);
            std::vector<uchar> encoded;
            cv::imencode(".png",resized,encoded);
            std::ofstream file(out/("feature_map_"+std::to_string(size.width)+".png"),std::ios::binary);
            file.write(reinterpret_cast<const char*>(encoded.data()),encoded.size());
        }
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["ok_ids_match"]=result.at("ok").dump()==reference.at("ok").dump();
        audit["best_pair_match"]=result.at("best").dump()==reference.at("best").dump();
        audit["kept"]=int(ok.size());
        audit["excluded"]=int(excluded.size());
        audit["best"]=ids_json(best);
        audit["changed"]="Same strict exclusion boxes and modulo4 fit/selection/final split, native C++ selection and GDI/OpenCV rendering. Full-data prefiltering remains a historical evaluation limitation. Original Pillow stroke and Lanczos pixels differ.";
        write(out/"checks.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
