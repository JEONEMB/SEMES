#include "selection.hpp"
#include <iostream>
#include <tuple>
using namespace semes;
using namespace historical_v1;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try{
        if(argc!=3)throw std::runtime_error("historical_v1_combos original-workspace new-output");
        fs::path root=fs::u8path(argv[1]),out=fs::u8path(argv[2]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Tracks data(root);
        auto reference=Json::parse(read(root/"analysis_v3/v1_data/combos.json"));
        std::vector<int> train,test,good;
        for(int i=0; i<data.x.rows; ++i)(i%2==0?train:test).push_back(i);
        Json info=Json::list();
        std::vector<double> sigma;
        double max_info_difference=0;
        for(int k=0; k<data.dx.cols; ++k){
            auto coefficient=least_squares(data.x,data.dx.col(k));
            cv::Mat residual=data.dx.col(k)-data.x*coefficient;
            double squared=0,maximum=0,minscore=1;
            for(int i=0; i<residual.rows; ++i){
                double value=residual.at<double>(i);
                squared+=value*value;
                maximum=std::max(maximum,std::abs(value));
                minscore=std::min(minscore,data.score.at<double>(i,k));
            }
            Json row=Json::dict();
            row["k"]=k;
            row["x"]=data.points.at<double>(k,0);
            row["y"]=data.points.at<double>(k,1);
            row["a"]=coefficient.at<double>(0);
            row["b"]=coefficient.at<double>(1);
            row["ratio"]=coefficient.at<double>(1)/coefficient.at<double>(0);
            row["sig"]=std::sqrt(squared/residual.rows);
            row["maxres"]=maximum;
            row["minscore"]=minscore;
            for(auto field:{
                "x","y","a","b","ratio","sig","maxres","minscore"
            })max_info_difference=std::max(max_info_difference,std::abs(row.at(field).num()-reference.at("info").at(k).at(field).num()));
            sigma.push_back(row.at("sig").num());
            if(maximum<2&&minscore>.8)good.push_back(k);
            info.push(row);
        }
        using Pair=std::tuple<double,double,double,int,int>;
        std::vector<Pair> pairs;
        for(size_t i=0; i<good.size(); ++i)for(size_t j=i+1; j<good.size(); ++j){
            auto result=evaluate(data,{
                good[i],good[j]
            },train,test);
            pairs.emplace_back(result.y,result.t,result.condition,good[i],good[j]);
        }
        std::sort(pairs.begin(),pairs.end());
        if(pairs.empty())throw std::runtime_error("No pair candidates");
        int bolt=good.front();
        double closest=std::numeric_limits<double>::infinity();
        for(int id:good){
            double d=std::hypot(data.points.at<double>(id,0)-3576,data.points.at<double>(id,1)-1802);
            if(d<closest){
                closest=d;
                bolt=id;
            }
        }
        std::vector<int> selected_ids={
            std::get<3>(pairs[0]),std::get<4>(pairs[0])
        },remaining;
        for(int id:good)if(std::find(selected_ids.begin(),selected_ids.end(),id)==selected_ids.end())remaining.push_back(id);
        Json history=Json::list();
        auto add_history=[&](int count,double t,double y){
            Json row=Json::list();
            row.push(count);
            row.push(t);
            row.push(y);
            history.push(row);
        };
        add_history(2,std::get<1>(pairs[0]),std::get<0>(pairs[0]));
        while(!remaining.empty()&&selected_ids.size()<30){
            size_t best=0;
            double score=std::numeric_limits<double>::infinity();
            for(size_t i=0; i<remaining.size(); ++i){
                auto candidate=selected_ids;
                candidate.push_back(remaining[i]);
                double value=evaluate(data,candidate,train,test).y;
                if(value<score){
                    score=value;
                    best=i;
                }
            }
            selected_ids.push_back(remaining[best]);
            remaining.erase(remaining.begin()+best);
            auto result=evaluate(data,selected_ids,train,test);
            add_history(int(selected_ids.size()),result.t,result.y);
        }
        int best_count=2;
        double best_y=std::numeric_limits<double>::infinity();
        for(auto& row:history.array)if(row.at(2).num()<best_y){
            best_y=row.at(2).num();
            best_count=int(row.at(0).num());
        }
        std::vector<int> greedy(selected_ids.begin(),selected_ids.begin()+best_count);
        Json result=Json::dict();
        result["info"]=info;
        result["good"]=ids_json(good);
        result["bolt"]=bolt;
        result["best_pair"]=ids_json({
            std::get<3>(pairs[0]),std::get<4>(pairs[0])
        });
        result["greedy"]=ids_json(greedy);
        result["hist"]=history;
        fs::create_directories(out);
        write(out/"combos.json",result.dump());
        std::ofstream rankings(out/"pair_rankings.csv");
        rankings<<"rank,id1,id2,Y_RMS,T_RMS,condition\n";
        for(size_t i=0; i<pairs.size(); ++i)rankings<<i+1<<','<<std::get<3>(pairs[i])<<','<<std::get<4>(pairs[i])<<','<<std::setprecision(17)<<std::get<0>(pairs[i])<<','<<std::get<1>(pairs[i])<<','<<std::get<2>(pairs[i])<<'\n';
        std::vector<double> weights;
        for(int id:good)weights.push_back(1/sigma[id]);
        auto uniform=evaluate(data,good,train,test),weighted=evaluate(data,good,train,test,weights);
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["points"]=data.dx.cols;
        audit["good_points"]=int(good.size());
        audit["pairs"]=int(pairs.size());
        audit["max_info_difference"]=max_info_difference;
        audit["good_ids_match"]=result.at("good").dump()==reference.at("good").dump();
        audit["best_pair_match"]=result.at("best_pair").dump()==reference.at("best_pair").dump();
        audit["greedy_ids_match"]=result.at("greedy").dump()==reference.at("greedy").dump();
        audit["bolt_match"]=bolt==int(reference.at("bolt").num());
        audit["uniform_T_RMS"]=uniform.t;
        audit["uniform_Y_RMS"]=uniform.y;
        audit["noise_weighted_T_RMS"]=weighted.t;
        audit["noise_weighted_Y_RMS"]=weighted.y;
        audit["limitation"]="Original full-data point filtering and same held-out subset used repeatedly for pair/greedy selection are retained as historical conditions, not independent evaluation.";
        write(out/"checks.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
