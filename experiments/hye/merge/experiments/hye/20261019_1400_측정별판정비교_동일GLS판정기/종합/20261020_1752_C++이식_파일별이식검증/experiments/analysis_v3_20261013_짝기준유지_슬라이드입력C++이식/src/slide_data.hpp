#pragma once
#include "../../analysis_v3_20261013_대응짝유지_그림렌더러C++변경/src/common.hpp"
#include <array>
namespace slide_history {
    using namespace twin_history;
    struct Row {
        std::string file;
        double value;
        std::array<double,3> dx;
    };
    struct Method {
        std::string id,name,basis;
        std::map<std::string,std::string> yflags,tflags;
        std::map<std::string,double> ytwins,ttwins;
        std::map<std::string,std::array<double,2>> estimates;
    };
    inline double value(const std::string& file){
        return file=="center.bmp"?0:std::stod(file.substr(1,file.size()-5));
    }
    inline double mean(const Row& row,const std::vector<int>& basis){
        double result=0;
        for(int col:basis)result+=row.dx[col];
        return result/basis.size();
    }
    inline double interpolation(const std::vector<double>& xs,const std::vector<double>& ys,double x){
        auto it=std::upper_bound(xs.begin(),xs.end(),x);
        if(it==xs.begin())return ys.front();
        if(it==xs.end())return ys.back();
        size_t b=size_t(it-xs.begin()),a=b-1;
        return ys[a]+(ys[b]-ys[a])*(x-xs[a])/(xs[b]-xs[a]);
    }
    struct SlideData {
        std::vector<Row> ys,ts;
        std::vector<Method> methods;
        const std::array<double,3> thresholds={
            .0449,.0274,.0478
        };
        SlideData(const fs::path& root,const fs::path& judge,const fs::path& predictions){
            CsvTable q(root/fs::u8path(u8"14_데이터_QR마커위치.csv"));
            for(size_t i=0; i<q.rows.size(); ++i){
                Row r{
                    q.text(i,"file"),q.number(i,"value"),{
                        q.number(i,"dLx"),q.number(i,"dCx"),q.number(i,"dRx")
                    }
                };
                if(r.file=="center.bmp"){
                    ys.push_back(r);
                    ts.push_back(r);
                }
                else if(r.file[0]=='Y')ys.push_back(r);
                else if(r.file[0]=='T')ts.push_back(r);
            }
            if(ys.size()!=81||ts.size()!=201)throw std::runtime_error("Expected center-inclusive81/201 rows");
            auto sort=[](const Row& a,const Row& b){
                return a.value<b.value;
            };
            std::stable_sort(ys.begin(),ys.end(),sort);
            std::stable_sort(ts.begin(),ts.end(),sort);
            const std::vector<std::pair<std::string,std::vector<int>>> bases={
                {
                    u8"좌",{
                        0
                    }
                },{
                    u8"중",{
                        1
                    }
                },{
                    u8"우",{
                        2
                    }
                },{
                    u8"좌중",{
                        0,1
                    }
                },{
                    u8"좌우",{
                        0,2
                    }
                },{
                    u8"중우",{
                        1,2
                    }
                },{
                    u8"좌중우",{
                        0,1,2
                    }
                }
            };
            for(auto& basis:bases)methods.push_back(pairing(basis.first,basis.second));
            methods.push_back(pairing_best());
            auto common=methods[6];
            CsvTable v2(judge);
            for(auto spec:std::vector<std::pair<std::string,std::string>>{
                {
                    "v2-b",u8"v2-b (QR 네모 가로, 세로, 폭, 높이 12개)"
                },{
                    "v2-a1",u8"v2-a1 (QR별 내부 픽셀 ECC affine 18개)"
                },{
                    "v2-a2",u8"v2-a2 (판 전체 원근 변환 14개)"
                }
            }){
                Method m;
                m.id="method_"+spec.first;
                m.name=spec.first;
                m.basis=u8"좌중우";
                m.ytwins=common.ytwins;
                m.ttwins=common.ttwins;
                for(size_t i=0; i<v2.rows.size(); ++i)if(v2.text(i,"configuration")==spec.second){
                    auto file=v2.text(i,"file");
                    (file[0]=='Y'?m.yflags:m.tflags)[file]=v2.text(i,"rule2");
                }
                if(m.yflags.size()!=80||m.tflags.size()!=200)throw std::runtime_error("Incomplete v2 method");
                methods.push_back(m);
            }
            CsvTable v3(predictions);
            for(auto spec:std::vector<std::pair<std::string,std::string>>{
                {
                    "Q7+C5","Q7_C5"
                },{
                    "G1","G1"
                },{
                    "O3+C5","O3_C5"
                }
            }){
                Method m;
                m.id="method_"+spec.first;
                m.name=spec.first;
                m.basis=u8"좌중우";
                m.ytwins=common.ytwins;
                m.ttwins=common.ttwins;
                for(size_t i=0; i<v3.rows.size(); ++i)if(v3.text(i,"key")==spec.second){
                    auto file=v3.text(i,"file");
                    m.estimates[file]={
                        v3.number(i,"pred_T"),v3.number(i,"pred_Y")
                    };
                    if(file=="center.bmp")continue;
                    std::string flag=v3.text(i,"twin_T").empty()?u8"범위 밖":v3.number(i,"can_distinguish")?u8"가능":u8"불가";
                    (v3.number(i,"true_T")==0?m.yflags:m.tflags)[file]=flag;
                }
                if(m.yflags.size()!=80||m.tflags.size()!=200||m.estimates.size()!=281)throw std::runtime_error("Incomplete v3 method");
                methods.push_back(m);
            }
        }
        Method pairing(const std::string& name,const std::vector<int>& basis)const{
            Method m;
            m.id="basis_"+name;
            m.name=name;
            m.basis=name;
            for(int axis=0; axis<2; ++axis){
                const auto& source=axis?ts:ys;
                auto reference=axis?ys:ts;
                std::stable_sort(reference.begin(),reference.end(),[&](const Row& a,const Row& b){
                    return mean(a,basis)<mean(b,basis);
                });
                std::vector<double> xs,command;
                std::array<std::vector<double>,3> observed;
                for(auto& row:reference){
                    xs.push_back(mean(row,basis));
                    command.push_back(row.value);
                    for(int j=0; j<3; ++j)observed[j].push_back(row.dx[j]);
                }
                auto& flags=axis?m.tflags:m.yflags;
                auto& twins=axis?m.ttwins:m.ytwins;
                for(auto& row:source)if(row.file!="center.bmp"){
                    double at=mean(row,basis);
                    if(at<xs.front()||at>xs.back()){
                        flags[row.file]=u8"범위 밖";
                        continue;
                    }
                    bool bad=true;
                    for(int j=0; j<3; ++j)bad=bad&&std::abs(row.dx[j]-interpolation(xs,observed[j],at))<=thresholds[j];
                    flags[row.file]=bad?u8"불가":u8"가능";
                    twins[row.file]=interpolation(xs,command,at);
                }
            }
            return m;
        }
        Method pairing_best()const{
            Method m;
            m.id=u8"basis_최적";
            m.name=u8"최적";
            m.basis=u8"최적";
            for(int axis=0; axis<2; ++axis){
                const auto& source=axis?ts:ys;
                const auto& reference=axis?ys:ts;
                std::vector<double> commands;
                std::array<std::vector<double>,3> observed;
                for(auto& row:reference){
                    commands.push_back(row.value);
                    for(int j=0; j<3; ++j)observed[j].push_back(row.dx[j]);
                }
                std::vector<Row> curve;
                double lo=1e100,hi=-1e100;
                for(int i=0; i<=20000; ++i){
                    Row row;
                    row.value=commands.front()+(commands.back()-commands.front())*(i/20000.);
                    for(int j=0; j<3; ++j)row.dx[j]=interpolation(commands,observed[j],row.value);
                    double a=mean(row,{
                        0,1,2
                    });
                    lo=std::min(lo,a);
                    hi=std::max(hi,a);
                    curve.push_back(row);
                }
                auto& flags=axis?m.tflags:m.yflags;
                auto& twins=axis?m.ttwins:m.ytwins;
                for(auto& row:source)if(row.file!="center.bmp"){
                    double at=mean(row,{
                        0,1,2
                    });
                    if(at<lo||at>hi){
                        flags[row.file]=u8"범위 밖";
                        continue;
                    }
                    double best=1e100,twin=0;
                    for(auto& candidate:curve){
                        double score=0;
                        for(int j=0; j<3; ++j)score=std::max(score,std::abs(row.dx[j]-candidate.dx[j])/thresholds[j]);
                        if(score<best){
                            best=score;
                            twin=candidate.value;
                        }
                    }
                    flags[row.file]=best<=1?u8"불가":u8"가능";
                    twins[row.file]=twin;
                }
            }
            return m;
        }
    };
    inline Json summary(const std::map<std::string,std::string>& flags){
        std::map<int,std::vector<std::string>> by_magnitude;
        int bad=0,total=0,maxbad=0;
        for(auto& flag:flags)if(flag.second!=u8"범위 밖"){
            int v=std::abs(key(value(flag.first)));
            by_magnitude[v].push_back(flag.second);
            ++total;
            if(flag.second==u8"불가"){
                ++bad;
                maxbad=std::max(maxbad,v);
            }
        }
        int allbad=0;
        for(auto& group:by_magnitude){
            if(std::all_of(group.second.begin(),group.second.end(),[](const std::string& s){
                return s==u8"불가";
            }))allbad=group.first;
            else break;
        }
        Json allgood;
        for(auto it=by_magnitude.rbegin(); it!=by_magnitude.rend(); ++it){
            if(std::all_of(it->second.begin(),it->second.end(),[](const std::string& s){
                return s==u8"가능";
            }))allgood=it->first/100.;
            else break;
        }
        Json j=Json::dict();
        j["n_bad"]=bad;
        j["n"]=total;
        j["allbad"]=allbad/100.;
        j["allgood"]=allgood;
        j["maxbad"]=maxbad/100.;
        return j;
    }
    inline Json serialize(const Method& m){
        Json j=Json::dict();
        j["id"]=m.id;
        j["name"]=m.name;
        j["basis"]=m.basis;
        for(auto pair:std::vector<std::pair<std::string,const std::map<std::string,std::string>*>>{
            {
                "flagY",&m.yflags
            },{
                "flagT",&m.tflags
            }
        }){
            Json flags=Json::dict();
            for(auto& x:*pair.second)flags[x.first]=x.second;
            j[pair.first]=flags;
        }
        for(auto pair:std::vector<std::pair<std::string,const std::map<std::string,double>*>>{
            {
                "twinY",&m.ytwins
            },{
                "twinT",&m.ttwins
            }
        }){
            Json twins=Json::dict();
            for(auto& x:*pair.second)twins[x.first]=x.second;
            j[pair.first]=twins;
        }
        Json estimates=Json::dict();
        for(auto& x:m.estimates){
            Json values=Json::list();
            for(double v:x.second)values.push(v);
            estimates[x.first]=values;
        }
        j["est"]=estimates;
        j["summaryY"]=summary(m.yflags);
        j["summaryT"]=summary(m.tflags);
        return j;
    }
}
