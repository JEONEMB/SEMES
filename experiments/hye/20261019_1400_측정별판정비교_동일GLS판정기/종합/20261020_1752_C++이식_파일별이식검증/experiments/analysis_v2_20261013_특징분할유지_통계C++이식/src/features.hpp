#pragma once
#include "alignment.hpp"
#include "table.hpp"
namespace v2_history {
    using namespace semes;
    struct Observation {
        std::string file;
        double value=0,mean_shift=0;
        std::map<std::string,double> features;
    };
    struct Features {
        std::vector<Observation> ys,ts;
        std::vector<std::pair<std::string,std::vector<std::string>>> configurations;
        bool joined_center_present=false;
        explicit Features(const fs::path& root) {
            CsvTable qr(root/fs::u8path(u8"14_데이터_QR마커위치.csv")),finder(root/fs::u8path(u8"15_데이터_파인더_볼트이동량.csv")),ecc(root/"analysis_agent/qr_measurements.csv"),hom(root/"analysis_v2/homography_features.csv");
            auto index=[](const CsvTable& t){
                std::map<std::string,size_t> m;
                for(size_t i=0; i<t.rows.size(); ++i)if(!m.emplace(t.text(i,"file"),i).second)throw std::runtime_error("Duplicate join key");
                return m;
            };
            auto fi=index(finder),ei=index(ecc),hi=index(hom);
            std::vector<std::string> affine,homography;
            for(auto& c:ecc.header)if(c.size()>2&&c[0]=='q'&&c[1]>='0'&&c[1]<='2'&&c.substr(c.size()-2)!="cc"&&(c.size()<5||c.substr(c.size()-5)!="match"))affine.push_back(c);
            for(auto& c:hom.header)if(c!="file"&&c!="cc"&&c!="h20"&&c!="h21")homography.push_back(c);
            Observation center;
            for(size_t i=0; i<qr.rows.size(); ++i) {
                std::string name=qr.text(i,"file");
                if(!fi.count(name)||!ei.count(name)||!hi.count(name))continue;
                Observation r;
                r.file=name;
                r.value=qr.number(i,"value");
                for(auto& c:qr.header)if(c!="axis"&&c!="file")r.features[c]=qr.number(i,c);
                for(auto& c:{
                    "finder_dx","finder_dy","bolt_dx"
                })r.features[c]=finder.number(fi.at(name),c);
                for(auto& c:affine)r.features[c]=ecc.number(ei.at(name),c);
                for(auto& c:homography)r.features[c]=hom.number(hi.at(name),c);
                r.mean_shift=(r.features.at("dLx")+r.features.at("dCx")+r.features.at("dRx"))/3;
                if(name=="center.bmp"){
                    center=r;
                    joined_center_present=true;
                }
                else if(name[0]=='Y')ys.push_back(r);
                else if(name[0]=='T')ts.push_back(r);
            }
            if(ys.size()!=80||ts.size()!=200)throw std::runtime_error("Expected joined 280 moving images");
            if(joined_center_present){
                ys.insert(ys.begin(),center);
                ts.insert(ts.begin(),center);
            }
            auto order=[](const Observation& a,const Observation& b){
                return a.mean_shift<b.mean_shift;
            };
            std::stable_sort(ys.begin(),ys.end(),order);
            std::stable_sort(ts.begin(),ts.end(),order);
            std::vector<std::string> x3={
                "dLx","dCx","dRx"
            },geometry=x3;
            for(auto& c:{
                "dLy","dCy","dRy","Lw","Cw","Rw","Lh","Ch","Rh"
            })geometry.push_back(c);
            auto combined=geometry;
            combined.insert(combined.end(),affine.begin(),affine.end());
            combined.insert(combined.end(),homography.begin(),homography.end());
            auto bolt=x3;
            bolt.push_back("bolt_dx");
            configurations={
                {
                    u8"v1 (QR 네모 가로 3개)",x3
                },{
                    u8"v2-b (QR 네모 가로, 세로, 폭, 높이 12개)",geometry
                }, {
                    u8"v2-a1 (QR별 내부 픽셀 ECC affine 18개)",affine
                },{
                    u8"v2-a2 (판 전체 원근 변환 14개)",homography
                }, {
                    u8"v2-a3 (QR만 전부 합침)",combined
                },{
                    u8"v2-d (참고: QR + 뒤쪽 볼트)",bolt
                }
            };
        }
    };
}
