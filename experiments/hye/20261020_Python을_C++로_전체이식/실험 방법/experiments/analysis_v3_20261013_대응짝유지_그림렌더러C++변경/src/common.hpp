#pragma once
#include "../../analysis_v2_20261013_특징분할유지_통계C++이식/src/features.hpp"
#include "ooxml.hpp"
#include "wintext.hpp"
#include <iostream>
#include <set>
namespace twin_history {
    using namespace semes;
    using v2_history::Observation;
    const cv::Scalar orange(49,125,237),green(71,173,112),blue(120,78,31),red(0,0,192),ink(30,31,31),sub(88,94,95);
    inline std::string signed_value(double v,int digits=2){
        std::ostringstream s;
        s<<std::showpos<<std::fixed<<std::setprecision(digits)<<v;
        return s.str();
    }
    inline int key(double v){
        return int(std::lround(v*100));
    }
    inline int text_width(const std::string& value,int size,bool bold=false){
        int n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,value.data(),int(value.size()),nullptr,0);
        if(!n)throw std::runtime_error("Invalid UTF8 label");
        std::wstring wide(n,L' ');
        MultiByteToWideChar(CP_UTF8,0,value.data(),int(value.size()),wide.data(),n);
        HDC dc=CreateCompatibleDC(nullptr);
        HFONT font=CreateFontW(-size,0,0,0,bold?FW_BOLD:FW_NORMAL,0,0,0,DEFAULT_CHARSET,0,0,0,0,L"Malgun Gothic");
        auto old=SelectObject(dc,font);
        SIZE extent;
        BOOL ok=GetTextExtentPoint32W(dc,wide.data(),n,&extent);
        SelectObject(dc,old);
        DeleteObject(font);
        DeleteDC(dc);
        if(!ok)throw std::runtime_error("Cannot measure text");
        return extent.cx;
    }
    inline void text(cv::Mat& image,const std::string& value,cv::Point origin,int size,cv::Scalar color=ink,bool bold=false,bool centered=false){
        if(value.empty())return;
        int width=text_width(value,size,bold);
        if(centered)origin.x-=width/2;
        cv::Rect box(origin.x-3,origin.y-3,width+8,size*2+8);
        box&=cv::Rect(0,0,image.cols,image.rows);
        if(box.empty())return;
        auto roi=image(box);
        wintext(roi,value,origin-box.tl(),size,RGB(int(color[2]),int(color[1]),int(color[0])),bold);
    }
    inline void connection(cv::Mat& image,cv::Point2d a,cv::Point2d b,cv::Scalar color,int width=1,bool dashed=false){
        if(!dashed){
            cv::line(image,a,b,color,width,cv::LINE_AA);
            return;
        }
        for(int s=0; s<40; s+=2)cv::line(image,a+(b-a)*(s/40.),a+(b-a)*((s+1)/40.),color,width,cv::LINE_AA);
    }
    inline void save(const fs::path& path,const cv::Mat& image){
        std::vector<unsigned char> encoded;
        if(!cv::imencode(".png",image,encoded))throw std::runtime_error("PNG encode failed");
        write(path,std::string(encoded.begin(),encoded.end()));
        auto check=cv::imdecode(bytes(path),cv::IMREAD_UNCHANGED);
        if(check.size()!=image.size())throw std::runtime_error("PNG decode dimension mismatch");
    }
    inline cv::Mat read_image(const fs::path& data,const std::string& name){
        auto path=name=="center.bmp"?data/"center.bmp":data/(name[0]=='T'?"Theta":"Y")/fs::u8path(name);
        auto image=cv::imdecode(bytes(path),cv::IMREAD_GRAYSCALE);
        if(image.size()!=cv::Size(3840,2160))throw std::runtime_error("Expected full 3840x2160 image");
        return image;
    }
    struct Pair {
        double source,target,mean_shift;
        std::string file;
    };
    struct Panel {
        std::string label;
        std::vector<Pair> yt,ty;
    };
    struct Dataset {
        v2_history::Features features;
        std::vector<Panel> panels;
        int v1_flag_mismatches=0,compared_v1_rows=0;
        double interpolation_max_difference=0,noncenter_interpolation_max_difference=0;
        int zero_bridge_rows=0;
        std::map<std::string,bool> old_flags;
        std::map<std::string,double> old_targets;
        fs::path workbook,decisions;
        Dataset(const fs::path& root,const fs::path& decision_path):features(root),workbook(root/fs::u8path(u8"22_YΘ구분표_v1.xlsx")),decisions(decision_path){
            ooxml::Reader excel(workbook);
            for(int sheet=2; sheet<=3; ++sheet){
                auto cells=excel.sheet(sheet);
                if(cells.at("B1").value!=u8"파일명"||cells.at("P1").value!=u8"판정")throw std::runtime_error("Unexpected v1 workbook headers");
                for(int row=2; row<300; ++row){
                    auto name=cells.find("B"+std::to_string(row));
                    if(name==cells.end()||name->second.value.empty())continue;
                    std::string file=name->second.value;
                    if(file=="center.bmp"||file.size()<4||file.substr(file.size()-4)!=".bmp")continue;
                    auto decision=cells.at("P"+std::to_string(row)).value;
                    if(!old_flags.emplace(file,decision==u8"구분 불가").second)throw std::runtime_error("Duplicate Excel filename: "+file);
                    auto target=cells.find("L"+std::to_string(row));
                    if(target!=cells.end()&&!target->second.value.empty())old_targets[file]=std::stod(target->second.value);
                }
            }
            if(old_flags.size()!=280)throw std::runtime_error("Expected 280 original v1 decisions");
            CsvTable table(decision_path);
            std::map<std::string,bool> v1,v2;
            for(size_t i=0; i<table.rows.size(); ++i){
                auto config=table.text(i,"configuration");
                if(config==u8"v1 (QR 네모 가로 3개)")v1.emplace(table.text(i,"file"),table.text(i,"rule1")==u8"불가");
                if(config==u8"v2-a2 (판 전체 원근 변환 14개)")v2.emplace(table.text(i,"file"),table.text(i,"rule2")==u8"불가");
            }
            if(v1.size()!=280||v2.size()!=280)throw std::runtime_error("Expected complete C++ v1/v2 decisions");
            for(auto& item:old_flags){
                ++compared_v1_rows;
                v1_flag_mismatches+=item.second!=v1.at(item.first);
            }
            panels={
                {
                    u8"v1: QR 네모 가로 위치",{
                    },{
                    }
                },{
                    u8"v2-a2: 판 전체 원근 변환 (과거 비교)",{
                    },{
                    }
                }
            };
            for(size_t p=0; p<panels.size(); ++p){
                auto& flags=p==0?old_flags:v2;
                for(auto& row:features.ys)if(flags.at(row.file))panels[p].yt.push_back({
                    row.value,interpolate(features.ts,row.mean_shift),row.mean_shift,row.file
                });
                for(auto& row:features.ts)if(flags.at(row.file))panels[p].ty.push_back({
                    row.value,interpolate(features.ys,row.mean_shift),row.mean_shift,row.file
                });
            }
            auto compare=[&](const std::vector<Observation>& rows,const std::vector<Observation>& reference){
                for(auto& row:rows)if(old_targets.count(row.file)){
                    double difference=std::abs(interpolate(reference,row.mean_shift)-old_targets.at(row.file));
                    interpolation_max_difference=std::max(interpolation_max_difference,difference);
                    auto upper=std::upper_bound(reference.begin(),reference.end(),row.mean_shift,[](double x,const Observation& r){
                        return x<r.mean_shift;
                    });
                    bool bridges=upper!=reference.begin()&&upper!=reference.end()&&(upper-1)->value*upper->value<0;
                    if(bridges)++zero_bridge_rows;
                    else noncenter_interpolation_max_difference=std::max(noncenter_interpolation_max_difference,difference);
                }
            };
            compare(features.ys,features.ts);
            compare(features.ts,features.ys);
            if(v1_flag_mismatches||noncenter_interpolation_max_difference>1e-10)throw std::runtime_error("Alternative Excel input mismatch: flags="+std::to_string(v1_flag_mismatches)+" noncenter interpolation="+std::to_string(noncenter_interpolation_max_difference));
        }
        static double interpolate(const std::vector<Observation>& rows,double mean){
            auto at=std::upper_bound(rows.begin(),rows.end(),mean,[](double x,const Observation& r){
                return x<r.mean_shift;
            });
            if(at==rows.begin())return rows.front().value;
            if(at==rows.end())return rows.back().value;
            auto& a=*(at-1);
            auto& b=*at;
            return a.value+(b.value-a.value)*(mean-a.mean_shift)/(b.mean_shift-a.mean_shift);
        }
        Json audit(const fs::path& root,const fs::path& out)const{
            Json report=Json::dict();
            report["created_kst"]=now();
            report["original_v1_temp_pickle_available"]=false;
            report["replacement_v1_workbook_sha256"]=sha256(workbook);
            report["cpp_decisions_sha256"]=sha256(decisions);
            report["v1_decisions_compared"]=compared_v1_rows;
            report["v1_decision_mismatches"]=v1_flag_mismatches;
            report["v1_workbook_all_interpolation_max_difference"]=interpolation_max_difference;
            report["v1_workbook_noncenter_interpolation_max_difference"]=noncenter_interpolation_max_difference;
            report["zero_bridge_rows_where_workbook_uses_extra_center"]=zero_bridge_rows;
            report["Y_images"]=int(features.ys.size());
            report["T_images"]=int(features.ts.size());
            Json sizes=Json::list();
            std::ofstream table(out/"twins.csv");
            table<<"configuration,direction,source_file,source_command,target_interpolated_command,source_mean_qr_dx\n";
            for(auto& p:panels){
                Json item=Json::dict();
                item["label"]=p.label;
                item["Y_to_T_pairs"]=int(p.yt.size());
                item["T_to_Y_pairs"]=int(p.ty.size());
                sizes.push(item);
                for(auto& x:p.yt)table<<csv(p.label)<<",Y_to_T,"<<csv(x.file)<<','<<std::setprecision(17)<<x.source<<','<<x.target<<','<<x.mean_shift<<'\n';
                for(auto& x:p.ty)table<<csv(p.label)<<",T_to_Y,"<<csv(x.file)<<','<<std::setprecision(17)<<x.source<<','<<x.target<<','<<x.mean_shift<<'\n';
            }
            report["panels"]=sizes;
            report["retained"]= "All original Y/T command grids, mean-QR-dx interpolation, v1 flags checked against original workbook and C++ rule1, v2-a2 rule2, direction/color/dashed connection meanings.";
            report["changed"]="Missing original temporary Yo/To pickle replaced by existing original v1 workbook, cross-checked all280 flags and noncenter interpolated pairs. Original chart join excludes center; Excel interpolation includes it, so zero-bridging pairs intentionally differ and are reported. C++ OpenCV/GDI rendering replaces Matplotlib/Pillow; no pixel-identical rendering claim.";
            report["interpretation"]= "Historical command-space paired-position interpolation. Not a measured motion trajectory, not a modern independent evaluation or a physical unit assertion for T.";
            report["source_lines_sha256"]=sha256(root/"analysis_v3/draw_twin_lines.py");
            report["source_photos_sha256"]=sha256(root/"analysis_v3/draw_twin_photos.py");
            return report;
        }
    };
    inline std::set<int> bad(const std::vector<Pair>& pairs){
        std::set<int> result;
        for(auto& p:pairs)result.insert(key(p.source));
        return result;
    }
}
