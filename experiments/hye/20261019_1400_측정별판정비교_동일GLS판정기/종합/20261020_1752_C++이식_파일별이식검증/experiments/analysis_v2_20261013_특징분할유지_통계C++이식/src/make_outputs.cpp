#include "features.hpp"
#include "ooxml.hpp"
#include "wintext.hpp"
#include <iostream>
using namespace semes;
using namespace v2_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try {
        if(argc!=5)throw std::runtime_error("historical_v2_outputs original-workspace cpp-judge-output cpp-estimate-output new-output");
        fs::path root=fs::u8path(argv[1]),judge=fs::u8path(argv[2]),estimate=fs::u8path(argv[3]),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Features data(root);
        CsvTable summary(judge/"judge_summary.csv"),decisions(judge/"decisions.csv"),est(estimate/"estimate_result.csv"),heatmap(judge/"heatmap.csv");
        std::map<std::pair<std::string,std::string>,size_t> decision_index;
        for(size_t i=0; i<decisions.rows.size(); ++i)decision_index.emplace(std::make_pair(decisions.text(i,"configuration"),decisions.text(i,"file")),i);
        std::vector<ooxml::Sheet> sheets(3);
        auto& overview=sheets[0];
        overview.name=u8"요약";
        overview.header_row=4;
        overview.widths={
            40,55,10,13,13,16,13,13,16,18,18,18
        };
        overview.text(1,0,u8"Y, Θ 구분표 v2: 특징점 조합과 판정 알고리즘 비교",10);
        overview.text(2,0,u8"과거 실험 재현: Y 80장·Θ 71장 범위 내 비교. 현재 OOF 분류 정확도가 아님.");
        const std::vector<std::string> headers={
            u8"버전",u8"사용한 값",u8"값 개수",u8"규칙1 Y 불가",u8"규칙1 Θ 불가",u8"규칙1 같은 축 오판율",u8"규칙2 Y 불가",u8"규칙2 Θ 불가",u8"규칙2 같은 축 오판율",u8"추정 Y 오차 RMS (µm)",u8"추정 Y 최대 (µm)",u8"Y 오차 90µm 초과 (280장 중)"
        };
        for(size_t col=0; col<headers.size(); ++col)overview.text(4,int(col),headers[col],2);
        const std::vector<std::string> descriptions={
            u8"QR 네모 중심 가로 위치 (좌, 중, 우)",u8"v1 + QR 네모 세로 위치, 폭, 높이",u8"QR별 내부 픽셀 전체 ECC affine",u8"판 귀퉁이 4개와 QR 중심 3개의 원근 변환 이동",u8"QR 위치·크기 + QR affine + 판 homography",u8"v1 + 뒤쪽 볼트 가로 위치 (QR 밖, 참고)"
        };
        for(size_t i=0; i<data.configurations.size(); ++i){
            int row=int(i+5),bold=i==3;
            overview.text(row,0,data.configurations[i].first,bold);
            overview.text(row,1,descriptions[i],bold);
            overview.number(row,2,summary.number(i,"dimensions"),bold);
            const std::vector<std::string> fields={
                "rule1_Y_unavailable","rule1_T_unavailable","rule1_fp_rate","rule2_Y_unavailable","rule2_T_unavailable","rule2_fp_rate"
            };
            for(int c=0; c<6; ++c)overview.number(row,c+3,summary.number(i,fields[c]),c==2||c==5?7+bold:bold);
            overview.number(row,9,est.number(i,"Y_RMS_um"),5+bold);
            overview.number(row,10,est.number(i,u8"Y_최대_um"),5+bold);
            overview.number(row,11,est.number(i,u8"Y_90um초과"),bold);
        }
        overview.text(12,0,u8"규칙1: 두 축의 같은 축 보간 잔차 표준편차 중 큰 값 × 3.");
        overview.text(13,0,u8"규칙2: 공분산을 반영한 마할라노비스 거리, chi-square 99.73% 기준.");
        overview.text(14,0,u8"보정·임계값은 당시와 같이 전체 자료에서 계산. 독립 시험 성능으로 해석하지 않음.");
        overview.text(15,0,u8"선형 회귀는 원본의 전체 자료 표준화·교대 2-fold를 재현.");
        overview.text(16,0,u8"파일명 값은 촬영 명령값. T의 물리 단위와 별도 실측 정확도는 확정하지 않음.");
        for(int axis=0; axis<2; ++axis){
            auto& sheet=sheets[axis+1];
            sheet.name=axis==0?"Y":u8"Θ";
            sheet.frozen_rows=1;
            sheet.frozen_columns=2;
            sheet.widths={
                9,16,12,13,13,13,13,13,13,14,19,14,14,14,14,14,14,18
            };
            auto& rows=axis==0?data.ys:data.ts;
            const std::vector<std::string> labels={
                u8"인덱스",u8"파일명",axis==0?u8"값 (mm)":u8"값 (Θ)",u8"좌 QR px",u8"중 QR px",u8"우 QR px",u8"좌중 평균",u8"좌우 평균",u8"중우 평균",u8"좌중우 평균",u8"v1 규칙1"
            };
            for(size_t col=0; col<labels.size(); ++col)sheet.text(1,int(col),labels[col],2);
            for(int c=0; c<6; ++c)sheet.text(1,11+c,data.configurations[c].first.substr(0,data.configurations[c].first.find(' '))+u8" 규칙2",2);
            sheet.text(1,17,u8"v2-a2 통계 거리",2);
            for(size_t i=0; i<rows.size(); ++i){
                auto& observation=rows[i];
                int row=int(i+2);
                size_t best=decision_index.at({
                    data.configurations[3].first,observation.file
                });
                int bold=decisions.text(best,"rule2")==u8"불가";
                sheet.number(row,0,double(i+1),bold);
                sheet.text(row,1,observation.file,bold);
                sheet.number(row,2,observation.value,bold);
                double left=observation.features.at("dLx"),middle=observation.features.at("dCx"),right=observation.features.at("dRx");
                sheet.number(row,3,left,3+bold);
                sheet.number(row,4,middle,3+bold);
                sheet.number(row,5,right,3+bold);
                std::string r=std::to_string(row);
                sheet.number(row,6,(left+middle)/2,3+bold,"AVERAGE(D"+r+":E"+r+")");
                sheet.number(row,7,(left+right)/2,3+bold,"AVERAGE(D"+r+",F"+r+")");
                sheet.number(row,8,(middle+right)/2,3+bold,"AVERAGE(E"+r+":F"+r+")");
                sheet.number(row,9,(left+middle+right)/3,3+bold,"AVERAGE(D"+r+":F"+r+")");
                auto initial=decision_index.at({
                    data.configurations[0].first,observation.file
                });
                auto first=decisions.text(initial,"rule1");
                sheet.text(row,10,first,first==u8"불가"?9:bold);
                for(int c=0; c<6; ++c){
                    auto decision=decision_index.at({
                        data.configurations[c].first,observation.file
                    });
                    auto value=decisions.text(decision,"rule2");
                    sheet.text(row,11+c,value,value==u8"불가"?9:bold);
                }
                if(!decisions.text(best,"distance").empty())sheet.number(row,17,decisions.number(best,"distance"),5+bold);
            }
            sheet.text(int(rows.size()+3),0,u8"통계 거리 기준값 (v2-a2): "+ooxml::numeric(summary.number(3,"distance_threshold"))+u8" 이하면 구분 불가");
        }
        fs::create_directories(out);
        fs::path workbook=out/fs::u8path(u8"23_YΘ구분표_v2_cpp.xlsx");
        ooxml::save(workbook,sheets);
        cv::Mat plot(660,1540,CV_8UC3,cv::Scalar(255,255,255));
        wintext(plot,u8"Y 사진 중 같은 QR 위치의 Θ와 구분 불가인 비율",cv::Point(20,15),27,RGB(25,25,25),true);
        wintext(plot,u8"과거 규칙2 재현 · 명령값 순 분석 · 독립 시험 성적 아님",cv::Point(20,55),18,RGB(90,90,90));
        const std::vector<std::string> bins={
            "0~0.25","0.25~0.5","0.5~0.75","0.75~1","1~1.5","1.5~2"
        };
        for(int bin=0; bin<6; ++bin)wintext(plot,"|Y| "+bins[bin]+" mm",cv::Point(650+bin*145,95),17,RGB(25,25,25));
        for(int i=0; i<6; ++i){
            wintext(plot,data.configurations[i].first,cv::Point(20,146+i*72),18,RGB(25,25,25));
            for(int bin=0; bin<6; ++bin){
                size_t row=size_t(i*6+bin);
                double value=heatmap.number(row,"fraction");
                cv::Scalar color(255-value*100,247-value*150,240-value*220);
                cv::rectangle(plot,cv::Rect(640+bin*145,128+i*72,143,70),color,-1);
                std::ostringstream label;
                label<<std::fixed<<std::setprecision(0)<<value*100<<'%';
                wintext(plot,label.str(),cv::Point(682+bin*145,146+i*72),23,value>.55?RGB(255,255,255):RGB(25,25,25),true);
            }
        }
        wintext(plot,u8"구간마다 Y 사진 10~20장. 짝은 QR 가로 평균 이동이 같은 Θ 지점의 선형 보간.",cv::Point(20,590),19,RGB(70,70,70));
        std::vector<uchar> encoded;
        cv::imencode(".png",plot,encoded);
        std::ofstream png(out/"24_v2_heatmap_cpp.png",std::ios::binary);
        png.write(reinterpret_cast<const char*>(encoded.data()),encoded.size());
        png.close();
        ooxml::Reader reread(workbook);
        int formulas=0;
        for(int sheet:{
            2,3
        })for(auto& cell:reread.sheet(sheet))formulas+=!cell.second.formula.empty();
        if(formulas!=1120)throw std::runtime_error("Expected 1120 average formulas");
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["worksheets"]=3;
        audit["formula_count"]=formulas;
        audit["workbook_sha256"]=sha256(workbook);
        audit["changed"]="C++ OOXML/GDI replaces openpyxl/matplotlib and pickle input. Same three sheet roles, data rows and average formulas; historical caveats clarified; raster/layout/ZIP bytes differ. Formula caches are calculated in C++ and fullCalcOnLoad requested; interactive Excel recalculation not yet checked.";
        write(out/"audit.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
