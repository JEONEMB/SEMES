#include "../../analysis_v3_20261013_대응짝유지_그림렌더러C++변경/src/common.hpp"
#include "source_provenance.hpp"
using namespace twin_history;
namespace {
    const std::vector<std::string> fields={
        u8"영역",u8"특징점",u8"알고리즘","T_RMS",u8"T_최대",u8"Y_평균","Y_RMS",u8"Y_최대","Y_009안","Y_005안",u8"Y불가",u8"Θ불가",u8"오판율",u8"σΘ"
    };
    const std::vector<std::string> headers={
        u8"영역",u8"특징점 조합",u8"알고리즘",u8"T 오차 RMS",u8"T 오차 최대",u8"Y 오차 평균 (mm)",u8"Y 오차 RMS (mm)",u8"Y 오차 최대 (mm)",u8"Y 0.09mm 안 (%)",u8"Y 0.05mm 안 (%)",u8"Y 구분 불가 (/80)",u8"T 구분 불가 (/71)",u8"같은 축 오판율",u8"σT 판정 기준"
    };
    std::string style_xml(){
        auto xml=ooxml::styles();
        auto replace=[&](const std::string& from,const std::string& to){
            size_t at=xml.find(from);
            if(at==std::string::npos)throw std::runtime_error("Unexpected base style");
            xml.replace(at,from.size(),to);
        };
        replace("<numFmts count=\"3\">","<numFmts count=\"4\"><numFmt numFmtId=\"167\" formatCode=\"0.0000\"/>");
        replace("<fills count=\"4\">","<fills count=\"5\">");
        replace("</fills>","<fill><patternFill patternType=\"solid\"><fgColor rgb=\"FFE2EFDA\"/></patternFill></fill></fills>");
        replace("<cellXfs count=\"11\">","<cellXfs count=\"15\">");
        replace("</cellXfs>","<xf numFmtId=\"167\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyNumberFormat=\"1\"><alignment vertical=\"center\"/></xf><xf numFmtId=\"167\" fontId=\"1\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyNumberFormat=\"1\"><alignment vertical=\"center\"/></xf><xf numFmtId=\"164\" fontId=\"0\" fillId=\"4\" borderId=\"0\" xfId=\"0\" applyNumberFormat=\"1\"><alignment vertical=\"center\"/></xf><xf numFmtId=\"164\" fontId=\"1\" fillId=\"4\" borderId=\"0\" xfId=\"0\" applyNumberFormat=\"1\"><alignment vertical=\"center\"/></xf></cellXfs>");
        return xml;
    }
    int metric_style(int col,bool bold,double value){
        if(col==3||col==4||col==13)return 11+bold;
        if(col==5||col==6||col==7)return (col==6&&value<=.09?13:3)+bold;
        if(col==8||col==9)return 5+bold;
        if(col==12)return 7+bold;
        return bold;
    }
    void table(ooxml::Sheet& sheet,int header,const CsvTable& results,const std::vector<size_t>& order,const std::set<size_t>& best){
        sheet.header_row=header;
        sheet.row_heights[header]=60;
        sheet.widths={
            12,46,40,13,13,16,16,16,16,16,16,16,17,16
        };
        for(int c=0; c<14; ++c)sheet.text(header,c,headers[c],2);
        for(size_t i=0; i<order.size(); ++i){
            size_t r=order[i];
            int row=header+1+int(i);
            bool bold=best.count(r);
            for(int c=0; c<14; ++c)if(c<3)sheet.text(row,c,results.text(r,fields[c]),bold);
            else{
                double value=results.number(r,fields[c]);
                sheet.number(row,c,value,metric_style(c,bold,value));
            }
        }
    }
}
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try{
        if(argc!=4)throw std::runtime_error("historical_v3_outputs original-workspace cpp-v3-evaluation new-output");
        fs::path root=fs::u8path(argv[1]),input=fs::u8path(argv[2]),out=fs::u8path(argv[3]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        CsvTable results(input/"v3_results.csv"),predictions(input/"predictions.csv");
        if(results.rows.size()!=66||predictions.rows.size()!=66*281)throw std::runtime_error("Full v3 evaluation required");
        std::vector<size_t> order;
        for(int phase=0; phase<3; ++phase)for(size_t i=0; i<results.rows.size(); ++i){
            bool g=results.text(i,"key")=="G1",outside=results.text(i,u8"영역")!=u8"QR만";
            if((phase==0&&!g&&!outside)||(phase==1&&g)||(phase==2&&outside))order.push_back(i);
        }
        std::vector<std::string> combos,algorithms;
        std::map<std::string,size_t> best_by_feature,by_key;
        for(size_t i:order){
            auto feature=results.text(i,u8"특징점"),algorithm=results.text(i,u8"알고리즘");
            if(std::find(combos.begin(),combos.end(),feature)==combos.end())combos.push_back(feature);
            if(std::find(algorithms.begin(),algorithms.end(),algorithm)==algorithms.end())algorithms.push_back(algorithm);
            auto old=best_by_feature.find(feature);
            if(old==best_by_feature.end()||results.number(i,"Y_RMS")<results.number(old->second,"Y_RMS"))best_by_feature[feature]=i;
            if(!by_key.emplace(results.text(i,"key"),i).second)throw std::runtime_error("Duplicate metric key");
        }
        std::vector<size_t> best;
        std::set<size_t> highlighted;
        for(auto& feature:combos){
            best.push_back(best_by_feature.at(feature));
            highlighted.insert(best.back());
        }
        size_t qbest=results.rows.size(),obest=qbest;
        for(auto row:best){
            auto& at=results.text(row,u8"영역")==u8"QR만"?qbest:obest;
            if(at==results.rows.size()||results.number(row,"Y_RMS")<results.number(at,"Y_RMS"))at=row;
        }
        if(qbest==results.rows.size()||obest==results.rows.size())throw std::runtime_error("Missing QR/outside best");
        std::vector<std::string> keys;
        for(auto candidate:{
            std::string("Q1_C1"),std::string("Q7_C5"),results.text(qbest,"key"),std::string("G1"),std::string("O3_C5"),results.text(obest,"key")
        })if(std::find(keys.begin(),keys.end(),candidate)==keys.end())keys.push_back(candidate);
        std::vector<ooxml::Sheet> sheets(5);
        auto& summary=sheets[0];
        summary.name=u8"요약";
        summary.frozen_rows=5;
        summary.frozen_columns=3;
        summary.merges={
            "A1:N1","A2:N2","A3:N3"
        };
        summary.row_heights[1]=28;
        summary.row_heights[2]=30;
        summary.row_heights[3]=30;
        summary.text(1,0,u8"Y/T 구분표 v3: 과거 홀짝 2-fold 특징모델·픽셀 정합 비교",10);
        summary.text(2,0,u8"QR만 최소 Y RMS: "+results.text(qbest,u8"특징점")+" + "+results.text(qbest,u8"알고리즘"));
        summary.text(3,0,u8"굵은 행: 조합별 최소 Y RMS. 초록: Y RMS≤0.09mm. 과거 구분 판정과 현재 두 축 허용오차 동시 통과율은 다름.");
        table(summary,5,results,order,highlighted);
        sheets[1].name=u8"조합별 최고";
        table(sheets[1],1,results,best,{
        });
        std::map<std::pair<std::string,std::string>,size_t> pi;
        for(size_t i=0; i<predictions.rows.size(); ++i)if(!pi.emplace(std::make_pair(predictions.text(i,"key"),predictions.text(i,"file")),i).second)throw std::runtime_error("Duplicate per-image key");
        int detail_rows=0;
        for(int axis=0; axis<2; ++axis){
            auto& sheet=sheets[2+axis];
            sheet.name=axis?u8"Θ":"Y";
            sheet.frozen_rows=1;
            sheet.frozen_columns=4;
            sheet.widths={
                9,16,13,15
            };
            std::vector<std::string> heads={
                u8"인덱스",u8"파일명",axis?u8"명령값 T (단위 미확정)":u8"명령값 Y (mm)",u8"짝의 T (보간)"
            };
            for(auto& key:keys){
                heads.push_back(key+u8" 추정 T");
                heads.push_back(u8"추정 Y (mm)");
                heads.push_back(u8"판정");
                sheet.widths.insert(sheet.widths.end(),{
                    16,16,13
                });
            }
            for(size_t c=0; c<heads.size(); ++c)sheet.text(1,int(c),heads[c],2);
            sheet.row_heights[1]=60;
            std::vector<size_t> rows;
            for(size_t i=0; i<predictions.rows.size(); ++i)if(predictions.text(i,"key")=="Q1_C1"&&predictions.number(i,axis?"true_T":"true_Y")!=0)rows.push_back(i);
            std::stable_sort(rows.begin(),rows.end(),[&](size_t a,size_t b){
                return predictions.number(a,axis?"true_T":"true_Y")<predictions.number(b,axis?"true_T":"true_Y");
            });
            if(rows.size()!=(axis?200:80))throw std::runtime_error("Wrong detail population");
            for(size_t j=0; j<rows.size(); ++j){
                int row=int(j+2);
                size_t i=rows[j];
                auto file=predictions.text(i,"file");
                size_t q=pi.at({
                    results.text(qbest,"key"),file
                });
                bool inrange=!predictions.text(q,"twin_T").empty(),bold=inrange&&predictions.number(q,"can_distinguish")==0;
                sheet.number(row,0,int(j+1),bold);
                sheet.text(row,1,file,bold);
                sheet.number(row,2,predictions.number(i,axis?"true_T":"true_Y"),3+bold);
                if(!predictions.text(i,"twin_T").empty())sheet.number(row,3,predictions.number(i,"twin_T"),11+bold);
                for(size_t k=0; k<keys.size(); ++k){
                    size_t p=pi.at({
                        keys[k],file
                    });
                    int c=4+int(k)*3;
                    sheet.number(row,c,predictions.number(p,"pred_T"),11+bold);
                    sheet.number(row,c+1,predictions.number(p,"pred_Y"),3+bold);
                    std::string flag=predictions.text(p,"twin_T").empty()?u8"범위 밖":predictions.number(p,"can_distinguish")?u8"가능":u8"불가";
                    sheet.text(row,c+2,flag,flag==u8"불가"?9:bold);
                }
                ++detail_rows;
            }
        }
        auto& explanation=sheets[4];
        explanation.name=u8"설명";
        explanation.widths={
            150
        };
        const std::vector<std::string> notes={
            u8"원본 make_v3_outputs.py의 요약·조합별 최고·Y·Θ·설명 5개 시트 역할 유지. CSV 입력은 native C++ v3 평가 출력.",u8"원본의 홀짝 2-fold·전체 OOF 잔차 기반 σ와 규칙3을 재현한 개발자료 분석이다. 새 독립 시험 성적이 아니다.",u8"C1~C5는 초기 특징모델. C6는 이차 곡선과 감쇠 Gauss–Newton. G1은 픽셀 직접 정합이며 입력 평가의 kind를 확인한다.",u8"구분 가능: 추정 T가 상대 축 보간 짝의 T에서 3σ보다 멀다. 같은 축 오판율과 Y/T 오차 지표를 별도로 기록한다.",u8"단일축 학습에서 T×Y 상호작용은 식별되지 않는다. 반복 관찰한 개발 자료이며 현재 규칙은 루트 ALGORITHM_RULES.md를 따른다.",u8"원본 표의 Θ 구분불가 분모71은 보간 범위 안 사진에 대한 과거 표기이며, 상세 Θ 시트는200장 전체를 포함한다.",u8"원본의 정답 자체 약40µm 오차 설명은 별도 실측으로 확인하지 못한 당시 주장이다. T 물리 단위도 미확정이다.",u8"Python pickle/openpyxl/Matplotlib을 C++ CSV/OOXML/OpenCV/GDI로 대체. 출력 값은 동결 평가 결과이며 편집 시 모델을 재학습하는 Excel 수식은 원본에도 없다."
        };
        for(size_t i=0; i<notes.size(); ++i){
            explanation.text(int(i+1),0,notes[i]);
            explanation.row_heights[int(i+1)]=36;
        }
        fs::create_directories(out);
        fs::path workbook=out/fs::u8path(u8"25_YΘ구분표_v3_cpp.xlsx");
        ooxml::save(workbook,sheets,style_xml());
        ooxml::Reader reread(workbook);
        int cell_mismatch=0,checked_cells=0;
        for(int s=0; s<5; ++s){
            auto cells=reread.sheet(s+1);
            for(auto& row:sheets[s].rows)for(auto& entry:row.second){
                auto address=ooxml::column(entry.first)+std::to_string(row.first);
                if(!cells.count(address))++cell_mismatch;
                ++checked_cells;
            }
        }
        // Independent original workbook control for every metric and selected per-image output.
        ooxml::Reader original(root/fs::u8path(u8"25_YΘ구분표_v3.xlsx"));
        auto original_summary=original.sheet(1);
        std::map<std::pair<std::string,std::string>,int> original_rows;
        for(int row=6; row<100; ++row){
            auto feature=original_summary.find("B"+std::to_string(row)),algorithm=original_summary.find("C"+std::to_string(row));
            if(feature!=original_summary.end()&&algorithm!=original_summary.end())original_rows.emplace(std::make_pair(feature->second.value,algorithm->second.value),row);
        }
        int compared=0,numeric_failed=0;
        double max_difference=0;
        for(size_t i:order){
            int row=original_rows.at({
                results.text(i,u8"특징점"),results.text(i,u8"알고리즘")
            });
            for(int c=3; c<14; ++c){
                double difference=std::abs(results.number(i,fields[c])-std::stod(original_summary.at(ooxml::column(c)+std::to_string(row)).value));
                max_difference=std::max(max_difference,difference);
                numeric_failed+=difference>1e-4;
                ++compared;
            }
        }
        int original_detail_values=0,original_detail_failures=0,original_detail_flag_failures=0;
        double original_detail_max_difference=0;
        for(int axis=0; axis<2; ++axis){
            auto control=original.sheet(3+axis),written=reread.sheet(3+axis);
            std::map<std::string,int> control_rows;
            for(auto& cell:control)if(cell.first[0]=='B'&&cell.second.value.size()>4&&cell.second.value.substr(cell.second.value.size()-4)==".bmp")control_rows.emplace(cell.second.value,std::stoi(cell.first.substr(1)));
            for(int row=2; row<=(axis?201:81); ++row){
                int expected_row=control_rows.at(written.at("B"+std::to_string(row)).value);
                for(int col=2; col<4+int(keys.size())*3; ++col){
                    auto address=ooxml::column(col);
                    auto a=control.find(address+std::to_string(expected_row)),b=written.find(address+std::to_string(row));
                    std::string expected=a==control.end()?"":a->second.value,actual=b==written.end()?"":b->second.value;
                    if(col>=4&&(col-4)%3==2){
                        original_detail_flag_failures+=expected!=actual;
                        continue;
                    }
                    if(expected.empty()||actual.empty()){
                        original_detail_failures+=expected!=actual;
                        continue;
                    }
                    double difference=std::abs(std::stod(expected)-std::stod(actual));
                    original_detail_max_difference=std::max(original_detail_max_difference,difference);
                    original_detail_failures+=difference>1e-4;
                    ++original_detail_values;
                }
            }
        }
        for(size_t index=0; index<order.size(); ++index){
            auto written=reread.sheet(1);
            size_t source=order[index];
            int row=int(index+6);
            for(int col=0; col<14; ++col){
                auto value=written.at(ooxml::column(col)+std::to_string(row)).value;
                if(col<3)cell_mismatch+=value!=results.text(source,fields[col]);
                else cell_mismatch+=std::abs(std::stod(value)-results.number(source,fields[col]))>1e-12;
            }
        }
        int left=600,top=180,cw=180,ch=54;
        cv::Mat plot(top+int(combos.size())*ch+115,left+int(algorithms.size())*cw+35,CV_8UC3,cv::Scalar::all(255));
        text(plot,u8"특징점 조합 × 알고리즘: Y RMS (µm)와 Y 사진 구분 불가 수",{
            25,15
        },29,ink,true);
        text(plot,u8"과거 홀짝 2-fold / 로그 색상 10~1000µm / Y80장 판정 / 새 독립 시험 아님",{
            25,60
        },21,sub);
        for(size_t j=0; j<algorithms.size(); ++j){
            auto label=algorithms[j];
            auto split=label.find(' ');
            text(plot,label.substr(0,split),{
                left+int(j)*cw+cw/2,125
            },25,ink,true,true);
        }
        int heat_values=0;
        for(size_t i=0; i<combos.size(); ++i){
            int y=top+int(i)*ch;
            text(plot,combos[i],{
                18,y+14
            },17,ink);
            for(size_t j=0; j<algorithms.size(); ++j){
                auto found=std::find_if(order.begin(),order.end(),[&](size_t r){
                    return results.text(r,u8"특징점")==combos[i]&&results.text(r,u8"알고리즘")==algorithms[j];
                });
                int x=left+int(j)*cw;
                if(found==order.end()){
                    text(plot,"-",{
                        x+cw/2,y+15
                    },20,sub,false,true);
                    continue;
                }
                double value=1000*results.number(*found,"Y_RMS"),level=(std::log10(std::clamp(value,10.,1000.))-1)/2;
                cv::Scalar color(107+148*level,48+203*level,8+239*level);
                cv::rectangle(plot,cv::Rect(x,y,cw-2,ch-2),color,cv::FILLED);
                auto font=level<.45?cv::Scalar::all(255):ink;
                std::ostringstream label;
                label<<std::fixed<<std::setprecision(0)<<value<<u8"µm / 불가 "<<int(results.number(*found,u8"Y불가"));
                text(plot,label.str(),{
                    x+cw/2,y+16
                },17,font,false,true);
                ++heat_values;
            }
            if(combos[i].rfind("O1",0)==0)cv::line(plot,{
                0,y
            },{
                plot.cols,y
            },orange,3);
        }
        text(plot,u8"위: QR만 / 아래: QR 밖 포함. 같은 자료·역사적 판정 재현이며 현재 모델 채택 기준이 아님.",{
            25,plot.rows-65
        },21,sub);
        save(out/fs::u8path(u8"26_v3_조합x알고리즘_히트맵.png"),plot);
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["source_sha256"]=semes::original_source_hash(root,"analysis_v3/make_v3_outputs.py");
        audit["input_metrics_sha256"]=sha256(input/"v3_results.csv");
        audit["input_predictions_sha256"]=sha256(input/"predictions.csv");
        audit["original_workbook_sha256"]=sha256(root/fs::u8path(u8"25_YΘ구분표_v3.xlsx"));
        audit["worksheets"]=5;
        audit["summary_rows"]=int(order.size());
        audit["best_rows"]=int(best.size());
        audit["detail_rows"]=detail_rows;
        audit["key_methods"]=int(keys.size());
        audit["heatmap_numeric_cells"]=heat_values;
        audit["saved_cells_presence_checked"]=checked_cells;
        audit["missing_saved_cells"]=cell_mismatch;
        audit["original_metric_values_compared"]=compared;
        audit["max_original_metric_difference"]=max_difference;
        audit["original_metric_tolerance"]=1e-4;
        audit["failed_original_metrics"]=numeric_failed;
        audit["qbest_key"]=results.text(qbest,"key");
        audit["obest_key"]=results.text(obest,"key");
        audit["changed"]="Native C++ OOXML/OpenCV/GDI export replaces pickle/openpyxl/Matplotlib. Same five worksheet roles, ranking and selected-method deduplication. Layout/font/raster/ZIP bytes differ. Reads C++ evaluation CSVs; G1 kind remains as stated by that evaluation. No workbook model-training formulas invented.";
        audit["original_detail_numeric_values_compared"]=original_detail_values;
        audit["original_detail_max_difference"]=original_detail_max_difference;
        audit["original_detail_numeric_failures"]=original_detail_failures;
        audit["original_detail_flag_failures"]=original_detail_flag_failures;
        audit["G1_input_kind"]=predictions.text(pi.at({
            "G1","center.bmp"
        }),"kind");
        audit["passed"]=cell_mismatch==0&&numeric_failed==0&&original_detail_failures==0&&original_detail_flag_failures==0;
        write(out/"checks.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return cell_mismatch||numeric_failed||original_detail_failures||original_detail_flag_failures?2:0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
