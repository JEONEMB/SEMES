#include "../../analysis_v3_20261013_대응짝유지_그림렌더러C++변경/src/common.hpp"
#include "../../analysis_v3_20261013_정합모델유지_픽셀GNC++이식/src/spline.hpp"
#include "source_provenance.hpp"
using namespace twin_history;
struct Sample {
    std::string file,label,segment;
};
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try{
        if(argc!=4)throw std::runtime_error("historical_zero_offset original-workspace bmp-20261014 new-output");
        fs::path root=fs::u8path(argv[1]),data=fs::u8path(argv[2]),out=fs::u8path(argv[3]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        cv::setNumThreads(4);
        CsvTable q(root/fs::u8path(u8"14_데이터_QR마커위치.csv")),reference(root/"analysis_v3/zero_offset_photos.csv");
        std::map<std::string,size_t> qi,ri;
        for(size_t i=0; i<q.rows.size(); ++i)qi.emplace(q.text(i,"file"),i);
        for(size_t i=0; i<reference.rows.size(); ++i)ri.emplace(reference.text(i,u8"사진"),i);
        size_t center=qi.at("center.bmp");
        int x0=int(q.number(center,"Lx")-100),x1=int(q.number(center,"Rx")+100),y0=int(q.number(center,"Cy")-105),y1=int(q.number(center,"Cy")+105);
        int width=x1-x0,height=y1-y0;
        auto load=[&](const std::string& file){
            auto full=read_image(data,file);
            cv::Mat raw,blurred;
            full(cv::Rect(x0-40,y0-20,width+80,height+40)).convertTo(raw,CV_64F);
            cv::GaussianBlur(raw,blurred,cv::Size(),1.);
            return blurred;
        };
        auto cen=load("center.bmp");
        cv::Mat padded;
        cv::copyMakeBorder(cen,padded,12,12,12,12,cv::BORDER_REPLICATE);
        auto coefficients=pixel_history::spline_coefficients(padded);
        const std::vector<Sample> samples={
            {
                "Y-0.10.bmp",u8"Y -0.10mm",u8"1차 -쪽, - 방향"
            },{
                "Y+0.10.bmp",u8"Y +0.10mm",u8"1차 +쪽, + 방향"
            },{
                "Y-0.05.bmp",u8"Y -0.05mm",u8"2차 -쪽, + 방향 (-쪽에서 올라옴)"
            },{
                "Y+0.05.bmp",u8"Y +0.05mm",u8"2차 +쪽, + 방향"
            },{
                "T-0.01.bmp",u8"Θ -0.01",u8"Θ -쪽, - 방향"
            },{
                "T+0.01.bmp",u8"Θ +0.01",u8"Θ +쪽, + 방향"
            }
        };
        fs::create_directories(out);
        std::ofstream table(out/"zero_offset_photos.csv"),pixels(out/"edge_measurements.csv");
        table<<u8"사진,구간,예상px,실제px,차이px,모서리중앙값px,차이um\n";
        pixels<<"file,x,y,expected_intensity,actual_intensity,expected_gradient_x,shift_px,rendered_every_third\n";
        const int panel_height=int(std::round(height*970./width)),row_stride=panel_height+105;
        cv::Mat figure(170+3*row_stride,2100,CV_8UC3,cv::Scalar::all(255));
        text(figure,u8"명령값으로 옮긴 center와 실제 사진의 원점 근처 차이 (과거 진단)",{
            30,15
        },31,ink,true);
        Json details=Json::list();
        int failed=0,total_edges=0,drawn=0;
        double max_difference=0;
        for(size_t index=0; index<samples.size(); ++index){
            auto sample=samples[index];
            double command=std::stod(sample.file.substr(1,sample.file.size()-5)),slope=sample.file[0]=='Y'?14.37:80.5,shift=slope*command;
            auto actual=load(sample.file);
            cv::Mat photo;
            actual(cv::Rect(40,20,width,height)).convertTo(photo,CV_8U);
            cv::cvtColor(photo,photo,cv::COLOR_GRAY2BGR);
            photo=photo*.55+cv::Scalar::all(255)*.45;
            std::vector<double> displacements;
            int rendered=0;
            for(int y=0; y<height; ++y)for(int x=0; x<width; ++x){
                double yy=y+20+12,xx=x+40+12-shift;
                double expected=pixel_history::spline_value(coefficients,xx,yy),gx=(pixel_history::spline_value(coefficients,xx+1,yy)-pixel_history::spline_value(coefficients,xx-1,yy))/2.;
                if(std::abs(gx)<=12)continue;
                double observed=actual.at<double>(y+20,x+40),delta=-(observed-expected)/gx;
                bool render=displacements.size()%3==0;
                displacements.push_back(delta);
                pixels<<csv(sample.file)<<','<<x<<','<<y<<','<<std::setprecision(17)<<expected<<','<<observed<<','<<gx<<','<<delta<<','<<render<<'\n';
                if(render){
                    double magnitude=std::min(std::abs(delta)/.8,1.);
                    cv::Scalar color=delta<0?cv::Scalar(172,102,33):cv::Scalar(31,24,178);
                    color=color*magnitude+cv::Scalar::all(247)*(1-magnitude);
                    cv::circle(photo,{
                        x,y
                    },1,color,cv::FILLED,cv::LINE_AA);
                    ++rendered;
                }
            }
            if(displacements.empty())throw std::runtime_error("No diagnostic edges");
            std::sort(displacements.begin(),displacements.end());
            size_t mid=displacements.size()/2;
            double median=displacements.size()%2?displacements[mid]:(displacements[mid-1]+displacements[mid])/2;
            size_t row=qi.at(sample.file);
            double measured=(q.number(row,"dLx")+q.number(row,"dCx")+q.number(row,"dRx"))/3.,offset=measured-shift;
            std::vector<double> values={
                shift,measured,offset,median
            };
            if(sample.file[0]=='Y')values.push_back(offset/14.37*1000);
            table<<csv(sample.label)<<','<<csv(sample.segment);
            for(double v:values)table<<','<<std::setprecision(17)<<v;
            if(values.size()==4)table<<',';
            table<<'\n';
            const std::vector<std::string> columns={
                u8"예상px",u8"실제px",u8"차이px",u8"모서리중앙값px",u8"차이um"
            };
            for(size_t j=0; j<values.size(); ++j){
                double difference=std::abs(values[j]-reference.number(ri.at(sample.label),columns[j]));
                max_difference=std::max(max_difference,difference);
                failed+=difference>1e-6;
            }
            int px=30+int(index%2)*1030,py=90+int(index/2)*row_stride;
            std::ostringstream title;
            title<<sample.label<<u8"  예상 "<<std::showpos<<std::fixed<<std::setprecision(2)<<shift<<u8"px / 측정 "<<measured<<u8"px / 차이 "<<offset<<"px";
            text(figure,title.str(),{
                px,py
            },22,ink,true);
            text(figure,sample.segment+u8" (원본이 가정한 구간명)",{
                px,py+32
            },18,sub);
            int target_width=970,target_height=int(std::round(height*target_width/double(width)));
            cv::Mat resized;
            cv::resize(photo,resized,cv::Size(target_width,target_height),0,0,cv::INTER_AREA);
            resized.copyTo(figure(cv::Rect(px,py+70,target_width,target_height)));
            Json detail=Json::dict();
            detail["file"]=sample.file;
            detail["edge_points"]=int(displacements.size());
            detail["rendered_every_third"]=rendered;
            detail["median_shift_px"]=median;
            detail["measured_minus_expected_px"]=offset;
            details.push(detail);
            total_edges+=int(displacements.size());
            drawn+=rendered;
        }
        std::ostringstream note;
        note<<u8"좌표 그림 배율 "<<std::setprecision(4)<<970./width<<u8"배. |예상 가로 기울기|>12 경계의 매3번째만 표시. 파랑: 예상보다 왼쪽 / 빨강: 오른쪽. 색 범위 ±0.8px.";
        text(figure,note.str(),{
            30,figure.rows-60
        },21,sub);
        text(figure,u8"예상 사진은 center의 명령값 기반 합성 이동이다. 구간·방향 명칭과 백래시 해석은 당시 가정이며 실제 시간순 이동/독립 위치 계측을 증명하지 않는다.",{
            30,figure.rows-30
        },18,sub);
        save(out/fs::u8path(u8"29_0점어긋남_실제사진.png"),figure);
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["source_sha256"]=semes::original_source_hash(root,"analysis_v3/zero_offset_photos.py");
        audit["reference_csv_sha256"]=sha256(root/"analysis_v3/zero_offset_photos.csv");
        audit["photos"]=6;
        audit["numeric_values_compared"]=28;
        audit["max_difference_from_original_csv"]=max_difference;
        audit["tolerance"]=1e-6;
        audit["failed_values"]=failed;
        audit["edge_points"]=total_edges;
        audit["drawn_edge_points"]=drawn;
        audit["details"]=details;
        audit["retained"]="Float64 Gaussian blur, center shift by original14.37/80.5 coefficients, cubic spline nearest-edge prepadding12, central gradient, abs-gradient>12 mask, row-major every-third scatter and median, original six source images.";
        audit["changed"]="Native C++ float64 spline/OpenCV/GDI; scatter palette/marker/font/layout differ. Full edge table saved including non-rendered points. Historical directional/backlash assumptions explicitly labelled rather than verified as trajectory.";
        audit["passed"]=failed==0;
        write(out/"checks.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return failed?2:0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
