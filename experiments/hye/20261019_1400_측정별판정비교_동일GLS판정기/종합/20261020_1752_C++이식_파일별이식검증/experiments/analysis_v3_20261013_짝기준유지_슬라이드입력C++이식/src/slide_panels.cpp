#include "slide_data.hpp"
#include "source_provenance.hpp"
using namespace slide_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try{
        if(argc!=5)throw std::runtime_error("historical_slide_panels original-workspace bmp-20261014 cpp-methods.json new-output");
        fs::path root=fs::u8path(argv[1]),data=fs::u8path(argv[2]),input=fs::u8path(argv[3]),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        auto methods=Json::parse(read(input));
        if(methods.size()!=14)throw std::runtime_error("Expected all14 methods");
        CsvTable q(root/fs::u8path(u8"14_데이터_QR마커위치.csv"));
        int cx=0,cy=0;
        for(size_t i=0; i<q.rows.size(); ++i)if(q.text(i,"file")=="center.bmp"){
            cx=int(q.number(i,"Cx"));
            cy=int(q.number(i,"Cy"));
        }
        if(!cx||!cy)throw std::runtime_error("No center crop");
        std::map<std::string,cv::Mat> thumbnails;
        auto crop=[&](const std::string& file){
            auto old=thumbnails.find(file);
            if(old==thumbnails.end()){
                auto image=read_image(data,file);
                cv::Mat reduced;
                cv::resize(image(cv::Rect(cx-450,cy-450,900,900)),reduced,cv::Size(96,96),0,0,cv::INTER_AREA);
                old=thumbnails.emplace(file,reduced).first;
            }
            cv::Mat resized;
            cv::resize(old->second,resized,cv::Size(44,44),0,0,cv::INTER_AREA);
            cv::cvtColor(resized,resized,cv::COLOR_GRAY2BGR);
            return resized;
        };
        fs::create_directories(out);
        int pasted=0;
        Json figures=Json::list();
        cv::Mat contact(14*156,1460,CV_8UC3,cv::Scalar::all(255));
        for(size_t index=0; index<methods.size(); ++index){
            const auto& method=methods.at(index);
            cv::Mat image(360,3948,CV_8UC3,cv::Scalar::all(255));
            auto xY=[](double v){
                return 30+(v+2)/.05*48+22;
            };
            auto xT=[](double v){
                return 30+(v+.4)/.01*48+22;
            };
            const auto& fy=method.at("flagY");
            const auto& ft=method.at("flagT");
            for(auto& twin:method.at("twinY").object)if(fy.at(twin.first).str()==u8"불가"&&std::abs(twin.second.num())<=.4)connection(image,{
                xY(value(twin.first)),82
            },{
                xT(twin.second.num()),244
            },blue,2);
            for(auto& twin:method.at("twinT").object)if(ft.at(twin.first).str()==u8"불가"&&std::abs(value(twin.first))<=.4){
                cv::Point2d a(xT(value(twin.first)),244),b(xY(twin.second.num()),82);
                for(int part=0; part<30; part+=2)cv::line(image,a+(b-a)*(part/30.),a+(b-a)*((part+1)/30.),red,1,cv::LINE_AA);
            }
            for(int axis=0; axis<2; ++axis){
                int y=axis?248:34;
                auto& flags=axis?ft:fy;
                auto color=axis?green:orange;
                for(int i=0; i<=80; ++i){
                    int command=axis?-40+i:-200+5*i;
                    std::string file=command?std::string(axis?"T":"Y")+signed_value(command/100.)+".bmp":"center.bmp";
                    bool bad=flags.has(file)&&flags.at(file).str()==u8"불가";
                    auto photo=crop(file);
                    if(!bad&&command)photo=photo*.35+cv::Scalar::all(255)*.65;
                    int x=30+i*48;
                    photo.copyTo(image(cv::Rect(x,y,44,44)));
                    int border=bad||!command?4:1;
                    cv::rectangle(image,cv::Rect(x-border,y-border,44+2*border,44+2*border),command?color:ink,border);
                    ++pasted;
                    if(i%10==0||!command)text(image,command?std::string(axis?"T ":"Y ")+signed_value(command/100.,axis?2:1):"center",{
                        x+22,axis?300:4
                    },20,sub,false,true);
                }
            }
            text(image,u8"동일 900×900px 영역 → 96×96 → 44×44 (약0.04889배). 명령값 보간 짝이며 실제 궤적이 아님. T 단위 미확정.",{
                30,334
            },18,sub);
            auto filename=method.at("id").str()+".png";
            save(out/fs::u8path(filename),image);
            text(contact,method.at("name").str(),{
                10,int(index)*156+2
            },20,ink,true);
            cv::Mat reduced;
            cv::resize(image,reduced,cv::Size(1420,129),0,0,cv::INTER_AREA);
            reduced.copyTo(contact(cv::Rect(20,int(index)*156+25,1420,129)));
            Json fig=Json::dict();
            fig["file"]=filename;
            fig["width"]=image.cols;
            fig["height"]=image.rows;
            fig["source_canvas_height"]=332;
            figures.push(fig);
        }
        save(out/"contact.png",contact);
        Json report=Json::dict();
        report["created_kst"]=now();
        report["source_sha256"]=semes::original_source_hash(root,"analysis_v3/slide_panels.py");
        report["input_methods_sha256"]=sha256(input);
        report["panels"]=figures;
        report["photos_pasted"]=pasted;
        report["unique_source_photos_read"]=int(thumbnails.size());
        report["retained"]="Original14 panels, Y81/T81 grid, 900px fixed crop then96px cache then44px thumbnail, white blend0.65, direction/color, original30-segment red dashes, sz44 gap4 mid170.";
        report["changed"]="Native C++ OpenCV/GDI renderer and explicit input/output. Adds28px caption below original332px layout to state shrink factor and interpolated-command meaning. Renderer pixels differ; original files remain untouched.";
        report["passed"]=pasted==2268&&thumbnails.size()==161;
        write(out/"checks.json",report.dump());
        std::cout<<report.dump()<<'\n';
        return report["passed"].num()?0:2;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
