#include "common.hpp"
using namespace twin_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try{
        if(argc!=5)throw std::runtime_error("historical_twin_lines original-workspace bmp-20261014 cpp-v2-decisions.csv new-output");
        fs::path root=fs::u8path(argv[1]),data=fs::u8path(argv[2]),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Dataset dataset(root,fs::u8path(argv[3]));
        fs::create_directories(out);
        cv::Mat image(1560,2640,CV_8UC3,cv::Scalar::all(255));
        auto center=read_image(data,"center.bmp");
        cv::Mat thumb;
        cv::resize(center(cv::Rect(1100,200,1600,1800)),thumb,cv::Size(24,27),0,0,cv::INTER_AREA);
        cv::cvtColor(thumb,thumb,cv::COLOR_GRAY2BGR);
        for(int p=0; p<2; ++p){
            auto& panel=dataset.panels[p];
            int top=50+p*700;
            auto xy=[&](double x,double y){
                return cv::Point(int(90+(x+2)*2460/104),int(top+80+(12.6-y)*530/15.8));
            };
            auto xY=[](double v){
                return 50+25*v;
            };
            auto xT=[](double v){
                return 50+50*v;
            };
            auto by=bad(panel.yt),bt=bad(panel.ty);
            text(image,panel.label+u8"  |  Y 쪽 "+std::to_string(panel.yt.size())+u8"세트, T 쪽 "+std::to_string(panel.ty.size())+u8"세트",{
                70,top
            },29,ink,true);
            for(auto& pair:panel.yt)connection(image,xy(xY(pair.source),8),xy(xT(pair.target),2.9),blue,2);
            for(auto& pair:panel.ty)connection(image,xy(xT(pair.source),2.9),xy(xY(pair.target),8),red,1,true);
            for(int k=-40; k<=40; ++k)if(k){
                double x=xY(k*.05),w=100./81*.82;
                bool b=by.count(k*5);
                cv::rectangle(image,xy(x-w/2,10.4),xy(x+w/2,8),b?orange:orange*.35+cv::Scalar::all(255)*.65,cv::FILLED);
                if(b)cv::rectangle(image,xy(x-w/2,10.4),xy(x+w/2,8),cv::Scalar(18,74,156),2);
            }
            for(int k=-100; k<=100; ++k)if(k){
                double x=xT(k*.01),w=100./201*.78;
                bool b=bt.count(k);
                cv::rectangle(image,xy(x-w/2,2.9),xy(x+w/2,.5),b?green:green*.35+cv::Scalar::all(255)*.65,cv::FILLED);
                if(b)cv::rectangle(image,xy(x-w/2,2.9),xy(x+w/2,.5),cv::Scalar(30,122,78),1);
            }
            for(double y:{
                9.2,1.7
            }){
                auto c=xy(50,y);
                thumb.copyTo(image(cv::Rect(c.x-12,c.y-13,24,27)));
                cv::rectangle(image,cv::Rect(c.x-13,c.y-14,26,29),blue,1);
            }
            for(double v:{
                -2.,-1.5,-1.,-.5,.5,1.,1.5,2.
            })text(image,"Y "+signed_value(v),xy(xY(v),11.05),19,sub,false,true);
            for(double v:{
                -1.,-.75,-.5,-.25,.25,.5,.75,1.
            })text(image,"T "+signed_value(v),xy(xT(v),-.5),19,sub,false,true);
            text(image,u8"Y 80장 + center",xy(-2,12.25),23,orange);
            text(image,u8"T 200장 + center",xy(-2,-1.9),23,green);
        }
        text(image,u8"진한 상자: 구분 불가 · 파란 실선: Y→T · 빨간 점선: T→Y · 명령값/QR 위치 보간 짝이며 실제 이동 궤적이 아님",{
            70,1470
        },25,sub);
        text(image,u8"center 미리보기: 원본 1600×1800px 영역을 24×27px로 축소(0.015배). T 물리 단위 미확정.",{
            70,1510
        },22,sub);
        save(out/fs::u8path(u8"27_구분안되는짝_연결도.png"),image);
        auto audit=dataset.audit(root,out);
        audit["width"]=image.cols;
        audit["height"]=image.rows;
        audit["passed"]=true;
        write(out/"checks.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
