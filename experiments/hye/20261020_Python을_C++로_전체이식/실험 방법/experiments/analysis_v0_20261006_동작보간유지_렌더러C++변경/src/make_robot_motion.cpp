#include "alignment.hpp"
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <iostream>
#include <fstream>
using namespace semes;
using Microsoft::WRL::ComPtr;
namespace {
    void check(HRESULT result,const char* operation){
        if(FAILED(result))throw std::runtime_error(std::string(operation)+" HRESULT="+std::to_string(unsigned(result)));
    }
    cv::Scalar color(unsigned rgb){
        return {
            double(rgb&255),double((rgb>>8)&255),double((rgb>>16)&255),255
        };
    }
    double lerp(double a,double b,double t){
        return a+(b-a)*std::clamp(t,0.,1.);
    }
    struct Canvas {
        HDC dc=nullptr;
        HBITMAP bitmap=nullptr;
        HGDIOBJ previous=nullptr;
        cv::Mat image;
        Canvas(){
            dc=CreateCompatibleDC(nullptr);
            BITMAPINFO info{
            };
            info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
            info.bmiHeader.biWidth=960;
            info.bmiHeader.biHeight=-560;
            info.bmiHeader.biPlanes=1;
            info.bmiHeader.biBitCount=32;
            info.bmiHeader.biCompression=BI_RGB;
            void* bits=nullptr;
            bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&bits,nullptr,0);
            if(!bitmap||!bits)throw std::runtime_error("Cannot create drawing surface");
            previous=SelectObject(dc,bitmap);
            image=cv::Mat(560,960,CV_8UC4,bits);
        }
        ~Canvas(){
            SelectObject(dc,previous);
            DeleteObject(bitmap);
            DeleteDC(dc);
        }
        void text(int x,int y,const wchar_t* value,int pixels,unsigned rgb){
            HFONT font=CreateFontW(-pixels,0,0,0,FW_NORMAL,FALSE,FALSE,FALSE,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,ANTIALIASED_QUALITY,DEFAULT_PITCH,L"Malgun Gothic");
            auto old=SelectObject(dc,font);
            SetBkMode(dc,TRANSPARENT);
            SetTextColor(dc,RGB((rgb>>16)&255,(rgb>>8)&255,rgb&255));
            RECT rect{
                x,y,960,560
            };
            DrawTextW(dc,value,-1,&rect,DT_LEFT|DT_TOP|DT_NOPREFIX);
            SelectObject(dc,old);
            DeleteObject(font);
            GdiFlush();
        }
        void rect(int x0,int y0,int x1,int y1,unsigned fill,unsigned outline=0,int width=1){
            cv::rectangle(image,{
                x0,y0
            },{
                x1,y1
            },color(fill),-1);
            if(outline)cv::rectangle(image,{
                x0,y0
            },{
                x1,y1
            },color(outline),width);
        }
        void ellipse(double x0,double y0,double x1,double y1,unsigned fill,unsigned outline=0,int width=1){
            cv::Point c(int(std::nearbyint((x0+x1)/2)),int(std::nearbyint((y0+y1)/2)));
            cv::Size axes(std::max(1,int((x1-x0)/2)),std::max(1,int((y1-y0)/2)));
            cv::ellipse(image,c,axes,0,0,360,color(fill),-1,cv::LINE_AA);
            if(outline)cv::ellipse(image,c,axes,0,0,360,color(outline),width,cv::LINE_AA);
        }
        void line(double x0,double y0,double x1,double y1,unsigned rgb,int width){
            cv::line(image,{
                int(std::nearbyint(x0)),int(std::nearbyint(y0))
            },{
                int(std::nearbyint(x1)),int(std::nearbyint(y1))
            },color(rgb),width,cv::LINE_AA);
        }
        void round(int x0,int y0,int x1,int y1,int radius,unsigned fill,unsigned outline=0,int width=1){
            auto shape=[&](int a,int b,int c,int d,int r,unsigned rgb){
                cv::rectangle(image,{
                    a+r,b
                },{
                    c-r,d
                },color(rgb),-1);
                cv::rectangle(image,{
                    a,b+r
                },{
                    c,d-r
                },color(rgb),-1);
                for(int x:{
                    a+r,c-r
                })for(int y:{
                    b+r,d-r
                })cv::circle(image,{
                    x,y
                },r,color(rgb),-1,cv::LINE_AA);
            };
            if(outline){
                shape(x0,y0,x1,y1,radius,outline);
                shape(x0+width,y0+width,x1-width,y1-width,std::max(1,radius-width),fill);
            }
            else shape(x0,y0,x1,y1,radius,fill);
        }
        void polygon(std::vector<cv::Point> points,unsigned rgb){
            cv::fillConvexPoly(image,points,color(rgb),cv::LINE_AA);
        }
    };
    void qr(Canvas& c,int x,int y){
        c.rect(x,y,x+20,y+20,0xffffff,0x263c56);
        for(auto d:{
            cv::Point(2,2),cv::Point(13,2),cv::Point(2,13)
        }){
            cv::rectangle(c.image,{
                x+d.x,y+d.y
            },{
                x+d.x+5,y+d.y+5
            },color(0x17273a),2);
        }
        for(auto d:{
            cv::Point(11,10),cv::Point(14,14),cv::Point(9,16)
        })c.rect(x+d.x,y+d.y,x+d.x+2,y+d.y+2,0x17273a);
    }
    void png(const fs::path& p,const cv::Mat& image){
        std::vector<unsigned char> buffer;
        cv::imencode(".png",image,buffer);
        write(p,std::string(buffer.begin(),buffer.end()));
    }
    class Gif {
        ComPtr<IWICImagingFactory> factory_;
        ComPtr<IWICStream> stream_;
        ComPtr<IWICBitmapEncoder> encoder_;
        ComPtr<IWICPalette> palette_;
        public: explicit Gif(const fs::path& p){
            check(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory_)),"WIC factory");
            check(factory_->CreateStream(&stream_),"GIF stream");
            check(stream_->InitializeFromFilename(p.c_str(),GENERIC_WRITE),"GIF output");
            check(factory_->CreateEncoder(GUID_ContainerFormatGif,nullptr,&encoder_),"GIF encoder");
            check(encoder_->Initialize(stream_.Get(),WICBitmapEncoderNoCache),"GIF initialize");
            check(factory_->CreatePalette(&palette_),"GIF palette");
            WICColor colors[256];
            for(int i=0; i<256; ++i){
                unsigned r=((i>>5)&7)*255/7,g=((i>>2)&7)*255/7,b=(i&3)*255/3;
                colors[i]=0xff000000|(r<<16)|(g<<8)|b;
            }
            check(palette_->InitializeCustom(colors,256),"Palette colors");
            check(encoder_->SetPalette(palette_.Get()),"Global palette");
            // WIC metadata types and paths: Microsoft native format metadata query specification.
            // https://learn.microsoft.com/en-us/windows/win32/wic/-wic-native-image-format-metadata-queries
            ComPtr<IWICMetadataQueryWriter> meta;
            check(encoder_->GetMetadataQueryWriter(&meta),"GIF metadata");
            unsigned char application[]="NETSCAPE2.0",loop[]={
                3,1,0,0,0
            };
            PROPVARIANT value{
            };
            value.vt=VT_VECTOR|VT_UI1;
            value.caub.cElems=11;
            value.caub.pElems=application;
            check(meta->SetMetadataByName(L"/appext/Application",&value),"GIF application");
            value.caub.cElems=5;
            value.caub.pElems=loop;
            check(meta->SetMetadataByName(L"/appext/Data",&value),"GIF loop");
        }
        void frame(const cv::Mat& bgr){
            ComPtr<IWICBitmapFrameEncode> frame;
            ComPtr<IPropertyBag2> options;
            check(encoder_->CreateNewFrame(&frame,&options),"GIF frame");
            check(frame->Initialize(options.Get()),"Frame initialize");
            check(frame->SetSize(bgr.cols,bgr.rows),"Frame size");
            WICPixelFormatGUID format=GUID_WICPixelFormat8bppIndexed;
            check(frame->SetPixelFormat(&format),"Frame pixel format");
            check(frame->SetPalette(palette_.Get()),"Frame palette");
            std::vector<unsigned char> indexed(bgr.total());
            for(int y=0; y<bgr.rows; ++y)for(int x=0; x<bgr.cols; ++x){
                auto p=bgr.at<cv::Vec3b>(y,x);
                indexed[size_t(y)*bgr.cols+x]=(p[2]&224)|((p[1]>>3)&28)|(p[0]>>6);
            }
            ComPtr<IWICMetadataQueryWriter> meta;
            check(frame->GetMetadataQueryWriter(&meta),"Frame metadata");
            PROPVARIANT delay{
            };
            delay.vt=VT_UI2;
            delay.uiVal=10;
            check(meta->SetMetadataByName(L"/grctlext/Delay",&delay),"Frame delay");
            check(frame->WritePixels(bgr.rows,bgr.cols,UINT(indexed.size()),indexed.data()),"GIF pixels");
            check(frame->Commit(),"Frame commit");
        }
        void finish(){
            check(encoder_->Commit(),"GIF commit");
            encoder_.Reset();
            stream_.Reset();
        }
        Json verify(const fs::path& p){
            ComPtr<IWICBitmapDecoder> decoder;
            check(factory_->CreateDecoderFromFilename(p.c_str(),nullptr,GENERIC_READ,WICDecodeMetadataCacheOnLoad,&decoder),"GIF decode");
            UINT count=0;
            check(decoder->GetFrameCount(&count),"GIF frame count");
            unsigned duration=0;
            for(UINT i=0; i<count; ++i){
                ComPtr<IWICBitmapFrameDecode> frame;
                check(decoder->GetFrame(i,&frame),"Decoded frame");
                UINT w,h;
                frame->GetSize(&w,&h);
                if(w!=960||h!=560)throw std::runtime_error("GIF frame dimensions changed");
                ComPtr<IWICMetadataQueryReader> meta;
                check(frame->GetMetadataQueryReader(&meta),"Decoded metadata");
                PROPVARIANT delay{
                };
                check(meta->GetMetadataByName(L"/grctlext/Delay",&delay),"Decoded delay");
                duration+=delay.uiVal*10;
                PropVariantClear(&delay);
            }
            Json j=Json::dict();
            j["decoded_frames"]=int(count);
            j["duration_ms"]=int(duration);
            if(count!=70||duration!=7000)throw std::runtime_error("GIF timing validation failed");
            return j;
        }
    };
}
int wmain(int argc,wchar_t** wide_argv) {
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try {
        if(argc!=2)throw std::runtime_error("historical_robot_motion new-output");
        fs::path out=fs::u8path(argv[1]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        fs::create_directories(out);
        check(CoInitializeEx(nullptr,COINIT_MULTITHREADED),"COM initialize");
        Gif gif(out/"robot_teaching_motion.gif");
        Canvas c;
        std::ofstream timeline(out/"frames.csv");
        timeline<<"frame,time_s,phase,x,y,angle_degrees\n";
        for(int i=0; i<70; ++i){
            double t=i/10.,x,y,angle;
            int phase;
            const wchar_t* caption;
            if(t<1.8){
                phase=1;
                caption=L"① 로봇팔이 선반 앞에 접근";
                x=lerp(324,550,t/1.8);
                y=322;
                angle=0;
            }
            else if(t<3.1){
                phase=2;
                caption=L"② 카메라가 QR 3개를 촬영";
                x=550;
                y=322;
                angle=0;
            }
            else if(t<4.8){
                phase=3;
                caption=L"③ 영상으로 Y·Θ 차이를 계산하고 보정";
                x=550;
                y=lerp(322,309,(t-3.1)/1.7);
                angle=lerp(-7,0,(t-3.1)/1.7);
            }
            else{
                phase=4;
                caption=L"④ 위치가 맞으면 좁은 공간에 진입";
                x=lerp(550,705,(t-4.8)/1.8);
                y=309;
                angle=0;
            }
            timeline<<std::setprecision(17)<<i<<','<<t<<','<<phase<<','<<x<<','<<y<<','<<angle<<'\n';
            c.image.setTo(color(0xf4f8fc));
            c.text(35,22,L"클린 공정 로봇 티칭: 가능한 동작 흐름",29,0x17324d);
            c.text(36,61,L"실제 장비 구조가 확인되지 않은 개념 애니메이션",16,0x6c8297);
            c.round(625,112,935,421,12,0xd7e2eb,0x748ba0,3);
            for(int sy:{
                145,225,345,405
            })c.rect(625,sy,935,sy+19,0x9aafc0,0x60788e,2);
            c.rect(646,247,906,276,0xa6b8c8,0x526d83,2);
            for(int qx:{
                685,765,845
            })qr(c,qx,251);
            c.text(662,181,L"선반 사이의 좁은 공간",21,0x17324d);
            c.line(653,299,921,299,0xbcd2e0,3);
            c.round(45,245,188,405,18,0x7e9cb4,0x425e76,4);
            c.ellipse(157,290,210,343,0x546f87,0x233d57,3);
            c.line(183,317,292,258,0x557c9d,42);
            c.ellipse(268,235,316,283,0x355a7c,0x17324d,3);
            c.line(292,259,x-75,y,0x668bac,34);
            c.ellipse(x-88,y-14,x-60,y+14,0x355a7c,0x17324d,2);
            double rad=angle*CV_PI/180,ex=x+85*std::cos(rad),ey=y+85*std::sin(rad);
            c.line(x-70,y,ex,ey,0x456c8e,10);
            c.ellipse(ex-50,ey-9,ex+50,ey+9,0x86d0dd,0x2b8394,2);
            c.text(112,425,L"로봇팔: 대상물을 옮기는 역할 (추정)",16,0x3b5b76);
            int cx=int(x-18),cy=int(y-74);
            c.round(cx-23,cy-14,cx+23,cy+14,5,0x263f5c,0x85b5d8,2);
            c.ellipse(cx+12,cy-7,cx+25,cy+7,0x7acbe7,0x102e49,2);
            if(phase==2){
                c.polygon({
                    {
                        cx+25,cy-10
                    },{
                        cx+25,cy+10
                    },{
                        754,255
                    }
                },0xe0f8ff);
                c.line(cx+25,cy,754,255,0x40afd5,3);
                c.text(344,139,L"QR 위치·기울기 측정",21,0x1787b1);
            }
            else if(phase==3){
                c.line(x+40,y+40,x+40,y+8,0xe67943,5);
                c.polygon({
                    {
                        int(x+32),int(y+12)
                    },{
                        int(x+48),int(y+12)
                    },{
                        int(x+40),int(y)
                    }
                },0xe67943);
                cv::ellipse(c.image,{
                    int(x+25.5),int(y+18.5)
                },{
                    44,41
                },0,210,320,color(0xe67943),4,cv::LINE_AA);
                c.text(350,139,L"Y 이동 + Θ 회전 보정",21,0xc46939);
            }
            c.round(35,472,925,538,15,0x173f63);
            c.text(59,486,caption,29,0xffffff);
            c.text(748,24,L"개념도",16,0x7f92a4);
            cv::Mat bgr;
            cv::cvtColor(c.image,bgr,cv::COLOR_BGRA2BGR);
            gif.frame(bgr);
            if(i==0||i==18||i==31||i==48||i==69)png(out/("frame_"+std::to_string(i)+".png"),bgr);
        }
        gif.finish();
        auto report=gif.verify(out/"robot_teaching_motion.gif");
        report["source"]=u8"make_robot_motion.py, observed creation 2026-10-06 17:04:35 KST";
        report["preserved"]="960x560, 70 logical frames at 100ms, four motion phases and original Korean concept disclaimers, geometry and time equations";
        report["changed"]="Pillow raster/text and optimized GIF palette replaced by OpenCV/GDI drawing and WIC fixed 256-color GIF. Pixel identity and identical compressed frame layout are not claimed.";
        report["created_kst"]=now();
        write(out/"checks.json",report.dump());
        std::cout<<report.dump()<<'\n';
        return 0;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
