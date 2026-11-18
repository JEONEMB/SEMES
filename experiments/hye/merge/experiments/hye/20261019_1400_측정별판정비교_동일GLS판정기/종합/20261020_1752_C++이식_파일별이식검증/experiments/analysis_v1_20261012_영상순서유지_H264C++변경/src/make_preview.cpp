#include "alignment.hpp"
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>
#include <iostream>
#include <cstring>
using namespace semes;
using Microsoft::WRL::ComPtr;
namespace {
    constexpr int width=1920,height=1080,fps=20;
    const std::array<cv::Point,3> points={
        cv::Point(1644,1065),cv::Point(1902,1065),cv::Point(2157,1065)
    };
    const cv::Scalar reference_color(255,230,0),current_color(240,90,255),white(245,245,245);
    void check(HRESULT hr,const char* operation) {
        if(FAILED(hr)){
            std::ostringstream s;
            s<<operation<<" failed: 0x"<<std::hex<<static_cast<unsigned long>(hr);
            throw std::runtime_error(s.str());
        }
    }
    struct MediaRuntime {
        MediaRuntime(){
            check(CoInitializeEx(nullptr,COINIT_MULTITHREADED),"CoInitializeEx");
            check(MFStartup(MF_VERSION),"MFStartup");
        }
        ~MediaRuntime(){
            MFShutdown();
            CoUninitialize();
        }
    };
    // Native C++ NV12 -> H.264 sink. Explicit timestamps preserve 20 fps.
    // API sequence: Microsoft Learn, "Using the Sink Writer to Encode Video".
    class VideoWriter {
        ComPtr<IMFSinkWriter> writer;
        DWORD stream=0;
        long long frame_number=0;
        public: explicit VideoWriter(const fs::path& path) {
            ComPtr<IMFAttributes> options;
            check(MFCreateAttributes(&options,1),"MFCreateAttributes");
            check(options->SetUINT32(MF_SINK_WRITER_DISABLE_THROTTLING,TRUE),"disable throttling");
            check(MFCreateSinkWriterFromURL(path.c_str(),nullptr,options.Get(),&writer),"create MP4 sink");
            ComPtr<IMFMediaType> output,input;
            check(MFCreateMediaType(&output),"output type");
            check(output->SetGUID(MF_MT_MAJOR_TYPE,MFMediaType_Video),"output major type");
            check(output->SetGUID(MF_MT_SUBTYPE,MFVideoFormat_H264),"output H264");
            check(output->SetUINT32(MF_MT_AVG_BITRATE,8000000),"bitrate");
            check(output->SetUINT32(MF_MT_INTERLACE_MODE,MFVideoInterlace_Progressive),"progressive");
            check(MFSetAttributeSize(output.Get(),MF_MT_FRAME_SIZE,width,height),"size");
            check(MFSetAttributeRatio(output.Get(),MF_MT_FRAME_RATE,fps,1),"fps");
            check(MFSetAttributeRatio(output.Get(),MF_MT_PIXEL_ASPECT_RATIO,1,1),"pixel aspect");
            check(writer->AddStream(output.Get(),&stream),"AddStream");
            check(MFCreateMediaType(&input),"input type");
            check(input->SetGUID(MF_MT_MAJOR_TYPE,MFMediaType_Video),"input major type");
            check(input->SetGUID(MF_MT_SUBTYPE,MFVideoFormat_NV12),"input NV12");
            check(input->SetUINT32(MF_MT_INTERLACE_MODE,MFVideoInterlace_Progressive),"input progressive");
            check(MFSetAttributeSize(input.Get(),MF_MT_FRAME_SIZE,width,height),"input size");
            check(MFSetAttributeRatio(input.Get(),MF_MT_FRAME_RATE,fps,1),"input fps");
            check(MFSetAttributeRatio(input.Get(),MF_MT_PIXEL_ASPECT_RATIO,1,1),"input aspect");
            check(input->SetUINT32(MF_MT_DEFAULT_STRIDE,width),"input stride");
            check(writer->SetInputMediaType(stream,input.Get(),nullptr),"SetInputMediaType");
            check(writer->BeginWriting(),"BeginWriting");
        }
        void append(const cv::Mat& image) {
            cv::Mat planar;
            cv::cvtColor(image,planar,cv::COLOR_BGR2YUV_I420);
            constexpr DWORD size=width*height*3/2;
            ComPtr<IMFMediaBuffer> buffer;
            check(MFCreateMemoryBuffer(size,&buffer),"video buffer");
            BYTE* data=nullptr;
            check(buffer->Lock(&data,nullptr,nullptr),"buffer lock");
            const BYTE* y=planar.ptr<BYTE>(),*u=y+width*height,*v=u+width*height/4;
            std::memcpy(data,y,width*height);
            for(int i=0; i<width*height/4; ++i){
                data[width*height+2*i]=u[i];
                data[width*height+2*i+1]=v[i];
            }
            check(buffer->Unlock(),"buffer unlock");
            check(buffer->SetCurrentLength(size),"buffer length");
            ComPtr<IMFSample> sample;
            check(MFCreateSample(&sample),"create sample");
            check(sample->AddBuffer(buffer.Get()),"sample buffer");
            check(sample->SetSampleTime(frame_number*10000000/fps),"sample time");
            check(sample->SetSampleDuration(10000000/fps),"sample duration");
            check(writer->WriteSample(stream,sample.Get()),"WriteSample");
            ++frame_number;
        }
        void finish(){
            check(writer->Finalize(),"Finalize");
            writer.Reset();
        }
    };
    cv::Mat color_image(const fs::path& path) {
        cv::Mat image=cv::imdecode(bytes(path),cv::IMREAD_COLOR);
        if(image.size()!=cv::Size(3840,2160))throw std::runtime_error("Expected 3840x2160 image: "+path.u8string());
        return image;
    }
    void save_png(const fs::path& path,const cv::Mat& image) {
        std::vector<uchar> encoded;
        if(!cv::imencode(".png",image,encoded))throw std::runtime_error("PNG encoding failed");
        std::ofstream file(path,std::ios::binary);
        file.write(reinterpret_cast<const char*>(encoded.data()),encoded.size());
    }
    void text(cv::Mat& image,const std::string& s,cv::Point at,double scale,cv::Scalar color=white,int weight=2) {
        cv::putText(image,s,at,cv::FONT_HERSHEY_SIMPLEX,scale,cv::Scalar(8,12,18),weight+3,cv::LINE_AA);
        cv::putText(image,s,at,cv::FONT_HERSHEY_SIMPLEX,scale,color,weight,cv::LINE_AA);
    }
    std::string number(double value,int precision=2){
        std::ostringstream s;
        s<<std::showpos<<std::fixed<<std::setprecision(precision)<<value;
        return s.str();
    }
    struct Item{
        char axis;
        double value;
        fs::path path;
    };
    struct Shift{
        int dx,dy;
        double score;
    };
    std::array<Shift,3> offsets(const cv::Mat& image,const std::array<cv::Mat,3>& templates,bool center) {
        std::array<Shift,3> result{
        };
        if(center){
            for(auto& s:result)s.score=1;
            return result;
        }
        cv::Mat gray;
        cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
        for(int i=0; i<3; ++i){
            auto p=points[i];
            cv::Mat correlation;
            cv::matchTemplate(gray(cv::Rect(p.x-155,p.y-155,310,310)),templates[i],correlation,cv::TM_CCOEFF_NORMED);
            double score;
            cv::Point location;
            cv::minMaxLoc(correlation,nullptr,&score,nullptr,&location);
            if(score<.85)throw std::runtime_error("Unreliable QR match");
            result[i]={
                location.x-100,location.y-100,score
            };
        }
        return result;
    }
    cv::Mat frame(const cv::Mat& image,const Item& item,const std::array<Shift,3>& shifts,int index,int count) {
        cv::Mat canvas(height,width,CV_8UC3,cv::Scalar(28,31,37));
        image(cv::Rect(1220,690,1400,840)).copyTo(canvas(cv::Rect(30,120,1400,840)));
        for(int i=0; i<3; ++i){
            cv::Point base=points[i]+cv::Point(-1190,-570),current=base+cv::Point(shifts[i].dx,shifts[i].dy);
            cv::line(canvas,base-cv::Point(0,135),base+cv::Point(0,135),reference_color,2,cv::LINE_AA);
            cv::line(canvas,current-cv::Point(0,115),current+cv::Point(0,115),current_color,2,cv::LINE_AA);
            cv::circle(canvas,base,7,reference_color,-1,cv::LINE_AA);
            cv::circle(canvas,current,7,current_color,-1,cv::LINE_AA);
            if(std::abs(shifts[i].dx)>3)cv::arrowedLine(canvas,base+cv::Point(0,125),current+cv::Point(0,125),current_color,3,cv::LINE_AA,0,.16);
            text(canvas,std::to_string(i+1),base+cv::Point(-12,-155),.85,reference_color);
        }
        cv::rectangle(canvas,cv::Point(30,120),cv::Point(1430,960),cv::Scalar(95,100,105),2);
        text(canvas,"QR MARKER MOTION  |  CENTER vs CURRENT",cv::Point(30,75),1.1);
        cv::rectangle(canvas,cv::Point(1450,120),cv::Point(1890,690),cv::Scalar(42,47,55),-1);
        cv::rectangle(canvas,cv::Point(1450,120),cv::Point(1890,690),cv::Scalar(85,92,100),2);
        text(canvas,item.axis=='T'?"THETA":"Y POSITION",cv::Point(1480,185),1.15,cv::Scalar(180,185,190));
        text(canvas,number(item.value),cv::Point(1470,295),2.55,cv::Scalar(255,255,255),5);
        text(canvas,"filename value",cv::Point(1480,340),.75,cv::Scalar(170,175,180));
        text(canvas,"QR SHIFT FROM CENTER",cv::Point(1480,410),.8,reference_color);
        for(int i=0; i<3; ++i){
            std::ostringstream label;
            label<<i+1<<"   "<<std::showpos<<std::setw(4)<<shifts[i].dx<<" px";
            text(canvas,label.str(),cv::Point(1480,465+i*62),1.05,current_color);
        }
        text(canvas,"cyan = center / pink = current",cv::Point(1480,670),.55,cv::Scalar(210,210,210));
        cv::resize(image,canvas(cv::Rect(1450,720,440,248)),cv::Size(440,248),0,0,cv::INTER_AREA);
        cv::rectangle(canvas,cv::Point(1450,720),cv::Point(1890,968),cv::Scalar(85,92,100),2);
        text(canvas,item.path.filename().u8string()+"   ["+std::to_string(index)+"/"+std::to_string(count)+"]",cv::Point(30,1010),.78);
        text(canvas,"Single-axis images: theta and Y are separate command-ordered sequences.",cv::Point(30,1050),.70,cv::Scalar(190,195,205));
        return canvas;
    }
}
int wmain(int argc,wchar_t** wide_argv) {
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try {
        if(argc!=3)throw std::runtime_error("historical_video_preview data-bmp-20261014 new-output");
        fs::path root=fs::u8path(argv[1]),out=fs::u8path(argv[2]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        fs::create_directories(out);
        MediaRuntime runtime;
        cv::Mat center=color_image(root/"center.bmp"),gray;
        cv::cvtColor(center,gray,cv::COLOR_BGR2GRAY);
        std::array<cv::Mat,3> templates;
        for(int i=0; i<3; ++i)templates[i]=gray(cv::Rect(points[i].x-55,points[i].y-55,110,110)).clone();
        std::vector<Item> sequence;
        for(char axis:{
            'T','Y'
        })for(int j=axis=='T'?-100:-40; j<=(axis=='T'?100:40); ++j){
            double value=j/double(axis=='T'?100:20);
            fs::path file=j==0?root/"center.bmp":root/(axis=='T'?"Theta":"Y")/fs::u8path(std::string(1,axis)+number(value)+".bmp");
            for(int repeat=0; repeat<(j==0?10:1); ++repeat)sequence.push_back({
                axis,value,file
            });
        }
        std::ofstream measurements(out/"frames.csv");
        measurements<<"frame,time_seconds,axis,filename_command,file,q0_dx,q0_dy,q0_score,q1_dx,q1_dy,q1_score,q2_dx,q2_dy,q2_score\n";
        fs::path video=out/"SSAFY_TestImageSet_2x_annotated.mp4";
        VideoWriter writer(fs::absolute(video));
        cv::Mat reference_frame;
        for(size_t i=0; i<sequence.size(); ++i){
            auto& item=sequence[i];
            cv::Mat image=color_image(item.path);
            auto shifts=offsets(image,templates,item.value==0);
            auto rendered=frame(image,item,shifts,int(i+1),int(sequence.size()));
            writer.append(rendered);
            measurements<<i+1<<','<<std::setprecision(17)<<i/double(fps)<<','<<item.axis<<','<<item.value<<','<<csv(item.path.filename().u8string());
            for(auto& s:shifts)measurements<<','<<s.dx<<','<<s.dy<<','<<s.score;
            measurements<<'\n';
            if(i==0||i==100||i==210||i==250||i==299){
                save_png(out/("frame_"+std::to_string(i+1)+".png"),rendered);
                if(i==0)reference_frame=rendered.clone();
            }
            if((i+1)%50==0)std::cout<<i+1<<"/"<<sequence.size()<<std::endl;
        }
        writer.finish();
        measurements.close();
        cv::VideoCapture decoder(video.u8string());
        if(!decoder.isOpened())throw std::runtime_error("Independent OpenCV video decoder failed");
        double decoded_fps=decoder.get(cv::CAP_PROP_FPS);
        cv::Mat decoded;
        int count=0;
        double first_frame_error=0;
        while(decoder.read(decoded)){
            if(decoded.size()!=cv::Size(width,height))throw std::runtime_error("Decoded dimensions differ");
            if(count==0){
                cv::Mat error;
                cv::absdiff(reference_frame,decoded,error);
                auto mean=cv::mean(error);
                first_frame_error=(mean[0]+mean[1]+mean[2])/3;
                save_png(out/"decoded_first.png",decoded);
            }
            ++count;
        }
        Json report=Json::dict();
        report["created_kst"]=now();
        report["frames"]=count;
        report["fps"]=decoded_fps;
        report["seconds"]=count/decoded_fps;
        report["width"]=width;
        report["height"]=height;
        report["first_frame_mean_abs_compression_error"]=first_frame_error;
        report["container_sha256"]=sha256(video);
        report["measurement_method"]="historical three-QR integer NCC, not 630-boundary inference";
        report["changes"]="Native Media Foundation H.264 8Mbit/s instead of ffmpeg/libx264 CRF22; explicit command-order caption. Encoded bytes and compression artifacts differ.";
        report["passed"]=count==300&&std::abs(decoded_fps-fps)<1e-6&&first_frame_error<15;
        write(out/"checks.json",report.dump());
        std::cout<<report.dump()<<'\n';
        return report.at("passed").num()?0:2;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
