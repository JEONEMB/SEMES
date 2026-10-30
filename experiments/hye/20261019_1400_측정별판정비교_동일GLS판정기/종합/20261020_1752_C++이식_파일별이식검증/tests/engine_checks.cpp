#include "alignment.hpp"
#include <iostream>
#include <limits>
using namespace semes;
static void require(bool v,const char* message){
    if(!v)throw std::runtime_error(message);
}
template<class F> static void rejected(F&& f,const char* message){
    bool caught=false;
    try{
        f();
    }
    catch(const std::exception&){
        caught=true;
    }
    require(caught,message);
}
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc!=4)throw std::runtime_error("engine_checks assets center scratch-root");
        fs::path assets=fs::u8path(argv[1]),center=fs::u8path(argv[2]),scratch=fs::u8path(argv[3]);
        Engine e(assets,center);
        require(e.corners().size()==630,"Fixed point count");
        require(e.predict(cv::Mat::zeros(1,12,CV_64F)).axis=="Y","Equal-cost branch must keep saved Python branch order");
        cv::Mat B(630,6,CV_64F),co(6,2,CV_64F);
        for(int k=0; k<12; k++)co.ptr<double>()[k]=.02*(k-5);
        for(int i=0; i<630; i++){
            auto p=e.corners()[i].reference;
            double u=(p.x-1906)/486,v=(p.y-1070)/122;
            double row[]={
                1,u,v,u*u,u*v,v*v
            };
            for(int j=0; j<6; j++)B.at<double>(i,j)=row[j];
        }
        cv::Mat d=B*co;
        require(cv::norm(e.features(d)-co.reshape(1,1),cv::NORM_INF)<1e-10,"P12 exact polynomial recovery");
        require(e.features(d,"QR36").cols==36,"QR36 dimension");
        require(e.features(d,"QR18").cols==18,"QR18 dimension");
        require(e.features(d,"raw1260").cols==1260,"Raw dimension");
        rejected([&]{
            e.features(cv::Mat::zeros(629,2,CV_64F));
        },"Wrong displacement count accepted");
        cv::Mat invalid=cv::Mat::zeros(630,2,CV_64F);
        invalid.at<double>(1,0)=std::numeric_limits<double>::quiet_NaN();
        rejected([&]{
            e.features(invalid);
        },"Nonfinite displacement accepted");
        rejected([&]{
            e.measure(cv::Mat::zeros(40,40,CV_8U));
        },"Cropped image accepted");
        rejected([&]{
            e.infer(std::vector<unsigned char>{
                1,2,3
            });
        },"Invalid image bytes accepted");
        std::string stamp=now();
        for(char& c:stamp)if(c==':'||c=='+'||c=='.')c='_';
        scratch/=fs::u8path(stamp);
        fs::create_directories(scratch);
        write_zip(scratch/"roundtrip.zip",{
            {
                "empty",""
            },{
                "한글.txt",u8"C++ 보존 검증"
            }
        });
        auto restored=read_zip(scratch/"roundtrip.zip");
        require(restored.at("empty").empty(),"Empty ZIP entry changed");
        require(std::string(restored.at("한글.txt").begin(),restored.at("한글.txt").end())==u8"C++ 보존 검증","Unicode ZIP content changed");
        auto corrupted=bytes(scratch/"roundtrip.zip");
        // Change a central-directory CRC while keeping the ZIP readable.
        for(size_t i=0; i+46<corrupted.size(); ++i)if(corrupted[i]==0x50&&corrupted[i+1]==0x4b&&corrupted[i+2]==1&&corrupted[i+3]==2){
            corrupted[i+16]^=1;
            break;
        }
        write(scratch/"corrupted.zip",std::string(corrupted.begin(),corrupted.end()));
        rejected([&]{
            read_zip(scratch/"corrupted.zip");
        },"ZIP CRC corruption accepted");
        for(auto filename:{
            "corners.json","reference.npz","provenance.json","model.json"
        })fs::copy_file(assets/filename,scratch/filename);
        auto corners=Json::parse(read(scratch/"corners.json"));
        corners.array[0]["id"]=999;
        write(scratch/"corners.json",corners.dump());
        rejected([&]{
            Engine bad(scratch,center);
        },"Mismatched structure ID accepted");
        std::vector<unsigned char> raw;
        cv::imencode(".bmp",cv::Mat::zeros(40,40,CV_8U),raw);
        write(scratch/"wrong_center.bmp",std::string(raw.begin(),raw.end()));
        rejected([&]{
            Engine bad(assets,scratch/"wrong_center.bmp");
        },"Mismatched center SHA256 accepted");
        std::cout<<"Fixed reference, polynomial recovery, maps and rejection checks: PASS\n";
        return 0;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
