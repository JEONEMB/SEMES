#include "common/console.hpp"
#include "alignment.hpp"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <chrono>
namespace fs = std::filesystem;
using namespace alignment;
int checks=0, failures=0;
void expect(bool value,const char* name){++checks;if(!value){++failures;std::cerr<<"FAIL "<<name<<"\n";}}
void failure(const Result&r,const char* code){expect(r.status==Status::Failed && r.code==code && !r.correction_allowed && std::isnan(r.y_mm) && std::isnan(r.t_deg),code);}
std::string read(const fs::path&p){std::ifstream f(p);return {std::istreambuf_iterator<char>(f),{}};}
void write(const fs::path&p,const std::string&s){std::ofstream(p)<<s;}
int main(int argc,char**argv){
    tools::use_korean_console();
 if(argc!=3)return 2;cv::setNumThreads(1);fs::path model=fs::u8path(argv[1]);auto photo=read_photo(argv[2]);
 failure(measure_alignment(photo),"E_NOT_INITIALIZED");expect(initialize(model.u8string()).status==Status::Ok,"initialize");
 failure(measure_alignment(photo),"E_ZERO_REQUIRED");auto raw=measure_diagnostic(photo);expect(raw.status==Status::Ok&&!raw.correction_allowed,"diagnostic");
 expect(set_zero_from_photos({photo,photo}).status==Status::Ok,"zero photos");auto zero=measure_alignment(photo);expect(zero.correction_allowed&&std::abs(zero.y_mm)<1e-10&&std::abs(zero.t_deg)<1e-10,"zero once");
 failure(set_zero_from_photos({}),"E_PHOTO");failure(measure_alignment(photo),"E_ZERO_REQUIRED");
 for(double v:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}){failure(set_zero_value(v,0),"E_ZERO_VALUE");failure(measure_alignment(photo),"E_ZERO_REQUIRED");failure(set_zero_value(0,v),"E_ZERO_VALUE");}
 expect(set_zero_value(0,0).status==Status::Ok,"finite zero");clear_zero();failure(measure_alignment(photo),"E_ZERO_REQUIRED");
 set_zero_value(0,0);failure(measure_alignment({}),"E_PHOTO");failure(measure_alignment(cv::Mat::zeros(20,20,CV_8U)),"E_PHOTO");failure(measure_alignment(cv::Mat::zeros(photo.size(),CV_16U)),"E_PHOTO");
 auto ref=set_reference_from_photos({photo});expect(ref.status!=Status::Failed,"reference");failure(measure_alignment(photo),"E_ZERO_REQUIRED");
 initialize(model.u8string());set_zero_value(0,0);cv::Mat covered=photo.clone();covered(cv::Rect(2664,1278,400,160)).setTo(128);auto missing=measure_alignment(covered);expect(missing.code=="W_SCREW_MISSING"&&missing.correction_allowed,"missing alone");
 fs::path work=fs::current_path()/ ("safety_cases_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directories(work);
 int n=0;auto copy=[&](){auto p=work/std::to_string(n++);fs::copy(model,p,fs::copy_options::recursive);return p;};
 for(const auto&e:fs::directory_iterator(model/"checks")){auto p=copy();fs::remove(p/"checks"/e.path().filename());failure(initialize(p.u8string()),"E_DATA");failure(measure_alignment(photo),"E_NOT_INITIALIZED");}
 for(const auto&e:fs::directory_iterator(model/"coefficients")){auto p=copy();write(p/"coefficients"/e.path().filename(),"");failure(initialize(p.u8string()),"E_DATA");}
 auto p=copy();auto c=p/"coefficients/screws_1234.tsv";auto text=read(c);auto pos=text.find("qr\t");auto end=text.find('\n',pos);write(c,text.substr(0,pos)+text.substr(end+1));failure(initialize(p.u8string()),"E_DATA");
 p=copy();c=p/"coefficients/screws_1234.tsv";text=read(c);pos=text.find("qr\t");end=text.find('\n',pos);write(c,text+text.substr(pos,end-pos)+"\n");failure(initialize(p.u8string()),"E_DATA");
 p=copy();c=p/"checks/screws_1234.tsv";text=read(c);pos=text.find("threshold\t");end=text.find('\n',pos);text.replace(pos,end-pos,"threshold\t0");write(c,text);failure(initialize(p.u8string()),"E_DATA");
 p=copy();c=p/"checks/screws_134.tsv";text=read(c);pos=text.find("threshold\t");end=text.find('\n',pos);text.replace(pos,end-pos,"threshold\t0.0000000001");write(c,text);expect(initialize(p.u8string()).status==Status::Ok,"strict model");set_zero_value(0,0);auto mixed=measure_alignment(covered);expect(mixed.notices.size()>=2&&mixed.code=="W_VALUES_DISAGREE"&&!mixed.correction_allowed&&mixed.action==Action::Recapture,"mixed warning");
 auto settings=work/"strict.json";write(settings,"{\"fail_on_disagreement\":1}");initialize(p.u8string(),settings.u8string());set_zero_value(0,0);failure(measure_alignment(covered),"W_VALUES_DISAGREE");
 p=copy();c=p/"coefficients/screws_1234.tsv";text=read(c);pos=text.find("qr\t");end=text.find('\n',pos);text.replace(pos,end-pos,"qr\t0\t1e-300\t1e300\t1e300");write(c,text);expect(initialize(p.u8string()).status==Status::Ok,"finite extreme model");failure(measure_diagnostic(photo),"E_NUMERIC");
 for (const std::string bad : {"qr\t0\t0\t1\t1", "qr\t0\t1\tnan\t1", "qr\t0\t1\tinf\t1", "qr\t0\t1\t1\t1\textra"}) { p=copy();c=p/"coefficients/screws_1234.tsv";text=read(c);pos=text.find("qr\t");end=text.find('\n',pos);text.replace(pos,end-pos,bad);write(c,text);failure(initialize(p.u8string()),"E_DATA"); }
 p=copy();write(p/"reference.json","{\"qr\":[],\"screws\":[]}");failure(initialize(p.u8string()),"E_DATA");
 initialize(model.u8string());set_zero_from_photos({photo});failure(set_reference_from_photos({}),"E_PHOTO");failure(measure_alignment(photo),"E_ZERO_REQUIRED");
 initialize(model.u8string());set_zero_value(0,0);failure(measure_alignment(photo,[](int,bool,double,const std::string&){throw std::runtime_error("callback failure");}),"E_INTERNAL");
 expect(measure_alignment(photo).correction_allowed,"callback lock released");
 auto unicode=work/fs::u8path(u8"한글 공백 모델");fs::copy(model,unicode,fs::copy_options::recursive);expect(initialize(unicode.u8string()).status==Status::Ok,"UTF8 model");
 std::cout<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;
}
