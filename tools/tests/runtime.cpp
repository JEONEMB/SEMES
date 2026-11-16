#include "alignment.hpp"
#include "common/reference.hpp"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <limits>
namespace fs=std::filesystem;
using namespace alignment;
int checks=0,failures=0;
// expect: 계약 조건 확인
//   입력: 조건과 검사 이름
//   출력: 검사 수와 실패 수 갱신
void expect(bool value,const std::string& name){++checks;if(!value){++failures;std::cerr<<"FAIL "<<name<<'\n';}}
// notice: 특정 알림 확인
//   입력: 결과와 코드
//   출력: 일치하는 코드가 있으면 true
bool notice(const Result&r,const std::string&code){for(const auto&n:r.notices)if(n.code==code)return true;return false;}
// failed: 실패 시 수치와 보정 금지 계약 확인
//   입력: 결과와 예상 코드
//   출력: 조건 검사 기록
void failed(const Result&r,const char*code){expect(r.status==Status::Failed&&r.code==code&&!r.correction_allowed&&std::isnan(r.y_mm)&&std::isnan(r.t_deg),code);}
// main: 영점, 누락 나사, 학습 범위와 검출 실패 구분 검사
//   입력: 모델 폴더
//   출력: 검사 결과, 실패 시 종료 코드 1
int main(int argc,char**argv){
 if(argc!=2)return 2;cv::setNumThreads(1);const fs::path model=fs::u8path(argv[1]);
 Reference ref;std::string error;if(!load_reference(model.u8string(),ref,error)){std::cerr<<error;return 2;}
 const auto photo=ref.photo;
 failed(measure_alignment(photo),"E_NOT_INITIALIZED");expect(initialize(model.u8string()).status==Status::Ok,"initialize");
 failed(measure_alignment(photo),"E_ZERO_REQUIRED");const auto diagnostic=measure_diagnostic(photo);expect(diagnostic.status!=Status::Failed&&!diagnostic.correction_allowed,"diagnostic only");
 expect(set_zero_from_photos({photo}).status!=Status::Failed,"zero setup");auto r=measure_alignment(photo);expect(r.status!=Status::Failed&&r.correction_allowed&&std::abs(r.y_mm)<1e-10&&std::abs(r.t_deg)<1e-10,"zero operational");
 failed(set_zero_from_photos({}),"E_PHOTO");failed(measure_alignment(photo),"E_ZERO_REQUIRED");
 expect(set_zero_from_photos({photo}).status!=Status::Failed,"zero again");failed(set_zero_value(std::numeric_limits<double>::quiet_NaN(),0),"E_ZERO_VALUE");failed(measure_alignment(photo),"E_ZERO_REQUIRED");
 set_zero_from_photos({photo});clear_zero();failed(measure_alignment(photo),"E_ZERO_REQUIRED");
 initialize(model.u8string());set_zero_from_photos({photo});failed(set_reference_from_photos({}),"E_PHOTO");failed(measure_alignment(photo),"E_ZERO_REQUIRED");
 initialize(model.u8string());set_zero_from_photos({photo});
 auto missing=photo.clone();missing(cv::Rect(ref.screw[1].x-135,ref.screw[1].y-135,270,270)).setTo(128);
 r=measure_alignment(missing);expect(r.status!=Status::Failed&&r.screws_used=="134"&&notice(r,"W_SCREW_MISSING")&&r.correction_allowed,"three screws usable");
 missing(cv::Rect(ref.screw[3].x-135,ref.screw[3].y-135,270,270)).setTo(128);
 r=measure_alignment(missing);expect(r.status!=Status::Failed&&notice(r,"W_COARSE")&&!r.correction_allowed,"one side insufficient");
 auto blank=cv::Mat::zeros(photo.size(),photo.type());failed(measure_alignment(blank),"E_QR_NOT_FOUND");
 const fs::path work=fs::current_path()/("range_cases_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directories(work);
 for(int axis=0;axis<2;++axis)for(int side:{-1,1}){
    const auto dest=work/(std::to_string(axis)+"_"+std::to_string(side));fs::copy(model,dest,fs::copy_options::recursive);
    cv::FileStorage file((dest/"reference.json").string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
    file<<"qr"<<"[";for(const auto&q:ref.qr)file<<"["<<q.x<<q.y<<q.width<<q.height<<"]";file<<"]";
    file<<"screws"<<"[";for(const auto&s:ref.screw)file<<"["<<s.x<<s.y<<"]";file<<"]";
    for(int j=0;j<2;++j){const double*original=j==0?ref.trained_y_mm:ref.trained_t_deg;
        file<<(j==0?"trained_y_mm":"trained_t_deg")<<"["<<(j==axis?(side>0?10000.0:-10001.0):original[0])<<(j==axis?(side>0?10001.0:-10000.0):original[1])<<"]";}
    file.release();expect(initialize(dest.u8string()).status==Status::Ok,"range model");
    r=measure_diagnostic(photo);expect(r.status==Status::Warning&&notice(r,"W_OUT_OF_RANGE")&&std::isfinite(r.y_mm)&&std::isfinite(r.t_deg),"range returns values");
    r=set_zero_from_photos({photo});expect(r.status==Status::Warning&&notice(r,"W_OUT_OF_RANGE"),"range zero succeeds with warning");
    r=measure_alignment(photo);expect(r.status==Status::Warning&&notice(r,"W_OUT_OF_RANGE")&&std::abs(r.y_mm)<1e-10&&std::abs(r.t_deg)<1e-10&&r.correction_allowed&&r.action==Action::Correct,"range warning allows correction");
    expect(set_zero_value(0,0).status==Status::Ok,"restore absolute zero for fixture");
    const auto raw=measure_diagnostic(photo);
    expect(!raw.correction_allowed,"diagnostic remains nonoperational");
    expect(set_zero_value(raw.y_mm-3.0,raw.t_deg).status==Status::Ok,"restore zero for 3 mm output");
    r=measure_alignment(photo);
    expect(r.status==Status::Warning&&notice(r,"W_OUT_OF_RANGE")&&std::abs(r.y_mm-3.0)<1e-10&&r.correction_allowed&&r.action==Action::Correct,"3 mm range warning allows correction");
    r=measure_alignment(missing);
    expect(notice(r,"W_OUT_OF_RANGE")&&notice(r,"W_COARSE")&&!r.correction_allowed,"range warning does not override other blocks");
    failed(measure_alignment(blank),"E_QR_NOT_FOUND");
    failed(set_zero_from_photos({blank}),"E_ZERO");failed(measure_alignment(photo),"E_ZERO_REQUIRED");
 }
 std::cout<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;
}
