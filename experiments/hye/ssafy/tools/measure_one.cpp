#include "alignment.hpp"
#include <iostream>
#include <iomanip>
// main: 영점 설정 후 한 장 측정 또는 명시적 진단 실행
//   입력: 모델, 사진, 영점 사진 또는 --diagnostic
//   출력: 측정값, 상태, 보정 허용 여부. 실패 시 종료 코드 1
int main(int argc,char**argv) {
    if(argc!=4){std::cerr<<"usage: measure_one MODEL IMAGE ZERO_IMAGE|--diagnostic\n";return 2;}
    auto r=alignment::initialize(argv[1]);if(r.status==alignment::Status::Failed)return 1;
    const bool diagnostic=std::string(argv[3])=="--diagnostic";
    if(!diagnostic){r=alignment::set_zero_from_photos({alignment::read_photo(argv[3])});if(r.status==alignment::Status::Failed)return 1;}
    const auto image=alignment::read_photo(argv[2]);
    r=diagnostic?alignment::measure_diagnostic(image):alignment::measure_alignment(image);
    std::cout<<std::setprecision(17)<<"y="<<r.y_mm<<" t="<<r.t_deg<<" status="<<int(r.status)<<" correction_allowed="<<r.correction_allowed<<" screws="<<r.screws_used<<" code="<<r.code<<"\n"<<r.message<<"\n";
    return r.status==alignment::Status::Failed?1:0;
}
