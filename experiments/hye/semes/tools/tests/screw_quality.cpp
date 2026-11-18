#include "common/reference.hpp"
#include "alignment.hpp"
#include "steps/02_find_screws.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
using namespace alignment;
// main: 큰 예상 위치 오차와 실제 검출 불량을 구분하는 회귀 검사
//   입력: 모델 폴더
//   출력: 검사 개수와 실패 개수, 실패 시 종료 코드 1
int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    cv::setNumThreads(1);
    Reference ref; std::string error;
    if (!load_reference(argv[1], ref, error)) { std::cerr << error; return 2; }
    ScrewFinder finder(ref);
    int checks = 0, failures = 0;
    const auto expect = [&](bool ok, const std::string& label) {
        ++checks; if (!ok) { ++failures; std::cerr << "FAIL " << label << '\n'; }
    };
    const std::array<bool,4> wanted{{true,true,true,true}};
    // 영상 전체를 평행 이동하고 QR 예상 이동을 0으로 주어 나사 검출기의 위치 오차 허용을 독립 검사한다.
    // 실제 장비의 Y/T 회전이나 보정 정확도를 재현하는 시험은 아니다.
    for (const cv::Point delta : {cv::Point(0,0),cv::Point(17,0),cv::Point(-17,0),cv::Point(0,17),cv::Point(0,-17),cv::Point(24,18)}) {
        cv::Mat moved;
        const cv::Mat affine = (cv::Mat_<double>(2,3) << 1,0,delta.x,0,1,delta.y);
        cv::warpAffine(ref.photo,moved,affine,ref.photo.size(),cv::INTER_NEAREST,cv::BORDER_CONSTANT,cv::Scalar(128));
        const auto result = finder.find_screw_shifts(moved,0,0,wanted,true);
        for (int k=0;k<4;++k) {
            const auto& s=result[k];
            expect(s.found && !s.partial,"visible screw "+std::to_string(k+1)+" offset "+std::to_string(delta.x)+","+std::to_string(delta.y)+" "+s.problem);
            expect(std::hypot(s.dx-delta.x,s.dy-delta.y)<0.15,"shift accuracy");
            expect(s.shape_similarity>=0.93,"shape remains valid");
        }
    }
    // 나사를 완전히 가리면 거리 제한 제거 후에도 검출에 성공하면 안 된다.
    auto hidden=ref.photo.clone();
    for (const auto& p:ref.screw) hidden(cv::Rect(p.x-120,p.y-120,240,240)).setTo(128);
    for (const auto& s:finder.find_screw_shifts(hidden,0,0,wanted)) expect(!s.found,"covered screw rejected");
    // 위치 계산이 끝난 불안정 관측은 경고로 반환하고, 계산 실패만 제외한다.
    auto partial=ref.photo.clone();
    for (const auto& p:ref.screw) partial(cv::Rect(p.x,p.y-35,35,70)).setTo(128);
    for (const auto& s:finder.find_screw_shifts(partial,0,0,wanted)) expect(!s.found || (s.quality_warning && std::isfinite(s.dx) && std::isfinite(s.dy)),"unstable estimate retained with warning or calculation failure");
    if (argc == 3) {
        const auto problem = read_photo(argv[2]);
        expect(initialize(argv[1]).status == Status::Ok, "regression initialize");
        const auto r = measure_diagnostic(problem);
        bool warned = false;
        for (const auto& n : r.notices) if (n.code == "W_SCREW_UNSTABLE") warned = true;
        expect(r.status == Status::Warning && warned && std::isfinite(r.y_mm) && std::isfinite(r.t_deg)
               && r.screws_used == "1234", "previously rejected photo returns all-screw Y/T");
        expect(set_zero_from_photos({ref.photo}).status != Status::Failed, "regression zero");
        const auto measured = measure_alignment(problem);
        expect(measured.status != Status::Failed && std::isfinite(measured.y_mm) && std::isfinite(measured.t_deg), "operational measurement retains values");
        std::cout << "Regression Y=" << r.y_mm << " T=" << r.t_deg << " screws=" << r.screws_used << "\n" << r.message << "\n";
    }
    std::cout<<checks<<" checks, "<<failures<<" failures\n";
    return failures?1:0;
}
