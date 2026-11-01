// =====================================================================================
// main.cpp — alignment() 사용 예 (제출 형식)
//
//  실행 방법 (작업 폴더 = 이 프로젝트 폴더)
//   1) 인자 없음            : alignment.yml 의 demo.image 한 장을 추정하고 결과 창 표시
//   2) <사진> [기준 사진]    : 지정한 사진 한 장 추정 (기준을 주지 않으면 reference_path)
//  모델 학습은 align_train, 정확도 평가는 align_eval (tools/) 로 분리
// =====================================================================================
#include <iostream>
#include <opencv2/opencv.hpp>

#include "alignment.hpp"
#include "ksh/align_config.hpp"
#include "ksh/dataset.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>   // 콘솔 한글 출력 (UTF-8)
#endif

// 이 파일에서만 쓸 네임스페이스 정의
namespace {

const char *kConfigPath = "config/alignment.yml";

// =====================================================================================
// show_result — 결과 창 : 측정한 ROI 를 '움직인 위치'에 사각형으로 그리고 추정값을 글자로 표시
//  img              : 추정에 쓴 현재 사진 (컬러 또는 회색)
//  r                : alignment() 결과 (y [mm], t [°])
//  display_width_px : 창 가로 크기. 3840 px 원본은 모니터보다 커서 이 폭으로 줄여서 보여 줌 (설정 demo.display_width_px)
// =====================================================================================
void show_result(const cv::Mat &img, const Alignment_t &r, int display_width_px) {
    // 그림을 그릴 사본 view : 회색(1채널)이면 색을 칠할 수 있게 BGR(3채널)로 바꾸고, 컬러면 복사
    //  (원본 img 에 직접 그리면 호출한 쪽의 사진이 바뀌므로 사본에 그림)
    cv::Mat view;
    if (img.channels() == 1) cv::cvtColor(img, view, cv::COLOR_GRAY2BGR);
    else view = img.clone();

    // 마지막 alignment() 호출의 측정 상세 (ROI 별 이동량, 성공 여부, s·d, 처리 시간)
    //  & 참조라 복사하지 않고 alignment.cpp 안에 저장된 값을 그대로 읽음
    const AlignmentInfo &info = alignment_last_info();

    // ROI 마다 (QR_L, QR_C, QR_R, W2, W3)
    for (const auto &roi : info.rois) {
        // 측정한 이동량(소수 px)을 반올림해 정수 좌표로 → 기준 위치 + 이동 = 현재 사진에서 ROI 가 있는 곳
        const cv::Point off(int(std::lround(roi.shift_px.x)), int(std::lround(roi.shift_px.y)));
        // 측정 성공 = 초록 (B, G, R 순서라 (0, 200, 0)), 실패 = 빨강 (0, 0, 255)
        const cv::Scalar col = roi.ok ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 255);
        // 사각형 (두께 4 px) : 기준 사진의 ROI 사각형을 off 만큼 옮겨서 그림
        cv::rectangle(view, roi.ref_rect_px + off, col, 4);
        // ROI 이름 : 사각형 왼쪽 위(tl = top-left)에서 12 px 위에, 글자 크기 1.4, 두께 3
        cv::putText(view, roi.name, roi.ref_rect_px.tl() + off + cv::Point(0, -12), cv::FONT_HERSHEY_SIMPLEX, 1.4, col, 3);
    }

    // 화면 위쪽에 쓸 문장. snprintf = 버퍼 크기(160)를 넘지 않게 형식대로 문자열 만들기
    //  %+.4f = 부호(+/−)를 항상 붙이고 소수 4자리, %.0f = 소수 없이 (ms)
    char buf[160];
    if (info.ok) std::snprintf(buf, sizeof(buf), "Y = %+.4f mm   Theta = %+.4f deg   (%.0f ms)", r.y, r.t, info.elapsed_ms);
    else std::snprintf(buf, sizeof(buf), "FAILED");
    // 같은 글자를 두 번 그림 : 굵은 검정(두께 12) 위에 흰색(두께 4) → 밝은·어두운 배경 어디서나 읽히는 외곽선 글씨
    //  (Θ 같은 한글·기호는 putText 가 못 그려서 영문 'Theta' 로 씀)
    cv::putText(view, buf, cv::Point(60, 120), cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(0, 0, 0), 12);
    cv::putText(view, buf, cv::Point(60, 120), cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(255, 255, 255), 4);

    // 창 크기 줄이기 : 가로가 display_width_px 보다 크면 가로세로 비율을 유지하며 축소
    //  INTER_AREA = 축소할 때 계단·물결 무늬가 적은 보간
    if (display_width_px > 0 && view.cols > display_width_px)
        cv::resize(view, view, cv::Size(display_width_px, view.rows * display_width_px / view.cols), 0, 0, cv::INTER_AREA);

    // "Alignment" 이름의 창에 띄우고, 아무 키나 누를 때까지 기다림 (0 = 무한 대기)
    cv::imshow("Alignment", view);
    cv::waitKey(0);
}

} // namespace


// 추가 인자 갯수, 추가 인자량
// 기본적으로 이 프로젝트인 alignment.exe가 포함
// 즉 아무것도 인자를 주지 않고 시작하면 argc = 1, argv[0] = "alignment.exe" 가 설정
int main(int argc, char **argv)
{
    #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
    #endif

    // 현재 OpenCV 버전 출력
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // 기존 안내문만 출력하고 종료되는 요소를 대신 안내
    if (argc >= 2 && std::string(argv[1]) == "--eval") {
        std::cerr << "평가는 align_eval 도구로 옮겼다 (tools/align_eval.cpp)\n";
        return 1;
    }


    ksh::AlignConfig cfg;   // 이미지 분석을 위한 변수
    std::string error;      // 에러 메시지를 담을 변수

    // 이미지 모두를 잘 불러왔는지 확인하는 조건문
    if (!ksh::load_align_config(kConfigPath, cfg, error) || !alignment_init(kConfigPath)) {
        std::cerr << error << "\n설정 파일 " << kConfigPath << " 을 읽지 못함 (작업 폴더를 프로젝트 폴더로 설정)\n";
        return 1;
    }

    // ========== 1. 추정할 사진과 기준(센터) 사진 정하기 ==========
    // 추정할 사진 경로
    //  인자가 있으면(argc ≥ 2) 첫 인자 argv[1], 없으면 설정 파일의 demo.image
    //  예) alignment.exe "Y+0.50 T+0.10 (2).bmp"  →  image_path = "Y+0.50 T+0.10 (2).bmp"
    const std::string image_path = argc >= 2 ? argv[1] : cfg.demo.image;

    // 기준 사진 경로 (삼항 연산자 두 번 = 경우 세 가지)
    //  argc ≥ 3 (사진 + 기준 둘 다 줌)  → 두 번째 인자 argv[2]
    //  argc = 2 (사진만 줌)             → 빈 문자열 = 기준을 바꾸지 않음 → alignment_init 이 읽어 둔 reference_path 를 그대로 사용
    //  argc = 1 (인자 없음, 데모)       → 설정의 demo.reference (데모 사진과 같은 날 찍은 center1)
    //  기준 사진은 추정할 사진과 같은 날 찍은 것이어야 정확함 (날마다 원점이 조금씩 다름)
    const std::string ref_path = argc >= 3 ? argv[2] : (argc >= 2 ? std::string() : cfg.demo.reference);

    // 기준 사진을 바꿔야 할 때만(ref_path 가 비어 있지 않을 때만) 읽어서 alignment 에 등록
    //  cv::imread(..., IMREAD_GRAYSCALE) : 회색 1채널로 읽음. 파일이 없으면 빈 Mat
    //  alignment_set_reference : 빈 사진이거나 크기가 설정(3840×2160)과 다르면 false
    //  && 는 앞이 거짓이면 뒤를 실행하지 않으므로, ref_path 가 비어 있으면 사진을 읽지 않음
    if (!ref_path.empty() && !alignment_set_reference(cv::imread(ref_path, cv::IMREAD_GRAYSCALE))) {
        std::cerr << "기준 사진을 읽지 못함 : " << ref_path << "\n";
        return 1;
    }

    // 추정할 사진 읽기 (컬러로 읽어도 alignment 안에서 회색으로 바꿈. 결과 창에 색을 칠하려고 컬러로 읽음)
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {   // 경로가 틀렸거나 사진 파일이 아님
        std::cerr << "사진을 읽지 못함 : " << image_path << "\n";
        return 1;
    }

    // ========== 2. 추정 : 사진 한 장 → Y [mm], Θ [°] ==========
    //  안에서 ① Coarse Alignment(위상 상관) → ② Fine Alignment(IC-GN) → ③ 특징 s, d → ④ 보정식 역산 순서로 계산
    //  결과는 기준 사진 대비 어긋남 (라벨과 같은 부호). 측정이 실패하면 y, t 에 NaN
    Alignment_t alignment_result = alignment(image);

    // ========== 3. 결과 출력 ==========
    // 방금 호출의 측정 상세 (성공 여부, 실패 이유, ROI 별 이동, s·d, 처리 시간)
    const AlignmentInfo &info = alignment_last_info();
    std::cout << "Image: " << image_path << std::endl;
    if (!info.ok) {
        // 실패 : 이유만 출력 (예 "ROI 측정 실패 (Fine Alignment 미수렴 또는 사진 밖)")
        std::cout << "Alignment failed: " << info.message << std::endl;
    } else {
        // 성공 : std::fixed + setprecision(n) = 소수점 아래 n 자리로 고정해서 출력
        std::cout << std::fixed << std::setprecision(4)
                  // 추정한 어긋남
                  << "Alignment result: y = " << alignment_result.y << " mm"
                  << ", t = " << alignment_result.t << " deg" << std::endl
                  // 로봇에 내릴 보정 이동 = 어긋남의 반대 부호 (어긋난 만큼 되돌림)
                  << "Correction move : y = " << -alignment_result.y << " mm"
                  << ", t = " << -alignment_result.t << " deg" << std::endl
                  // 중간값 : s = QR 3개 평균 이동, d = s − 하단 문 홀 평균 이동 (시차)
                  << std::setprecision(3) << "Features        : s = " << info.s_px << " px, d = " << info.d_px << " px" << std::endl
                  // 처리 시간 (회색 변환 ~ 역산, 사진 읽기 제외)
                  << std::setprecision(1) << "Elapsed         : " << info.elapsed_ms << " ms" << std::endl;

        // 파일 이름이 라벨 형식이면(예 "Y+0.50 T+0.10 (2)") 정답과 오차도 출력 — 확인용, 추정에는 쓰지 않음
        //  u8path(...).stem() = 경로에서 폴더와 확장자를 뗀 이름 (한글 경로도 깨지지 않게 UTF-8 로)
        //  parse_label : 이름을 읽어 L 에 정답 Y·Θ 를 채움. 형식이 아니면 false
        //  is_reference : center 류(기준 사진 자체)는 정답 비교에서 제외
        ksh::LabeledPhoto L;
        if (ksh::parse_label(std::filesystem::u8path(image_path).stem().u8string(), L) && !L.is_reference)
            std::cout << std::setprecision(4) << "Label (truth)   : y = " << L.y_mm << " mm, t = " << L.t_deg << " deg  → error "
                      // Y 오차는 mm → µm (×1000), Θ 오차는 ° 그대로
                      << std::setprecision(1) << (alignment_result.y - L.y_mm) * 1000.0 << " um, " << std::setprecision(4)
                      << alignment_result.t - L.t_deg << " deg" << std::endl;
    }

    // ========== 4. 결과 창 ==========
    //  ROI 를 측정 위치에 그린 사진을 띄우고 키를 누를 때까지 기다림 (실패해도 어느 ROI 가 틀렸는지 빨강으로 보여 줌)
    show_result(image, alignment_result, cfg.demo.display_width_px);

    // 정상 종료 (0 = 성공, 0 이 아니면 실패라는 관례)
    return 0;
}
