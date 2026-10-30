// =====================================================================================
// slides_common.hpp  -  sweep, grid 슬라이드가 같이 쓰는 레이아웃과 그림
//   위치·글자 크기는 v4 발표자료(SEMES_01_데이터EDA_정리본_v4)와 같게 맞췄다.
// =====================================================================================
#pragma once

#include "analysis.hpp"

namespace hw {

// 측정 방법 두 가지
enum class Method { Corner, Ecc };
inline const char *methodKey(Method m) { return m == Method::Corner ? "corner" : "ecc"; }
inline const char *methodName(Method m) { return m == Method::Corner ? "특징점" : "ECC"; }
inline double shotDx(const ShotMeasure &s, Method m) { return m == Method::Corner ? s.cornerDx : s.eccDx; }
inline double shotDy(const ShotMeasure &s, Method m) { return m == Method::Corner ? s.cornerDy : s.eccDy; }

// 표지
void coverSlide(Context &ctx, const std::string &subtitle);
// 내용 슬라이드: 머리글 + 제목 + 쪽 번호
Slide &contentSlide(Context &ctx, const std::string &section, const std::string &title);
// 제목 없는 슬라이드 (2x2 그림처럼 꽉 채울 때): 쪽 번호만
Slide &bareSlide(Context &ctx);

// 오른쪽 키-값 표 (v4 의 7.21 ~ 12.54 인치 영역)
struct KvRow { std::string key; std::vector<std::string> values; };
void kvTable(Slide &s, const std::vector<KvRow> &rows, double top = 2.03, double bottom = 6.43);
// 열이 여러 개인 표 (머리줄 + 행)
void table(Slide &s, const std::vector<std::string> &header, const std::vector<std::vector<std::string>> &rows,
           const std::vector<double> &colX, const std::vector<double> &colW, double top, double rowH, double fontSize = 11);
// 그림 아래 설명
void caption(Slide &s, double x, double y, double w, const std::string &text);
// 2x2 칸: 칸 제목 + 한 줄 설명 위치를 돌려줌 (그림/차트 넣을 영역)
struct Cell { double x, y, w, h; };
Cell gridCell(Slide &s, int index, const std::string &title, const std::string &desc);

// 기준 위치 슬라이드 3장 (QR 전체, 특징점, ECC). sweep, grid 공통
void referenceSlides(Context &ctx);
// 기초 정보 왼쪽 그림 (center 사진 축소 + QR 위치)
fs::path centerThumbnail(Context &ctx);

// 그림 저장 후 경로 (fig/<name>.png)
fs::path saveFig(Context &ctx, const std::string &name, const cv::Mat &img);

// 차트 색 (trial, 반복 순서)
const std::string &seriesColor(size_t i);

}  // namespace hw
