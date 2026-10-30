// =====================================================================================
// pptx.hpp  -  C++ 로 pptx 만들기 (라이브러리 없이 OOXML 을 직접 쓴다)
//
//   pptx = zip 안의 XML 묶음.
//   1. 템플릿 pptx 를 Windows 기본 tar 로 풀고, 기존 슬라이드·그림·차트·메모를 지운다 (디자인만 남김)
//   2. 슬라이드마다 도형(글상자, 그림, 선, 사각형, 차트) XML 을 만든다
//   3. presentation.xml, 관계(rels), [Content_Types].xml 에 슬라이드를 등록하고 tar 로 다시 zip
//
//   좌표 단위는 인치 (슬라이드 13.333 x 7.5 인치). 내부에서 EMU(1인치 = 914400)로 바꾼다.
//   그래프는 PowerPoint 기본 차트 (한글 축 이름, PPT 안에서 수정 가능)
// =====================================================================================
#pragma once

#include "nan.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace hw {

namespace fs = std::filesystem;

// v4 발표자료 색
namespace color {
const std::string ink = "111827";     // 제목, 본문 진한 글자
const std::string sub = "4B5563";     // 본문
const std::string faint = "9CA3AF";   // 설명, 쪽 번호
const std::string line = "E5E7EB";    // 표 구분선
const std::string accent = "0071E3";  // 강조 파랑
const std::string orange = "D9480F";
const std::string teal = "0F766E";
const std::string brown = "B45309";
const std::string green = "16A34A";
const std::string gray = "9CA3AF";
const std::string bg = "F7F7F8";      // 슬라이드 배경
}  // namespace color

// 글자 한 조각
struct Run {
    std::string text;
    double size = 11;            // pt
    std::string color = color::sub;
    bool bold = false;
};
// 문단 = 조각 여러 개
struct Para {
    std::vector<Run> runs;
    std::string align = "l";     // l, ctr, r
};
inline Para para(const std::string &text, double size, const std::string &col = color::sub, bool bold = false,
                 const std::string &align = "l") {
    return Para{{Run{text, size, col, bold}}, align};
}

// 차트 데이터
struct Series {
    std::string name;
    std::string color = color::accent;
    std::vector<double> x, y;    // 산점도: x, y / 막대: y 만 (x 무시)
    bool line = false;           // 점끼리 선으로 잇기
    bool marker = true;
    double markerSize = 3;
    bool dashed = false;
};
struct ChartSpec {
    std::vector<Series> series;
    std::string xTitle, yTitle;
    double xMin = kNaN, xMax = kNaN, yMin = kNaN, yMax = kNaN;
    bool legend = true;
    double fontSize = 8;
    std::string numFmtX = "General", numFmtY = "General";
    // 막대 차트용
    std::vector<std::string> categories;
    bool horizontal = false;
    bool dataLabels = false;
    std::string labelFmt = "0.00";
};

class Slide {
public:
    void text(double x, double y, double w, double h, const std::vector<Para> &paras, const std::string &anchor = "t");
    void picture(const fs::path &png, double x, double y, double w, double h);
    void line(double x1, double y1, double x2, double y2, const std::string &col = color::line, double widthPt = 0.75);
    void rect(double x, double y, double w, double h, const std::string &fill, const std::string &lineCol = "");
    void scatter(const ChartSpec &spec, double x, double y, double w, double h);
    void bars(const ChartSpec &spec, double x, double y, double w, double h);

private:
    friend class Deck;
    std::string layout_;                  // 레이아웃 이름 (CONTENT_GRAY, DIVIDER ...)
    bool grayBackground_ = false;
    std::vector<std::string> shapes_;     // 도형 XML
    std::vector<fs::path> images_;        // 슬라이드에 들어갈 그림 (rId 순서)
    std::vector<std::string> charts_;     // 차트 XML (rId 순서)
    int nextId_ = 2;
    std::string nextRel(const std::string &kind);   // 그림·차트 관계 id
    std::vector<std::pair<std::string, std::string>> rels_;  // (rId, kind:index)
};

class Deck {
public:
    explicit Deck(const fs::path &templatePptx);    // 템플릿을 풀고 기존 슬라이드를 지운다
    ~Deck();
    Slide &add(const std::string &layoutName, bool grayBackground = true);
    void save(const fs::path &out);                 // 슬라이드를 쓰고 zip

private:
    fs::path work_;                                  // 풀어 놓은 임시 폴더
    std::vector<Slide> slides_;
    std::string layoutTarget(const std::string &name) const;
};

std::string xmlEscape(const std::string &s);

}  // namespace hw
