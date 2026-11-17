#include "pptx.hpp"

#include "core.hpp"

#include <windows.h>

#include <cmath>
#include <regex>
#include <sstream>

namespace hw {

namespace {

const char *kNsA = "http://schemas.openxmlformats.org/drawingml/2006/main";
const char *kNsR = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const char *kNsP = "http://schemas.openxmlformats.org/presentationml/2006/main";
const char *kNsC = "http://schemas.openxmlformats.org/drawingml/2006/chart";
const char *kRelSlide = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide";
const char *kRelLayout = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout";
const char *kRelImage = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image";
const char *kRelChart = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart";
const char *kFont = "Noto Sans KR";

long long emu(double inch) { return std::llround(inch * 914400.0); }

std::string readText(const fs::path &p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
void writeText(const fs::path &p, const std::string &s) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out << s;
}

// Windows 기본 tar (bsdtar, zip 읽기·쓰기 가능) 실행. Git Bash 의 tar 와 헷갈리지 않게 전체 경로로 부른다.
void runTar(const std::wstring &args) {
    wchar_t sys[MAX_PATH];
    GetSystemDirectoryW(sys, MAX_PATH);
    std::wstring cmd = L"\"" + std::wstring(sys) + L"\\tar.exe\" " + args;
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
        fail("tar 를 실행하지 못했습니다");
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (code != 0) fail("tar 실패 (코드 " + std::to_string(code) + ")");
}

std::string fmtNum(double v) {
    std::ostringstream s;
    s.precision(10);
    s << v;
    return s.str();
}

// 글자 속성 (v4 와 같은 Noto Sans KR)
std::string runProps(const Run &r) {
    std::ostringstream s;
    s << "<a:rPr lang=\"ko-KR\" altLang=\"en-US\" sz=\"" << std::lround(r.size * 100) << "\"" << (r.bold ? " b=\"1\"" : " b=\"0\"")
      << " dirty=\"0\"><a:solidFill><a:srgbClr val=\"" << r.color << "\"/></a:solidFill>"
      << "<a:latin typeface=\"" << kFont << "\"/><a:ea typeface=\"" << kFont << "\"/><a:cs typeface=\"" << kFont << "\"/></a:rPr>";
    return s.str();
}

std::string axisTitle(const std::string &t, double size) {
    if (t.empty()) return "";
    std::ostringstream s;
    s << "<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz=\"" << std::lround(size * 100)
      << "\" b=\"0\"/></a:pPr><a:r>" << runProps(Run{t, size, color::sub, false}) << "<a:t>" << xmlEscape(t)
      << "</a:t></a:r></a:p></c:rich></c:tx><c:overlay val=\"0\"/></c:title>";
    return s.str();
}

std::string textProps(double size) {
    std::ostringstream s;
    s << "<c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz=\"" << std::lround(size * 100)
      << "\"><a:solidFill><a:srgbClr val=\"" << color::sub << "\"/></a:solidFill><a:latin typeface=\"" << kFont
      << "\"/><a:ea typeface=\"" << kFont << "\"/></a:defRPr></a:pPr><a:endParaRPr lang=\"ko-KR\"/></a:p></c:txPr>";
    return s.str();
}

std::string scaling(double mn, double mx) {
    std::ostringstream s;
    s << "<c:scaling><c:orientation val=\"minMax\"/>";
    if (std::isfinite(mx)) s << "<c:max val=\"" << fmtNum(mx) << "\"/>";
    if (std::isfinite(mn)) s << "<c:min val=\"" << fmtNum(mn) << "\"/>";
    s << "</c:scaling>";
    return s.str();
}

std::string gridLines() {
    return std::string("<c:majorGridlines><c:spPr><a:ln w=\"6350\"><a:solidFill><a:srgbClr val=\"") + color::line +
           "\"/></a:solidFill></a:ln></c:spPr></c:majorGridlines>";
}

std::string axisLine() {
    return std::string("<c:spPr><a:ln w=\"6350\"><a:solidFill><a:srgbClr val=\"") + color::line + "\"/></a:solidFill></a:ln></c:spPr>";
}

// 숫자 목록 -> c:numLit (차트 안에 값을 직접 넣음, 엑셀 데이터 없이 표시)
std::string numLit(const std::vector<double> &v) {
    std::ostringstream s;
    s << "<c:numLit><c:formatCode>General</c:formatCode><c:ptCount val=\"" << v.size() << "\"/>";
    for (size_t i = 0; i < v.size(); ++i)
        if (std::isfinite(v[i])) s << "<c:pt idx=\"" << i << "\"><c:v>" << fmtNum(v[i]) << "</c:v></c:pt>";
    s << "</c:numLit>";
    return s.str();
}

std::string chartHead() {
    return std::string("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<c:chartSpace xmlns:c=\"") + kNsC +
           "\" xmlns:a=\"" + kNsA + "\" xmlns:r=\"" + kNsR + "\"><c:roundedCorners val=\"0\"/><c:chart><c:autoTitleDeleted val=\"1\"/><c:plotArea><c:layout/>";
}

std::string chartTail(const ChartSpec &sp) {
    std::ostringstream s;
    s << "</c:plotArea>";
    if (sp.legend) s << "<c:legend><c:legendPos val=\"t\"/><c:overlay val=\"0\"/>" << textProps(sp.fontSize - 1) << "</c:legend>";
    s << "<c:plotVisOnly val=\"1\"/><c:dispBlanksAs val=\"gap\"/></c:chart>"
      << "<c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>" << textProps(sp.fontSize) << "</c:chartSpace>";
    return s.str();
}

}  // namespace

std::string xmlEscape(const std::string &s) {
    std::string r;
    r.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&': r += "&amp;"; break;
            case '<': r += "&lt;"; break;
            case '>': r += "&gt;"; break;
            case '"': r += "&quot;"; break;
            default: r += c;
        }
    }
    return r;
}

// -------------------------------------------------------------------------------------
// Slide: 도형
// -------------------------------------------------------------------------------------

std::string Slide::nextRel(const std::string &kind) {
    const std::string id = "rId" + std::to_string(rels_.size() + 2);   // rId1 = 레이아웃
    rels_.emplace_back(id, kind);
    return id;
}

void Slide::text(double x, double y, double w, double h, const std::vector<Para> &paras, const std::string &anchor) {
    const int id = nextId_++;
    std::ostringstream s;
    s << "<p:sp><p:nvSpPr><p:cNvPr id=\"" << id << "\" name=\"Text " << id << "\"/><p:cNvSpPr txBox=\"1\"/><p:nvPr/></p:nvSpPr>"
      << "<p:spPr><a:xfrm><a:off x=\"" << emu(x) << "\" y=\"" << emu(y) << "\"/><a:ext cx=\"" << emu(w) << "\" cy=\"" << emu(h)
      << "\"/></a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>"
      << "<p:txBody><a:bodyPr wrap=\"square\" lIns=\"0\" tIns=\"0\" rIns=\"0\" bIns=\"0\" rtlCol=\"0\" anchor=\"" << anchor
      << "\"><a:normAutofit/></a:bodyPr><a:lstStyle/>";
    for (const auto &p : paras) {
        s << "<a:p><a:pPr algn=\"" << p.align << "\"><a:buNone/></a:pPr>";
        for (const auto &r : p.runs) s << "<a:r>" << runProps(r) << "<a:t xml:space=\"preserve\">" << xmlEscape(r.text) << "</a:t></a:r>";
        s << "</a:p>";
    }
    if (paras.empty()) s << "<a:p/>";
    s << "</p:txBody></p:sp>";
    shapes_.push_back(s.str());
}

void Slide::picture(const fs::path &png, double x, double y, double w, double h) {
    const int id = nextId_++;
    images_.push_back(png);
    const std::string rid = nextRel("image:" + std::to_string(images_.size() - 1));
    std::ostringstream s;
    s << "<p:pic><p:nvPicPr><p:cNvPr id=\"" << id << "\" name=\"Picture " << id << "\"/><p:cNvPicPr><a:picLocks noChangeAspect=\"1\"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>"
      << "<p:blipFill><a:blip r:embed=\"" << rid << "\"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>"
      << "<p:spPr><a:xfrm><a:off x=\"" << emu(x) << "\" y=\"" << emu(y) << "\"/><a:ext cx=\"" << emu(w) << "\" cy=\"" << emu(h)
      << "\"/></a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></p:spPr></p:pic>";
    shapes_.push_back(s.str());
}

void Slide::line(double x1, double y1, double x2, double y2, const std::string &col, double widthPt) {
    const int id = nextId_++;
    std::ostringstream s;
    s << "<p:cxnSp><p:nvCxnSpPr><p:cNvPr id=\"" << id << "\" name=\"Line " << id << "\"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>"
      << "<p:spPr><a:xfrm><a:off x=\"" << emu(x1) << "\" y=\"" << emu(y1) << "\"/><a:ext cx=\"" << emu(x2 - x1) << "\" cy=\"" << emu(y2 - y1)
      << "\"/></a:xfrm><a:prstGeom prst=\"line\"><a:avLst/></a:prstGeom><a:ln w=\"" << std::lround(widthPt * 12700)
      << "\"><a:solidFill><a:srgbClr val=\"" << col << "\"/></a:solidFill></a:ln></p:spPr></p:cxnSp>";
    shapes_.push_back(s.str());
}

void Slide::rect(double x, double y, double w, double h, const std::string &fill, const std::string &lineCol) {
    const int id = nextId_++;
    std::ostringstream s;
    s << "<p:sp><p:nvSpPr><p:cNvPr id=\"" << id << "\" name=\"Rect " << id << "\"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
      << "<p:spPr><a:xfrm><a:off x=\"" << emu(x) << "\" y=\"" << emu(y) << "\"/><a:ext cx=\"" << emu(w) << "\" cy=\"" << emu(h)
      << "\"/></a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val=\"" << fill << "\"/></a:solidFill>"
      << (lineCol.empty() ? "<a:ln><a:noFill/></a:ln>" : "<a:ln w=\"9525\"><a:solidFill><a:srgbClr val=\"" + lineCol + "\"/></a:solidFill></a:ln>")
      << "</p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang=\"ko-KR\"/></a:p></p:txBody></p:sp>";
    shapes_.push_back(s.str());
}

// 차트를 담는 그래픽 프레임
static std::string chartFrame(int id, const std::string &rid, double x, double y, double w, double h) {
    std::ostringstream s;
    s << "<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id=\"" << id << "\" name=\"Chart " << id << "\"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>"
      << "<p:xfrm><a:off x=\"" << emu(x) << "\" y=\"" << emu(y) << "\"/><a:ext cx=\"" << emu(w) << "\" cy=\"" << emu(h) << "\"/></p:xfrm>"
      << "<a:graphic><a:graphicData uri=\"" << kNsC << "\"><c:chart xmlns:c=\"" << kNsC << "\" r:id=\"" << rid
      << "\"/></a:graphicData></a:graphic></p:graphicFrame>";
    return s.str();
}

void Slide::scatter(const ChartSpec &sp, double x, double y, double w, double h) {
    std::ostringstream c;
    c << chartHead() << "<c:scatterChart><c:scatterStyle val=\"lineMarker\"/><c:varyColors val=\"0\"/>";
    for (size_t i = 0; i < sp.series.size(); ++i) {
        const Series &s = sp.series[i];
        c << "<c:ser><c:idx val=\"" << i << "\"/><c:order val=\"" << i << "\"/><c:tx><c:v>" << xmlEscape(s.name) << "</c:v></c:tx><c:spPr>";
        if (s.line) c << "<a:ln w=\"12700\"><a:solidFill><a:srgbClr val=\"" << s.color << "\"/></a:solidFill>"
                      << (s.dashed ? "<a:prstDash val=\"dash\"/>" : "") << "</a:ln>";
        else c << "<a:ln><a:noFill/></a:ln>";
        c << "</c:spPr><c:marker>";
        if (s.marker) c << "<c:symbol val=\"circle\"/><c:size val=\"" << std::lround(s.markerSize) << "\"/><c:spPr><a:solidFill><a:srgbClr val=\""
                        << s.color << "\"/></a:solidFill><a:ln><a:noFill/></a:ln></c:spPr>";
        else c << "<c:symbol val=\"none\"/>";
        c << "</c:marker><c:xVal>" << numLit(s.x) << "</c:xVal><c:yVal>" << numLit(s.y) << "</c:yVal><c:smooth val=\"0\"/></c:ser>";
    }
    c << "<c:axId val=\"1001\"/><c:axId val=\"1002\"/></c:scatterChart>";
    c << "<c:valAx><c:axId val=\"1001\"/>" << scaling(sp.xMin, sp.xMax) << "<c:delete val=\"0\"/><c:axPos val=\"b\"/>"
      << axisTitle(sp.xTitle, sp.fontSize) << "<c:numFmt formatCode=\"" << sp.numFmtX << "\" sourceLinked=\"0\"/><c:majorTickMark val=\"out\"/><c:minorTickMark val=\"none\"/>"
      << "<c:tickLblPos val=\"low\"/>" << axisLine() << textProps(sp.fontSize) << "<c:crossAx val=\"1002\"/><c:crosses val=\"autoZero\"/><c:crossBetween val=\"midCat\"/></c:valAx>";
    c << "<c:valAx><c:axId val=\"1002\"/>" << scaling(sp.yMin, sp.yMax) << "<c:delete val=\"0\"/><c:axPos val=\"l\"/>" << gridLines()
      << axisTitle(sp.yTitle, sp.fontSize) << "<c:numFmt formatCode=\"" << sp.numFmtY << "\" sourceLinked=\"0\"/><c:majorTickMark val=\"out\"/><c:minorTickMark val=\"none\"/>"
      << "<c:tickLblPos val=\"low\"/>" << axisLine() << textProps(sp.fontSize) << "<c:crossAx val=\"1001\"/><c:crosses val=\"autoZero\"/><c:crossBetween val=\"midCat\"/></c:valAx>";
    c << chartTail(sp);
    charts_.push_back(c.str());
    const std::string rid = nextRel("chart:" + std::to_string(charts_.size() - 1));
    shapes_.push_back(chartFrame(nextId_++, rid, x, y, w, h));
}

void Slide::bars(const ChartSpec &sp, double x, double y, double w, double h) {
    std::ostringstream c;
    c << chartHead() << "<c:barChart><c:barDir val=\"" << (sp.horizontal ? "bar" : "col") << "\"/><c:grouping val=\"clustered\"/><c:varyColors val=\"0\"/>";
    for (size_t i = 0; i < sp.series.size(); ++i) {
        const Series &s = sp.series[i];
        c << "<c:ser><c:idx val=\"" << i << "\"/><c:order val=\"" << i << "\"/><c:tx><c:v>" << xmlEscape(s.name) << "</c:v></c:tx>"
          << "<c:spPr><a:solidFill><a:srgbClr val=\"" << s.color << "\"/></a:solidFill></c:spPr><c:invertIfNegative val=\"0\"/>";
        if (sp.dataLabels)
            c << "<c:dLbls><c:numFmt formatCode=\"" << sp.labelFmt << "\" sourceLinked=\"0\"/><c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>"
              << textProps(sp.fontSize) << "<c:dLblPos val=\"outEnd\"/><c:showLegendKey val=\"0\"/><c:showVal val=\"1\"/><c:showCatName val=\"0\"/>"
              << "<c:showSerName val=\"0\"/><c:showPercent val=\"0\"/><c:showBubbleSize val=\"0\"/></c:dLbls>";
        c << "<c:cat><c:strLit><c:ptCount val=\"" << sp.categories.size() << "\"/>";
        for (size_t k = 0; k < sp.categories.size(); ++k) c << "<c:pt idx=\"" << k << "\"><c:v>" << xmlEscape(sp.categories[k]) << "</c:v></c:pt>";
        c << "</c:strLit></c:cat><c:val>" << numLit(s.y) << "</c:val></c:ser>";
    }
    c << "<c:gapWidth val=\"60\"/><c:axId val=\"2001\"/><c:axId val=\"2002\"/></c:barChart>";
    c << "<c:catAx><c:axId val=\"2001\"/><c:scaling><c:orientation val=\"" << (sp.horizontal ? "maxMin" : "minMax") << "\"/></c:scaling><c:delete val=\"0\"/>"
      << "<c:axPos val=\"" << (sp.horizontal ? "l" : "b") << "\"/>" << axisTitle(sp.xTitle, sp.fontSize)
      << "<c:numFmt formatCode=\"General\" sourceLinked=\"0\"/><c:majorTickMark val=\"none\"/><c:minorTickMark val=\"none\"/><c:tickLblPos val=\"low\"/>"
      << axisLine() << textProps(sp.fontSize) << "<c:crossAx val=\"2002\"/><c:crosses val=\"autoZero\"/><c:auto val=\"1\"/><c:lblAlgn val=\"ctr\"/><c:lblOffset val=\"100\"/><c:noMultiLvlLbl val=\"0\"/></c:catAx>";
    c << "<c:valAx><c:axId val=\"2002\"/>" << scaling(sp.yMin, sp.yMax) << "<c:delete val=\"0\"/><c:axPos val=\"" << (sp.horizontal ? "b" : "l") << "\"/>"
      << gridLines() << axisTitle(sp.yTitle, sp.fontSize) << "<c:numFmt formatCode=\"" << sp.numFmtY << "\" sourceLinked=\"0\"/><c:majorTickMark val=\"out\"/>"
      << "<c:minorTickMark val=\"none\"/><c:tickLblPos val=\"nextTo\"/>" << axisLine() << textProps(sp.fontSize)
      << "<c:crossAx val=\"2001\"/><c:crosses val=\"autoZero\"/><c:crossBetween val=\"between\"/></c:valAx>";
    c << chartTail(sp);
    charts_.push_back(c.str());
    const std::string rid = nextRel("chart:" + std::to_string(charts_.size() - 1));
    shapes_.push_back(chartFrame(nextId_++, rid, x, y, w, h));
}

// -------------------------------------------------------------------------------------
// Deck: 템플릿 정리, 저장
// -------------------------------------------------------------------------------------

Deck::Deck(const fs::path &templatePptx) {
    if (!fs::exists(templatePptx)) fail("템플릿이 없습니다: " + u8(templatePptx));
    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    work_ = fs::path(tmp) / (L"hw_eda_pptx_" + std::to_wstring(GetCurrentProcessId()) + L"_" + std::to_wstring(GetTickCount64()));
    fs::create_directories(work_);
    runTar(L"-xf \"" + templatePptx.wstring() + L"\" -C \"" + work_.wstring() + L"\"");

    // 슬라이드, 메모, 그림, 차트, 내장 파일은 지우고 디자인(마스터, 레이아웃, 테마)만 남긴다
    for (const char *d : {"ppt/slides", "ppt/notesSlides", "ppt/media", "ppt/charts", "ppt/embeddings"}) fs::remove_all(work_ / d);

    // presentation.xml: 슬라이드 목록 비우기
    std::string pres = readText(work_ / "ppt/presentation.xml");
    pres = std::regex_replace(pres, std::regex("<p:sldIdLst>[\\s\\S]*?</p:sldIdLst>"), "");
    writeText(work_ / "ppt/presentation.xml", pres);
    // 관계: 슬라이드 관계 지우기
    std::string rels = readText(work_ / "ppt/_rels/presentation.xml.rels");
    rels = std::regex_replace(rels, std::regex("<Relationship [^>]*Target=\"slides/[^\"]*\"/>"), "");
    writeText(work_ / "ppt/_rels/presentation.xml.rels", rels);
    // 콘텐츠 형식: 슬라이드, 메모, 차트 등록 지우기
    std::string ct = readText(work_ / "[Content_Types].xml");
    ct = std::regex_replace(ct, std::regex("<Override PartName=\"/ppt/(slides|notesSlides|charts|embeddings)/[^\"]*\"[^>]*/>"), "");
    writeText(work_ / "[Content_Types].xml", ct);
}

Deck::~Deck() {
    std::error_code ec;
    fs::remove_all(work_, ec);
}

Slide &Deck::add(const std::string &layoutName, bool grayBackground) {
    slides_.emplace_back();
    slides_.back().layout_ = layoutName;
    slides_.back().grayBackground_ = grayBackground;
    return slides_.back();
}

std::string Deck::layoutTarget(const std::string &name) const {
    for (const auto &e : fs::directory_iterator(work_ / "ppt/slideLayouts")) {
        if (e.path().extension() != ".xml") continue;
        if (readText(e.path()).find("cSld name=\"" + name + "\"") != std::string::npos) return "../slideLayouts/" + u8(e.path().filename());
    }
    fail("템플릿에 레이아웃이 없습니다: " + name);
}

void Deck::save(const fs::path &out) {
    std::string pres = readText(work_ / "ppt/presentation.xml");
    std::string rels = readText(work_ / "ppt/_rels/presentation.xml.rels");
    std::string ct = readText(work_ / "[Content_Types].xml");
    if (ct.find("Extension=\"png\"") == std::string::npos)
        ct = std::regex_replace(ct, std::regex("<Default "), "<Default Extension=\"png\" ContentType=\"image/png\"/><Default ", std::regex_constants::format_first_only);

    std::ostringstream idList, relAdd, ctAdd;
    int chartNo = 0, imageNo = 0;
    for (size_t i = 0; i < slides_.size(); ++i) {
        const Slide &s = slides_[i];
        const std::string no = std::to_string(i + 1);
        // 슬라이드 본문
        std::ostringstream x;
        x << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<p:sld xmlns:a=\"" << kNsA << "\" xmlns:r=\"" << kNsR << "\" xmlns:p=\"" << kNsP << "\"><p:cSld>";
        if (s.grayBackground_) x << "<p:bg><p:bgPr><a:solidFill><a:srgbClr val=\"" << color::bg << "\"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>";
        x << "<p:spTree><p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/>"
          << "<a:ext cx=\"0\" cy=\"0\"/><a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>";
        for (const auto &sh : s.shapes_) x << sh;
        x << "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>";
        writeText(work_ / ("ppt/slides/slide" + no + ".xml"), x.str());

        // 슬라이드 관계: 레이아웃, 그림, 차트
        std::ostringstream r;
        r << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
          << "<Relationship Id=\"rId1\" Type=\"" << kRelLayout << "\" Target=\"" << layoutTarget(s.layout_) << "\"/>";
        for (const auto &[rid, kind] : s.rels_) {
            const size_t colon = kind.find(':');
            const int idx = std::stoi(kind.substr(colon + 1));
            if (kind.rfind("image", 0) == 0) {
                const std::string name = "image" + std::to_string(++imageNo) + ".png";
                fs::create_directories(work_ / "ppt/media");
                fs::copy_file(s.images_[idx], work_ / "ppt/media" / name, fs::copy_options::overwrite_existing);
                r << "<Relationship Id=\"" << rid << "\" Type=\"" << kRelImage << "\" Target=\"../media/" << name << "\"/>";
            } else {
                const std::string name = "chart" + std::to_string(++chartNo) + ".xml";
                writeText(work_ / "ppt/charts" / name, s.charts_[idx]);
                r << "<Relationship Id=\"" << rid << "\" Type=\"" << kRelChart << "\" Target=\"../charts/" << name << "\"/>";
                ctAdd << "<Override PartName=\"/ppt/charts/" << name << "\" ContentType=\"application/vnd.openxmlformats-officedocument.drawingml.chart+xml\"/>";
            }
        }
        r << "</Relationships>";
        writeText(work_ / ("ppt/slides/_rels/slide" + no + ".xml.rels"), r.str());

        idList << "<p:sldId id=\"" << 256 + i << "\" r:id=\"rIdS" << no << "\"/>";
        relAdd << "<Relationship Id=\"rIdS" << no << "\" Type=\"" << kRelSlide << "\" Target=\"slides/slide" << no << ".xml\"/>";
        ctAdd << "<Override PartName=\"/ppt/slides/slide" << no << ".xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slide+xml\"/>";
    }

    // 슬라이드 목록은 notesMasterIdLst 뒤(없으면 sldMasterIdLst 뒤)에 넣는다 (스키마 순서)
    const std::string list = slides_.empty() ? "" : "<p:sldIdLst>" + idList.str() + "</p:sldIdLst>";
    size_t at = pres.find("</p:notesMasterIdLst>");
    if (at != std::string::npos) at += std::string("</p:notesMasterIdLst>").size();
    else at = pres.find("</p:sldMasterIdLst>") + std::string("</p:sldMasterIdLst>").size();
    pres.insert(at, list);
    rels.insert(rels.rfind("</Relationships>"), relAdd.str());
    ct.insert(ct.rfind("</Types>"), ctAdd.str());
    writeText(work_ / "ppt/presentation.xml", pres);
    writeText(work_ / "ppt/_rels/presentation.xml.rels", rels);
    writeText(work_ / "[Content_Types].xml", ct);

    // zip 으로 묶기 (최상위 항목만 지정해서 "./" 접두어가 붙지 않게)
    fs::create_directories(out.parent_path());
    const fs::path zip = out.parent_path() / (out.stem().wstring() + L".zip");
    fs::remove(zip);
    runTar(L"-a -c -f \"" + zip.wstring() + L"\" -C \"" + work_.wstring() + L"\" \"[Content_Types].xml\" _rels docProps ppt");
    std::error_code ec;
    fs::remove(out, ec);
    if (ec) fail("저장할 수 없습니다 (PowerPoint 에서 열려 있나요?): " + u8(out));
    fs::rename(zip, out);
}

}  // namespace hw
