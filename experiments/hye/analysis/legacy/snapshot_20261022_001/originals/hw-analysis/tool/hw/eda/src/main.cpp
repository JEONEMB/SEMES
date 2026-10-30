// =====================================================================================
// hw_eda  -  데이터 묶음 하나를 분석하고 발표자료(pptx)까지 만든다
//
// [실행]  프로젝트 루트에서
//   .\build\hw\eda\Release\hw_eda.exe                          묶음을 물어봄
//   .\build\hw\eda\Release\hw_eda.exe 20261014                 sweep 묶음 (1차)
//   .\build\hw\eda\Release\hw_eda.exe 20261020 --ref 20261014  grid 묶음 (2차), 1차 결과를 기준으로 평가
//   .\build\hw\eda\Release\hw_eda.exe --make-template <v4.pptx>  발표 템플릿 다시 만들기 (슬라이드 뺀 디자인만)
//
// [입력]  data/hw/bmp/<묶음>/                 (data/hw/data-automation 으로 정리한 원본)
//         docs/hw/eda/sets/<묶음>/notes.md    직접 쓰는 해석 (있으면 마지막 슬라이드로 들어감)
// [출력]  docs/validation-runs/hw/eda/<묶음>/ (Git 제외)
//           csv/          측정값, 분석표
//           fig/          사진 그림
//           summary.yml   핵심 숫자
//           SEMES_EDA_<묶음>.pptx
//         docs/hw/eda/sets/<묶음>/summary.yml (Git 추적용 복사본, 묶음끼리 비교)
// =====================================================================================
#include "analysis.hpp"
#include "core.hpp"
#include "measurements.hpp"
#include "pptx.hpp"

#include <fcntl.h>
#include <io.h>

#include <iostream>

using namespace hw;

static std::string askSet() {
    const auto sets = listSets();
    if (sets.empty()) fail("data/hw/bmp 에 묶음이 없습니다. data-automation 으로 먼저 정리하세요.");
    std::cout << "묶음 목록:\n";
    for (size_t i = 0; i < sets.size(); ++i) std::cout << "  " << i + 1 << ". " << sets[i] << "\n";
    std::cout << "번호나 이름을 입력하세요: ";
    std::wstring w;
    std::getline(std::wcin, w);
    const std::string s = u8(fs::path(w));
    for (size_t i = 0; i < sets.size(); ++i)
        if (s == std::to_string(i + 1) || s == sets[i]) return sets[i];
    fail("목록에 없는 묶음입니다: " + s);
}

int wmain(int argc, wchar_t **argv) {
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdin), _O_U16TEXT);

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.push_back(u8(fs::path(argv[i])));

    // 템플릿 만들기
    if (!args.empty() && args[0] == "--make-template") {
        if (args.size() < 2) fail("사용법: hw_eda --make-template <원본.pptx>");
        const fs::path out = fs::current_path() / "tool" / "hw" / "eda" / "report" / "template.pptx";
        Deck deck(u8path(args[1]));
        deck.save(out);
        log("템플릿 저장: " + u8(out));
        return 0;
    }

    std::string setName, refName;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--ref" && i + 1 < args.size()) refName = args[++i];
        else setName = args[i];
    }
    if (setName.empty()) setName = askSet();

    const fs::path out = outRoot() / u8path(setName);
    fs::create_directories(out / "csv");
    fs::create_directories(out / "fig");
    openLog(out / "log.txt");
    log("=== hw_eda: " + setName + " ===");

    const DataSet ds = loadSet(setName);
    log(std::string("종류: ") + (ds.kind == SetKind::Sweep ? "sweep (한 축씩 연속 촬영)" : "grid (Y, Θ 조합 반복 촬영)") +
        ", 사진 " + std::to_string(ds.shots.size()) + "장, center " + std::to_string(ds.centers.size()) + "장");

    const Measurements m = measureSet(ds);
    writeMeasureCsv(ds, m, out / "csv");

    const fs::path tmpl = fs::current_path() / "tool" / "hw" / "eda" / "report" / "template.pptx";
    if (!fs::exists(tmpl)) fail("발표 템플릿이 없습니다: " + u8(tmpl));
    Deck deck(tmpl);
    Summary summary;
    Context ctx{ds, m, out, deck, summary, nullptr};

    Summary ref;
    if (!refName.empty()) {
        const fs::path refFile = outRoot() / u8path(refName) / "summary.yml";
        if (!fs::exists(refFile)) fail("기준 묶음 결과가 없습니다. 먼저 hw_eda " + refName + " 를 실행하세요: " + u8(refFile));
        ref = Summary::read(refFile);
        ctx.ref = &ref;
        ctx.refName = refName;
    }

    if (ds.kind == SetKind::Sweep) buildSweepDeck(ctx);
    else buildGridDeck(ctx);
    addNotesSlides(ctx);

    const fs::path pptx = out / ("SEMES_EDA_" + setName + ".pptx");
    deck.save(pptx);
    summary.write(out / "summary.yml", "hw_eda " + setName + " 핵심 숫자 (자동 생성)");
    summary.write(docsRoot() / "sets" / u8path(setName) / "summary.yml", "hw_eda " + setName + " 핵심 숫자 (자동 생성)");
    log("발표자료: " + u8(pptx));
    log("완료");
    return 0;
}
