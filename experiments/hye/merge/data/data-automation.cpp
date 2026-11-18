// =====================================================================================
// data-automation.cpp
//
// 받은 원본 이미지 zip 을 data/ 아래 bmp/, png/ 폴더로 나눠 정리한다.
//
// [실행하면]
//   1. zip 경로를 묻는다.   (탐색기 "경로로 복사" 결과, 창에 끌어다 놓은 경로 모두 가능)
//   2. 폴더명을 정한다.
//        - zip 이름에 날짜(20261014, 2026-10-14, 2026_10_14 ...)가 있으면 그 날짜만 YYYYMMDD 로 폴더명에 쓴다.
//          (예: zip_20261014.zip -> 20261014)
//        - 날짜가 없으면 폴더명을 입력하라고 묻는다.
//   3. zip 을 풀어서 (zip 안에 또 zip 이 있으면 그것까지 풀어서)
//        data/bmp/<폴더명>/...   원본 BMP   (zip 안의 폴더 구조 그대로, 맨 위 폴더 하나만 벗김)
//        data/png/<폴더명>/...   같은 경로의 무손실 PNG
//      로 나눠 넣는다.
//   4. 받은 zip(과 분할 조각 .z01 ...)은 data/raw/zip/ 에 복사해 보관한다.
//
// [zip 풀기]
//   C++ 표준 라이브러리와 OpenCV 에는 zip 기능이 없어서 설치된 압축 프로그램을 불러 쓴다.
//   7-Zip -> 반디집(bz.exe) -> Windows 기본 tar 순서로 찾는다.
//   분할 zip(.z01, .z02 ...)은 7-Zip 이나 반디집이 있어야 풀린다.
//
// [빌드]  프로젝트 루트에서 (팀 개발환경 설정과 같은 옵션)
//   cmake -S data -B build/data -G "Visual Studio 18 2026" -A x64 -T v141 ^
//         -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" ^
//         -DOpenCV_DIR="C:/opencv/build"
//   cmake --build build/data --config Release
//
// [실행]  프로젝트 루트에서
//   .\build\data\Release\data-automation.exe              -> zip 경로를 물어봄
//   .\build\data\Release\data-automation.exe "<zip 경로>"   -> 묻지 않고 바로 처리 (폴더명은 필요하면 물어봄)
// =====================================================================================

#include <windows.h>
#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// -------------------------------------------------------------------------------------
// 출력, 입력 도우미
// -------------------------------------------------------------------------------------

// 한글이 섞인 경로를 콘솔에 찍을 때 쓴다 (경로는 내부적으로 UTF-16, 콘솔 출력은 UTF-8)
static std::string u8(const fs::path &p) { return p.u8string(); }

// 오류를 알리고 끝낸다. 창이 바로 닫히지 않도록 엔터를 기다린다.
[[noreturn]] static void fail(const std::string &msg) {
    std::cerr << "\n[오류] " << msg << "\n엔터를 누르면 종료합니다.";
    std::wstring dummy;
    std::getline(std::wcin, dummy);
    std::exit(1);
}

// 한 줄 입력받기. 앞뒤 공백과, 탐색기 "경로로 복사"가 붙이는 따옴표를 벗긴다.
static std::wstring ask(const std::string &prompt) {
    std::cout << prompt;
    std::wstring line;
    if (!std::getline(std::wcin, line)) fail("입력을 받지 못했습니다.");
    auto notSpace = [](wchar_t c) { return !std::iswspace(c); };
    line.erase(line.begin(), std::find_if(line.begin(), line.end(), notSpace));
    line.erase(std::find_if(line.rbegin(), line.rend(), notSpace).base(), line.end());
    if (line.size() >= 2 && line.front() == L'"' && line.back() == L'"') line = line.substr(1, line.size() - 2);
    return line;
}

// -------------------------------------------------------------------------------------
// 폴더명 정하기
// -------------------------------------------------------------------------------------

// zip 이름에서 날짜를 찾아 YYYYMMDD 로 돌려준다. 없으면 빈 문자열.
// 20261014, 2026-10-14, 2026_10_14, 2026.10.14 를 모두 20261014 로 맞춘다.
static std::wstring dateIn(const std::wstring &name) {
    static const std::wregex date(L"((?:19|20)\\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\\d|3[01])");
    std::wsmatch m;
    if (!std::regex_search(name, m, date)) return L"";
    return m[1].str() + m[2].str() + m[3].str();
}

// 폴더명으로 쓸 수 없는 글자(\ / : * ? " < > |)가 있는지 본다.
static bool validFolderName(const std::wstring &name) {
    return !name.empty() && name.find_first_of(L"\\/:*?\"<>|") == std::wstring::npos;
}

// zip 이름에 날짜가 있으면 그 날짜(YYYYMMDD)를 폴더명으로, 없으면 입력받는다. 이미 있는 폴더명이면 다시 묻는다.
static std::wstring chooseFolderName(const fs::path &zip, const fs::path &bmpRoot) {
    std::wstring name = dateIn(zip.stem().wstring());
    if (!name.empty() && !fs::exists(bmpRoot / name)) {
        std::cout << "zip 이름의 날짜를 폴더명으로 씁니다: " << u8(name) << "\n";
        return name;
    }
    if (!name.empty()) std::cout << "같은 날짜 폴더가 이미 있습니다: bmp/" << u8(name) << "\n";
    else std::cout << "zip 이름에 날짜가 없습니다: " << u8(zip.filename()) << "\n";

    while (true) {
        name = ask("폴더명을 입력하세요 (예: 20261014): ");
        if (!validFolderName(name)) { std::cout << "  폴더명에 \\ / : * ? \" < > | 는 쓸 수 없습니다.\n"; continue; }
        if (fs::exists(bmpRoot / name)) { std::cout << "  이미 있는 폴더입니다: bmp/" << u8(name) << "\n"; continue; }
        return name;
    }
}

// -------------------------------------------------------------------------------------
// zip 풀기
// -------------------------------------------------------------------------------------

// zip 과 같은 이름의 분할 조각을 찾는다. (세트.zip -> 세트.z01, 세트.z02 ...)
static std::vector<fs::path> splitParts(const fs::path &zip) {
    std::vector<fs::path> parts;
    for (int i = 1; i < 100; ++i) {
        wchar_t ext[8];
        swprintf(ext, 8, L".z%02d", i);
        fs::path part = zip;
        part.replace_extension(ext);
        if (!fs::exists(part)) break;
        parts.push_back(part);
    }
    return parts;
}

// 외부 프로그램을 실행하고 끝날 때까지 기다린다. 성공하면 0 을 돌려준다.
// (std::system 은 한글 경로를 깨뜨릴 수 있어서 Windows API 를 직접 쓴다)
static int runProgram(std::wstring commandLine) {
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, commandLine.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
        return -1;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 1;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(exitCode);
}

// zip 을 outDir 에 푼다. 7-Zip -> 반디집 -> Windows tar 순서로 시도한다.
static void unzip(const fs::path &zip, const fs::path &outDir) {
    const std::wstring z = L"\"" + zip.wstring() + L"\"";
    const std::wstring o = outDir.wstring();
    const fs::path sevenZip = L"C:\\Program Files\\7-Zip\\7z.exe";
    const fs::path bandizip = L"C:\\Program Files\\Bandizip\\bz.exe";

    if (fs::exists(sevenZip)) {
        std::cout << "압축 해제 (7-Zip) ...\n";
        if (runProgram(L"\"" + sevenZip.wstring() + L"\" x " + z + L" \"-o" + o + L"\" -y") == 0) return;
        fail("7-Zip 으로 풀지 못했습니다: " + u8(zip.filename()));
    }
    if (fs::exists(bandizip)) {
        std::cout << "압축 해제 (반디집) ...\n";
        if (runProgram(L"\"" + bandizip.wstring() + L"\" x \"-o:" + o + L"\" -y " + z) == 0) return;
        fail("반디집으로 풀지 못했습니다: " + u8(zip.filename()));
    }
    if (splitParts(zip).empty()) {  // 분할이 아닌 zip 은 Windows 10 이상 기본 tar 로 풀린다
        std::cout << "압축 해제 (Windows tar) ...\n";
        if (runProgram(L"tar -xf " + z + L" -C \"" + o + L"\"") == 0) return;
    }
    fail("zip 을 풀 프로그램이 없습니다. 7-Zip 이나 반디집을 설치하세요.");
}

// dir 아래(하위 폴더 포함)에서 확장자가 ext 인 파일을 찾아 정렬해 돌려준다. 확장자 대소문자는 무시한다.
static std::vector<fs::path> filesWithExtension(const fs::path &dir, const std::wstring &ext) {
    std::vector<fs::path> found;
    for (const auto &entry : fs::recursive_directory_iterator(dir)) {
        std::wstring e = entry.path().extension().wstring();
        std::transform(e.begin(), e.end(), e.begin(), ::towlower);
        if (entry.is_regular_file() && e == ext) found.push_back(entry.path());
    }
    std::sort(found.begin(), found.end());
    return found;
}

// -------------------------------------------------------------------------------------
// 이미지 읽기, 쓰기
// OpenCV 3.4 의 imread / imwrite 는 Windows 에서 한글 경로를 열지 못한다.
// 그래서 파일은 C++ 스트림(UTF-16 경로 지원)으로 바이트째 읽고 쓰고, 변환만 OpenCV 에 맡긴다.
// -------------------------------------------------------------------------------------

static cv::Mat readImage(const fs::path &file) {
    std::ifstream in(file, std::ios::binary);
    std::vector<uchar> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return cv::imdecode(bytes, cv::IMREAD_UNCHANGED);  // IMREAD_UNCHANGED: 채널, 비트 깊이 그대로
}

static void writePng(const fs::path &file, const cv::Mat &image) {
    std::vector<uchar> bytes;
    if (!cv::imencode(".png", image, bytes)) fail("PNG 로 바꾸지 못했습니다: " + u8(file));  // PNG 는 무손실
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

// -------------------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------------------

int wmain(int argc, wchar_t **argv) {
    // 콘솔 입출력 설정: 출력은 UTF-8, 입력은 UTF-16 (한글 경로, 한글 폴더명 입력용)
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdin), _O_U16TEXT);

    // data 폴더 위치: 이 프로그램을 실행한 곳(프로젝트 루트)의 data/
    const fs::path dataDir = fs::current_path() / "data";
    const fs::path rawDir = dataDir / "raw" / "zip";
    const fs::path bmpRoot = dataDir / "bmp";
    const fs::path pngRoot = dataDir / "png";
    if (!fs::exists(dataDir)) fail("data 폴더가 없습니다. 프로젝트 루트에서 실행하세요: " + u8(fs::current_path()));

    std::cout << "=== 원본 zip -> data/bmp, data/png 정리 ===\n\n";

    // 1. zip 경로: 실행 인자로 받았으면 그대로, 아니면 물어본다.
    fs::path zip = (argc >= 2) ? fs::path(argv[1]) : fs::path(ask("zip 경로를 입력하세요: "));
    zip = fs::absolute(zip);
    if (!fs::exists(zip)) fail("파일이 없습니다: " + u8(zip));
    if (zip.extension() != ".zip" && zip.extension() != ".ZIP") fail("zip 파일이 아닙니다: " + u8(zip.filename()));
    const std::vector<fs::path> parts = splitParts(zip);
    std::cout << "zip: " << u8(zip.filename());
    if (!parts.empty()) std::cout << "  (분할 조각 " << parts.size() << "개 함께)";
    std::cout << "\n";

    // 2. 폴더명
    const std::wstring folder = chooseFolderName(zip, bmpRoot);
    const fs::path bmpDir = bmpRoot / folder;
    const fs::path pngDir = pngRoot / folder;

    // 3. 받은 zip 보관: raw/zip/ 으로 복사 (이미 raw/zip/ 안의 파일이면 건너뜀)
    fs::create_directories(rawDir);
    if (!fs::equivalent(zip.parent_path(), rawDir)) {
        std::cout << "raw/zip/ 에 보관 ...\n";
        std::vector<fs::path> files = parts;
        files.push_back(zip);
        for (const auto &f : files) fs::copy_file(f, rawDir / f.filename(), fs::copy_options::overwrite_existing);
    }

    // 4. 임시 폴더에 풀기 (같은 디스크라 BMP 를 옮길 때 복사 없이 이름만 바뀜)
    const fs::path tempDir = dataDir / (L"tmp_" + folder);
    fs::remove_all(tempDir);
    fs::create_directories(tempDir);
    fs::path extracted = tempDir / L"0";
    fs::create_directories(extracted);
    unzip(zip, extracted);

    // 5. 풀린 파일 중 BMP 만 모은다.
    //    SSAFY 에서 받은 원본처럼 "zip 안에 분할 zip" 인 경우가 있어서,
    //    BMP 가 없고 안에 zip 이 있으면 그 zip 을 다시 푼다 (최대 3단계).
    std::vector<fs::path> bmps = filesWithExtension(extracted, L".bmp");
    for (int depth = 1; bmps.empty() && depth <= 3; ++depth) {
        const std::vector<fs::path> inner = filesWithExtension(extracted, L".zip");
        if (inner.empty()) break;
        std::cout << "zip 안에 zip 이 있어서 한 번 더 풉니다: " << u8(inner[0].filename()) << "\n";
        const fs::path next = tempDir / std::to_wstring(depth);
        fs::create_directories(next);
        for (const auto &z : inner) unzip(z, next);  // 분할 조각(.z01 ...)은 같은 폴더에 있어서 함께 풀린다
        extracted = next;
        bmps = filesWithExtension(extracted, L".bmp");
    }
    if (bmps.empty()) { fs::remove_all(tempDir); fail("zip 안에 BMP 가 없습니다."); }

    // zip 안 맨 위에 폴더가 하나뿐이면(예: "SSAFY Test Image Set/") 그 폴더는 벗긴다
    fs::path base = extracted;
    std::vector<fs::path> top(fs::directory_iterator(extracted), fs::directory_iterator{});
    if (top.size() == 1 && fs::is_directory(top[0])) base = top[0];

    // 6. BMP 는 bmp/<폴더명>/ 으로 옮기고, 같은 경로에 PNG 를 만든다
    std::cout << "BMP " << bmps.size() << "장 -> bmp/" << u8(folder) << ", png/" << u8(folder) << "\n";
    for (size_t i = 0; i < bmps.size(); ++i) {
        const fs::path rel = fs::relative(bmps[i], base);           // 예: Y/Y+0.50.bmp
        const fs::path bmpOut = bmpDir / rel;
        const fs::path pngOut = (pngDir / rel).replace_extension(".png");
        fs::create_directories(bmpOut.parent_path());
        fs::create_directories(pngOut.parent_path());

        fs::rename(bmps[i], bmpOut);                                  // 옮기기 (복사 아님)
        const cv::Mat image = readImage(bmpOut);
        if (image.empty()) fail("BMP 를 읽지 못했습니다: " + u8(rel));
        writePng(pngOut, image);

        if ((i + 1) % 20 == 0 || i + 1 == bmps.size())
            std::cout << "  " << (i + 1) << " / " << bmps.size() << "\r" << std::flush;
    }

    // 7. 임시 폴더 정리
    fs::remove_all(tempDir);
    std::cout << "\n\n완료: " << bmps.size() << "장\n"
              << "  " << u8(bmpDir) << "\n"
              << "  " << u8(pngDir) << "\n";
    return 0;
}
