// 원본 이미지 묶음(zip)을 data/ 아래로 정리한다.
//
//   data_ingest                                   data/raw/zip/ 에 있는 새 zip 전부 처리
//   data_ingest "D:/받은파일/세트.zip"             다른 곳의 zip 을 raw/zip/ 으로 복사한 뒤 처리
//   data_ingest 세트.zip --date 20261014           날짜 폴더 이름 지정 (기본: zip 파일 수정일)
//   data_ingest --data <data 폴더>                 data 폴더 위치 (기본: 현재 폴더의 data)
//
// 결과
//   data/raw/zip/<zip 과 분할 조각 .z01 ...>   받은 그대로 보관
//   data/bmp/<YYYYMMDD>/...                  zip 안의 BMP (zip 안 폴더 구조 유지, 맨 위 폴더 하나는 벗김)
//   data/png/<YYYYMMDD>/...                  같은 경로의 무손실 PNG
//   data/manifest.csv                         처리한 묶음 목록 (zip 이름, 날짜, 장수, sha256)
//
// zip 풀기는 7-Zip, 반디집(bz.exe) 순서로 찾아서 쓴다. 분할 zip(.z01 ...)은 Windows 기본 tar 로는 못 푼다.

#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace {

struct Paths {
    fs::path data, raw, bmp, png, manifest;
    explicit Paths(const fs::path &root)
        : data(root), raw(root / "raw" / "zip"), bmp(root / "bmp"), png(root / "png"), manifest(root / "manifest.csv") {}
};

[[noreturn]] void fail(const std::string &msg) {
    std::cerr << "오류: " << msg << "\n";
    std::exit(1);
}

std::string u8(const fs::path &p) { return p.u8string(); }

// zip 과 같은 이름의 분할 조각 (.z01, .z02 ...)
std::vector<fs::path> partsOf(const fs::path &archive) {
    std::vector<fs::path> parts;
    for (int i = 1; i < 100; ++i) {
        wchar_t ext[8];
        swprintf(ext, 8, L".z%02d", i);
        fs::path p = archive;
        p.replace_extension(ext);
        if (!fs::exists(p)) break;
        parts.push_back(p);
    }
    return parts;
}

std::string sha256(const fs::path &file) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) fail("sha256 초기화 실패");
    BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0);
    std::ifstream in(file, std::ios::binary);
    std::vector<char> buf(1 << 20);
    while (in.read(buf.data(), buf.size()) || in.gcount() > 0)
        BCryptHashData(hash, reinterpret_cast<PUCHAR>(buf.data()), static_cast<ULONG>(in.gcount()), 0);
    unsigned char digest[32];
    BCryptFinishHash(hash, digest, sizeof(digest), 0);
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    std::ostringstream out;
    for (unsigned char b : digest) out << std::hex << (b < 16 ? "0" : "") << static_cast<int>(b);
    return out.str();
}

// 외부 프로그램 실행 (한글 경로 그대로 전달)
int run(const std::wstring &cmdline) {
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    std::wstring cmd = cmdline;
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return -1;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(code);
}

void extract(const fs::path &archive, const fs::path &out) {
    const std::wstring a = L"\"" + archive.wstring() + L"\"", o = out.wstring();
    const fs::path sevenZip = L"C:\\Program Files\\7-Zip\\7z.exe", bandizip = L"C:\\Program Files\\Bandizip\\bz.exe";
    if (fs::exists(sevenZip)) {
        if (run(L"\"" + sevenZip.wstring() + L"\" x " + a + L" \"-o" + o + L"\" -y") == 0) return;
        fail("7-Zip 압축 해제 실패: " + u8(archive.filename()));
    }
    if (fs::exists(bandizip)) {
        if (run(L"\"" + bandizip.wstring() + L"\" x \"-o:" + o + L"\" -y " + a) == 0) return;
        fail("반디집 압축 해제 실패: " + u8(archive.filename()));
    }
    if (partsOf(archive).empty() && run(L"tar -xf " + a + L" -C \"" + o + L"\"") == 0) return;  // 분할이 아닌 zip 은 Windows 기본 tar
    fail("분할 zip 을 풀 도구가 없습니다. 7-Zip 이나 반디집을 설치하세요: " + u8(archive.filename()));
}

// OpenCV 3.4 의 imread/imwrite 는 한글 경로를 못 읽어서 바이트로 읽고 쓴다
cv::Mat readImage(const fs::path &p) {
    std::ifstream in(p, std::ios::binary);
    std::vector<uchar> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
}

void writePng(const fs::path &p, const cv::Mat &img) {
    std::vector<uchar> bytes;
    if (!cv::imencode(".png", img, bytes)) fail("PNG 변환 실패: " + u8(p));
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::string dateOf(const fs::path &file) {
    // C++17 에는 file_time -> system_clock 변환이 없어서 지금 시각 차이로 옮긴다
    auto ft = fs::last_write_time(file);
    auto sys = std::chrono::system_clock::now() + std::chrono::duration_cast<std::chrono::system_clock::duration>(ft - fs::file_time_type::clock::now());
    std::time_t t = std::chrono::system_clock::to_time_t(sys);
    std::tm tm{};
    localtime_s(&tm, &t);
    char s[16];
    std::strftime(s, sizeof(s), "%Y%m%d", &tm);
    return s;
}

std::string now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);
    char s[32];
    std::strftime(s, sizeof(s), "%Y-%m-%dT%H:%M:%S", &tm);
    return s;
}

std::set<std::string> ingested(const Paths &P) {
    std::set<std::string> names;
    std::ifstream in(P.manifest);
    std::string line;
    std::getline(in, line);  // 머리줄
    while (std::getline(in, line))
        if (!line.empty()) names.insert(line.substr(0, line.find(',')));
    return names;
}

void appendManifest(const Paths &P, const std::string &row) {
    const bool fresh = !fs::exists(P.manifest);
    std::ofstream out(P.manifest, std::ios::app | std::ios::binary);
    if (fresh) out << "zip,parts,date,images,sha256,ingested\n";
    out << row << "\n";
}

int ingest(const Paths &P, const fs::path &archive, const std::string &date) {
    const fs::path bmpDir = P.bmp / date, pngDir = P.png / date;
    if (fs::exists(bmpDir)) fail(u8(bmpDir) + " 가 이미 있습니다. --date 로 다른 날짜를 주거나 폴더를 지우고 다시 실행하세요.");

    const fs::path tmp = P.data / ("tmp_ingest_" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(tmp);
    extract(archive, tmp);

    std::vector<fs::path> bmps;
    for (auto &e : fs::recursive_directory_iterator(tmp)) {
        std::wstring ext = e.path().extension().wstring();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
        if (e.is_regular_file() && ext == L".bmp") bmps.push_back(e.path());
    }
    if (bmps.empty()) { fs::remove_all(tmp); fail("BMP 가 없습니다: " + u8(archive.filename())); }
    std::sort(bmps.begin(), bmps.end());

    // 맨 위 폴더가 하나뿐이면 벗김 (예: "SSAFY Test Image Set/")
    fs::path root = tmp;
    std::vector<fs::path> tops(fs::directory_iterator(tmp), fs::directory_iterator{});
    if (tops.size() == 1 && fs::is_directory(tops[0])) root = tops[0];

    for (size_t i = 0; i < bmps.size(); ++i) {
        const fs::path rel = fs::relative(bmps[i], root);
        const fs::path dst = bmpDir / rel, png = (pngDir / rel).replace_extension(".png");
        fs::create_directories(dst.parent_path());
        fs::create_directories(png.parent_path());
        fs::rename(bmps[i], dst);
        cv::Mat img = readImage(dst);
        if (img.empty()) fail("BMP 를 읽지 못했습니다: " + u8(rel));
        writePng(png, img);
        if ((i + 1) % 50 == 0 || i + 1 == bmps.size()) std::cout << "  " << i + 1 << "/" << bmps.size() << std::endl;
    }
    fs::remove_all(tmp);
    return static_cast<int>(bmps.size());
}

}  // namespace

int wmain(int argc, wchar_t **argv) {
    SetConsoleOutputCP(CP_UTF8);

    std::optional<fs::path> archiveArg;
    std::optional<std::string> dateArg;
    fs::path dataRoot = fs::current_path() / "data";
    for (int i = 1; i < argc; ++i) {
        const std::wstring a = argv[i];
        if (a == L"--date" && i + 1 < argc) dateArg = fs::path(argv[++i]).u8string();
        else if (a == L"--data" && i + 1 < argc) dataRoot = argv[++i];
        else if (a == L"-h" || a == L"--help") {
            std::cout << "사용법: data_ingest [zip 경로] [--date YYYYMMDD] [--data data 폴더]\n";
            return 0;
        } else archiveArg = fs::path(a);
    }
    if (dateArg && (dateArg->size() != 8 || !std::all_of(dateArg->begin(), dateArg->end(), ::isdigit)))
        fail("--date 는 YYYYMMDD 형식이어야 합니다: " + *dateArg);

    const Paths P(fs::absolute(dataRoot));
    fs::create_directories(P.raw);

    std::vector<fs::path> targets;
    if (archiveArg) {
        const fs::path src = fs::absolute(*archiveArg);
        if (!fs::exists(src)) fail("zip 이 없습니다: " + u8(src));
        if (fs::equivalent(src.parent_path(), P.raw) == false) {  // 받은 zip 과 분할 조각을 raw/zip/ 으로 복사
            std::vector<fs::path> files = partsOf(src);
            files.push_back(src);
            for (auto &f : files) fs::copy_file(f, P.raw / f.filename(), fs::copy_options::overwrite_existing);
        }
        targets.push_back(P.raw / src.filename());
    } else {
        for (auto &e : fs::directory_iterator(P.raw))
            if (e.path().extension() == ".zip") targets.push_back(e.path());
        std::sort(targets.begin(), targets.end());
    }

    const auto seen = ingested(P);
    for (auto &archive : targets) {
        const std::string name = u8(archive.filename());
        if (seen.count(name)) { std::cout << "건너뜀 (이미 처리): " << name << "\n"; continue; }
        const std::string date = dateArg ? *dateArg : dateOf(archive);
        std::cout << name << " -> bmp/" << date << ", png/" << date << std::endl;
        const int n = ingest(P, archive, date);
        appendManifest(P, name + "," + std::to_string(partsOf(archive).size()) + "," + date + "," + std::to_string(n) + "," +
                              sha256(archive) + "," + now());
        std::cout << "완료: " << n << "장\n";
    }
    return 0;
}
