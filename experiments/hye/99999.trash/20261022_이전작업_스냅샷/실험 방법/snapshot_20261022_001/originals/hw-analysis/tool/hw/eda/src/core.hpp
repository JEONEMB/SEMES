// =====================================================================================
// core.hpp  -  EDA 공통: 데이터 묶음 읽기, 파일명 해석, 이미지 입출력, CSV, 요약값, 통계
//
// [데이터 묶음]  data/hw/bmp/<묶음>/   (data/hw/data-automation 으로 정리한 폴더)
//
//   sweep 묶음 (1차, 예: 20261014)  한 번에 한 축만 움직이며 연속 촬영
//       center.bmp
//       Y/Y+0.50.bmp ...         Y 명령값(mm)
//       Theta/T-0.35.bmp ...     Θ 명령값
//
//   grid 묶음 (2차, 예: 20261020)   Y 와 Θ 를 같이 움직인 조합을 여러 번 촬영
//       기준 위치/center1.bmp ...           기준(center) 여러 장
//       Y+0.05 T+0.01/Y+0.05 T+0.01 (1).bmp  조합마다 반복 촬영 (1), (2), (3)
// =====================================================================================
#pragma once

#include <windows.h>

#include "nan.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace hw {

namespace fs = std::filesystem;

// -------------------------------------------------------------------------------------
// 데이터 묶음
// -------------------------------------------------------------------------------------

// 사진 한 장
struct Shot {
    fs::path file;            // 전체 경로
    std::string rel;          // 묶음 폴더 기준 경로 (UTF-8)
    std::string name;         // 파일명 (UTF-8)
    bool center = false;      // 기준(center) 사진인지
    double y = 0.0;           // Y 명령값 (mm)
    double t = 0.0;           // Θ 명령값
    int rep = 0;              // 반복 번호 (grid 묶음의 (1), (2) ... / center1, center2 ...)
    std::string group;        // sweep: trial 이름 ("Y 0.1 간격 +" ...), grid: 조합 이름 ("Y+0.05 T+0.01")
    double minutes = 0.0;     // 첫 사진부터 지난 시간 (파일 수정 시각 = 촬영 시각)
    std::string takenAt;      // 촬영 시각 "YYYY-MM-DD HH:MM:SS"
};

enum class SetKind { Sweep, Grid };

struct DataSet {
    std::string name;         // 묶음 이름 (폴더명)
    fs::path dir;             // data/hw/bmp/<묶음>
    SetKind kind = SetKind::Sweep;
    std::vector<Shot> shots;  // 촬영 시각 순서
    std::vector<size_t> centers;  // center 사진 번호 (shots 안의 위치)
};

// 프로젝트 루트(실행 위치) 기준 경로
fs::path dataRoot();          // data/hw/bmp
fs::path outRoot();           // docs/validation-runs/hw/eda
fs::path docsRoot();          // docs/hw/eda

// 묶음 목록 (이름순)
std::vector<std::string> listSets();

// 묶음 읽기: 파일명 해석, 종류 판별, 촬영 시각 순 정렬
DataSet loadSet(const std::string &name);

// sweep trial 이름 (축, 명령값 -> "Y 0.1 간격 +" 등)
std::string trialOf(char axis, double value);

// -------------------------------------------------------------------------------------
// 이미지, 파일
// -------------------------------------------------------------------------------------

// OpenCV 3.4 imread/imwrite 는 한글 경로를 못 열어서 바이트로 읽고 쓴다
cv::Mat readImage(const fs::path &p, int flags = cv::IMREAD_UNCHANGED);
cv::Mat readGray(const fs::path &p);              // 3채널이 같은 흑백 사진이라 첫 채널만
void writeImage(const fs::path &p, const cv::Mat &img);   // 확장자로 형식 결정 (.png)

// UTF-8 문자열 <-> 경로
inline fs::path u8path(const std::string &s) { return fs::u8path(s); }
inline std::string u8(const fs::path &p) { return p.u8string(); }

[[noreturn]] void fail(const std::string &msg);
void log(const std::string &msg);                 // 콘솔 + 실행 로그 파일
void openLog(const fs::path &file);

// -------------------------------------------------------------------------------------
// CSV, 요약값
// -------------------------------------------------------------------------------------

std::string num(double v, int prec);              // 고정 소수점 문자열 (-0.000 은 0.000)
std::string signedNum(double v, int prec);        // +0.12 / -0.12

class Csv {
public:
    Csv(const fs::path &file, const std::vector<std::string> &header);
    void row(const std::vector<std::string> &cells);
private:
    std::ofstream out_;
};

// 슬라이드 문장과 묶음끼리 비교에 쓰는 핵심 숫자. "key: value" 한 줄씩 summary.yml 로 저장한다.
class Summary {
public:
    void set(const std::string &key, double v, int prec = 4) { put(key, num(v, prec)); }
    void set(const std::string &key, const std::string &v) { put(key, "\"" + v + "\""); }
    bool has(const std::string &key) const;
    double get(const std::string &key) const;     // 없으면 NaN
    void write(const fs::path &file, const std::string &header) const;
    static Summary read(const fs::path &file);
private:
    void put(const std::string &key, const std::string &v);
    std::vector<std::pair<std::string, std::string>> kv_;
};

// -------------------------------------------------------------------------------------
// 통계
// -------------------------------------------------------------------------------------

double mean(const std::vector<double> &v);
double stdev(const std::vector<double> &v, int ddof = 1);
double median(std::vector<double> v);
double maxAbs(const std::vector<double> &v);

struct LineFit { double slope = 0, intercept = 0, residSd = 0; };
LineFit fitLine(const std::vector<double> &x, const std::vector<double> &y);      // 잔차 σ 는 ddof=2

// 2차 곡선 맞춤 후 잔차 (점마다 흔들림 계산용)
std::vector<double> quadResiduals(const std::vector<double> &x, const std::vector<double> &y);

}  // namespace hw
