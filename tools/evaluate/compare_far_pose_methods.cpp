// 먼 자세의 T 오차: 계산식 모양을 바꾸는 방법과 두 번 보정하는 방법 비교
// 사용: compare_far_pose_methods <tables 폴더> <출력 폴더>
// 입력: data/new_photos/tables (prepare_dataset 출력. 측정값 5개와 정답)
// 출력: result.txt, by_range.tsv
// 방법: 1차식(지금), 자세 2차, 자세 3차, 값 2차, 두 번 보정
//   자세 2차: 1차식의 (Y, T) 를 [1, Y, T, Y², YT, T²] 식으로 한 번 더 보정
//   자세 3차: 위에 [Y³, Y²T, YT², T³] 추가
//   값 2차: 측정값 5개와 그 곱 15개
//   두 번 보정: 1차식으로 재서 보정한 뒤 센터 근처에서 다시 측정. 로봇 이동이 아닌 계산
//     첫 오차가 e 이면 보정 뒤 판은 센터에서 -e 위치. 둘째 측정의 오차 = 같은 회차 센터 근처 사진의 오차
// 시험: 섞음(5묶음 x 10번), 처음 보는 날, 범위 밖, 학습 범위 밖. 모두 학습에 안 쓴 사진으로 채점
// 계산식 모양이 측정 코드(CoefficientSet)와 달라 학습식을 이 파일에 따로 둠. 학습 방법은 tools/common/coefficient_fit 과 같음
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

const int kValueCount = 5, kRepeatCount = 10, kFoldCount = 5;
const double kToleranceYmm = 0.045, kToleranceTdeg = 0.01;
const int kHuberRounds = 6;
const double kRidgePerPhoto = 1e-7;
const double kInnerYmm = 1.0, kInnerTdeg = 0.5;
const double kRangeYmm = 2.0, kRangeTdeg = 1.0;
const double kNearCenterYmm = 0.25, kNearCenterTdeg = 0.05;
const double kCommandMismatchTdeg = 0.1;   // 범위 밖 사진 중 명령값이 틀려 보이는 것을 거른다

enum Method { kLinear, kPoseQuadratic, kPoseCubic, kValueQuadratic, kMethodCount };
const char* const kMethodNames[kMethodCount] = {"1차식 (지금)", "자세 2차", "자세 3차", "값 2차"};

struct Photo {
    int id = 0;
    std::string session;
    double command_y = 0, command_t = 0, label_y = 0, label_t = 0;
    std::array<double, kValueCount> values{};
    bool measured = false, usable = false, final_check = false;
    int fold[kRepeatCount] = {};
};

struct Pose { double y = 0, t = 0; };

// passes: 두 방향의 오차가 허용 범위인지 확인
//   입력: error_y (Y 오차 mm), error_t (T 오차 도)
//   출력: 둘 다 허용 범위 안이면 true
bool passes(double error_y, double error_t) { return std::abs(error_y) <= kToleranceYmm && std::abs(error_t) <= kToleranceTdeg; }

//
// 표 읽기
//

// read_rows: 탭 구분 표 읽기
//   입력: path (파일 경로)
//   출력: 첫 줄(열 이름)을 뺀 줄 목록
std::vector<std::vector<std::string>> read_rows(const fs::path& path) {
    std::vector<std::vector<std::string>> rows;
    std::ifstream file(path, std::ios::binary);
    std::string line;
    bool header = true;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (header) { header = false; continue; }
        if (line.empty()) continue;
        std::vector<std::string> cells;
        std::stringstream stream(line);
        std::string cell;
        while (std::getline(stream, cell, '\t')) cells.push_back(cell);
        rows.push_back(cells);
    }
    return rows;
}

// to_number: 숫자 읽기
//   입력: text (숫자 또는 nan)
//   출력: 숫자. 빈 칸과 nan 은 NaN
double to_number(const std::string& text) { return text == "nan" || text.empty() ? std::nan("") : std::stod(text); }

// load_photos: 사진별 측정값, 정답, 검증 묶음 읽기
//   입력: tables (tables 폴더)
//   출력: 사진 목록. 못 읽으면 빈 목록
std::vector<Photo> load_photos(const fs::path& tables) {
    std::map<int, Photo> by_id;
    for (const auto& r : read_rows(tables / "photos.tsv")) {
        Photo p; p.id = std::stoi(r[0]); p.session = r[1]; p.command_y = std::stod(r[3]); p.command_t = std::stod(r[4]);
        by_id[p.id] = p;
    }
    for (const auto& r : read_rows(tables / "features.tsv")) {
        Photo& p = by_id[std::stoi(r[0])];
        p.measured = r[1] == "1";
        for (int k = 0; k < kValueCount; ++k) p.values[k] = to_number(r[2 + k]);
    }
    for (const auto& r : read_rows(tables / "labels.tsv")) {
        Photo& p = by_id[std::stoi(r[0])];
        p.label_y = std::stod(r[1]); p.label_t = std::stod(r[2]);
    }
    for (const auto& r : read_rows(tables / "splits.tsv")) {
        Photo& p = by_id[std::stoi(r[0])];
        p.usable = r[1] == "1"; p.final_check = r[2] == "1";
        for (int k = 0; k < kRepeatCount; ++k) p.fold[k] = std::stoi(r[3 + k]);
    }
    std::vector<Photo> photos;
    for (const auto& item : by_id) photos.push_back(item.second);
    return photos;
}

//
// 학습: 큰 오차의 영향을 줄인 최소제곱 (tools/common/coefficient_fit 과 같은 방법)
//

using Row = std::vector<double>;

// solve: 연립방정식 A x = b 풀기
//   입력: a (정사각 행렬), b (오른쪽 값. Y 열과 T 열)
//   출력: x (항마다 Y 계수, T 계수)
std::vector<std::array<double, 2>> solve(std::vector<Row> a, std::vector<std::array<double, 2>> b) {
    const int n = int(a.size());
    for (int c = 0; c < n; ++c) {
        int pivot = c;
        for (int r = c + 1; r < n; ++r) if (std::abs(a[r][c]) > std::abs(a[pivot][c])) pivot = r;
        std::swap(a[c], a[pivot]); std::swap(b[c], b[pivot]);
        for (int r = c + 1; r < n; ++r) {
            const double f = a[r][c] / a[c][c];
            for (int k = c; k < n; ++k) a[r][k] -= f * a[c][k];
            b[r][0] -= f * b[c][0]; b[r][1] -= f * b[c][1];
        }
    }
    for (int c = n - 1; c >= 0; --c) {
        for (int k = c + 1; k < n; ++k) { b[c][0] -= a[c][k] * b[k][0]; b[c][1] -= a[c][k] * b[k][1]; }
        b[c][0] /= a[c][c]; b[c][1] /= a[c][c];
    }
    return b;
}

// solve_huber: 큰 오차의 영향을 줄인 최소제곱
//   입력: design (사진마다의 항. 첫 칸은 절편용 1), truth (사진마다의 정답)
//   출력: 항마다 Y 계수, T 계수
std::vector<std::array<double, 2>> solve_huber(const std::vector<Row>& design, const std::vector<Pose>& truth) {
    const int n = int(design.size()), p = int(design[0].size());
    std::vector<double> weight(n, 1.0);
    std::vector<std::array<double, 2>> coefficients;
    for (int round = 0; round < kHuberRounds; ++round) {
        std::vector<Row> a(p, Row(p, 0.0));
        std::vector<std::array<double, 2>> b(p, {0.0, 0.0});
        for (int i = 0; i < n; ++i)
            for (int r = 0; r < p; ++r) {
                const double wr = weight[i] * design[i][r];
                for (int c = 0; c < p; ++c) a[r][c] += wr * design[i][c];
                b[r][0] += wr * truth[i].y; b[r][1] += wr * truth[i].t;
            }
        for (int r = 1; r < p; ++r) a[r][r] += kRidgePerPhoto * n;
        coefficients = solve(a, b);
        double weight_sum = 0;
        for (int i = 0; i < n; ++i) {
            double y = 0, t = 0;
            for (int c = 0; c < p; ++c) { y += design[i][c] * coefficients[c][0]; t += design[i][c] * coefficients[c][1]; }
            const double badness = std::max(std::abs(y - truth[i].y) / kToleranceYmm, std::abs(t - truth[i].t) / kToleranceTdeg);
            weight[i] = std::min(1.0, 1.0 / std::max(badness, 1e-12));
            weight_sum += weight[i];
        }
        for (double& w : weight) w *= n / weight_sum;
    }
    return coefficients;
}

//
// 방법별 계산식
//

struct Model {
    Method method = kLinear;
    double mean[kValueCount] = {}, scale[kValueCount] = {};
    std::vector<std::array<double, 2>> first;    // 값으로 (Y, T) 를 내는 식
    std::vector<std::array<double, 2>> second;   // 자세 2차, 3차: (Y, T) 를 고치는 식
};

// value_terms: 측정값으로 계산식의 항 구성
//   입력: model (평균, 표준편차, 방법), photo (사진)
//   출력: [1, 표준화한 값 5개]. 값 2차는 곱 15개 추가
Row value_terms(const Model& model, const Photo& photo) {
    Row row = {1.0};
    double z[kValueCount];
    for (int k = 0; k < kValueCount; ++k) { z[k] = (photo.values[k] - model.mean[k]) / model.scale[k]; row.push_back(z[k]); }
    if (model.method == kValueQuadratic)
        for (int i = 0; i < kValueCount; ++i) for (int j = i; j < kValueCount; ++j) row.push_back(z[i] * z[j]);
    return row;
}

// pose_terms: 자세로 보정식의 항 구성
//   입력: method (자세 2차 또는 3차), pose (1차식의 Y, T)
//   출력: 2차 항 6개. 3차는 4개 추가. Y 는 2 mm, T 는 1 도로 나눈 값 사용
Row pose_terms(Method method, const Pose& pose) {
    const double y = pose.y / kRangeYmm, t = pose.t / kRangeTdeg;
    Row row = {1.0, y, t, y * y, y * t, t * t};
    if (method == kPoseCubic) { row.push_back(y * y * y); row.push_back(y * y * t); row.push_back(y * t * t); row.push_back(t * t * t); }
    return row;
}

// evaluate_terms: 항과 계수로 Y, T 계산
//   입력: coefficients (항마다 Y, T 계수), row (항)
//   출력: Pose (Y mm, T 도)
Pose evaluate_terms(const std::vector<std::array<double, 2>>& coefficients, const Row& row) {
    Pose pose;
    for (size_t c = 0; c < row.size(); ++c) { pose.y += row[c] * coefficients[c][0]; pose.t += row[c] * coefficients[c][1]; }
    return pose;
}

// predict: 사진 한 장의 Y, T 계산
//   입력: model (학습한 계산식), photo (사진)
//   출력: Pose (Y mm, T 도)
Pose predict(const Model& model, const Photo& photo) {
    Pose pose = evaluate_terms(model.first, value_terms(model, photo));
    if (!model.second.empty()) pose = evaluate_terms(model.second, pose_terms(model.method, pose));
    return pose;
}

// train: 방법별 계산식 학습
//   입력: method (방법), photos (학습 사진)
//   출력: Model (계산식)
Model train(Method method, const std::vector<const Photo*>& photos) {
    Model model;
    model.method = method;
    const int n = int(photos.size());
    for (int k = 0; k < kValueCount; ++k) {
        double sum = 0, square = 0;
        for (const Photo* p : photos) sum += p->values[k];
        model.mean[k] = sum / n;
        for (const Photo* p : photos) square += std::pow(p->values[k] - model.mean[k], 2);
        model.scale[k] = std::max(std::sqrt(square / n), 1e-6);
    }
    std::vector<Row> design;
    std::vector<Pose> truth;
    for (const Photo* p : photos) { design.push_back(value_terms(model, *p)); truth.push_back({p->label_y, p->label_t}); }
    model.first = solve_huber(design, truth);
    if (method == kPoseQuadratic || method == kPoseCubic) {
        std::vector<Row> pose_design;
        for (size_t i = 0; i < photos.size(); ++i) pose_design.push_back(pose_terms(method, evaluate_terms(model.first, design[i])));
        model.second = solve_huber(pose_design, truth);
    }
    return model;
}

//
// 채점과 보고
//

struct Scored { const Photo* photo; Pose error; };   // error = 측정 − 정답

struct Summary {
    int total = 0, passed = 0;
    double square_y = 0, square_t = 0, max_y = 0, max_t = 0, outward_t = 0;
    double truth_square_t = 0, truth_times_error_t = 0;   // T 오차 = k × 정답 T 의 기울기 k (비례 오차) 를 구할 때 쓴다
    void add(const Scored& s) {
        ++total; passed += passes(s.error.y, s.error.t);
        square_y += s.error.y * s.error.y; square_t += s.error.t * s.error.t;
        max_y = std::max(max_y, std::abs(s.error.y)); max_t = std::max(max_t, std::abs(s.error.t));
        if (s.photo->label_t != 0) outward_t += s.error.t * (s.photo->label_t > 0 ? 1 : -1);
        truth_square_t += s.photo->label_t * s.photo->label_t; truth_times_error_t += s.photo->label_t * s.error.t;
    }
    double rate() const { return total ? 100.0 * passed / total : 0; }
    double rms_y() const { return total ? std::sqrt(square_y / total) : 0; }
    double rms_t() const { return total ? std::sqrt(square_t / total) : 0; }
    double proportional_t_percent() const { return truth_square_t > 0 ? 100.0 * truth_times_error_t / truth_square_t : 0; }
};

std::ostringstream report;
std::ofstream table_file;

// print: 화면과 보고서에 한 줄 기록
//   입력: line (내용)
//   출력: 없음
void print(const std::string& line) { std::cout << line << "\n"; report << line << "\n"; }

// fixed: 소수 자릿수를 정한 숫자 표시
//   입력: value (숫자), digits (소수 자릿수)
//   출력: 숫자 문자열
std::string fixed(double value, int digits) { std::ostringstream o; o << std::fixed << std::setprecision(digits) << value; return o.str(); }

// pad: 화면 폭에 맞춘 칸 채우기
//   입력: text (글자), width (화면 칸 수. 한글은 2칸)
//   출력: 뒤에 빈칸을 채운 글자
std::string pad(const std::string& text, int width) {
    int shown = 0;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = text[i];
        const int bytes = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        shown += bytes >= 3 ? 2 : 1; i += bytes;
    }
    return text + std::string(std::max(0, width - shown), ' ');
}

const double kTBinEdges[] = {0.1, 0.3, 0.6, 1.0, 1e9};
const char* const kTBinNames[] = {"|T| 0 ~ 0.1", "|T| 0.1 ~ 0.3", "|T| 0.3 ~ 0.6", "|T| 0.6 ~ 1.0", "|T| 1.0 넘음"};
const double kYBinEdges[] = {0.25, 1.0, 2.0, 1e9};
const char* const kYBinNames[] = {"|Y| 0 ~ 0.25", "|Y| 0.25 ~ 1", "|Y| 1 ~ 2", "|Y| 2 넘음"};

// bin_of: 구간 번호
//   입력: value (명령값), edges (구간 끝값 목록)
//   출력: |value| 가 속한 구간 번호
int bin_of(double value, const double* edges) { int b = 0; while (std::abs(value) > edges[b] + 1e-9) ++b; return b; }

// near_center: 센터 근처 사진 확인
//   입력: p (사진)
//   출력: 명령이 |Y| <= 0.25 mm, |T| <= 0.05 도 이면 true
bool near_center(const Photo& p) { return std::abs(p.command_y) <= kNearCenterYmm && std::abs(p.command_t) <= kNearCenterTdeg; }

// print_method: 시험 하나, 방법 하나의 구간별 결과 기록
//   입력: test (시험 이름), method (방법), scored (채점한 사진), pool (센터 근처 사진. 두 번 보정 계산용, 1차식에서만 사용)
//   출력: 없음 (보고서와 표 기록)
void print_method(const std::string& test, Method method, const std::vector<Scored>& scored, const std::vector<Scored>& pool) {
    Summary all, by_t[5], by_y[4];
    for (const Scored& s : scored) { all.add(s); by_t[bin_of(s.photo->command_t, kTBinEdges)].add(s); by_y[bin_of(s.photo->command_y, kYBinEdges)].add(s); }
    auto line = [&](const std::string& name, const Summary& s) {
        if (!s.total) return;
        print("    " + pad(name, 16) + pad(std::to_string(s.total), 7) + pad(fixed(s.rate(), 1) + "%", 9) + pad(fixed(s.rms_t(), 4), 9) + pad(fixed(s.max_t, 4), 9) +
              pad(fixed(s.outward_t / s.total, 4), 10) + pad(fixed(s.rms_y(), 4), 9) + fixed(s.max_y, 4));
        table_file << test << "\t" << kMethodNames[method] << "\t" << name << "\t" << s.total << "\t" << fixed(s.rate(), 1) << "\t" << fixed(s.rms_t(), 5) << "\t" << fixed(s.max_t, 5)
                   << "\t" << fixed(s.outward_t / s.total, 5) << "\t" << fixed(s.rms_y(), 5) << "\t" << fixed(s.max_y, 5) << "\n";
    };
    print("  " + std::string(kMethodNames[method]));
    print("    " + pad("구간", 16) + pad("사진", 7) + pad("통과율", 9) + pad("T RMS", 9) + pad("T 최대", 9) + pad("T 바깥쪽", 10) + pad("Y RMS", 9) + "Y 최대");
    line("전체", all);
    for (int b = 0; b < 5; ++b) line(kTBinNames[b], by_t[b]);
    for (int b = 0; b < 4; ++b) line(kYBinNames[b], by_y[b]);
    print("    T 비례 오차 " + fixed(all.proportional_t_percent(), 2) + "% (T 오차 = 비율 × 정답 T 로 맞춘 비율)");

    if (method != kLinear) return;
    // 두 번 보정: 사진마다, 같은 회차의 센터 근처 사진이 통과한 비율
    std::map<std::string, Summary> pool_by_session;
    Summary pool_all;
    for (const Scored& s : pool) { pool_by_session[s.photo->session].add(s); pool_all.add(s); }
    if (!pool_all.total) { print("  두 번 보정: 센터 근처 사진이 없어 계산하지 못함"); return; }
    double expected[5] = {}, expected_all = 0;
    int count[5] = {}, outside = 0, borrowed = 0;
    for (const Scored& s : scored) {
        if (std::abs(s.error.y) > kNearCenterYmm || std::abs(s.error.t) > kNearCenterTdeg) ++outside;   // 한 번 보정해도 센터 근처에 못 옴
        const auto it = pool_by_session.find(s.photo->session);
        const Summary& use = it != pool_by_session.end() ? it->second : pool_all;
        borrowed += it == pool_by_session.end();
        const int b = bin_of(s.photo->command_t, kTBinEdges);
        expected[b] += use.rate(); ++count[b]; expected_all += use.rate();
    }
    print("  두 번 보정 (식은 1차식 그대로)");
    print("    둘째 측정 오차로 쓴 센터 근처 사진 " + std::to_string(pool_all.total) + "장: 통과율 " + fixed(pool_all.rate(), 1) + "%, T RMS " + fixed(pool_all.rms_t(), 4) + ", T 최대 " +
          fixed(pool_all.max_t, 4) + ", Y RMS " + fixed(pool_all.rms_y(), 4));
    for (const auto& s : pool_by_session) print("      " + pad(s.first, 18) + std::to_string(s.second.total) + "장, 통과율 " + fixed(s.second.rate(), 1) + "%");
    print("    " + pad("전체", 16) + pad(std::to_string(int(scored.size())), 7) + fixed(expected_all / scored.size(), 1) + "%");
    table_file << test << "\t두 번 보정\t전체\t" << scored.size() << "\t" << fixed(expected_all / scored.size(), 1) << "\t" << fixed(pool_all.rms_t(), 5) << "\t" << fixed(pool_all.max_t, 5)
               << "\t\t" << fixed(pool_all.rms_y(), 5) << "\t" << fixed(pool_all.max_y, 5) << "\n";
    for (int b = 0; b < 5; ++b) {
        if (!count[b]) continue;
        print("    " + pad(kTBinNames[b], 16) + pad(std::to_string(count[b]), 7) + fixed(expected[b] / count[b], 1) + "%");
        table_file << test << "\t두 번 보정\t" << kTBinNames[b] << "\t" << count[b] << "\t" << fixed(expected[b] / count[b], 1) << "\t\t\t\t\t\n";
    }
    print("    한 번 보정한 뒤에도 센터 근처(|Y| <= 0.25, |T| <= 0.05)에 못 온 사진 " + std::to_string(outside) + "장, 센터 근처 사진이 없어 다른 회차 것을 쓴 사진 " + std::to_string(borrowed) + "장");
}

using TrainTest = std::pair<std::vector<const Photo*>, std::vector<const Photo*>>;

// run_test: 시험 하나 실행
//   입력: name (시험 이름), splits (학습 사진과 시험 사진의 나누기 목록)
//   출력: 없음 (방법마다 학습, 채점, 기록)
void run_test(const std::string& name, const std::vector<TrainTest>& splits) {
    print("\n[" + name + "]");
    for (int m = 0; m < kMethodCount; ++m) {
        std::vector<Scored> scored, pool;
        for (const TrainTest& split : splits) {
            const Model model = train(Method(m), split.first);
            for (const Photo* p : split.second) {
                const Pose pose = predict(model, *p);
                scored.push_back({p, {pose.y - p->label_y, pose.t - p->label_t}});
            }
            // 센터 근처 사진: 시험 사진에 있으면 그것, 없으면(범위 밖 시험) 학습 사진의 것
            bool test_has_center = false;
            for (const Photo* p : split.second) test_has_center |= near_center(*p);
            for (const Photo* p : (test_has_center ? split.second : split.first)) {
                if (!near_center(*p)) continue;
                const Pose pose = predict(model, *p);
                pool.push_back({p, {pose.y - p->label_y, pose.t - p->label_t}});
            }
        }
        print_method(name, Method(m), scored, pool);
    }
}

}   // 파일 내부

// main: 먼 자세 T 오차 방법 비교
//   입력: 실행할 때 지정한 경로(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "사용: compare_far_pose_methods <tables 폴더> <출력 폴더>\n"; return 2; }
    const fs::path tables = fs::u8path(argv[1]), output = fs::u8path(argv[2]);
    const std::vector<Photo> photos = load_photos(tables);
    if (photos.empty()) { std::cerr << "표를 읽지 못했습니다: " << tables.u8string() << "\n"; return 1; }
    fs::create_directories(output);
    table_file.open(output / "by_range.tsv", std::ios::binary);
    table_file << "시험\t방법\t구간\t사진\t통과율\tT_RMS_도\tT_최대_도\tT_바깥쪽_평균_도\tY_RMS_mm\tY_최대_mm\n";

    std::vector<const Photo*> usable, development, beyond;
    std::vector<std::string> sessions;
    for (const Photo& p : photos) {
        if (p.usable) {
            usable.push_back(&p);
            if (!p.final_check) development.push_back(&p);
            if (std::find(sessions.begin(), sessions.end(), p.session) == sessions.end()) sessions.push_back(p.session);
        } else if (p.measured && (std::abs(p.command_y) > kRangeYmm + 1e-9 || std::abs(p.command_t) > kRangeTdeg + 1e-9)) {
            bool all_found = true;
            for (double v : p.values) all_found &= std::isfinite(v);
            if (all_found) beyond.push_back(&p);
        }
    }
    print("통과 기준: |Y 오차| <= 0.045 mm 그리고 |T 오차| <= 0.01 도. 단위: T 도, Y mm");
    print("T 바깥쪽 = T 오차 × (정답 T 의 부호) 의 평균. 양수면 실제보다 크게, 음수면 작게 잰 것");
    print("쓴 사진 " + std::to_string(usable.size()) + "장, 학습 범위를 넘는 사진 " + std::to_string(beyond.size()) + "장");

    {   // 1. 섞음
        std::vector<TrainTest> splits;
        for (int r = 0; r < kRepeatCount; ++r)
            for (int f = 0; f < kFoldCount; ++f) {
                TrainTest split;
                for (const Photo* p : development) (p->fold[r] == f ? split.second : split.first).push_back(p);
                splits.push_back(split);
            }
        run_test("1. 섞음 (5묶음 x 10번, 사진 수는 10번을 더한 것)", splits);
    }
    {   // 2. 처음 보는 날
        std::vector<TrainTest> splits;
        for (const std::string& session : sessions) {
            TrainTest split;
            for (const Photo* p : usable) (p->session == session ? split.second : split.first).push_back(p);
            splits.push_back(split);
        }
        run_test("2. 처음 보는 날 (회차 하나를 빼고 학습)", splits);
        for (size_t i = 0; i < sessions.size(); ++i) run_test("2-" + std::to_string(i + 1) + ". 처음 보는 날: " + sessions[i], {splits[i]});
    }
    {   // 3. 범위 밖
        TrainTest split;
        for (const Photo* p : usable) (std::abs(p->command_y) <= kInnerYmm && std::abs(p->command_t) <= kInnerTdeg ? split.first : split.second).push_back(p);
        run_test("3. 범위 밖 (안쪽 |Y| <= 1, |T| <= 0.5 로 학습, 바깥 시험)", {split});
    }
    if (!beyond.empty()) {   // 4. 학습 범위 밖
        print("\n학습 범위를 넘는 사진");
        const Model linear = train(kLinear, usable);
        std::vector<const Photo*> kept;
        for (const Photo* p : beyond) {
            const Pose pose = predict(linear, *p);
            const bool mismatch = std::abs(pose.t - p->label_t) > kCommandMismatchTdeg;
            print("  " + p->session + "  명령 Y " + fixed(p->command_y, 2) + ", T " + fixed(p->command_t, 2) + (mismatch ? "   (명령값이 틀려 보여 뺌)" : ""));
            if (!mismatch) kept.push_back(p);
        }
        run_test("4. 학습 범위 밖 (전부로 학습, 범위를 넘는 사진 시험)", {TrainTest(usable, kept)});
    }
    std::ofstream(output / "result.txt", std::ios::binary) << report.str();
    return 0;
}
