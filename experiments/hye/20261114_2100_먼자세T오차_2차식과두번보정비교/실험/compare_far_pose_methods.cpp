// 먼 자세에서 T 오차가 커지는 문제: 식을 바꾸는 방법과 두 번 보정하는 방법을 비교한다.
//
// 사용: compare_far_pose_methods <tables 폴더> <출력 폴더>
//   tables 폴더 = semes/data/new_photos/tables (prepare_dataset 이 만든 표. 값 5개와 정답이 들어 있다)
//
// 비교하는 방법
//   1차식        지금 식. Y, T = 절편 + Σ 계수 × 값 5개
//   자세 2차     1차식으로 낸 (Y, T) 를 2차식 [1, Y, T, Y², YT, T²] 으로 한 번 더 고친다
//   자세 3차     위에 3차 항 [Y³, Y²T, YT², T³] 을 더한다
//   값 2차       값 5개와 그 곱 15개를 모두 넣은 식
//   두 번 보정   1차식으로 재서 로봇을 보정한 뒤, 센터 근처에서 다시 잰다 (식은 그대로)
//
// 두 번 보정의 계산 (실제 로봇을 움직인 것이 아니다)
//   첫 측정의 오차가 e1 이면, 보정 뒤 판은 센터에서 −e1 만큼 떨어진 곳에 있다 (로봇이 명령대로 움직인다고 가정).
//   그 자리는 센터 근처이므로, 둘째 측정의 오차는 같은 시험에서 센터 근처 사진들이 보인 오차와 같다고 본다.
//   센터 근처 = |명령 Y| <= 0.25 mm, |명령 T| <= 0.05 도. 같은 회차의 센터 근처 사진들 중 통과한 비율이 그 사진의 통과 확률.
//
// 시험 (모두 학습에 안 쓴 사진으로 채점. 정답은 tables/labels.tsv)
//   1. 섞음           5묶음 × 10번
//   2. 처음 보는 날   회차 하나를 통째로 빼고 학습
//   3. 범위 밖        안쪽(|Y| <= 1, |T| <= 0.5)으로만 학습, 바깥 시험
//   4. 학습 범위 밖   전부로 학습, 범위(|Y| <= 2, |T| <= 1)를 넘는 사진 시험
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

bool passes(double error_y, double error_t) { return std::abs(error_y) <= kToleranceYmm && std::abs(error_t) <= kToleranceTdeg; }

// ---------- 표 읽기 ----------
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

double to_number(const std::string& text) { return text == "nan" || text.empty() ? std::nan("") : std::stod(text); }

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

// ---------- Huber 가중 최소제곱 (tools/common 의 계수 학습과 같은 식) ----------
using Row = std::vector<double>;

// A x = b 를 푼다 (부분 피벗 소거). b 는 열 2개 (Y, T).
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

// design 의 첫 칸은 1 (절편). 돌려주는 값: 항마다 (Y 계수, T 계수)
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

// ---------- 방법마다의 식 ----------
struct Model {
    Method method = kLinear;
    double mean[kValueCount] = {}, scale[kValueCount] = {};
    std::vector<std::array<double, 2>> first;    // 값으로 (Y, T) 를 내는 식
    std::vector<std::array<double, 2>> second;   // 자세 2차, 3차: (Y, T) 를 고치는 식
};

Row value_terms(const Model& model, const Photo& photo) {
    Row row = {1.0};
    double z[kValueCount];
    for (int k = 0; k < kValueCount; ++k) { z[k] = (photo.values[k] - model.mean[k]) / model.scale[k]; row.push_back(z[k]); }
    if (model.method == kValueQuadratic)
        for (int i = 0; i < kValueCount; ++i) for (int j = i; j < kValueCount; ++j) row.push_back(z[i] * z[j]);
    return row;
}

// 자세 항. 크기를 맞추려고 Y 는 범위 2 mm, T 는 범위 1 도로 나눈다.
Row pose_terms(Method method, const Pose& pose) {
    const double y = pose.y / kRangeYmm, t = pose.t / kRangeTdeg;
    Row row = {1.0, y, t, y * y, y * t, t * t};
    if (method == kPoseCubic) { row.push_back(y * y * y); row.push_back(y * y * t); row.push_back(y * t * t); row.push_back(t * t * t); }
    return row;
}

Pose evaluate_terms(const std::vector<std::array<double, 2>>& coefficients, const Row& row) {
    Pose pose;
    for (size_t c = 0; c < row.size(); ++c) { pose.y += row[c] * coefficients[c][0]; pose.t += row[c] * coefficients[c][1]; }
    return pose;
}

Pose predict(const Model& model, const Photo& photo) {
    Pose pose = evaluate_terms(model.first, value_terms(model, photo));
    if (!model.second.empty()) pose = evaluate_terms(model.second, pose_terms(model.method, pose));
    return pose;
}

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

// ---------- 채점 ----------
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

void print(const std::string& line) { std::cout << line << "\n"; report << line << "\n"; }

std::string fixed(double value, int digits) { std::ostringstream o; o << std::fixed << std::setprecision(digits) << value; return o.str(); }

std::string pad(const std::string& text, int width) {   // 한글은 화면에서 2칸
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

int bin_of(double value, const double* edges) { int b = 0; while (std::abs(value) > edges[b] + 1e-9) ++b; return b; }   // 구간은 명령값으로 나눈다

bool near_center(const Photo& p) { return std::abs(p.command_y) <= kNearCenterYmm && std::abs(p.command_t) <= kNearCenterTdeg; }

// 한 시험, 한 방법의 표를 낸다. pool = 두 번 보정의 둘째 측정 오차로 쓸 센터 근처 사진들 (1차식일 때만 쓴다)
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

// 시험 하나: 나누기(split)들마다 학습하고 시험 사진을 채점해 모은다.
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

}   // namespace

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
