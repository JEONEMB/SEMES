// 값 7개 구성(final00)의 측정·학습·평가 도구.
//
// 사용
//   final00_tool measure <assets> <photos.tsv> <출력 폴더>
//       photos.tsv: "id<TAB>사진 경로" (UTF-8, 머리줄 없음). 출력: features7.tsv. 모델 없이 값 7개만 잰다.
//   final00_tool eval <features7.tsv> <labels.tsv> <splits.tsv> <출력 폴더>
//       labels.tsv: id, source, Y_cmd, T_cmd, Y_abs, T_abs (절대 영점 라벨)
//       splits.tsv: id, final, cv0..cv9, half0..half9 (자세 묶음 단위 분할, 마지막 확인 사진의 cv/half 는 -1)
//       출력: eval.txt, model7.tsv (전체 사진으로 학습한 계수. assets 에 넣는 파일)
//       맨 끝에 no-bars 를 붙이면 바 기울기를 빼고 값 5개로 학습한다(model7.tsv 의 바 두 줄은 가중치 0).
//   final00_tool predict <assets> <photos.tsv> <labels.tsv> <출력 폴더>
//       공개 API(initialize_joint, joint)로 사진마다 (Y, T) 를 내고 라벨과 비교한다. 출력: predictions.tsv, predict.txt
//   final00_tool predict <assets> <photos.tsv> <labels.tsv> <출력 폴더> [ref=사진 ...] [zero=사진 ...]
//       ref=(또는 그냥 경로): set_joint_reference 로 측정 기준을 다시 잡는다(절대 영점은 잇는다).
//       zero=: set_joint_zero 로 영점을 그 사진으로 바꾼다(출력의 (0, 0) 이 그 사진). 둘 다 없으면 절대 영점.
//       predictions.tsv 의 health 는 측정값 일치 점검값(alignment_diagnostics().model_cost). 기준을 넘으면 값은 그대로 내고 message 에 WARNING 이 남는다
//       (predict.txt 의 health_warning = 그런 사진 수). 끝에 health=stop 을 붙이면 set_joint_health_check(true) 로 그런 사진에서 멈춘다.
//   final00_tool health <model7.tsv> <features7.tsv> <labels.tsv> <출력 health.tsv> [leave=회차]
//       측정값 일치 점검식(assets 의 health.tsv)을 라벨 파일에 있는 사진으로 만든다. leave= 를 주면 그 회차를 빼고 만들고 그 회차로 거짓 경보를 잰다.
//       점검식 파일의 SHA-256 을 출력한다(provenance.json 의 health.tsv_sha256 또는 alt_models.<방식>.health_sha256 에 적는다).
//   final00_tool health-check <model7.tsv> <health.tsv> <features7.tsv>
//       다른 사진에서 점검값 분포, 기준 초과 수, 그날 영점 업데이트를 흉내 낸 경우(센터 근처 평균 / 센터 사진 1장)를 보인다.
//   final00_tool range <features7.tsv> <labels.tsv> <splits.tsv> <출력 폴더> [no-bars]
//       학습 범위 밖 예측 시험: 안쪽(|Y| ≤ 1, |T| ≤ 0.5)으로만 학습하고 바깥쪽을 예측한다. range.txt, predictions_range.tsv
//   final00_tool eval 은 탈락 사진을 cv_failures.tsv 에도 남긴다(시험, 반복, 사진, 라벨, 예측, 오차).
//   final00_tool register <assets> <출력 폴더> <기준 사진> [기준 사진 ...]
//       기준 사진(여러 장이면 화소 평균)으로 새 측정 기준 자료를 출력 폴더에 쓰고 찾은 이동을 보인다.
// 인자 경로는 ASCII 로 준다(목록 파일 안의 사진 경로는 한글이어도 된다). 기존 출력은 덮어쓰지 않는다.
#include "final00.hpp"
#include "joint_reference.hpp"
#include "huber_fit.hpp"
#include "sha256.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;

namespace {
std::vector<std::string> split(const std::string& line, char delimiter) {
    std::vector<std::string> out; std::string item; std::istringstream s(line);
    while (std::getline(s, item, delimiter)) out.push_back(item);
    return out;
}
std::vector<std::vector<std::string>> read_table(const fs::path& file, bool header) {
    std::ifstream f(file); if (!f) throw std::runtime_error("파일을 열 수 없다: " + file.u8string());
    std::vector<std::vector<std::string>> rows; std::string line; bool first = true;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (first && line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);   // BOM
        if (first && header) { first = false; continue; }
        first = false;
        if (!line.empty()) rows.push_back(split(line, '\t'));
    }
    return rows;
}
cv::Mat read_image(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return bytes.empty() ? cv::Mat() : cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
}

std::string g_method;   // method= 인자 (QR 방식). 비면 site.json
bool g_health_stop = false;   // health=stop 인자: 측정값 일치 점검에 걸린 사진에서 멈춘다(predict)

int measure(const fs::path& assets, const fs::path& photos, const fs::path& out, bool no_bars = false) {
    if (fs::exists(out / "features7.tsv")) { std::cerr << "출력이 이미 있다\n"; return 3; }
    const auto rows = read_table(photos, false);
    const int n = int(rows.size());
    final00::Measurer measurer(assets.u8string(), g_method);
    if (no_bars) measurer.set_use_bars(false);   // 바를 재지 않는다(바 기울기 값 0). 바가 밀려 측정이 실패하는 사진용
    std::cout << "qr_method " << measurer.qr_method() << "\n";
    std::vector<final00::Measurement> m(n); std::vector<int> ok(n, 0); std::vector<std::string> error(n);
    cv::setNumThreads(1);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        try {
            const cv::Mat gray = read_image(fs::u8path(rows[i].at(1)));
            if (gray.empty()) throw std::runtime_error("사진을 읽지 못했다");
            m[i] = measurer.measure(gray); ok[i] = 1;
        } catch (const std::exception& e) { error[i] = e.what(); }
    }
    fs::create_directories(out);
    std::ofstream f(out / "features7.tsv", std::ios::binary); f << std::setprecision(17) << "id";
    for (const char* name : final00::kValueNames) f << '\t' << name;
    f << "\tvalid536\tqr_correlation\tline_correlation\tmeasured\terror\n";
    int good = 0;
    for (int i = 0; i < n; ++i) {
        f << rows[i][0];
        for (double v : m[i].values) f << '\t' << (ok[i] ? v : std::numeric_limits<double>::quiet_NaN());
        f << '\t' << m[i].valid_points << '\t' << m[i].qr_correlation << '\t' << m[i].line_correlation << '\t' << ok[i] << '\t' << error[i] << '\n'; good += ok[i];
    }
    std::cout << "measured " << good << "/" << n << "\n";
    return good == n ? 0 : 4;
}

struct Data { std::vector<int> id; std::vector<std::string> source; cv::Mat x, y; std::vector<int> final; std::vector<std::vector<int>> cv, half; };

Data load(const fs::path& features, const fs::path& labels, const fs::path& splits) {
    std::map<int, std::vector<std::string>> lab, sp;
    for (auto& r : read_table(labels, true)) lab[std::stoi(r.at(0))] = r;
    for (auto& r : read_table(splits, true)) sp[std::stoi(r.at(0))] = r;
    Data d; std::vector<std::array<double, 7>> x; std::vector<std::array<double, 2>> y;
    for (auto& r : read_table(features, true)) {
        const int id = std::stoi(r.at(0));
        if (r.at(11) != "1") throw std::runtime_error("측정 실패 사진이 있다: id " + r[0]);
        const auto& l = lab.at(id); const auto& s = sp.at(id);
        d.id.push_back(id); d.source.push_back(l.at(1));
        x.push_back({std::stod(r[1]), std::stod(r[2]), std::stod(r[3]), std::stod(r[4]), std::stod(r[5]), std::stod(r[6]), std::stod(r[7])});
        y.push_back({std::stod(l.at(4)), std::stod(l.at(5))});
        d.final.push_back(std::stoi(s.at(1)));
        std::vector<int> c, h; for (int k = 0; k < 10; ++k) { c.push_back(std::stoi(s.at(2 + k))); h.push_back(std::stoi(s.at(12 + k))); }
        d.cv.push_back(c); d.half.push_back(h);
    }
    const int n = int(d.id.size());
    d.x = cv::Mat(n, 7, CV_64F); d.y = cv::Mat(n, 2, CV_64F);
    for (int i = 0; i < n; ++i) { for (int j = 0; j < 7; ++j) d.x.at<double>(i, j) = x[i][j]; d.y.at<double>(i, 0) = y[i][0]; d.y.at<double>(i, 1) = y[i][1]; }
    return d;
}

cv::Mat take(const cv::Mat& m, const std::vector<int>& index) {
    cv::Mat out(int(index.size()), m.cols, CV_64F);
    for (int i = 0; i < int(index.size()); ++i) m.row(index[i]).copyTo(out.row(i));
    return out;
}
struct Score { int fail = 0, count = 0; double max_y = 0, max_t = 0, sum_y2 = 0, sum_t2 = 0; };
// 탈락 사진 기록 (2026-11-14): 어느 시험, 몇 번째 반복에서 어느 사진이 얼마나 틀렸는지
struct Miss { std::string test; int repeat; int row; double py, pt; };
// train 으로 학습해 test 를 채점한다. misses 가 있으면 탈락 사진을 남긴다.
void score(const Data& d, const std::vector<int>& train, const std::vector<int>& test, Score& s,
           std::vector<Miss>* misses = nullptr, const std::string& name = "", int repeat = 0) {
    final00::LinearModel model; model.fit(take(d.x, train), take(d.y, train));
    const cv::Mat p = model.predict(take(d.x, test));
    for (int i = 0; i < int(test.size()); ++i) {
        const double ey = p.at<double>(i, 0) - d.y.at<double>(test[i], 0), et = p.at<double>(i, 1) - d.y.at<double>(test[i], 1);
        const bool ok = final00::passes(ey, et);
        s.fail += !ok; ++s.count; s.max_y = std::max(s.max_y, std::abs(ey)); s.max_t = std::max(s.max_t, std::abs(et)); s.sum_y2 += ey * ey; s.sum_t2 += et * et;
        if (!ok && misses) misses->push_back({name, repeat, test[i], p.at<double>(i, 0), p.at<double>(i, 1)});
    }
}

int evaluate(const fs::path& features, const fs::path& labels, const fs::path& splits, const fs::path& out, bool no_bars = false) {
    if (fs::exists(out / "eval.txt")) { std::cerr << "출력이 이미 있다\n"; return 3; }
    Data d = load(features, labels, splits); const int n = int(d.id.size());
    // no_bars: 바 기울기 2개를 빼고 값 5개(QR 1 + 나사 4)로 학습·평가한다. model7.tsv 에는 바 두 줄을 가중치 0 으로 쓴다(측정기가 바를 재지 않는다).
    const std::vector<int> used = no_bars ? std::vector<int>{0, 3, 4, 5, 6} : std::vector<int>{0, 1, 2, 3, 4, 5, 6};
    if (no_bars) { cv::Mat x(n, int(used.size()), CV_64F); for (int j = 0; j < int(used.size()); ++j) d.x.col(used[j]).copyTo(x.col(j)); d.x = x; }
    std::vector<int> dev, fin;
    for (int i = 0; i < n; ++i) (d.final[i] ? fin : dev).push_back(i);
    std::ostringstream o; o << std::fixed;
    o << "photos " << n << " dev " << dev.size() << " final " << fin.size() << "\n";
    // 주 검증: 개발 자료 섞은 80/20, 10회
    std::vector<int> fails; double worst_y = 0, worst_t = 0, sy = 0, st = 0;
    std::vector<Miss> misses;   // 탈락 사진 (cv_failures.tsv)
    for (int k = 0; k < 10; ++k) {
        Score s;
        for (int fold = 0; fold < 5; ++fold) {
            std::vector<int> train, test;
            for (int i : dev) (d.cv[i][k] == fold ? test : train).push_back(i);
            score(d, train, test, s, &misses, "main_cv", k);
        }
        fails.push_back(s.fail); worst_y = std::max(worst_y, s.max_y); worst_t = std::max(worst_t, s.max_t);
        sy += std::sqrt(s.sum_y2 / s.count); st += std::sqrt(s.sum_t2 / s.count);
    }
    double mean = 0; for (int v : fails) mean += v / 10.0;
    o << "main_cv_fails"; for (int v : fails) o << ' ' << v;
    o << std::setprecision(2) << " mean " << mean << " worst " << *std::max_element(fails.begin(), fails.end())
      << std::setprecision(5) << " rmsY " << sy / 10 << " rmsT " << st / 10 << " max|eY| " << worst_y << " max|eT| " << worst_t << "\n";
    // 보조: 세션별 절반 50/50, 10회
    fails.clear();
    for (int k = 0; k < 10; ++k) {
        Score s;
        for (int part = 0; part < 2; ++part) {
            std::vector<int> train, test;
            for (int i : dev) (d.half[i][k] == part ? test : train).push_back(i);
            score(d, train, test, s, &misses, "half", k);
        }
        fails.push_back(s.fail);
    }
    mean = 0; for (int v : fails) mean += v / 10.0;
    o << "half_fails"; for (int v : fails) o << ' ' << v;
    o << std::setprecision(2) << " mean " << mean << "\n";
    // 참고: 세션 통째로 빼기
    std::set<std::string> sources(d.source.begin(), d.source.end()); int pass = 0;
    o << "leave_one_session";
    for (const auto& name : sources) {
        std::vector<int> train, test;
        for (int i : dev) (d.source[i] == name ? test : train).push_back(i);
        Score s; score(d, train, test, s, &misses, "leave_" + name, 0); pass += s.count - s.fail;
        o << ' ' << name << ' ' << std::setprecision(1) << 100.0 * (s.count - s.fail) / s.count << '%';
    }
    o << " all " << std::setprecision(1) << 100.0 * pass / dev.size() << "%\n";
    // 마지막 확인
    { Score s; score(d, dev, fin, s, &misses, "final_check", 0);
      o << "final_check fails " << s.fail << "/" << s.count << std::setprecision(5) << " max|eY| " << s.max_y << " max|eT| " << s.max_t << "\n"; }
    // 전체 사진으로 학습한 모델
    final00::LinearModel model; model.fit(d.x, d.y);
    double min_y = 1e9, max_y = -1e9, min_t = 1e9, max_t = -1e9;
    { int fail = 0; const cv::Mat p = model.predict(d.x);
      for (int i = 0; i < n; ++i) {
          fail += !final00::passes(p.at<double>(i, 0) - d.y.at<double>(i, 0), p.at<double>(i, 1) - d.y.at<double>(i, 1));
          min_y = std::min(min_y, d.y.at<double>(i, 0)); max_y = std::max(max_y, d.y.at<double>(i, 0)); min_t = std::min(min_t, d.y.at<double>(i, 1)); max_t = std::max(max_t, d.y.at<double>(i, 1));
      }
      o << "all_photos_resubstitution fails " << fail << "/" << n << "\n"
        << std::setprecision(6) << "label_range Y " << min_y << " " << max_y << " T " << min_t << " " << max_t << "\n"; }
    fs::create_directories(out);
    { std::ofstream f(out / "eval.txt", std::ios::binary); f << o.str(); }
    // 탈락 사진: 시험, 반복 번호, 사진 id, 회차, 라벨(절대 영점), 예측, 오차, 어느 쪽이 기준을 넘었는지
    { std::ofstream f(out / "cv_failures.tsv", std::ios::binary); f << std::setprecision(6) << std::fixed;
      f << "test\trepeat\tid\tsource\tY_abs\tT_abs\tY_pred\tT_pred\teY\teT\tover\n";
      for (const auto& m : misses) {
          const double ly = d.y.at<double>(m.row, 0), lt = d.y.at<double>(m.row, 1), ey = m.py - ly, et = m.pt - lt;
          f << m.test << '\t' << m.repeat << '\t' << d.id[m.row] << '\t' << d.source[m.row] << '\t' << ly << '\t' << lt << '\t' << m.py << '\t' << m.pt << '\t' << ey << '\t' << et << '\t'
            << (std::abs(ey) > 0.045 ? (std::abs(et) > 0.01 ? "Y,T" : "Y") : "T") << '\n';
      } }
    std::ofstream m(out / "model7.tsv", std::ios::binary); m << std::setprecision(17);
    m << "name\tmean\tscale\tweight_Y\tweight_T\n" << "intercept\t0\t1\t" << model.weights.at<double>(0, 0) << '\t' << model.weights.at<double>(0, 1) << '\n';
    for (int j = 0, u = 0; j < 7; ++j) {
        if (u < int(used.size()) && used[u] == j) { m << final00::kValueNames[j] << '\t' << model.mean.at<double>(u) << '\t' << model.scale.at<double>(u) << '\t' << model.weights.at<double>(u + 1, 0) << '\t' << model.weights.at<double>(u + 1, 1) << '\n'; ++u; }
        else m << final00::kValueNames[j] << "\t0\t1\t0\t0\n";   // 쓰지 않는 값(가중치 0)
    }
    std::cout << o.str();
    return 0;
}

// ---- 학습 범위 밖 예측 시험 (2026-11-14, 실험 20261114_0130) ----
// 안쪽(|Y| ≤ 1, |T| ≤ 0.5)으로만 학습하고 바깥쪽을 예측한다. 성현 님 평가처럼 범위가 넓을 때 오차가 어떻게 커지는지 흉내 낸다.
int range_test(const fs::path& features, const fs::path& labels, const fs::path& splits, const fs::path& out, bool no_bars) {
    if (fs::exists(out / "range.txt")) { std::cerr << "출력이 이미 있다\n"; return 3; }
    Data d = load(features, labels, splits); const int n = int(d.id.size());
    const std::vector<int> used = no_bars ? std::vector<int>{0, 3, 4, 5, 6} : std::vector<int>{0, 1, 2, 3, 4, 5, 6};
    if (no_bars) { cv::Mat x(n, int(used.size()), CV_64F); for (int j = 0; j < int(used.size()); ++j) d.x.col(used[j]).copyTo(x.col(j)); d.x = x; }
    std::map<int, double> qr_corr;   // 값 파일의 QR 상관(가장 낮은 QR)
    for (auto& r : read_table(features, true)) if (r.at(11) == "1") qr_corr[std::stoi(r[0])] = std::stod(r.at(9));
    auto Y = [&](int i) { return d.y.at<double>(i, 0); }; auto T = [&](int i) { return d.y.at<double>(i, 1); };
    std::vector<int> inner, outer;
    for (int i = 0; i < n; ++i) (std::abs(Y(i)) <= 1.0 && std::abs(T(i)) <= 0.5 ? inner : outer).push_back(i);
    final00::LinearModel model; model.fit(take(d.x, inner), take(d.y, inner));
    const cv::Mat p = model.predict(take(d.x, outer));
    // 비교: 안쪽 사진끼리 5-fold (고정 시드)
    std::vector<double> in_py(n), in_pt(n);
    { cv::RNG rng(20261114); std::vector<int> fold(inner.size()); for (size_t k = 0; k < fold.size(); ++k) fold[k] = int(k % 5);
      for (size_t k = fold.size(); k > 1; --k) std::swap(fold[k - 1], fold[size_t(rng.uniform(0, int(k)))]);
      for (int f = 0; f < 5; ++f) {
          std::vector<int> tr, te; for (size_t k = 0; k < inner.size(); ++k) (fold[k] == f ? te : tr).push_back(inner[k]);
          final00::LinearModel m; m.fit(take(d.x, tr), take(d.y, tr)); const cv::Mat q = m.predict(take(d.x, te));
          for (size_t k = 0; k < te.size(); ++k) { in_py[te[k]] = q.at<double>(int(k), 0); in_pt[te[k]] = q.at<double>(int(k), 1); }
      } }
    struct Acc { int n = 0, fail = 0; double sy = 0, st = 0, my = 0, mt = 0; void add(double ey, double et) { ++n; fail += !final00::passes(ey, et); sy += ey * ey; st += et * et; my = std::max(my, std::abs(ey)); mt = std::max(mt, std::abs(et)); } };
    auto line = [](const std::string& name, const Acc& a) {
        std::ostringstream o; o << std::fixed << std::setprecision(5) << name << " n " << a.n << " fails " << a.fail << " rmsY " << (a.n ? std::sqrt(a.sy / a.n) : 0) << " rmsT " << (a.n ? std::sqrt(a.st / a.n) : 0)
                               << " max|eY| " << a.my << " max|eT| " << a.mt << "\n"; return o.str(); };
    Acc in_acc, out_acc, bins[3];
    for (int i : inner) in_acc.add(in_py[i] - Y(i), in_pt[i] - T(i));
    // 비례 오차: 바깥쪽에서 eY = a + b·Y, eT = a + b·T 직선 기울기
    double sY = 0, sY2 = 0, seY = 0, sYeY = 0, sT = 0, sT2 = 0, seT = 0, sTeT = 0;
    fs::create_directories(out);
    std::ofstream f(out / "predictions_range.tsv", std::ios::binary); f << std::fixed << std::setprecision(5);
    f << "id\tsource\tregion\tY_given\tT_given\tY_pred\teY\tT_pred\teT\tpass\tqr_correlation\n";
    for (int i : inner) f << d.id[i] << '\t' << d.source[i] << "\tinner_5fold\t" << Y(i) << '\t' << T(i) << '\t' << in_py[i] << '\t' << in_py[i] - Y(i) << '\t' << in_pt[i] << '\t' << in_pt[i] - T(i) << '\t' << final00::passes(in_py[i] - Y(i), in_pt[i] - T(i)) << '\t' << qr_corr[d.id[i]] << '\n';
    for (int k = 0; k < int(outer.size()); ++k) {
        const int i = outer[k]; const double py = p.at<double>(k, 0), pt = p.at<double>(k, 1), ey = py - Y(i), et = pt - T(i);
        out_acc.add(ey, et);
        const double dist = std::sqrt(Y(i) * Y(i) + 4 * T(i) * T(i));   // 이동량 (T 1° ≈ 2 mm 로 환산)
        bins[dist < 1.5 ? 0 : dist < 2.0 ? 1 : 2].add(ey, et);
        sY += Y(i); sY2 += Y(i) * Y(i); seY += ey; sYeY += Y(i) * ey; sT += T(i); sT2 += T(i) * T(i); seT += et; sTeT += T(i) * et;
        f << d.id[i] << '\t' << d.source[i] << "\touter\t" << Y(i) << '\t' << T(i) << '\t' << py << '\t' << ey << '\t' << pt << '\t' << et << '\t' << final00::passes(ey, et) << '\t' << qr_corr[d.id[i]] << '\n';
    }
    const double m = double(outer.size());
    const double slopeY = (m * sYeY - sY * seY) / (m * sY2 - sY * sY), slopeT = (m * sTeT - sT * seT) / (m * sT2 - sT * sT);
    // QR 상관이 |T| 에 따라 떨어지는 정도 (모든 사진): 직선 맞춤 후 T 2° 에서의 값
    double c0 = 0, c1 = 0, c2 = 0, cc = 0, ct = 0; int cn = 0; double cmin[4] = {1, 1, 1, 1};
    for (int i = 0; i < n; ++i) { const double a = std::abs(T(i)), c = qr_corr[d.id[i]]; ++cn; c0 += a; c1 += a * a; cc += c; ct += a * c; const int b = std::min(3, int(a / 0.25)); cmin[b] = std::min(cmin[b], c); }
    const double cslope = (cn * ct - c0 * cc) / (cn * c1 - c0 * c0), cint = (cc - cslope * c0) / cn;
    std::ostringstream o; o << std::fixed << std::setprecision(5);
    o << "train inner |Y|<=1 |T|<=0.5 : " << inner.size() << " photos, test outer : " << outer.size() << " photos" << (no_bars ? " (no bars, 5 values)" : " (7 values)") << "\n";
    o << line("inner_5fold", in_acc) << line("outer", out_acc);
    o << line("outer dist<1.5", bins[0]) << line("outer dist1.5-2", bins[1]) << line("outer dist>=2", bins[2]);
    o << "outer proportional error: slope eY/Y " << slopeY << " (" << std::setprecision(2) << 100 * slopeY << "%), slope eT/T " << std::setprecision(5) << slopeT << " (" << std::setprecision(2) << 100 * slopeT << "%)\n";
    o << std::setprecision(4) << "qr_correlation min by |T| bin [0,.25) [.25,.5) [.5,.75) [.75,1]: " << cmin[0] << ' ' << cmin[1] << ' ' << cmin[2] << ' ' << cmin[3]
      << " | linear fit corr = " << cint << " + " << cslope << "·|T| -> at |T|=2: " << cint + 2 * cslope << "\n";
    { std::ofstream t(out / "range.txt", std::ios::binary); t << o.str(); }
    std::cout << o.str();
    return 0;
}

std::string report_text(const final00::ReferenceReport& r) {
    std::ostringstream o;
    o << "reference";
    for (int k = 0; k < 3; ++k) o << " QR" << k + 1 << " (" << r.qr_shift[k].x << ',' << r.qr_shift[k].y << ") ncc " << std::setprecision(4) << r.qr_ncc[k] << " ecc " << r.qr_ecc[k];
    for (int k = 0; k < 4; ++k) o << " screw" << k + 1 << " (" << r.screw_shift[k].x << ',' << r.screw_shift[k].y << ")";
    o << " bar_upper " << r.bar_shift_min[0] << '~' << r.bar_shift_max[0] << " bar_lower " << r.bar_shift_min[1] << '~' << r.bar_shift_max[1]
      << " absolute_kept " << r.absolute_kept << " bridge " << std::setprecision(6) << r.bridge_y << ' ' << r.bridge_t << " folder " << r.folder << '\n';
    return o.str();
}

std::vector<cv::Mat> read_images(char** first, char** last) {
    std::vector<cv::Mat> v;
    for (char** a = first; a != last; ++a) { cv::Mat m = read_image(fs::u8path(*a)); if (m.empty()) throw std::runtime_error(std::string("기준 사진을 읽지 못했다: ") + *a); v.push_back(m); }
    return v;
}
// predict 의 덧붙인 인자: "zero=경로" 는 영점 업데이트용, "ref=경로" 또는 그냥 경로는 측정 기준 다시 잡기용
void split_extra(char** first, char** last, std::vector<cv::Mat>& reference, std::vector<cv::Mat>& zero) {
    for (char** a = first; a != last; ++a) {
        std::string s(*a); std::vector<cv::Mat>* to = &reference;
        if (s.rfind("zero=", 0) == 0) { s = s.substr(5); to = &zero; } else if (s.rfind("ref=", 0) == 0) s = s.substr(4);
        cv::Mat m = read_image(fs::u8path(s)); if (m.empty()) throw std::runtime_error("사진을 읽지 못했다: " + s); to->push_back(m);
    }
}

int register_reference(const fs::path& assets, const fs::path& out, const std::vector<cv::Mat>& images) {
    if (fs::exists(out / "site.json")) { std::cerr << "출력이 이미 있다\n"; return 3; }
    cv::Mat sum = cv::Mat::zeros(2160, 3840, CV_64F);
    for (const auto& m : images) { cv::Mat d; m.convertTo(d, CV_64F); sum += d; }
    cv::Mat mean8; sum.convertTo(mean8, CV_8U, 1.0 / double(images.size()));
    const auto r = final00::make_reference(assets, mean8, out);
    std::cout << report_text(r);
    return 0;
}

int predict(const fs::path& assets, const fs::path& photos, const fs::path& labels, const fs::path& out, const std::vector<cv::Mat>& reference = {}, const std::vector<cv::Mat>& zero = {}) {
    if (fs::exists(out / "predictions.tsv")) { std::cerr << "출력이 이미 있다\n"; return 3; }
    std::map<int, std::vector<std::string>> lab;
    for (auto& r : read_table(labels, true)) lab[std::stoi(r.at(0))] = r;
    const auto rows = read_table(photos, false);
    if (g_method.empty()) initialize_joint(assets.u8string()); else initialize_joint_method(assets.u8string(), g_method);
    set_joint_health_check(g_health_stop);
    std::string ref_text = "qr_method " + joint_qr_method() + " model " + joint_model_id() + (g_health_stop ? " health=stop" : " health=warn") + "\n";
    if (!reference.empty()) ref_text = report_text(set_joint_reference(reference));
    if (!zero.empty()) { const Alignment_t z = set_joint_zero(zero); std::ostringstream s; s << std::setprecision(9) << "zero " << z.y << ' ' << z.t << " (절대 영점 기준)\n"; ref_text += s.str(); }
    fs::create_directories(out);
    std::ofstream f(out / "predictions.tsv", std::ios::binary); f << std::setprecision(17) << "id\tY\tT\tY_abs\tT_abs\tpass\tvalid\toutside_training_range\thealth\tmessage\n";
    int valid = 0, pass = 0, warned = 0; double max_y = 0, max_t = 0;
    cv::setNumThreads(1);
    for (const auto& r : rows) {   // joint() 는 내부 잠금으로 한 번에 한 장만 처리한다
        const int id = std::stoi(r.at(0)); const auto& l = lab.at(id); const double ly = std::stod(l.at(4)), lt = std::stod(l.at(5));
        cv::Mat image = read_image(fs::u8path(r.at(1)));
        try {
            const Alignment_t p = joint(image); const auto d = alignment_diagnostics();
            const bool okay = final00::passes(p.y - ly, p.t - lt); ++valid; pass += okay;
            max_y = std::max(max_y, std::abs(p.y - ly)); max_t = std::max(max_t, std::abs(p.t - lt));
            f << id << '\t' << p.y << '\t' << p.t << '\t' << ly << '\t' << lt << '\t' << okay << "\t1\t" << d.outside_training_range << '\t' << d.model_cost << '\t' << (d.normalized_cost_margin > 1 ? d.message.substr(0, d.message.find(" | ")) : std::string()) << '\n';
            warned += d.normalized_cost_margin > 1;
        } catch (const std::exception& e) {
            f << id << "\tnan\tnan\t" << ly << '\t' << lt << "\t0\t0\t0\t" << alignment_diagnostics().model_cost << '\t' << e.what() << '\n';
        }
    }
    std::ostringstream o; o << "joint_api photos " << rows.size() << " valid " << valid << " pass " << pass << " health_warning " << warned << std::fixed << std::setprecision(5) << " max|eY| " << max_y << " max|eT| " << max_t << "\n";
    o << ref_text;
    { std::ofstream t(out / "predict.txt", std::ios::binary); t << o.str(); }
    std::cout << o.str();
    return 0;
}

// ---- 측정값 일치 점검식 (2026-11-13, 실험 20261113_2230) ----
// 값마다 "모델 출력 자세 → 값" 2차식(1, Y, T, Y², YT, T²)을 학습 사진으로 맞추고, 정상 흔들림 = 1.4826 × 잔차 MAD,
// 점검값 H = max |잔차| / 흔들림, 기준값 = 1.5 × 학습 사진 H 의 99.9% 분위(계획.md 에 미리 정한 방법).
struct LinearPose {   // model7.tsv
    std::array<double, 7> mean{}, scale{}; std::array<std::array<double, 2>, 8> w{};
    std::vector<int> used;   // 가중치가 0 이 아닌 값
    std::array<double, 2> operator()(const std::array<double, 7>& x) const {
        double y = w[0][0], t = w[0][1];
        for (int j = 0; j < 7; ++j) { const double z = (x[j] - mean[j]) / scale[j]; y += w[j + 1][0] * z; t += w[j + 1][1] * z; }
        return {y, t};
    }
};
LinearPose read_model7(const fs::path& file) {
    LinearPose m; const auto rows = read_table(file, true);
    if (rows.size() != 8) throw std::runtime_error("model7.tsv 의 행이 8개가 아니다: " + file.u8string());
    for (int r = 0; r < 8; ++r) {
        if (rows[r].at(0) != (r == 0 ? std::string("intercept") : std::string(final00::kValueNames[r - 1]))) throw std::runtime_error("model7.tsv 의 값 순서가 다르다");
        m.w[r] = {std::stod(rows[r].at(3)), std::stod(rows[r].at(4))};
        if (r > 0) { m.mean[r - 1] = std::stod(rows[r].at(1)); m.scale[r - 1] = std::stod(rows[r].at(2)); if (m.w[r][0] != 0 || m.w[r][1] != 0) m.used.push_back(r - 1); }
    }
    return m;
}
struct HealthModel { double threshold = 0; std::vector<int> value; std::vector<std::array<double, 6>> c; std::vector<double> scale; };
HealthModel read_health(const fs::path& file) {
    HealthModel h; std::ifstream f(file); if (!f) throw std::runtime_error("파일을 열 수 없다: " + file.u8string());
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const auto r = split(line, '\t');
        if (r[0] == "threshold") { h.threshold = std::stod(r.at(1)); continue; }
        if (r[0] == "name") continue;
        int v = -1; for (int j = 0; j < 7; ++j) if (r[0] == final00::kValueNames[j]) v = j;
        if (v < 0 || r.size() != 8) throw std::runtime_error("점검식을 읽지 못했다: " + line);
        std::array<double, 6> c; for (int k = 0; k < 6; ++k) c[k] = std::stod(r[1 + k]);
        h.value.push_back(v); h.c.push_back(c); h.scale.push_back(std::stod(r[7]));
    }
    return h;
}
std::array<double, 6> terms(const std::array<double, 2>& p) { return {1, p[0], p[1], p[0] * p[0], p[0] * p[1], p[1] * p[1]}; }
double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end()); const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double quantile_of(std::vector<double> v, double q) {   // 선형 보간(numpy 기본과 같은 정의)
    std::sort(v.begin(), v.end()); const double pos = q * (v.size() - 1); const size_t lo = size_t(std::floor(pos));
    return lo + 1 < v.size() ? v[lo] + (pos - lo) * (v[lo + 1] - v[lo]) : v[lo];
}
struct FeatureRows { std::vector<int> id; std::vector<std::array<double, 7>> x; };
FeatureRows read_feature_rows(const fs::path& file, const std::set<int>* only) {
    FeatureRows f;
    for (auto& r : read_table(file, true)) {
        if (r.at(11) != "1") continue;   // 측정 실패 사진 제외
        const int id = std::stoi(r[0]); if (only && !only->count(id)) continue;
        std::array<double, 7> x; for (int j = 0; j < 7; ++j) x[j] = std::stod(r[1 + j]);
        f.id.push_back(id); f.x.push_back(x);
    }
    return f;
}
// 잔차(점검에 쓰는 값 순서)
std::vector<std::vector<double>> residuals(const LinearPose& m, const HealthModel& h, const FeatureRows& f, std::vector<std::array<double, 2>>* pose = nullptr) {
    std::vector<std::vector<double>> r(f.x.size(), std::vector<double>(h.value.size()));
    for (size_t i = 0; i < f.x.size(); ++i) {
        const auto p = m(f.x[i]); const auto a = terms(p); if (pose) pose->push_back(p);
        for (size_t k = 0; k < h.value.size(); ++k) { double v = 0; for (int q = 0; q < 6; ++q) v += h.c[k][q] * a[q]; r[i][k] = f.x[i][h.value[k]] - v; }
    }
    return r;
}
std::vector<double> scores(const HealthModel& h, const std::vector<std::vector<double>>& r, const std::vector<double>& offset, std::vector<int>* worst = nullptr) {
    std::vector<double> s(r.size(), 0);
    for (size_t i = 0; i < r.size(); ++i) {
        int w = -1;
        for (size_t k = 0; k < h.value.size(); ++k) { const double z = std::abs(r[i][k] - offset[k]) / h.scale[k]; if (z > s[i]) { s[i] = z; w = h.value[k]; } }
        if (worst) worst->push_back(w);
    }
    return s;
}

HealthModel fit_health(const LinearPose& m, const FeatureRows& f) {
    HealthModel h; const int n = int(f.x.size());
    cv::Mat a(n, 6, CV_64F);
    for (int i = 0; i < n; ++i) { const auto t = terms(m(f.x[i])); for (int q = 0; q < 6; ++q) a.at<double>(i, q) = t[q]; }
    for (int v : m.used) {
        cv::Mat b(n, 1, CV_64F), c; for (int i = 0; i < n; ++i) b.at<double>(i) = f.x[i][v];
        cv::solve(a, b, c, cv::DECOMP_SVD);
        std::array<double, 6> cc; for (int q = 0; q < 6; ++q) cc[q] = c.at<double>(q);
        std::vector<double> r(n); for (int i = 0; i < n; ++i) { double p = 0; for (int q = 0; q < 6; ++q) p += cc[q] * a.at<double>(i, q); r[i] = f.x[i][v] - p; }
        const double med = median_of(r); for (double& e : r) e = std::abs(e - med);
        h.value.push_back(v); h.c.push_back(cc); h.scale.push_back(1.4826 * median_of(r));
    }
    const auto s = scores(h, residuals(m, h, f), std::vector<double>(h.value.size(), 0));
    h.threshold = 1.5 * quantile_of(s, 0.999);
    return h;
}
void write_health(const HealthModel& h, const fs::path& out, const std::string& note, int photos) {
    std::ofstream o(out, std::ios::binary); o << std::setprecision(17);
    o << "# 측정값 일치 점검식. " << note << ". 학습 사진 " << photos << "장. 항 1 Y T Y^2 YT T^2 (Y, T = 모델 출력, 측정 기준 보정·영점 빼기 전)\n";
    o << "threshold\t" << h.threshold << "\nname\tc1\tcY\tcT\tcYY\tcYT\tcTT\tscale\n";
    for (size_t k = 0; k < h.value.size(); ++k) { o << final00::kValueNames[h.value[k]]; for (double c : h.c[k]) o << '\t' << c; o << '\t' << h.scale[k] << '\n'; }
}
// 점검값 분포와 기준 초과 수. 그날 영점 업데이트를 흉내 낸 두 경우도 보인다:
//   센터 근처(|Y|, |T| ≤ 0.1) 사진 잔차 평균을 0 으로 / 센터에 가장 가까운 사진 1장의 잔차를 0 으로
std::string check_text(const LinearPose& m, const HealthModel& h, const FeatureRows& f) {
    std::vector<std::array<double, 2>> pose; const auto r = residuals(m, h, f, &pose);
    std::vector<int> worst; const auto raw = scores(h, r, std::vector<double>(h.value.size(), 0), &worst);
    std::vector<double> near(h.value.size(), 0); int nn = 0; size_t centre = 0; double best = 1e18;
    for (size_t i = 0; i < r.size(); ++i) {
        if (std::abs(pose[i][0]) <= 0.1 && std::abs(pose[i][1]) <= 0.1) { ++nn; for (size_t k = 0; k < near.size(); ++k) near[k] += r[i][k]; }
        const double d = std::hypot(pose[i][0], 10 * pose[i][1]); if (d < best) { best = d; centre = i; }
    }
    for (double& v : near) v /= std::max(nn, 1);
    auto line = [&](const char* name, const std::vector<double>& s) {
        std::ostringstream o; int over = 0; for (double v : s) over += v > h.threshold;
        o << std::fixed << std::setprecision(2) << "  " << name << ": H median " << median_of(s) << " p99 " << quantile_of(s, .99) << " max " << *std::max_element(s.begin(), s.end())
          << " over " << over << "/" << s.size() << "\n";
        return o.str();
    };
    std::ostringstream o; o << std::fixed << std::setprecision(2) << "photos " << r.size() << " threshold " << h.threshold << "\n";
    o << line("no recentre", raw);
    std::map<int, int> count; for (size_t i = 0; i < raw.size(); ++i) if (raw[i] > h.threshold) ++count[worst[i]];
    if (!count.empty()) { o << "    worst value of over:"; for (auto& c : count) o << ' ' << final00::kValueNames[c.first] << ' ' << c.second; o << "\n"; }
    if (nn) o << line(("recentre near-centre mean (" + std::to_string(nn) + " photos)").c_str(), scores(h, r, near));
    o << line(("recentre on photo id " + std::to_string(f.id[centre])).c_str(), scores(h, r, r[centre]));
    return o.str();
}

std::set<int> label_ids(const fs::path& labels, const std::string& keep_source, const std::string& drop_source) {
    std::set<int> ids;
    for (auto& r : read_table(labels, true)) {
        if (!keep_source.empty() && r.at(1) != keep_source) continue;
        if (!drop_source.empty() && r.at(1) == drop_source) continue;
        ids.insert(std::stoi(r.at(0)));
    }
    return ids;
}
// health MODEL7 FEATURES LABELS OUT [leave=SOURCE]: 라벨 파일의 사진(leave 회차 빼고)으로 점검식을 만든다. leave 가 있으면 뺀 회차로 거짓 경보를 잰다.
int make_health(const fs::path& model, const fs::path& features, const fs::path& labels, const fs::path& out, const std::string& leave) {
    if (fs::exists(out)) { std::cerr << "출력이 이미 있다\n"; return 3; }
    const LinearPose m = read_model7(model);
    const std::set<int> train = label_ids(labels, "", leave);
    const FeatureRows f = read_feature_rows(features, &train);
    const HealthModel h = fit_health(m, f);
    write_health(h, out, "model " + model.filename().u8string() + (leave.empty() ? "" : ", leave " + leave), int(f.x.size()));
    std::cout << "photos " << f.x.size() << " used"; for (int v : h.value) std::cout << ' ' << final00::kValueNames[v];
    std::cout << "\nscale px" << std::setprecision(5); for (size_t k = 0; k < h.value.size(); ++k) std::cout << ' ' << h.scale[k];
    std::cout << "\nfit photos (resubstitution)\n" << check_text(m, h, f);
    if (!leave.empty()) {
        const std::set<int> test = label_ids(labels, leave, "");
        std::cout << "left-out session " << leave << "\n" << check_text(m, h, read_feature_rows(features, &test));
    }
    { std::ifstream b(out, std::ios::binary); const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(b)), std::istreambuf_iterator<char>());
      std::cout << "sha256 " << final00_sha::sha256_hex(bytes) << "\n"; }
    return 0;
}
// health-check MODEL7 HEALTH FEATURES: 다른 사진(측정 성공 전부)에서 점검값을 잰다
int check_health(const fs::path& model, const fs::path& health, const fs::path& features) {
    const LinearPose m = read_model7(model);
    std::cout << check_text(m, read_health(health), read_feature_rows(features, nullptr));
    return 0;
}
}

int main(int raw_argc, char** raw_argv) {
    try {
        // 어디에든 method=track536 또는 method=ncc 를 붙이면 QR 방식을 고른다(measure, predict). 없으면 assets/site.json 의 qr_method.
        std::vector<char*> kept;
        for (int i = 0; i < raw_argc; ++i) {
            const std::string s(raw_argv[i]);
            if (s.rfind("method=", 0) == 0) g_method = s.substr(7);
            else if (s == "health=stop") g_health_stop = true;
            else kept.push_back(raw_argv[i]);
        }
        const int argc = int(kept.size()); char** argv = kept.data();
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "measure" && argc == 5) return measure(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]));
        if (mode == "measure" && argc == 6 && std::string(argv[5]) == "no-bars") return measure(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), true);
        if (mode == "eval" && argc == 6) return evaluate(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), fs::u8path(argv[5]));
        if (mode == "eval" && argc == 7 && std::string(argv[6]) == "no-bars") return evaluate(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), fs::u8path(argv[5]), true);
        if (mode == "predict" && argc == 6) return predict(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), fs::u8path(argv[5]));
        if (mode == "predict" && argc > 6) { std::vector<cv::Mat> ref, zero; split_extra(argv + 6, argv + argc, ref, zero); return predict(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), fs::u8path(argv[5]), ref, zero); }
        if (mode == "register" && argc > 4) return register_reference(fs::u8path(argv[2]), fs::u8path(argv[3]), read_images(argv + 4, argv + argc));
        if (mode == "health" && (argc == 6 || argc == 7)) {
            std::string leave; if (argc == 7) { leave = argv[6]; if (leave.rfind("leave=", 0) != 0) throw std::invalid_argument("마지막 인자는 leave=회차"); leave = leave.substr(6); }
            return make_health(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), fs::u8path(argv[5]), leave);
        }
        if (mode == "range" && (argc == 6 || (argc == 7 && std::string(argv[6]) == "no-bars")))
            return range_test(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), fs::u8path(argv[5]), argc == 7);
        if (mode == "health-check" && argc == 5) return check_health(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]));
        std::cerr << "usage: final00_tool measure ASSETS PHOTOS_TSV OUTDIR [no-bars] | eval FEATURES_TSV LABELS_TSV SPLITS_TSV OUTDIR [no-bars] | predict ASSETS PHOTOS_TSV LABELS_TSV OUTDIR [ref=IMAGE...] [zero=IMAGE...] | register ASSETS OUTDIR REFERENCE_IMAGE..."
                     " | health MODEL7_TSV FEATURES_TSV LABELS_TSV OUT_TSV [leave=SOURCE] | health-check MODEL7_TSV HEALTH_TSV FEATURES_TSV  (measure, predict 에 method=track536|ncc 를 붙이면 QR 방식을 고른다)\n";
        return 2;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}
