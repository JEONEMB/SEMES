// 값 7개 구성(final00)의 측정·학습·평가 도구.
//
// 사용
//   final00_tool measure <assets> <photos.tsv> <출력 폴더>
//       photos.tsv: "id<TAB>사진 경로" (UTF-8, 머리줄 없음). 출력: features7.tsv. 모델 없이 값 7개만 잰다.
//   final00_tool eval <features7.tsv> <labels.tsv> <splits.tsv> <출력 폴더>
//       labels.tsv: id, source, Y_cmd, T_cmd, Y_abs, T_abs (절대 영점 라벨)
//       splits.tsv: id, final, cv0..cv9, half0..half9 (자세 묶음 단위 분할, 마지막 확인 사진의 cv/half 는 -1)
//       출력: eval.txt, model7.tsv (전체 사진으로 학습한 계수. assets 에 넣는 파일)
//   final00_tool predict <assets> <photos.tsv> <labels.tsv> <출력 폴더>
//       공개 API(initialize_joint, joint)로 사진마다 (Y, T) 를 내고 라벨과 비교한다. 출력: predictions.tsv, predict.txt
// 인자 경로는 ASCII 로 준다(목록 파일 안의 사진 경로는 한글이어도 된다). 기존 출력은 덮어쓰지 않는다.
#include "final00.hpp"
#include "huber_fit.hpp"
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

int measure(const fs::path& assets, const fs::path& photos, const fs::path& out) {
    if (fs::exists(out / "features7.tsv")) { std::cerr << "출력이 이미 있다\n"; return 3; }
    const auto rows = read_table(photos, false);
    const int n = int(rows.size());
    const final00::Measurer measurer(assets.u8string());
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
// train 으로 학습해 test 를 채점한다.
void score(const Data& d, const std::vector<int>& train, const std::vector<int>& test, Score& s) {
    final00::LinearModel model; model.fit(take(d.x, train), take(d.y, train));
    const cv::Mat p = model.predict(take(d.x, test));
    for (int i = 0; i < int(test.size()); ++i) {
        const double ey = p.at<double>(i, 0) - d.y.at<double>(test[i], 0), et = p.at<double>(i, 1) - d.y.at<double>(test[i], 1);
        s.fail += !final00::passes(ey, et); ++s.count; s.max_y = std::max(s.max_y, std::abs(ey)); s.max_t = std::max(s.max_t, std::abs(et)); s.sum_y2 += ey * ey; s.sum_t2 += et * et;
    }
}

int evaluate(const fs::path& features, const fs::path& labels, const fs::path& splits, const fs::path& out) {
    if (fs::exists(out / "eval.txt")) { std::cerr << "출력이 이미 있다\n"; return 3; }
    const Data d = load(features, labels, splits); const int n = int(d.id.size());
    std::vector<int> dev, fin;
    for (int i = 0; i < n; ++i) (d.final[i] ? fin : dev).push_back(i);
    std::ostringstream o; o << std::fixed;
    o << "photos " << n << " dev " << dev.size() << " final " << fin.size() << "\n";
    // 주 검증: 개발 자료 섞은 80/20, 10회
    std::vector<int> fails; double worst_y = 0, worst_t = 0, sy = 0, st = 0;
    for (int k = 0; k < 10; ++k) {
        Score s;
        for (int fold = 0; fold < 5; ++fold) {
            std::vector<int> train, test;
            for (int i : dev) (d.cv[i][k] == fold ? test : train).push_back(i);
            score(d, train, test, s);
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
            score(d, train, test, s);
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
        Score s; score(d, train, test, s); pass += s.count - s.fail;
        o << ' ' << name << ' ' << std::setprecision(1) << 100.0 * (s.count - s.fail) / s.count << '%';
    }
    o << " all " << std::setprecision(1) << 100.0 * pass / dev.size() << "%\n";
    // 마지막 확인
    { Score s; score(d, dev, fin, s);
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
    std::ofstream m(out / "model7.tsv", std::ios::binary); m << std::setprecision(17);
    m << "name\tmean\tscale\tweight_Y\tweight_T\n" << "intercept\t0\t1\t" << model.weights.at<double>(0, 0) << '\t' << model.weights.at<double>(0, 1) << '\n';
    for (int j = 0; j < 7; ++j) m << final00::kValueNames[j] << '\t' << model.mean.at<double>(j) << '\t' << model.scale.at<double>(j) << '\t' << model.weights.at<double>(j + 1, 0) << '\t' << model.weights.at<double>(j + 1, 1) << '\n';
    std::cout << o.str();
    return 0;
}

int predict(const fs::path& assets, const fs::path& photos, const fs::path& labels, const fs::path& out) {
    if (fs::exists(out / "predictions.tsv")) { std::cerr << "출력이 이미 있다\n"; return 3; }
    std::map<int, std::vector<std::string>> lab;
    for (auto& r : read_table(labels, true)) lab[std::stoi(r.at(0))] = r;
    const auto rows = read_table(photos, false);
    initialize_joint(assets.u8string());
    fs::create_directories(out);
    std::ofstream f(out / "predictions.tsv", std::ios::binary); f << std::setprecision(17) << "id\tY\tT\tY_abs\tT_abs\tpass\tvalid\toutside_training_range\tmessage\n";
    int valid = 0, pass = 0; double max_y = 0, max_t = 0;
    cv::setNumThreads(1);
    for (const auto& r : rows) {   // joint() 는 내부 잠금으로 한 번에 한 장만 처리한다
        const int id = std::stoi(r.at(0)); const auto& l = lab.at(id); const double ly = std::stod(l.at(4)), lt = std::stod(l.at(5));
        cv::Mat image = read_image(fs::u8path(r.at(1)));
        try {
            const Alignment_t p = joint(image); const auto d = alignment_diagnostics();
            const bool okay = final00::passes(p.y - ly, p.t - lt); ++valid; pass += okay;
            max_y = std::max(max_y, std::abs(p.y - ly)); max_t = std::max(max_t, std::abs(p.t - lt));
            f << id << '\t' << p.y << '\t' << p.t << '\t' << ly << '\t' << lt << '\t' << okay << "\t1\t" << d.outside_training_range << "\t\n";
        } catch (const std::exception& e) {
            f << id << "\tnan\tnan\t" << ly << '\t' << lt << "\t0\t0\t0\t" << e.what() << '\n';
        }
    }
    std::ostringstream o; o << "joint_api photos " << rows.size() << " valid " << valid << " pass " << pass << std::fixed << std::setprecision(5) << " max|eY| " << max_y << " max|eT| " << max_t << "\n";
    { std::ofstream t(out / "predict.txt", std::ios::binary); t << o.str(); }
    std::cout << o.str();
    return 0;
}
}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "measure" && argc == 5) return measure(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]));
        if (mode == "eval" && argc == 6) return evaluate(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), fs::u8path(argv[5]));
        if (mode == "predict" && argc == 6) return predict(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), fs::u8path(argv[5]));
        std::cerr << "usage: final00_tool measure ASSETS PHOTOS_TSV OUTDIR | eval FEATURES_TSV LABELS_TSV SPLITS_TSV OUTDIR | predict ASSETS PHOTOS_TSV LABELS_TSV OUTDIR\n";
        return 2;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}
