// 새 기준 구성(값 7개)의 측정과 평가 도구.
//   값 7개 = QR 536점 이동의 가로 평균 1 + 위·아래 바 기울기 2 + 나사 4개의 가로 이동(기울기 영상 맞춤) 4
//
// 사용
//   v2tool measure <00004 assets> <control assets> <photos.tsv> <출력 폴더>
//       photos.tsv: "id<TAB>사진 경로" (UTF-8, 머리줄 없음). 출력: features7.tsv
//   v2tool eval <features7.tsv> <labels.tsv> <splits.tsv> <출력 폴더>
//       labels.tsv: id, source, Y_cmd, T_cmd, Y_abs, T_abs (절대 영점 라벨)
//       splits.tsv: id, final, cv0..cv9, half0..half9 (자세 묶음 단위 분할, 마지막 확인 사진의 cv/half 는 -1)
//       출력: eval.txt, model7.tsv
// 인자 경로는 ASCII 로 준다(목록 파일 안의 사진 경로는 한글이어도 된다). 기존 출력은 덮어쓰지 않는다.
#include "screw_gradient.hpp"
#include "huber_fit.hpp"
#include "control_internal.hpp"
#include "final04.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
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
std::vector<unsigned char> selected_points(const fs::path& control_assets) {
    cv::FileStorage file((control_assets / "points.json").u8string(), cv::FileStorage::READ);
    if (!file.isOpened()) throw std::runtime_error("control points.json 이 없다");
    std::vector<unsigned char> selected;
    for (const auto& row : file["points"]) selected.push_back(static_cast<unsigned char>(int(row[3])));
    if (selected.size() != 630 || std::count(selected.begin(), selected.end(), 1) != 536) throw std::runtime_error("630점 중 536점 선택이 아니다");
    return selected;
}

int measure(const fs::path& assets04, const fs::path& assets_control, const fs::path& photos, const fs::path& out) {
    if (fs::exists(out / "features7.tsv")) { std::cerr << "출력이 이미 있다\n"; return 3; }
    const auto rows = read_table(photos, false);
    const int n = int(rows.size());
    const legacy_control::Tracker tracker(assets_control);
    const auto selected = selected_points(assets_control);
    const alignment_detail::Observer observer(assets04);
    const cv::Mat center = alignment_detail::read_gray(assets04 / "center.bmp");
    const final04_pin::Observer pins(center);
    const v2::ScrewGradient screws(center);
    const auto block_x = observer.block_centers();
    std::vector<std::array<double, 11>> values(n); std::vector<int> valid(n, 0), ok(n, 0); std::vector<std::string> error(n);
    cv::setNumThreads(1);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        try {
            const cv::Mat gray = read_image(fs::u8path(rows[i].at(1)));
            if (gray.empty()) throw std::runtime_error("사진을 읽지 못했다");
            const auto tracks = tracker.measure(gray);
            valid[i] = tracker.valid_selected(tracks);
            if (valid[i] != 536) throw std::runtime_error("536점이 모두 유효하지 않다");
            double dx = 0;
            for (int p = 0; p < 630; ++p) if (selected[p]) dx += tracks.displacement.at<float>(p, 0);
            const auto observation = observer.measure(gray);
            const auto pin = pins.measure(gray);
            const cv::Mat f66 = final04::features(observation.values, pin.displacement, block_x);
            const cv::Mat screw = screws.measure(gray, pin.displacement);
            if (!cv::checkRange(screw)) throw std::runtime_error("나사 기울기 영상 맞춤 실패");
            values[i] = {dx / 536.0, f66.at<double>(55), f66.at<double>(57),
                         screw.at<double>(0), screw.at<double>(2), screw.at<double>(4), screw.at<double>(6),
                         screw.at<double>(1), screw.at<double>(3), screw.at<double>(5), screw.at<double>(7)};
            ok[i] = 1;
        } catch (const std::exception& e) { error[i] = e.what(); }
    }
    fs::create_directories(out);
    std::ofstream f(out / "features7.tsv"); f << std::setprecision(17);
    f << "id\tqr_dx_mean\tbar_tilt_upper\tbar_tilt_lower\tscrew1_x\tscrew2_x\tscrew3_x\tscrew4_x\tscrew1_y\tscrew2_y\tscrew3_y\tscrew4_y\tvalid536\tmeasured\terror\n";
    int good = 0;
    for (int i = 0; i < n; ++i) {
        f << rows[i][0];
        for (double v : values[i]) f << '\t' << (ok[i] ? v : std::numeric_limits<double>::quiet_NaN());
        f << '\t' << valid[i] << '\t' << ok[i] << '\t' << error[i] << '\n'; good += ok[i];
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
        if (r.at(13) != "1") throw std::runtime_error("측정 실패 사진이 있다: id " + r[0]);
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
    v2::LinearModel model; model.fit(take(d.x, train), take(d.y, train));
    const cv::Mat p = model.predict(take(d.x, test));
    for (int i = 0; i < int(test.size()); ++i) {
        const double ey = p.at<double>(i, 0) - d.y.at<double>(test[i], 0), et = p.at<double>(i, 1) - d.y.at<double>(test[i], 1);
        s.fail += !v2::passes(ey, et); ++s.count; s.max_y = std::max(s.max_y, std::abs(ey)); s.max_t = std::max(s.max_t, std::abs(et)); s.sum_y2 += ey * ey; s.sum_t2 += et * et;
    }
}

int evaluate(const fs::path& features, const fs::path& labels, const fs::path& splits, const fs::path& out) {
    if (fs::exists(out / "eval.txt")) { std::cerr << "출력이 이미 있다\n"; return 3; }
    const Data d = load(features, labels, splits); const int n = int(d.id.size());
    std::vector<int> dev, fin, all;
    for (int i = 0; i < n; ++i) { all.push_back(i); (d.final[i] ? fin : dev).push_back(i); }
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
    // 전체 702장으로 학습한 모델
    v2::LinearModel model; model.fit(d.x, d.y);
    { Score s; const cv::Mat p = model.predict(d.x);
      for (int i = 0; i < n; ++i) s.fail += !v2::passes(p.at<double>(i, 0) - d.y.at<double>(i, 0), p.at<double>(i, 1) - d.y.at<double>(i, 1));
      o << "all_photos_resubstitution fails " << s.fail << "/" << n << "\n"; }
    fs::create_directories(out);
    { std::ofstream f(out / "eval.txt"); f << o.str(); }
    std::ofstream m(out / "model7.tsv"); m << std::setprecision(17);
    const char* names[7] = {"qr_dx_mean", "bar_tilt_upper", "bar_tilt_lower", "screw1_x", "screw2_x", "screw3_x", "screw4_x"};
    m << "name\tmean\tscale\tweight_Y\tweight_T\n" << "intercept\t0\t1\t" << model.weights.at<double>(0, 0) << '\t' << model.weights.at<double>(0, 1) << '\n';
    for (int j = 0; j < 7; ++j) m << names[j] << '\t' << model.mean.at<double>(j) << '\t' << model.scale.at<double>(j) << '\t' << model.weights.at<double>(j + 1, 0) << '\t' << model.weights.at<double>(j + 1, 1) << '\n';
    std::cout << o.str();
    return 0;
}
}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "measure" && argc == 6) return measure(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), fs::u8path(argv[5]));
        if (mode == "eval" && argc == 6) return evaluate(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]), fs::u8path(argv[5]));
        std::cerr << "usage: v2tool measure ASSETS04 ASSETS_CONTROL PHOTOS_TSV OUTDIR | v2tool eval FEATURES_TSV LABELS_TSV SPLITS_TSV OUTDIR\n";
        return 2;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}
