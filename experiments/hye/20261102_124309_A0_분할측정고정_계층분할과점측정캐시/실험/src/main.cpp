// A0: 분할 고정, 점 측정 캐시, 그리고 "우리가 쓸 수 있는 자극"의 확인.
//  1) data/clean 의 575장을 (세트, Y, T) 묶음으로 묶는다. 반복 3장은 같은 묶음.
//  2) 혼합 4-fold: 1014 280장은 기존 splits.json 배정을 그대로 쓰고, 1020·1029 묶음은 (Y,T) 정렬 순서로 계층 배정한다.
//  3) 날짜 3-fold: 1014=0, 1020=1, 1029=2.
//  4) 채택 측정기(control tracker)로 630점 변위를, joint 의 선 프로토콜로 48블록 이동량을 재서 캐시에 굳힌다.
//  5) 측정 영역·점·선 블록·자극 범위·변위장을 그림으로 남겨 눈으로 확인할 수 있게 한다.
// 난수를 쓰지 않는다. 다시 돌리면 같은 파일이 나온다.
#include "cache.hpp"
#include "control_internal.hpp"
#include "observer_internal.hpp"
#include "hye/plot.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <set>

using namespace a0;
namespace {

const std::string kCenterSha = "01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685";
// tracker.cpp 와 같은 상수. 그림에 그대로 그린다.
const cv::Rect kFullCrop(1340, 800, 1130, 530);
const cv::Rect kQr[3] = {cv::Rect(1552, 969, 184, 198), cv::Rect(1809, 971, 184, 198), cv::Rect(2067, 973, 185, 198)};

cv::Mat imread_u8(const fs::path& path, int flags) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return cv::imdecode(buf, flags);
}

struct ManifestRow {
    std::string set, rel;
    double y = 0, t = 0;
    int repeat = 1;
    std::string condition;
};

std::vector<ManifestRow> read_manifest(const fs::path& clean, const std::string& set) {
    std::ifstream in(clean / set / "_metadata" / "manifest.csv");
    if (!in) throw std::runtime_error("manifest 없음: " + set);
    std::string line;
    std::getline(in, line);
    if (line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);
    auto head = split_csv(line);
    auto col = [&](const char* n) {
        for (size_t i = 0; i < head.size(); ++i) if (head[i] == n) return int(i);
        throw std::runtime_error(std::string("manifest 열 없음: ") + n);
    };
    const int cp = col("relative_path"), cy = col("filename_Y_command"), ct = col("filename_T_command"),
              cr = col("capture_repeat"), cc = col("condition");
    std::vector<ManifestRow> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split_csv(line);
        ManifestRow r;
        r.set = set; r.rel = f[cp];
        r.y = std::stod(f[cy]); r.t = std::stod(f[ct]);
        r.repeat = std::stoi(f[cr]); r.condition = f[cc];
        out.push_back(r);
    }
    return out;
}

// splits.json 의 fold 별 "test" 목록을 파일명 -> fold 로 읽는다. 최소 파서다 (문자열 배열만 본다).
std::map<std::string, int> read_fixed_1014_folds(const fs::path& file) {
    std::ifstream f(file);
    if (!f) throw std::runtime_error("splits.json 없음: " + file.u8string());
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::map<std::string, int> out;
    size_t pos = 0;
    int fold = -1;
    while (true) {
        size_t kf = s.find("\"fold\"", pos), kt = s.find("\"test\"", pos);
        if (kt == std::string::npos) break;
        if (kf != std::string::npos && kf < kt) fold = std::stoi(s.substr(s.find(':', kf) + 1, 8));
        size_t open = s.find('[', kt), close = s.find(']', open);
        if (open == std::string::npos || close == std::string::npos) break;
        std::string body = s.substr(open + 1, close - open - 1);
        size_t q = 0;
        while ((q = body.find('"', q)) != std::string::npos) {
            size_t e = body.find('"', q + 1);
            if (e == std::string::npos) break;
            out[body.substr(q + 1, e - q - 1)] = fold;
            q = e + 1;
        }
        pos = close + 1;
    }
    return out;
}

std::string basename(const std::string& rel) {
    size_t p = rel.find_last_of("/\\");
    return p == std::string::npos ? rel : rel.substr(p + 1);
}

struct LineBlock {
    cv::Rect box;
    std::string id;
    std::vector<double> reference;
};

std::vector<LineBlock> read_line_blocks(const fs::path& protocol, const cv::Mat& center) {
    cv::FileStorage fsx(protocol.u8string(), cv::FileStorage::READ);
    if (!fsx.isOpened()) throw std::runtime_error("line_protocol.json 없음");
    std::vector<LineBlock> out;
    for (auto it = fsx["blocks"].begin(); it != fsx["blocks"].end(); ++it) {
        auto n = *it;
        auto b = n["read_box"];
        cv::Rect r(int(b[0]), int(b[1]), int(b[2]) - int(b[0]), int(b[3]) - int(b[1]));
        if (r.width != 64 || r.height != 49) throw std::runtime_error("선 블록 크기가 64x49 가 아니다");
        LineBlock lb;
        lb.box = r;
        lb.id = std::string(n["id"]);
        lb.reference = alignment_detail::line_profile(center, r);
        out.push_back(lb);
    }
    if (out.size() != kLines) throw std::runtime_error("선 블록이 48개가 아니다");
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const fs::path repo = fs::u8path(REPO_DIR);
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path cache = result / "cache";
        const fs::path results = result / "results";
        fs::create_directories(cache);
        fs::create_directories(results);
        hye::begin_results();
        const bool quick = argc > 1 && std::string(argv[1]) == "quick";

        const fs::path clean = repo / "data" / "clean";
        const fs::path control_assets = repo / fs::u8path(u8"experiments/hye/00000_final_Y_T control/assets");
        const fs::path joint_assets = repo / fs::u8path(u8"experiments/hye/00000_final_Y_T_joint/assets");
        const fs::path splits_json = repo / fs::u8path(u8"experiments/hye/00000_final_Y_T control/validation/splits.json");

        // ---- 1) 사진 목록 ----
        std::vector<ManifestRow> rows;
        for (auto set : {"clean_20261014", "clean_20261020", "clean_20261029"}) {
            auto r = read_manifest(clean, set);
            rows.insert(rows.end(), r.begin(), r.end());
        }
        std::cout << "사진 " << rows.size() << "장\n";

        // ---- 2) 묶음 ----
        std::map<std::tuple<std::string, long long, long long>, int> group_of;
        std::vector<Shot> shots;
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            auto key = std::make_tuple(r.set, llround(r.y * 1000), llround(r.t * 1000));
            auto it = group_of.find(key);
            int g;
            if (it == group_of.end()) { g = int(group_of.size()); group_of[key] = g; }
            else g = it->second;
            Shot s;
            s.idx = int(i); s.set = r.set; s.rel = r.rel; s.y = r.y; s.t = r.t;
            s.repeat = r.repeat; s.condition = r.condition; s.group = g;
            s.fold_date = r.set == "clean_20261014" ? 0 : r.set == "clean_20261020" ? 1 : 2;
            shots.push_back(s);
        }
        std::cout << "조건 묶음 " << group_of.size() << "개\n";

        // ---- 3) 혼합 4-fold ----
        auto fixed = read_fixed_1014_folds(splits_json);
        int matched = 0;
        for (auto& s : shots) {
            if (s.set != "clean_20261014") continue;
            if (s.condition == "center") { s.fold_mixed = -1; continue; }
            auto it = fixed.find(basename(s.rel));
            if (it == fixed.end()) throw std::runtime_error("기존 fold 배정에 없는 1014 사진: " + s.rel);
            s.fold_mixed = it->second;
            ++matched;
        }
        std::cout << "1014 기존 배정 일치 " << matched << "장\n";
        for (auto set : {"clean_20261020", "clean_20261029"}) {
            std::vector<std::pair<std::pair<double, double>, int>> groups;
            std::map<int, bool> seen;
            for (auto& s : shots) {
                if (s.set != set || seen[s.group]) continue;
                seen[s.group] = true;
                groups.push_back({{s.y, s.t}, s.group});
            }
            std::sort(groups.begin(), groups.end());
            std::map<int, int> fold_of;
            for (size_t k = 0; k < groups.size(); ++k) fold_of[groups[k].second] = int(k % 4);
            for (auto& s : shots) if (s.set == set) s.fold_mixed = fold_of[s.group];
        }

        // ---- 4) 측정 ----
        legacy_control::Tracker tracker(control_assets);
        if (legacy_control::sha256(clean / "clean_20261014" / "center.bmp") != kCenterSha)
            throw std::runtime_error("center.bmp 해시가 다르다");
        cv::Mat center = legacy_control::read_gray(clean / "clean_20261014" / "center.bmp");
        auto blocks = read_line_blocks(joint_assets / "line_protocol.json", center);

        // 점 목록 (그림과 캐시에 함께 쓴다)
        std::vector<Point> points;
        {
            cv::FileStorage pf((control_assets / "points.json").u8string(), cv::FileStorage::READ);
            auto node = pf["points"];
            for (auto it = node.begin(); it != node.end(); ++it) {
                auto r = *it;
                points.push_back({double(r[0]), double(r[1]), int(r[2]), int(r[3])});
            }
            if (points.size() != kPoints) throw std::runtime_error("points.json 이 630점이 아니다");
        }

        const size_t n = shots.size();
        std::vector<float> disp(n * kPoints * 2, 0), quality(n * kPoints * kQuality, 0), lines(n * kLines * 2, 0);
        std::vector<unsigned char> valid(n * kPoints, 0), lines_valid(n * kLines, 0);
        std::vector<std::string> failures(n);

        auto zero_tracks = tracker.measure(center);
        std::vector<float> zero(size_t(kPoints) * 2);
        for (int p = 0; p < kPoints; ++p) {
            zero[p * 2] = zero_tracks.displacement.at<float>(p, 0);
            zero[p * 2 + 1] = zero_tracks.displacement.at<float>(p, 1);
        }

        cv::setNumThreads(1);
        std::atomic<int> done{0};
        std::mutex log;
        auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < int(n); ++i) {
            if (quick && i % 40 != 0) continue;
            try {
                cv::Mat img = imread_u8(clean / fs::u8path(shots[i].set) / fs::u8path(shots[i].rel), cv::IMREAD_GRAYSCALE);
                if (img.empty()) throw std::runtime_error("읽기 실패");
                auto tr = tracker.measure(img);
                for (int p = 0; p < kPoints; ++p) {
                    disp[size_t(i) * kPoints * 2 + p * 2] = tr.displacement.at<float>(p, 0);
                    disp[size_t(i) * kPoints * 2 + p * 2 + 1] = tr.displacement.at<float>(p, 1);
                    valid[size_t(i) * kPoints + p] = tr.valid[p];
                    for (int c = 0; c < kQuality; ++c)
                        quality[size_t(i) * kPoints * kQuality + p * kQuality + c] = tr.quality.at<float>(p, c);
                }
                for (int k = 0; k < 3; ++k) shots[i].ncc[k] = tr.ncc[k];
                shots[i].valid_count = int(std::count(tr.valid.begin(), tr.valid.end(), 1));
                shots[i].valid_selected = tracker.valid_selected(tr);
                for (int b = 0; b < kLines; ++b) {
                    auto fit = alignment_detail::match_line(blocks[b].reference, alignment_detail::line_profile(img, blocks[b].box));
                    lines[size_t(i) * kLines * 2 + b * 2] = float(fit.shift);
                    lines[size_t(i) * kLines * 2 + b * 2 + 1] = float(fit.correlation);
                    lines_valid[size_t(i) * kLines + b] = fit.valid ? 1 : 0;
                }
                shots[i].measured = true;
            } catch (const std::exception& e) {
                failures[i] = e.what();
            }
            int k = ++done;
            if (k % 25 == 0) {
                std::lock_guard<std::mutex> g(log);
                auto sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                std::cout << k << "/" << n << "  " << int(sec) << "s\n" << std::flush;
            }
        }

        // ---- 5) 캐시 쓰기 ----
        Table st;
        st.head = {"idx", "set", "rel", "y", "t", "repeat", "condition", "group", "fold_mixed", "fold_date",
                   "ncc0", "ncc1", "ncc2", "valid_count", "valid_selected", "measured"};
        for (auto& s : shots)
            st.add({std::to_string(s.idx), s.set, s.rel, num(s.y, 3), num(s.t, 3), std::to_string(s.repeat),
                    s.condition, std::to_string(s.group), std::to_string(s.fold_mixed), std::to_string(s.fold_date),
                    num(s.ncc[0], 4), num(s.ncc[1], 4), num(s.ncc[2], 4), std::to_string(s.valid_count),
                    std::to_string(s.valid_selected), s.measured ? "1" : "0"});
        st.csv(cache / "shots.csv");

        Table pt;
        pt.head = {"id", "x", "y", "region", "selected"};
        for (size_t i = 0; i < points.size(); ++i)
            pt.add({std::to_string(i), num(points[i].x, 6), num(points[i].y, 6),
                    std::to_string(points[i].region), std::to_string(points[i].selected)});
        pt.csv(cache / "points.csv");

        write_binary(cache / "disp.f32", disp);
        write_binary(cache / "valid.u8", valid);
        write_binary(cache / "quality.f32", quality);
        write_binary(cache / "zero.f32", zero);
        write_binary(cache / "lines.f32", lines);
        write_binary(cache / "lines_valid.u8", lines_valid);

        // ---- 6) 자극(입력) 설계: 우리가 실제로 줄 수 있었던 명령의 범위 ----
        Table ex;
        ex.head = {"세트", "Y 명령 값", "T 명령 값", "조건 수", "Y 자극", "T 자극", "두 축 동시"};
        Table excsv;
        excsv.head = {"set", "y", "t", "shots"};
        for (auto set : {"clean_20261014", "clean_20261020", "clean_20261029"}) {
            std::set<double> ys, ts;
            std::map<std::pair<double, double>, int> cond;
            for (auto& s : shots) if (s.set == set) { ys.insert(s.y); ts.insert(s.t); cond[{s.y, s.t}]++; }
            for (auto& c : cond) excsv.add({std::string(set).substr(6), num(c.first.first, 2), num(c.first.second, 2), std::to_string(c.second)});
            auto range = [](const std::set<double>& v) {
                if (v.size() == 1) return num(*v.begin(), 2) + " 하나";
                return num(*v.begin(), 2) + " ~ " + num(*v.rbegin(), 2) + " (" + std::to_string(v.size()) + "종)";
            };
            bool ymove = ys.size() > 1, tmove = ts.size() > 1;
            int both = 0;
            for (auto& c : cond) if (c.first.first != 0 && c.first.second != 0) ++both;
            ex.add({std::string(set).substr(6), range(ys), range(ts), std::to_string(cond.size()),
                    ymove ? "있음" : "없음", tmove ? "있음" : "없음", both ? std::to_string(both) + " 조건" : "없음"});
        }
        ex.csv(results / "excitation.csv");
        excsv.csv(results / "excitation_conditions.csv");

        // ---- 7) 점검 표 ----
        Table fold;
        fold.head = {"fold", "1014", "1020", "1029", "합계", "묶음 수"};
        for (int f = -1; f < 4; ++f) {
            int a = 0, b = 0, c = 0;
            std::map<int, bool> gs;
            for (auto& s : shots) if (s.fold_mixed == f) {
                gs[s.group] = true;
                (s.set == "clean_20261014" ? a : s.set == "clean_20261020" ? b : c)++;
            }
            fold.add({f < 0 ? "-1 (center)" : std::to_string(f), std::to_string(a), std::to_string(b),
                      std::to_string(c), std::to_string(a + b + c), std::to_string(gs.size())});
        }
        fold.csv(results / "fold_mixed.csv");

        Table sets;
        sets.head = {"세트", "장수", "측정 성공", "유효점 중앙값/630", "선택점 유효 중앙값/536", "최소 NCC 중앙값", "선 블록 유효 중앙값/48"};
        Table failed;
        failed.head = {"idx", "사진", "이유"};
        for (auto set : {"clean_20261014", "clean_20261020", "clean_20261029"}) {
            std::vector<double> vc, vs, ncc, lv;
            int total = 0, ok = 0;
            for (auto& s : shots) {
                if (s.set != set) continue;
                ++total;
                if (!s.measured) continue;
                ++ok;
                vc.push_back(s.valid_count);
                vs.push_back(s.valid_selected);
                ncc.push_back(std::min({s.ncc[0], s.ncc[1], s.ncc[2]}));
                int c = 0;
                for (int b = 0; b < kLines; ++b) c += lines_valid[size_t(s.idx) * kLines + b];
                lv.push_back(c);
            }
            sets.add({std::string(set).substr(6), std::to_string(total), std::to_string(ok), num(median(vc), 0),
                      num(median(vs), 0), num(median(ncc), 4), num(median(lv), 0)});
        }
        for (auto& s : shots) if (!failures[s.idx].empty()) failed.add({std::to_string(s.idx), s.name(), failures[s.idx]});
        sets.csv(results / "measure_summary.csv");
        failed.csv(results / "failures.csv");

        // ---- 8) 그림 ----
        // 8-1 전체 측정 영역
        {
            std::vector<hye::Box> boxes;
            boxes.push_back({cv::Rect2d(kFullCrop), hye::color(2), "full crop 1130x530"});
            for (int k = 0; k < 3; ++k) {
                boxes.push_back({cv::Rect2d(kQr[k]), hye::kRed, "QR" + std::to_string(k + 1)});
                cv::Rect search(kQr[k].x - 110, kQr[k].y - 22, kQr[k].width + 220, kQr[k].height + 44);
                boxes.push_back({cv::Rect2d(search), hye::color(1), ""});
            }
            for (auto& b : blocks) boxes.push_back({cv::Rect2d(b.box), hye::color(4), ""});
            cv::Mat m = hye::zoom(center, cv::Rect(0, 0, 3840, 2160), 1600, {}, 0,
                                  "A0-1 measurement regions: red=QR template, orange=NCC search, green=crop, purple=48 line blocks", boxes);
            hye::save_png("01_측정영역_전체.png", m);
        }
        // 8-2 QR 세 영역의 630점 (선택 536 / 비선택 94)
        {
            std::vector<cv::Mat> tiles;
            for (int k = 0; k < 3; ++k) {
                std::vector<hye::Mark> marks;
                for (auto& p : points) {
                    if (p.region != k) continue;
                    marks.push_back({{p.x, p.y}, {0, 0}, "", p.selected ? hye::kRed : hye::color(1)});
                }
                int sel = 0, all = 0;
                for (auto& p : points) if (p.region == k) { ++all; sel += p.selected; }
                cv::Rect roi = kQr[k];
                roi.x -= 14; roi.y -= 14; roi.width += 28; roi.height += 28;
                tiles.push_back(hye::zoom(center, roi, 480, marks, 0,
                                          "QR" + std::to_string(k + 1) + ": " + std::to_string(sel) + "/" + std::to_string(all) + " selected (red)"));
            }
            hye::save_png("02_고정630점.png", hye::hstack(tiles));
        }
        // 8-3 선 48블록
        {
            std::vector<cv::Mat> tiles;
            for (auto line : {"upper", "lower"}) {
                std::vector<hye::Box> bx;
                int y0 = 1 << 30, y1 = 0, x0 = 1 << 30, x1 = 0, count = 0;
                for (auto& b : blocks) {
                    if (b.id.rfind(line, 0) != 0) continue;
                    bx.push_back({cv::Rect2d(b.box), hye::color(4), ""});
                    y0 = std::min(y0, b.box.y); y1 = std::max(y1, b.box.y + b.box.height);
                    x0 = std::min(x0, b.box.x); x1 = std::max(x1, b.box.x + b.box.width);
                    ++count;
                }
                cv::Rect roi(x0 - 30, y0 - 40, x1 - x0 + 60, y1 - y0 + 80);
                tiles.push_back(hye::zoom(center, roi, 1500, {}, 0,
                                          std::string(line) + " line: " + std::to_string(count) + " blocks 64x49", bx));
            }
            hye::save_png("03_선블록_위아래.png", hye::vstack(tiles));
        }
        // 8-4 쓸 수 있는 자극 범위 (Y,T 명령 지도)
        {
            hye::Chart c("A0-4 available excitation (commanded Y,T)", "Y command [mm]", "T command [deg]");
            const char* names[] = {"clean_20261014", "clean_20261020", "clean_20261029"};
            const char* labs[] = {"1014 single-axis", "1020 mixed", "1029 T sweep (Y fixed)"};
            for (int k = 0; k < 3; ++k) {
                std::vector<double> x, y;
                std::set<std::pair<double, double>> seen;
                for (auto& s : shots) if (s.set == names[k] && seen.insert({s.y, s.t}).second) { x.push_back(s.y); y.push_back(s.t); }
                c.scatter(x, y, hye::color(k), labs[k], 5);
            }
            c.save("04_자극범위_YT지도.png");
        }
        // 8-5 fold 배정 지도
        {
            hye::Chart c("A0-5 mixed 4-fold assignment", "Y command [mm]", "T command [deg]");
            for (int f = 0; f < 4; ++f) {
                std::vector<double> x, y;
                for (auto& s : shots) if (s.fold_mixed == f) { x.push_back(s.y); y.push_back(s.t); }
                c.scatter(x, y, hye::color(f), "fold " + std::to_string(f), 4);
            }
            c.save("05_fold배정_YT지도.png");
        }
        // 8-6 사진별 유효 선택점 수
        {
            hye::Chart c("A0-6 valid selected points per shot (of 536)", "shot index", "valid points");
            for (int k = 0; k < 3; ++k) {
                const char* names[] = {"clean_20261014", "clean_20261020", "clean_20261029"};
                std::vector<double> x, y;
                for (auto& s : shots) if (s.set == names[k] && s.measured) { x.push_back(s.idx); y.push_back(s.valid_selected); }
                c.scatter(x, y, hye::color(k), std::string(names[k]).substr(6), 2);
            }
            c.hl.push_back({536 * 0.95, hye::kRed});
            c.save("06_유효점수_사진별.png");
        }
        // 8-7 대표 변위장: Y 단독 최대, T 단독 최대, 혼합 하나
        {
            auto pick = [&](const std::string& set, double y, double t) {
                for (auto& s : shots) if (s.set == set && std::abs(s.y - y) < 1e-9 && std::abs(s.t - t) < 1e-9 && s.measured) return s.idx;
                return -1;
            };
            struct Pick { int idx; std::string cap; };
            std::vector<Pick> picks;
            int a = pick("clean_20261014", 1.00, 0.00), b = pick("clean_20261014", 0.00, 1.00), c = pick("clean_20261020", 1.00, 1.00);
            if (a >= 0) picks.push_back({a, "Y+1.00 T0"});
            if (b >= 0) picks.push_back({b, "Y0 T+1.00"});
            if (c >= 0) picks.push_back({c, "Y+1.00 T+1.00"});
            cv::Rect roi(kQr[0].x - 20, kQr[0].y - 30, kQr[2].x + kQr[2].width - kQr[0].x + 40, kQr[0].height + 60);
            std::vector<cv::Mat> tiles, tiles2;
            for (auto& p : picks) {
                // (a) 변위 그대로. 배율은 가장 긴 화살표가 40 px 가 되게 맞춘다.
                // (b) 점 평균(판 전체 평행이동)을 뺀 나머지. Y 와 T 를 가르는 것은 이 구조 성분이다.
                std::vector<cv::Point2d> d(kPoints);
                cv::Point2d m(0, 0);
                int cnt = 0;
                double big = 0;
                for (int i = 0; i < kPoints; ++i) {
                    d[i] = {disp[size_t(p.idx) * kPoints * 2 + i * 2] - zero[i * 2],
                            disp[size_t(p.idx) * kPoints * 2 + i * 2 + 1] - zero[i * 2 + 1]};
                    if (valid[size_t(p.idx) * kPoints + i]) { m += d[i]; ++cnt; }
                    big = std::max(big, cv::norm(d[i]));
                }
                if (cnt) m *= 1.0 / cnt;
                double g1 = big > 1e-9 ? 40.0 / big : 1;
                double big2 = 0;
                for (int i = 0; i < kPoints; ++i) big2 = std::max(big2, cv::norm(d[i] - m));
                double g2 = big2 > 1e-9 ? 40.0 / big2 : 1;
                std::vector<hye::Mark> a, b;
                for (int i = 0; i < kPoints; i += 7) {
                    if (!valid[size_t(p.idx) * kPoints + i]) continue;
                    a.push_back({{points[i].x, points[i].y}, d[i], "", hye::kRed});
                    b.push_back({{points[i].x, points[i].y}, d[i] - m, "", hye::color(2)});
                }
                char c1[160], c2[200];
                std::snprintf(c1, sizeof c1, "A0-7a %s  mean dx=%.2f dy=%.2f px  (arrow x%.2f)", p.cap.c_str(), m.x, m.y, g1);
                std::snprintf(c2, sizeof c2, "A0-7b %s  minus mean translation, spread %.3f px  (arrow x%.1f)", p.cap.c_str(), big2, g2);
                tiles.push_back(hye::zoom(center, roi, 1500, a, g1, c1));
                tiles2.push_back(hye::zoom(center, roi, 1500, b, g2, c2));
            }
            if (!tiles.empty()) hye::save_png("07_변위장_대표.png", hye::vstack(tiles));
            if (!tiles2.empty()) hye::save_png("09_변위장_평균뺀구조.png", hye::vstack(tiles2));
        }
        // 8-8 선 블록 이동량 (대표 3장)
        {
            hye::Chart c("A0-8 line block shift by block index (0-23 upper, 24-47 lower)", "block index", "shift [px]");
            auto pick = [&](const std::string& set, double y, double t) {
                for (auto& s : shots) if (s.set == set && std::abs(s.y - y) < 1e-9 && std::abs(s.t - t) < 1e-9 && s.measured) return s.idx;
                return -1;
            };
            const char* labs[] = {"Y+1.00 T0", "Y0 T+1.00", "Y+1.00 T+1.00"};
            int ids[] = {pick("clean_20261014", 1.00, 0.00), pick("clean_20261014", 0.00, 1.00), pick("clean_20261020", 1.00, 1.00)};
            for (int k = 0; k < 3; ++k) {
                if (ids[k] < 0) continue;
                std::vector<double> x, y;
                for (int b = 0; b < kLines; ++b) {
                    x.push_back(b);
                    y.push_back(lines[size_t(ids[k]) * kLines * 2 + b * 2]);
                }
                c.line(x, y, hye::color(k), labs[k]);
            }
            c.vl.push_back({23.5, hye::kGray});
            c.save("08_선블록_이동량.png");
        }

        // ---- 9) 결과.md ----
        std::ostringstream o;
        o << "# A0 결과 — 분할 고정, 점 측정 캐시, 자극 범위\n\n"
          << "이 파일은 `실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md)에 있다.\n\n"
          << "허용 절대오차 ±" << num(kTol, 3) << " (5회 반복 정밀도 0.09 의 절반, 2026-11-02 재정의), "
          << "판정용 '작다' 경계 " << num(kTight, 3) << ".\n\n"
          << "## 1. 측정 영역 확인\n\n"
          << hye::img_md("01_측정영역_전체.png", "measurement regions") << "\n\n"
          << hye::img_md("02_고정630점.png", "fixed 630 points") << "\n\n"
          << hye::img_md("03_선블록_위아래.png", "48 line blocks") << "\n\n"
          << "## 2. 쓸 수 있는 자극 (명령 범위)\n\n" << ex.md()
          << "\n" << hye::img_md("04_자극범위_YT지도.png", "excitation map") << "\n\n"
          << "조건별 장수는 [excitation_conditions.csv](results/excitation_conditions.csv) 에 있다.\n\n"
          << "## 3. 혼합 4-fold 구성\n\n" << fold.md()
          << "\n1014 의 " << matched << "장은 기존 `validation/splits.json` 배정을 그대로 썼다. "
          << "1020·1029 는 세트별 (Y,T) 오름차순 묶음에 fold 를 돌려 가며 준 계층 배정이다.\n\n"
          << hye::img_md("05_fold배정_YT지도.png", "fold map") << "\n\n"
          << "## 4. 날짜 3-fold\n\n| fold | 세트 | 장수 |\n|---|---|---|\n";
        for (int f = 0; f < 3; ++f) {
            int c = 0;
            std::string set;
            for (auto& s : shots) if (s.fold_date == f) { ++c; set = s.set.substr(6); }
            o << "| " << f << " | " << set << " | " << c << " |\n";
        }
        o << "\n## 5. 측정 요약\n\n" << sets.md() << "\n"
          << hye::img_md("06_유효점수_사진별.png", "valid points") << "\n\n"
          << "## 6. 측정 실패\n\n";
        if (failed.rows.empty()) o << "없다. " << shots.size() << "장 전부 측정했다.\n";
        else o << failed.md();
        o << "\n## 7. 측정이 실제로 무엇을 보았나\n\n"
          << "판 전체 평행이동(화살표 길이)은 Y 와 T 가 함께 만든다. 두 축을 가르는 것은 평균을 뺀 나머지 구조다.\n\n"
          << hye::img_md("07_변위장_대표.png", "displacement field") << "\n\n"
          << hye::img_md("09_변위장_평균뺀구조.png", "structure after removing mean translation") << "\n\n"
          << hye::img_md("08_선블록_이동량.png", "line shifts") << "\n\n"
          << "## 8. 캐시 파일\n\n"
          << "| 파일 | 크기(원소) |\n|---|---|\n"
          << "| cache/disp.f32 | " << disp.size() << " float32 |\n"
          << "| cache/valid.u8 | " << valid.size() << " |\n"
          << "| cache/quality.f32 | " << quality.size() << " float32 |\n"
          << "| cache/lines.f32 | " << lines.size() << " float32 |\n"
          << "| cache/zero.f32 | " << zero.size() << " float32 |\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());

        std::cout << "끝. 결과: " << result.u8string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
