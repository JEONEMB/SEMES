// 341장 촬영 기록 정리: 원본 zip 의 파일 시각으로 시간순 목록, 단일축 trial, 반복 조건, center 재촬영, 기록 공백을 만든다.
// 공통 이동량은 630점 경계 맞춤(joint_center_630 measurement.py)을 C++로 다시 재서 붙인다.
// 원본: eda_capture_inventory_20261022/build.py (+ hw-analysis EDA shots.csv 의 group/rep/taken_at 규칙)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include "struct630.hpp"
#include "zip_times.hpp"

using namespace hye;

namespace {

struct Row {
    size_t si;
    int ds;
    std::string file, source, group, trial, sha;
    bool center;
    double y, t;
    int rep;
    int64_t time = 0, dos = 0;
    int order = 0;
    double gap = kNaN, elapsed = 0;
    double dx = kNaN, dy = kNaN;
    int valid = 0;
};

cv::Mat read_img(const Shot& s, int flag) {
    std::ifstream f(fs::u8path(s.path), std::ios::binary);
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cv::Mat m = cv::imdecode(buf, flag);
    if (m.empty()) throw std::runtime_error("read failed: " + s.rel);
    return m;
}

// file_inventory.csv / manifest.csv 의 열 하나 (relative_path -> 값)
std::map<std::string, std::string> meta_column(const std::string& set, const std::string& file, const std::string& col) {
    std::ifstream in(fs::u8path(clean_dir() + "/" + set + "/_metadata/" + file));
    std::string line;
    std::getline(in, line);
    if (line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);
    auto head = split_csv(line);
    int a = -1, b = -1;
    for (int i = 0; i < (int)head.size(); ++i) { if (head[i] == "relative_path") a = i; if (head[i] == col) b = i; }
    if (a < 0 || b < 0) throw std::runtime_error("열 없음: " + col);
    std::map<std::string, std::string> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split_csv(line);
        out[f[a]] = f[b];
    }
    return out;
}

std::string source_archive(const std::string& set) {  // provenance.json 의 "source_archive"
    std::ifstream in(fs::u8path(clean_dir() + "/" + set + "/_metadata/provenance.json"));
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto k = s.find("\"source_archive\"");
    auto q1 = s.find('"', s.find(':', k) + 1), q2 = s.find('"', q1 + 1);
    return s.substr(q1 + 1, q2 - q1 - 1);
}

std::string cmd2(double v) {
    char b[16];
    std::snprintf(b, sizeof b, "%+.2f", v);
    return b;
}
std::string hms(int64_t t) { return ziptimes::format(t).substr(11); }

double sd1(const std::vector<double>& v) {
    double m = mean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}
double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }

}  // namespace

static int run();
int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

static int run() {
    begin_results();
    cv::setNumThreads(1);
    omp_set_num_threads(4);
    auto shots = load_341();
    const int N = (int)shots.size();

    // ---- 1. 파일 시각: clean 메타데이터의 source_member -> provenance 의 source_archive zip 안 NTFS 수정 시각 ----
    std::vector<Row> rows(N);
    int dos_agree = 0, ntfs_missing = 0;
    std::map<std::string, std::string> archive_note;
    for (const std::string set : {"clean_20261014", "clean_20261020"}) {
        auto member = meta_column(set, "file_inventory.csv", "source_member");
        auto sha = meta_column(set, "manifest.csv", "sha256");
        const fs::path zip = fs::u8path(clean_dir() + "/" + set) / fs::u8path(source_archive(set));
        std::string inner_name;
        auto inner = ziptimes::extract_member(zip, ".zip", &inner_name);
        std::map<std::string, ziptimes::Entry> byname;
        for (auto& e : ziptimes::central_directory(inner)) byname[e.name] = e;
        archive_note[set] = "`data/raw/" + u8(zip.filename()) + "` 안의 `" + inner_name + "` (분할 zip 마지막 조각, 중앙 디렉터리 " + std::to_string(byname.size()) + "개 항목)";
        for (int i = 0; i < N; ++i) {
            if (shots[i].set != set) continue;
            auto& r = rows[i];
            r.si = i;
            r.ds = set == "clean_20261014" ? 20261014 : 20261020;
            r.file = shots[i].rel;
            r.source = member.at(shots[i].rel);
            r.sha = sha.at(shots[i].rel);
            auto it = byname.find(r.source);
            if (it == byname.end()) throw std::runtime_error("zip 에 없음: " + r.source);
            if (it->second.ntfs < 0) { ++ntfs_missing; continue; }
            r.time = ziptimes::kst_seconds(it->second.ntfs);
            r.dos = ziptimes::dos_seconds(it->second.dos_time, it->second.dos_date);
            dos_agree += std::abs(r.dos - r.time) <= 2;
        }
    }

    // ---- 2. 그룹·반복 번호 (hw-analysis EDA shots.csv 규칙) ----
    for (int i = 0; i < N; ++i) {
        auto& r = rows[i];
        const auto& s = shots[i];
        r.center = s.condition == "center";
        r.y = s.y;
        r.t = s.t;
        if (r.ds == 20261014) {
            r.rep = r.center ? 1 : 0;
            if (r.center) r.group = "center";
            else if (s.condition == "T") r.group = std::string("Θ ") + (s.t > 0 ? "+" : "-");
            else r.group = std::string(std::lround(std::abs(s.y) / 0.05) % 2 ? "Y 0.05 홀수 " : "Y 0.1 간격 ") + (s.y > 0 ? "+" : "-");
        } else {
            r.rep = s.repeat;
            r.group = r.center ? "center" : "Y" + cmd2(s.y) + " T" + cmd2(s.t);
        }
    }

    // ---- 3. 630점 공통 이동 ----
    {
        Shot c = center_1014(shots);
        const cv::Mat center = read_img(c, cv::IMREAD_GRAYSCALE);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N; ++i) {
            try {
                auto R = s630::extract(center, read_img(shots[i], cv::IMREAD_GRAYSCALE));
                rows[i].dx = R.mean_dx();
                rows[i].dy = R.mean_dy();
                rows[i].valid = R.nvalid();
            } catch (const std::exception& e) {
#pragma omp critical
                std::cerr << "630 failed " << shots[i].rel << ": " << e.what() << "\n";
            }
        }
    }
    std::cout << "measured\n";

    // ---- 4. 시간순 정렬, trial ----
    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return std::tie(rows[a].ds, rows[a].time, rows[a].file) < std::tie(rows[b].ds, rows[b].time, rows[b].file); });
    std::map<int, int64_t> first_time;
    std::map<std::string, std::string> trial_of;
    int ntrial = 0, ties = 0;
    for (size_t k = 0; k < ord.size(); ++k) {
        auto& r = rows[ord[k]];
        if (!first_time.count(r.ds)) first_time[r.ds] = r.time;
        if (k > 0 && rows[ord[k - 1]].ds == r.ds) {
            r.gap = double(r.time - rows[ord[k - 1]].time);
            ties += r.time == rows[ord[k - 1]].time;
            r.order = rows[ord[k - 1]].order + 1;
        } else r.order = 1;
        r.elapsed = (r.time - first_time[r.ds]) / 60.0;
        if (r.ds == 20261014 && !r.center && !trial_of.count(r.group)) trial_of[r.group] = "S" + std::to_string(++ntrial);
        r.trial = r.center ? "center" : r.ds == 20261014 ? trial_of[r.group] : "M" + std::to_string(r.rep);
    }
    // 10-14 시간 블록: 그룹이 바뀌는 지점 수
    int blocks = 0;
    {
        std::string prev;
        for (int i : ord) if (rows[i].ds == 20261014) { if (rows[i].group != prev) ++blocks; prev = rows[i].group; }
    }
    std::set<std::string> uniq_sha;
    for (auto& r : rows) uniq_sha.insert(r.sha);

    Table all{{"dataset", "order_in_set", "trial", "group", "file", "taken_at", "y", "t", "rep", "elapsed_min", "gap_s", "common_dx_630", "common_dy_630", "valid_points", "sha256"}};
    long long time_sum = 0;
    for (int i : ord) {
        auto& r = rows[i];
        time_sum += r.time;
        all.add({std::to_string(r.ds), std::to_string(r.order), r.trial, r.group, r.file, ziptimes::format(r.time), fmt(r.y, 2), fmt(r.t, 2), std::to_string(r.rep),
                 fmt(r.elapsed, 3), std::isnan(r.gap) ? "" : fmt(r.gap, 0), fmt(r.dx, 6), fmt(r.dy, 6), std::to_string(r.valid), r.sha});
    }
    all.csv(out("all_341_shots.csv"));

    // ---- 5. 단일축 trial ----
    struct Trial { std::string name, group, axis; int count; int64_t start, end; double c0, c1, mn, mx, med_gap; std::string steps; bool mono; };
    std::vector<Trial> trials;
    for (int k = 1; k <= ntrial; ++k) {
        std::string nm = "S" + std::to_string(k);
        std::vector<int> ix;
        for (int i : ord) if (rows[i].trial == nm) ix.push_back(i);
        Trial T{nm, rows[ix[0]].group};
        bool isY = false;
        for (int i : ix) isY |= rows[i].y != 0;
        T.axis = isY ? "Y" : "T";
        std::vector<double> v, gaps;
        for (size_t j = 0; j < ix.size(); ++j) {
            v.push_back(isY ? rows[ix[j]].y : rows[ix[j]].t);
            if (j > 0) gaps.push_back(rows[ix[j]].gap);
        }
        T.count = (int)ix.size();
        T.start = rows[ix.front()].time;
        T.end = rows[ix.back()].time;
        T.c0 = v.front();
        T.c1 = v.back();
        T.mn = *std::min_element(v.begin(), v.end());
        T.mx = *std::max_element(v.begin(), v.end());
        std::set<double> st;
        bool up = true, dn = true;
        for (size_t j = 1; j < v.size(); ++j) {
            double d = v[j] - v[j - 1];
            st.insert(std::round(d * 100) / 100);
            up &= d >= 0;
            dn &= d <= 0;
        }
        for (double d : st) T.steps += (T.steps.empty() ? "" : ",") + cmd2(d);
        T.mono = up || dn;
        T.med_gap = median(gaps);
        trials.push_back(T);
    }
    Table trt{{"trial", "group", "axis", "count", "start", "end", "duration_min", "start_command", "end_command", "minimum", "maximum", "step_values", "monotonic", "median_gap_s"}};
    for (auto& T : trials)
        trt.add({T.name, T.group, T.axis, std::to_string(T.count), ziptimes::format(T.start), ziptimes::format(T.end), fmt((T.end - T.start) / 60.0, 3), fmt(T.c0, 2), fmt(T.c1, 2),
                 fmt(T.mn, 2), fmt(T.mx, 2), T.steps, T.mono ? "True" : "False", fmt(T.med_gap, 1)});
    trt.csv(out("single_axis_trials.csv"));

    // ---- 6. 명령 조건별 (나타난 순서) ----
    struct Cond { int ds; double y, t; std::vector<int> ix; };
    std::vector<Cond> conds;
    for (int i : ord) {
        auto& r = rows[i];
        auto it = std::find_if(conds.begin(), conds.end(), [&](const Cond& c) { return c.ds == r.ds && c.y == r.y && c.t == r.t; });
        if (it == conds.end()) conds.push_back({r.ds, r.y, r.t, {i}});
        else it->ix.push_back(i);
    }
    Table cat{{"dataset", "Y", "T", "count", "first", "last", "span_s", "repetitions", "times", "common_dx_sd_px", "common_dy_sd_px", "dx_range_px", "dy_range_px"}};
    Table rept = cat;
    struct RepStat { int c; double sdx, sdy, rdx, rdy; };
    std::vector<RepStat> reps;
    for (int c = 0; c < (int)conds.size(); ++c) {
        auto& C = conds[c];
        std::vector<double> dx, dy;
        std::string rp, tm;
        for (int i : C.ix) {
            dx.push_back(rows[i].dx);
            dy.push_back(rows[i].dy);
            rp += (rp.empty() ? "" : ",") + std::to_string(rows[i].rep);
            tm += (tm.empty() ? "" : " | ") + hms(rows[i].time);
        }
        const int n = (int)C.ix.size();
        const double rdx = *std::max_element(dx.begin(), dx.end()) - *std::min_element(dx.begin(), dx.end());
        const double rdy = *std::max_element(dy.begin(), dy.end()) - *std::min_element(dy.begin(), dy.end());
        std::vector<std::string> row{std::to_string(C.ds), fmt(C.y, 2), fmt(C.t, 2), std::to_string(n), ziptimes::format(rows[C.ix.front()].time), ziptimes::format(rows[C.ix.back()].time),
                                     fmt(double(rows[C.ix.back()].time - rows[C.ix.front()].time), 0), rp, tm, n > 1 ? fmt(sd1(dx), 6) : "", n > 1 ? fmt(sd1(dy), 6) : "", fmt(rdx, 6), fmt(rdy, 6)};
        cat.add(row);
        if (n > 1) { rept.add(row); reps.push_back({c, sd1(dx), sd1(dy), rdx, rdy}); }
    }
    cat.csv(out("all_conditions.csv"));
    rept.csv(out("repeated_conditions.csv"));

    // center, 공백
    Table cen{{"dataset", "file", "taken_at", "order_in_set", "elapsed_min", "common_dx_630", "common_dy_630", "valid_points"}};
    std::vector<int> centers;
    for (int i : ord) if (rows[i].center) {
        centers.push_back(i);
        cen.add({std::to_string(rows[i].ds), rows[i].file, ziptimes::format(rows[i].time), std::to_string(rows[i].order), fmt(rows[i].elapsed, 3), fmt(rows[i].dx, 6), fmt(rows[i].dy, 6), std::to_string(rows[i].valid)});
    }
    cen.csv(out("center_chronology.csv"));
    Table gapt{{"dataset", "order_in_set", "file", "taken_at", "gap_s"}};
    int ngaps = 0;
    for (int i : ord) if (rows[i].gap >= 120) {
        ++ngaps;
        gapt.add({std::to_string(rows[i].ds), std::to_string(rows[i].order), rows[i].file, ziptimes::format(rows[i].time), fmt(rows[i].gap, 0)});
    }
    gapt.csv(out("gaps_at_least_2min.csv"));

    // 혼합 조건: 반복 2 - 반복 1, 관측 범위
    std::vector<std::string> mixed_names;
    std::map<std::string, std::map<int, double>> piv;
    std::map<std::string, std::vector<int>> mixed_ix;
    for (int i : ord) {
        auto& r = rows[i];
        if (r.ds != 20261020 || r.center) continue;
        if (!piv.count(r.group)) mixed_names.push_back(r.group);
        piv[r.group][r.rep] = r.dx;
        mixed_ix[r.group].push_back(i);
    }
    int rep2_higher = 0;
    std::vector<double> d21;
    for (auto& nm : mixed_names) { rep2_higher += piv[nm][2] > piv[nm][1]; d21.push_back(piv[nm][2] - piv[nm][1]); }
    const double rep2_shift = median(d21);
    const double gy[4] = {.05, .1, .5, 1.}, gt[7] = {.01, .02, .03, .1, .2, 1., 2.};
    Table cov{{"y \\ t", "0.01", "0.02", "0.03", "0.1", "0.2", "1.0", "2.0"}};
    int covered = 0;
    for (double y : gy) {
        std::vector<std::string> r{fmt(y, 2)};
        for (double t : gt) {
            int n = 0;
            for (int i : ord) n += rows[i].ds == 20261020 && !rows[i].center && std::abs(rows[i].y - y) < 1e-9 && std::abs(rows[i].t - t) < 1e-9;
            covered += n > 0;
            r.push_back(std::to_string(n));
        }
        cov.add(r);
    }
    cov.csv(out("mixed_coverage_counts.csv"));

    // ---- 그림 ----
    {
        std::vector<cv::Mat> panels;
        for (int ds : {20261014, 20261020})
            for (int ax = 0; ax < 2; ++ax) {
                Chart c(std::to_string(ds) + " set - " + (ax ? "T" : "Y") + " command by file time", "minutes since first file of the set", ax ? "T command" : "Y command (mm)");
                c.W = 1100;
                c.H = 300;
                if (ds == 20261014) {
                    for (int k = 1; k <= ntrial; ++k) {
                        std::vector<double> x, y;
                        for (int i : ord) if (rows[i].trial == "S" + std::to_string(k)) { x.push_back(rows[i].elapsed); y.push_back(ax ? rows[i].t : rows[i].y); }
                        c.line(x, y, color(k - 1), "S" + std::to_string(k), true);
                    }
                } else {
                    for (int rp = 1; rp <= 3; ++rp) {
                        std::vector<double> x, y;
                        for (int i : ord) if (rows[i].ds == ds && !rows[i].center && rows[i].rep == rp) { x.push_back(rows[i].elapsed); y.push_back(ax ? rows[i].t : rows[i].y); }
                        c.scatter(x, y, color(rp - 1), "repeat " + std::to_string(rp), 4);
                    }
                }
                for (int i : centers) if (rows[i].ds == ds) {
                    std::string stem = rows[i].file.substr(rows[i].file.rfind('/') + 1);
                    c.note(rows[i].elapsed, 0, stem.substr(0, stem.rfind('.')));
                }
                c.hline(0);
                panels.push_back(c.render());
            }
        save_png("01_시간순_명령값.png", vstack(panels));
    }
    {
        std::vector<cv::Mat> panels;
        for (int ds : {20261014, 20261020}) {
            Chart c(std::to_string(ds) + ": command conditions (dot size = shots)", "Y command (mm)", "T command");
            c.W = 620;
            for (int n = 1; n <= 3; ++n) {
                std::vector<double> x, y;
                for (auto& C : conds) if (C.ds == ds && (int)C.ix.size() == n) { x.push_back(C.y); y.push_back(C.t); }
                if (!x.empty()) c.scatter(x, y, color(n - 1), std::to_string(n) + " shot" + (n > 1 ? "s" : ""), 2 + 2 * n);
            }
            panels.push_back(c.render());
        }
        save_png("02_명령조건_관측범위.png", hstack(panels));
    }
    // 반복 변동: 조건 평균 대비 dx, center 재촬영 이동
    int worst_c = -1;
    double worst_r = -1;
    for (auto& R : reps) if (!(conds[R.c].y == 0 && conds[R.c].t == 0) && R.rdx > worst_r) { worst_r = R.rdx; worst_c = R.c; }
    {
        Chart a("Mixed repeats: common dx minus condition mean", "condition index (time order)", "dx - mean (px)");
        a.W = 1100;
        a.H = 360;
        for (int rp = 1; rp <= 3; ++rp) {
            std::vector<double> x, y;
            for (size_t k = 0; k < mixed_names.size(); ++k) {
                double m = (piv[mixed_names[k]][1] + piv[mixed_names[k]][2] + piv[mixed_names[k]][3]) / 3;
                x.push_back((double)k);
                y.push_back(piv[mixed_names[k]][rp] - m);
            }
            a.scatter(x, y, color(rp - 1), "repeat " + std::to_string(rp), 4);
        }
        a.hline(0);
        for (size_t k = 0; k < mixed_names.size(); ++k)
            if (rows[conds[worst_c].ix[0]].group == mixed_names[k]) a.note((double)k, piv[mixed_names[k]][2] - (piv[mixed_names[k]][1] + piv[mixed_names[k]][2] + piv[mixed_names[k]][3]) / 3, "Y" + cmd2(conds[worst_c].y) + " T" + cmd2(conds[worst_c].t));
        Chart b("10-20 center re-shoots: shift from center1", "minutes since first file of 10-20 set", "630-point mean shift (px)");
        b.W = 1100;
        b.H = 320;
        std::vector<double> x, ydx, ydy;
        int c1 = -1;
        for (int i : centers) if (rows[i].ds == 20261020) {
            if (c1 < 0) c1 = i;
            x.push_back(rows[i].elapsed);
            ydx.push_back(rows[i].dx - rows[c1].dx);
            ydy.push_back(rows[i].dy - rows[c1].dy);
        }
        b.line(x, ydx, color(0), "dx", true).line(x, ydy, color(1), "dy", true).hline(0);
        save_png("03_반복변동_center재촬영.png", vstack({a.render(), b.render()}));
    }
    // 튀는 값 그림: 반복 dx 퍼짐이 가장 큰 혼합 조건의 세 사진, center 재촬영 세 장 (QR 가운데)
    auto qr_tiles = [&](const std::vector<int>& ix, const std::string& file, const std::string& cap) {
        const cv::Rect2d ref(s630::kBox[1][0], s630::kBox[1][1], s630::kBox[1][2] - s630::kBox[1][0], s630::kBox[1][3] - s630::kBox[1][1]);
        double base_dx = rows[ix[0]].dx, base_dy = rows[ix[0]].dy;
        std::vector<cv::Mat> tiles;
        for (size_t k = 0; k < ix.size(); ++k) {
            auto& r = rows[ix[k]];
            cv::Mat im = read_img(shots[r.si], cv::IMREAD_COLOR);
            cv::Rect2d now(ref.x + r.dx, ref.y + r.dy, ref.width, ref.height);
            // 첫 사진 기준 상자 (파랑)와 이 사진 상자 (빨강): 차이가 반복 변동
            cv::Rect2d first(ref.x + base_dx, ref.y + base_dy, ref.width, ref.height);
            cv::Rect roi = cv::Rect(now | first) + cv::Size(40, 40) - cv::Point(20, 20);
            std::string stem = r.file.substr(r.file.rfind('/') + 1);
            stem = stem.substr(0, stem.rfind('.'));
            tiles.push_back(zoom(im, roi, 420, {}, 0, k == 0 ? cap : "",
                                 {{first, color(0), ""}, {now, kRed, stem.substr(stem.size() > 3 ? stem.size() - 3 : 0) + " dx " + fmt(r.dx, 3) + " dy " + fmt(r.dy, 3)}}));
        }
        save_png(file, hstack(tiles));
    };
    const std::string wname = "Y" + cmd2(conds[worst_c].y) + "_T" + cmd2(conds[worst_c].t);
    qr_tiles(conds[worst_c].ix, "04_튀는값_반복_" + wname + ".png", wname + " repeats, dx range " + fmt(worst_r, 3) + " px");
    std::vector<int> c20;
    for (int i : centers) if (rows[i].ds == 20261020) c20.push_back(i);
    qr_tiles(c20, "05_튀는값_center재촬영.png", "10-20 center1..3");

    // 튀는 값 표: dx 퍼짐 상위 5 반복 조건
    std::vector<RepStat> sorted = reps;
    std::stable_sort(sorted.begin(), sorted.end(), [](const RepStat& a, const RepStat& b) { return a.rdx > b.rdx; });
    Table outl{{"조건 (Y, T)", "장수", "시각", "반복별 공통 dx (px)", "dx 범위 (px)", "dx 표준편차", "dy 범위 (px)"}};
    for (size_t k = 0; k < std::min<size_t>(5, sorted.size()); ++k) {
        auto& C = conds[sorted[k].c];
        std::string tm, dxs;
        for (int i : C.ix) { tm += (tm.empty() ? "" : " / ") + hms(rows[i].time); dxs += (dxs.empty() ? "" : " / ") + fmt(rows[i].dx, 3); }
        outl.add({fmts(C.y, 2) + ", " + fmts(C.t, 2), std::to_string(C.ix.size()), tm, dxs, fmt(sorted[k].rdx, 4), fmt(sorted[k].sdx, 4), fmt(sorted[k].rdy, 4)});
    }

    // ---- 원래 결과와 대조 ----
    // 원래 값: eda_capture_inventory_20261022/audit.json, single_axis_trials.csv, center_chronology.csv,
    //          repeated_conditions.csv, gaps_at_least_2min.csv, all_341_shots.csv(taken_at 합), report.html(반복 2 문단)
    Parity par;
    par.add("사진 수", 341, N, 0, 0);
    par.add("단일축 trial 수", 6, (double)trials.size(), 0, 0);
    par.add("10-14 시간 블록 수 (center 포함)", 7, blocks, 0, 0);
    par.add("혼합 조건 수", 19, (double)mixed_names.size(), 0, 0);
    par.add("반복 조건 수 (2장 이상)", 20, (double)reps.size(), 0, 0);
    par.add("center 수", 4, (double)centers.size(), 0, 0);
    par.add("고유 해시 수", 341, (double)uniq_sha.size(), 0, 0);
    par.add("같은 시각 겹침", 0, ties, 0, 0);
    par.add("2분 이상 공백 수", 12, ngaps, 0, 0);
    par.add("341장 시각 합 (초, KST)", 610830048058.0, (double)time_sum, 0, 0);
    {
        const char* o[6][2] = {{"2026-10-04 12:14:14", "2026-10-04 12:18:18"}, {"2026-10-04 12:20:56", "2026-10-04 12:23:31"}, {"2026-10-04 12:26:30", "2026-10-04 12:39:11"},
                               {"2026-10-04 12:40:27", "2026-10-04 12:46:12"}, {"2026-10-04 13:01:26", "2026-10-04 13:04:10"}, {"2026-10-04 13:09:00", "2026-10-04 13:11:04"}};
        const char* og[6] = {"Y 0.1 간격 +", "Y 0.1 간격 -", "Θ +", "Θ -", "Y 0.05 홀수 +", "Y 0.05 홀수 -"};
        for (int k = 0; k < 6 && k < (int)trials.size(); ++k) {
            std::string a = std::string(og[k]) + " " + o[k][0] + " ~ " + std::string(o[k][1]).substr(11);
            std::string b = trials[k].group + " " + ziptimes::format(trials[k].start) + " ~ " + ziptimes::format(trials[k].end).substr(11);
            par.add_text(trials[k].name + " 그룹·시작~끝", a, b, a == b);
        }
        const double omed[6] = {8, 7, 6, 3, 5, 4};
        std::string a, b;
        for (int k = 0; k < 6 && k < (int)trials.size(); ++k) { a += (k ? ", " : "") + fmt(omed[k], 0); b += (k ? ", " : "") + fmt(trials[k].med_gap, 0); }
        par.add_text("S1~S6 사진 간격 중앙값 (초)", a, b, a == b);
    }
    {
        std::string a = "12:13:03, 19:33:08, 19:43:08, 21:54:26", b;
        for (size_t k = 0; k < centers.size(); ++k) b += (k ? ", " : "") + hms(rows[centers[k]].time);
        par.add_text("center 4장 시각", a, b, a == b);
    }
    par.add("반복 2 > 반복 1 혼합 조건 수", 17, rep2_higher, 0, 0);
    par.add("반복 2-1 dx 중앙값 (px)", 0.314461, rep2_shift, 0.001);
    par.add("10-20 center1 공통 dx (px)", -0.213815, rows[centers[1]].dx, 0.001);
    par.add("10-20 center2 공통 dx (px)", -0.289594, rows[centers[2]].dx, 0.001);
    par.add("10-20 center3 공통 dy (px)", 0.494178, rows[centers[3]].dy, 0.001);
    par.add("(0,0) 3장 dx 표준편차 (px)", 0.038565, reps[0].sdx, 0.001);
    par.add("Y+0.10 T+0.10 dx 표준편차 (px)", 0.334853, [&] { for (auto& R : reps) if (conds[R.c].y == 0.1 && conds[R.c].t == 0.1) return R.sdx; return kNaN; }(), 0.001);
    int vsum = 0;
    for (auto& r : rows) vsum += r.valid;
    par.add("유효 점 합 (341 x 630 중)", 213843, vsum, 0, 0);

    // ---- 결과.md ----
    const std::string time_note =
        "시각 출처: `data/clean/*/_metadata/file_inventory.csv`(relative_path, source_member, bytes, sha256, zip_crc32)와 `manifest.csv`, `provenance.json`에는 사진별 시각이 없다. "
        "clean 사본의 파일 시스템 수정 시각은 2026-10-28 추출 시각이라 촬영 순서가 아니다. 원래 실험은 hw-analysis EDA `shots.csv`의 `taken_at`을 썼고, 그 값은 "
        "`hw-analysis/data/hw/bmp`에 풀린 BMP의 파일 수정 시각(Asia/Seoul, 초 아래 버림)이었다(원래 audit: 341/341 일치). 그 BMP들은 원본 zip에서 풀렸고, zip 중앙 디렉터리에 "
        "사진마다 NTFS 수정 시각(100 ns, UTC)이 들어 있다. 그래서 C++는 clean의 `provenance.json` `source_archive`가 가리키는 zip을 열어, `file_inventory.csv`의 `source_member` 이름으로 "
        "그 시각을 찾고 +9시간 한 뒤 초 아래를 버렸다. 이는 '오직 data/clean' 입력 규칙의 예외이며, 사진 바이트는 data/clean에서만 읽었다.";
    std::ostringstream R;
    R << "# 결과: 341장 촬영 기록 정리 - 파일 시각 정렬\n\n"
      << "입력: `data/clean/clean_20261014` 281장 + `clean_20261020` 60장. 공통 이동은 center.bmp 기준 630점 경계 맞춤 평균(px)이다.\n\n"
      << time_note << "\n\n"
      << "| 묶음 | 시각을 읽은 곳 |\n|---|---|\n| clean_20261014 | " << archive_note["clean_20261014"] << " |\n| clean_20261020 | " << archive_note["clean_20261020"] << " |\n\n"
      << "zip 항목의 DOS 시각(지역 시각, 2초 단위)과 NTFS 시각(+9시간)이 2초 안으로 맞은 사진: " << dos_agree << "/" << N << ", NTFS 시각 없는 사진: " << ntfs_missing << ".\n\n"
      << "## 단일축 trial (10-14)\n\nS1~S6은 파일명 그룹(Y 0.1 간격, Y 0.05 홀수, T 부호)을 시간순으로 번호 붙인 것이고, 각 그룹은 끊기지 않은 시간 블록이다(10-14 블록 " << blocks << "개 = center + 6).\n\n"
      << trt.md()
      << "\n## 혼합 조건 관측 범위 (10-20, 사진 수)\n\n" << cov.md() << "\n양수 Y 4수준 x T 7수준 28칸 중 " << covered << "칸만 찍혔다.\n"
      << "\n## 같은 명령 반복 (" << reps.size() << "조건)\n\n" << rept.md()
      << "\n혼합 " << mixed_names.size() << "조건 중 " << rep2_higher << "조건에서 반복 2의 공통 dx가 반복 1보다 크고, 반복 2-1 차이의 중앙값은 " << fmt(rep2_shift, 4) << " px다.\n"
      << "\n## center 재촬영\n\n" << cen.md()
      << "\n## 2분 이상 기록 공백 (" << ngaps << "개)\n\n" << gapt.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- 원래 report.html(검색 표)과 matplotlib 그림 스타일. 같은 내용을 CSV와 PNG로 만들었다.\n"
      << "- 630점 측정 자산(모서리 위치·모양)은 `실험/src/fixed630.hpp` 상수로 옮겼다(`실험/tools/gen_fixed630.py`). 자산을 만든 격자 해독은 옮기지 않았다.\n"
      << "\n## 파일\n\n- `all_341_shots.csv`: 시간순 341장 (trial, 그룹, 시각, 간격, 공통 이동, 유효 점 수, sha256)\n- `single_axis_trials.csv`, `all_conditions.csv`, `repeated_conditions.csv`, "
      << "`center_chronology.csv`, `gaps_at_least_2min.csv`, `mixed_coverage_counts.csv`\n- `이미지/01~05`\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    const auto& W = conds[worst_c];
    double kT = 0;  // 10-14 T 스캔의 공통 dx 기울기 (px/deg)
    {
        std::vector<double> x, y;
        for (auto& r : rows) if (r.ds == 20261014 && (r.t != 0 || r.center)) { x.push_back(r.t); y.push_back(r.dx); }
        double mx = mean(x), my = mean(y), sxy = 0, sxx = 0;
        for (size_t i = 0; i < x.size(); ++i) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
        kT = sxy / sxx;
    }
    std::ostringstream P;
    P << "# 보고서: 341장 촬영 기록 정리 - 파일 시각 정렬 (10-22 17:43)\n\n"
      << "## 문제\n\n341장의 촬영 순서와 trial 구성, 같은 명령 반복을 정리해야 했다.\n\n"
      << "## 해결 방법\n\n원본 zip에 남은 사진별 파일 시각으로 341장을 시간순으로 놓고, 파일명 그룹이 끊기지 않는 시간 블록을 trial로, 같은 명령 사진을 반복 조건으로 묶었다.\n\n"
      << "## 결과\n\n10-14는 단일축 trial " << trials.size() << "개와 center 1장, 10-20은 혼합 " << mixed_names.size() << "조건 x 3장과 center 3장이고, 같은 명령 반복은 "
      << reps.size() << "조건이다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 사진 / 고유 해시 | " << N << " / " << uniq_sha.size() << " |\n"
      << "| 단일축 trial (S1~S6) | " << trials.size() << " (Y 4, T 2) |\n"
      << "| 혼합 조건 / 관측 칸 | " << mixed_names.size() << " / 28 |\n"
      << "| 반복 조건 (혼합 19 + center 1) | " << reps.size() << " |\n"
      << "| center | " << centers.size() << " (10-14 1, 10-20 3) |\n"
      << "| 2분 이상 공백 / 최대 공백 | " << ngaps << " / " << fmt([&] { double m = 0; for (auto& r : rows) if (!std::isnan(r.gap)) m = std::max(m, r.gap); return m; }(), 0) << " 초 |\n"
      << "| 같은 시각 겹침 | " << ties << " |\n"
      << "| 시각 출처 | 원본 zip 중앙 디렉터리의 NTFS 수정 시각 (clean 메타데이터에는 시각 없음) |\n\n"
      << trt.md() << "\n"
      << img_md("01_시간순_명령값.png", "시간순 명령값") << "\n\n" << img_md("02_명령조건_관측범위.png", "관측 범위") << "\n\n"
      << "## 튀는 값\n\n같은 명령 반복에서 공통 dx가 가장 크게 흩어진 조건은 Y" << cmd2(W.y) << " T" << cmd2(W.t) << "이고, " << W.ix.size() << "장을 "
      << fmt(double(rows[W.ix.back()].time - rows[W.ix.front()].time), 0) << "초 안에 찍었는데 dx 범위가 " << fmt(worst_r, 3) << " px로 T 0.01이 만드는 이동(" << fmt(kT / 100, 3) << " px)의 "
      << fmt(worst_r / (kT / 100), 2) << "배다.\n\n"
      << outl.md() << "\n" << img_md("04_튀는값_반복_" + wname + ".png", "반복 튀는 값") << "\n\n" << img_md("05_튀는값_center재촬영.png", "center 재촬영") << "\n\n"
      << "파란 상자는 그 조건 첫 사진의 가운데 QR 위치, 빨간 상자는 이 사진의 위치다(630점 평균 이동만큼 옮긴 center 상자). 두 번째 그림은 같은 방식으로 본 10-20 center1~3이다(center1 기준).\n\n"
      << img_md("03_반복변동_center재촬영.png", "반복 변동") << "\n\n"
      << "## 한줄평\n\n파일 시각은 촬영 장비 기록이 아니라 파일 수정 시각이지만 341장 모두 서로 다르고(겹침 " << ties << ") 파일명 그룹이 7개 연속 블록으로 끊기지 않아 trial 경계를 시간으로 확정할 수 있었고, "
      << "같은 명령도 반복 사이 공통 dx가 최대 " << fmt(worst_r, 3) << " px 달라 반복을 같은 자세로 볼 수 없다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `eda_capture_inventory_20261022/`의 `audit.json`, `single_axis_trials.csv`, `center_chronology.csv`, `repeated_conditions.csv`, "
      << "`gaps_at_least_2min.csv`, `all_341_shots.csv`(taken_at 합), `report.html`에서 가져왔다. " << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md()
      << "\n" << time_note << "\n";
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
