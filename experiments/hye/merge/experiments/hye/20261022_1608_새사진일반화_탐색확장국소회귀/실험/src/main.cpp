// 새 사진 일반화: NCC 탐색 확장(±130 -> ±260/512)으로 630점 측정을 복구하고, 20261020 조건까지 교정에 넣은
// 전체점 회귀(global/local ForwardGLS)를 중첩 그룹 OOF 로 고른다.
// 원본: 00000.legacies/codex_experiments_20261014/joint_yt_generalization_20261022/
//   measurements/extractor_scipy.py, prepare.py, recover.py, data.py, forward_model.py, joint_model.py, evaluate.py
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include "extract630.hpp"
#include "protocol341.hpp"
#include "ytmodel.hpp"

using namespace hye;

namespace {

struct Img {
    Shot shot;
    int fold = 0;
    std::string group, scope, name;
    bool isnew = false;
    e630::Measure m;
    std::vector<double> D;  // 1260
    double h = 0, hy = 0;
    int nvalid = 0;
};

// 개발용: HYE_DEV_CACHE 가 있으면 측정을 저장·재사용한다(최종 실행에서는 쓰지 않는다)
bool load_cache(const std::string& p, std::vector<Img>& v, std::vector<cv::Point2d>& ref) {
    std::ifstream f(fs::u8path(p), std::ios::binary);
    if (!f) return false;
    ref.resize(630);
    f.read((char*)ref.data(), sizeof(cv::Point2d) * 630);
    for (auto& im : v) {
        auto& m = im.m;
        f.read((char*)&m.ncc_min, sizeof m.ncc_min);
        f.read((char*)&m.pad, sizeof m.pad);
        f.read((char*)m.shift, sizeof m.shift);
        m.ok = true;
        m.cur.resize(630); m.q.resize(630); m.valid.resize(630); m.why.resize(630);
        f.read((char*)m.cur.data(), sizeof(cv::Point2d) * 630);
        f.read((char*)m.q.data(), sizeof(double) * 5 * 630);
        f.read((char*)m.valid.data(), 630);
        f.read((char*)m.why.data(), sizeof(int) * 630);
    }
    return (bool)f;
}
void save_cache(const std::string& p, const std::vector<Img>& v, const std::vector<cv::Point2d>& ref) {
    std::ofstream f(fs::u8path(p), std::ios::binary);
    f.write((const char*)ref.data(), sizeof(cv::Point2d) * 630);
    for (auto& im : v) {
        auto& m = im.m;
        f.write((const char*)&m.ncc_min, sizeof m.ncc_min);
        f.write((const char*)&m.pad, sizeof m.pad);
        f.write((const char*)m.shift, sizeof m.shift);
        f.write((const char*)m.cur.data(), sizeof(cv::Point2d) * 630);
        f.write((const char*)m.q.data(), sizeof(double) * 5 * 630);
        f.write((const char*)m.valid.data(), 630);
        f.write((const char*)m.why.data(), sizeof(int) * 630);
    }
}

std::string short_name(const Img& im) {
    std::string r = im.shot.rel;
    r = r.substr(r.rfind('/') + 1);
    r = r.substr(0, r.rfind('.'));
    std::string out;
    for (char c : r) {
        if (c == ' ') out += '_';
        else if (c == '(') out += 'r';
        else if (c != ')') out += c;
    }
    if (im.shot.set == "clean_20261020" && im.scope == "center") out = "1020_" + out;
    return out;
}

using ytm::Config;

// hye::read_gray 는 cv::imread 라 한글 경로(clean_20261020/기준 위치)를 못 읽는다. 바이트로 읽어 imdecode 한다.
cv::Mat imread_u8(const Shot& s, int flag) {
    std::ifstream f(fs::u8path(s.path), std::ios::binary);
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cv::Mat m = cv::imdecode(buf, flag);
    if (m.empty()) throw std::runtime_error("읽기 실패: " + s.path);
    return m;
}

struct Rank {
    int fails;
    double worst, mean;
    bool operator<(const Rank& o) const { return std::tie(fails, worst, mean) < std::tie(o.fails, o.worst, o.mean); }
};

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
    omp_set_num_threads(4);
    cv::setNumThreads(1);
    begin_results();
    // ---- 사진 목록: 원래 평가 순서(0..340)로 data/clean 파일을 맞춘다 ----
    auto shots = load_341();
    std::map<std::string, Shot> by;
    for (auto& s : shots) by[s.set + "/" + s.rel] = s;
    std::vector<Img> imgs(341);
    for (int i = 0; i < 341; ++i) {
        auto& r = p341::kRows[i];
        auto it = by.find(std::string(r.set) + "/" + r.rel);
        if (it == by.end()) throw std::runtime_error(std::string("사진 없음: ") + r.rel);
        imgs[i].shot = it->second;
        imgs[i].fold = r.fold;
        imgs[i].group = r.cv_group;
        imgs[i].scope = r.scope;
        imgs[i].isnew = std::string(r.set) == "clean_20261020";
    }
    for (auto& im : imgs) im.name = short_name(im);

    // ---- 630점 측정 ----
    const cv::Mat center = imread_u8(center_1014(shots), cv::IMREAD_GRAYSCALE);
    std::vector<cv::Point2d> ref;
    const char* cache = std::getenv("HYE_DEV_CACHE");
    bool cached = cache && load_cache(cache, imgs, ref);
    if (!cached) {
        auto mc = e630::measure(center, center);  // 기준 좌표 = center 사진 자신의 측정
        ref = mc.cur;
        std::cout << "measuring 341..." << std::endl;
        std::vector<std::string> errs(341);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < 341; ++i) {
            try {
                imgs[i].m = e630::measure(imread_u8(imgs[i].shot, cv::IMREAD_GRAYSCALE), center);
            } catch (const std::exception& e) {
                errs[i] = e.what();
            }
        }
        for (int i = 0; i < 341; ++i)
            if (!errs[i].empty()) throw std::runtime_error("측정 오류 " + imgs[i].shot.rel + ": " + errs[i]);
        if (cache) save_cache(cache, imgs, ref);
    }
    double ref_dev = 0;
    for (int k = 0; k < 630; ++k)
        ref_dev = std::max(ref_dev, std::max(std::abs(ref[k].x - c630::kCorners[k].px), std::abs(ref[k].y - c630::kCorners[k].py)));
    ytm::Data data;
    data.n = 341;
    int recovered = 0, failed = 0, same_valid = 0;
    double max_dh = 0, max_dhy = 0, valid_sum = 0, ref_valid_sum = 0;
    for (int i = 0; i < 341; ++i) {
        auto& im = imgs[i];
        if (!im.m.ok) { ++failed; throw std::runtime_error("NCC 실패: " + im.shot.rel); }
        im.D.resize(1260);
        double sx = 0, sy = 0;
        for (int k = 0; k < 630; ++k) {
            im.D[2 * k] = im.m.cur[k].x - ref[k].x;
            im.D[2 * k + 1] = im.m.cur[k].y - ref[k].y;
            sx += im.D[2 * k];
            sy += im.D[2 * k + 1];
            im.nvalid += im.m.valid[k];
        }
        im.h = sx / 630;
        im.hy = sy / 630;
        recovered += im.m.pad > 130;
        same_valid += im.nvalid == p341::kRows[i].ref_valid;
        valid_sum += im.nvalid;
        ref_valid_sum += p341::kRows[i].ref_valid;
        max_dh = std::max(max_dh, std::abs(im.h - p341::kRows[i].ref_mean_dx));
        max_dhy = std::max(max_dhy, std::abs(im.hy - p341::kRows[i].ref_mean_dy));
        data.D.push_back(im.D);
        data.h.push_back(im.h);
        data.y.push_back(im.shot.y);
        data.t.push_back(im.shot.t);
    }
    std::cout << "measured; ref_dev " << ref_dev << " max_dh " << max_dh << " recovered " << recovered << "\n";

    // ---- 설정과 중첩 선택 ----
    std::vector<Config> configs;
    for (std::string f : {"raw1260", "raw_shift"})
        for (double s : {.001, .01, .1, .3}) configs.push_back({"global", f, s, 0});
    for (double w : {20., 100.})
        for (double s : {.01, .1}) configs.push_back({"local", "raw_shift", s, w});

    std::vector<std::string> strat(341), groups(341);
    for (int i = 0; i < 341; ++i) {
        strat[i] = imgs[i].isnew ? "new_" + imgs[i].scope : "old";
        groups[i] = imgs[i].group;
    }
    auto metric_of = [&](const std::vector<int>& ix, const std::vector<double>& py, const std::vector<double>& pt) {
        std::vector<double> y, t;
        for (int i : ix) { y.push_back(data.y[i]); t.push_back(data.t[i]); }
        return ytm::metric(y, py, t, pt);
    };
    struct Sel {
        uint32_t seed;
        std::vector<std::pair<int, long long>> inner;  // 검증 장수, 인덱스 합
        std::vector<Rank> ranks;
        int best;
    };
    auto select = [&](const std::vector<int>& tr, uint32_t seed) {
        std::vector<std::string> sy, sg;
        for (int i : tr) { sy.push_back(strat[i]); sg.push_back(groups[i]); }
        auto tests = ytm::sgkf_test(sy, sg, 3, seed);
        Sel s;
        s.seed = seed;
        std::vector<std::pair<std::vector<int>, std::vector<int>>> folds;
        for (auto& te : tests) {
            std::set<int> ts(te.begin(), te.end());
            std::vector<int> it, iv;
            long long sum = 0;
            for (int k = 0; k < (int)tr.size(); ++k) {
                if (ts.count(k)) { sum += tr[k]; if (imgs[tr[k]].isnew) iv.push_back(tr[k]); }
                else it.push_back(tr[k]);
            }
            s.inner.push_back({(int)te.size(), sum});
            folds.push_back({it, iv});
        }
        for (auto& c : configs) {
            int fails = 0;
            double worst = 0, sum = 0;
            int cnt = 0;
            for (auto& [it, iv] : folds) {
                std::vector<double> py, pt;
                ytm::fit_predict(data, c, it, iv, py, pt);
                for (size_t k = 0; k < iv.size(); ++k) {
                    double ey = std::abs(data.y[iv[k]] - py[k]), et = std::abs(data.t[iv[k]] - pt[k]);
                    fails += (ey > .09 + 1e-12 || et > .09 + 1e-12);
                    worst = std::max({worst, ey, et});
                    sum += ey + et;
                    cnt += 2;
                }
            }
            s.ranks.push_back({fails, worst, sum / cnt});
        }
        s.best = 0;
        for (int k = 1; k < (int)configs.size(); ++k)
            if (s.ranks[k] < s.ranks[s.best]) s.best = k;
        return s;
    };

    struct Pred { int i; double py, pt; std::string method, protocol; int fold; };
    std::vector<Pred> preds;
    std::vector<Sel> sels;
    std::vector<int> oof_fold_of(341, -1);
    std::vector<std::string> sel_cfg_of(341);
    const std::vector<std::pair<std::string, Config>> controls{{"raw_global", {"global", "raw1260", .1, 0}},
                                                               {"interaction_global", {"global", "raw_shift", .01, 0}},
                                                               {"interaction_local", {"local", "raw_shift", .01, 20.}}};
    for (int f = 0; f < 4; ++f) {
        std::vector<int> tr, te;
        for (int i = 0; i < 341; ++i) (imgs[i].fold == f ? te : tr).push_back(i);
        auto s = select(tr, 20261022 + f);
        sels.push_back(s);
        std::vector<double> py, pt;
        ytm::fit_predict(data, configs[s.best], tr, te, py, pt);
        for (size_t k = 0; k < te.size(); ++k) {
            preds.push_back({te[k], py[k], pt[k], "selected_joint", "nested_group_oof", f});
            oof_fold_of[te[k]] = f;
            sel_cfg_of[te[k]] = configs[s.best].str();
        }
        for (auto& [nm, c] : controls) {
            ytm::fit_predict(data, c, tr, te, py, pt);
            for (size_t k = 0; k < te.size(); ++k) preds.push_back({te[k], py[k], pt[k], nm, "fixed_config_group_oof", f});
        }
        std::cout << "outer " << f << " selected " << configs[s.best].str() << "\n";
    }
    std::vector<int> all(341);
    std::iota(all.begin(), all.end(), 0);
    auto fs_ = select(all, 20261101);
    sels.push_back(fs_);
    {
        std::vector<double> py, pt;
        ytm::fit_predict(data, configs[fs_.best], all, all, py, pt);
        for (int i = 0; i < 341; ++i) preds.push_back({i, py[i], pt[i], "selected_joint", "resub341", -1});
    }
    std::vector<int> oldi, newi;
    for (int i = 0; i < 341; ++i) (imgs[i].isnew ? newi : oldi).push_back(i);
    for (std::string feat : {"raw1260", "raw_shift"}) {
        std::vector<double> py, pt;
        ytm::fit_predict(data, {"global", feat, .01, 0}, oldi, newi, py, pt);
        for (size_t k = 0; k < newi.size(); ++k) preds.push_back({newi[k], py[k], pt[k], feat, "old_only_transfer", -1});
    }

    // ---- 요약 ----
    auto summarize = [&](const std::string& method, const std::string& protocol, int ds) {
        std::vector<int> ix;
        std::vector<double> py, pt;
        for (auto& p : preds)
            if (p.method == method && p.protocol == protocol && (ds == 0 || (ds == 1020) == imgs[p.i].isnew)) {
                ix.push_back(p.i); py.push_back(p.py); pt.push_back(p.pt);
            }
        return metric_of(ix, py, pt);
    };
    const std::vector<std::pair<std::string, std::string>> mp{{"selected_joint", "nested_group_oof"}, {"raw_global", "fixed_config_group_oof"},
                                                              {"interaction_global", "fixed_config_group_oof"}, {"interaction_local", "fixed_config_group_oof"},
                                                              {"selected_joint", "resub341"}, {"raw1260", "old_only_transfer"}, {"raw_shift", "old_only_transfer"}};
    Table sum{{"방법", "평가", "자료", "장수", "통과", "Y 평균 |오차|", "Y 최대", "T 평균 |오차|", "T 최대"}};
    std::map<std::string, ytm::Metric> M;
    for (auto& [me, pr] : mp)
        for (int ds : {0, 1014, 1020}) {
            if (pr == "old_only_transfer" && ds != 1020) continue;
            auto m = summarize(me, pr, ds);
            M[me + "|" + pr + "|" + std::to_string(ds)] = m;
            sum.add({me, pr, ds == 0 ? "전체" : ds == 1014 ? "20261014" : "20261020", std::to_string(m.n), std::to_string(m.passed), fmt(m.ymae, 4), fmt(m.ymax, 4),
                     fmt(m.tmae, 4), fmt(m.tmax, 4)});
        }
    auto& Mn = M["selected_joint|nested_group_oof|0"];

    // 예측 CSV
    Table pc{{"index", "file", "Y", "T", "scope", "fold", "method", "protocol", "predY", "predT", "eY", "eT", "pass"}};
    for (auto& p : preds) {
        auto& im = imgs[p.i];
        double ey = std::abs(p.py - im.shot.y), et = std::abs(p.pt - im.shot.t);
        pc.add({std::to_string(p.i), im.shot.name(), fmt(im.shot.y, 2), fmt(im.shot.t, 2), im.scope, std::to_string(p.fold), p.method, p.protocol, fmt(p.py, 5),
                fmt(p.pt, 5), fmt(ey, 5), fmt(et, 5), (ey <= .09 + 1e-12 && et <= .09 + 1e-12) ? "1" : "0"});
    }
    pc.csv(out("predictions.csv"));
    Table ms{{"index", "file", "Y", "T", "scope", "fold", "pad", "ncc_min", "valid", "mean_dx", "mean_dy", "orig_mean_dx", "orig_valid"}};
    for (int i = 0; i < 341; ++i) {
        auto& im = imgs[i];
        ms.add({std::to_string(i), im.shot.name(), fmt(im.shot.y, 2), fmt(im.shot.t, 2), im.scope, std::to_string(im.fold), std::to_string(im.m.pad), fmt(im.m.ncc_min, 4),
                std::to_string(im.nvalid), fmt(im.h, 5), fmt(im.hy, 5), fmt(p341::kRows[i].ref_mean_dx, 5), std::to_string(p341::kRows[i].ref_valid)});
    }
    ms.csv(out("measurements_summary.csv"));

    // 선택 표
    Table st{{"바깥 fold", "seed", "내부 검증 장수", "선택 설정", "실패 수", "최악 |오차|", "평균 |오차|"}};
    for (size_t k = 0; k < sels.size(); ++k) {
        auto& s = sels[k];
        std::string in;
        for (auto& q : s.inner) in += (in.empty() ? "" : " / ") + std::to_string(q.first);
        auto& r = s.ranks[s.best];
        st.add({k < 4 ? std::to_string(k) : "전체(최종)", std::to_string(s.seed), in, configs[s.best].str(), std::to_string(r.fails), fmt(r.worst, 4), fmt(r.mean, 4)});
    }
    Table cs{{"설정", "fold0 실패/최악", "fold1", "fold2", "fold3", "최종"}};
    for (size_t c = 0; c < configs.size(); ++c) {
        std::vector<std::string> row{configs[c].str()};
        for (auto& s : sels) row.push_back(std::to_string(s.ranks[c].fails) + " / " + fmt(s.ranks[c].worst, 3) + (s.best == (int)c ? " *" : ""));
        cs.add(row);
    }

    // ---- 그림 ----
    std::vector<double> ty, py, tt, pt, ey, et;
    std::vector<std::string> names;
    std::vector<int> evi;
    for (auto& p : preds)
        if (p.method == "selected_joint" && p.protocol == "nested_group_oof") {
            auto& im = imgs[p.i];
            ty.push_back(im.shot.y); py.push_back(p.py); tt.push_back(im.shot.t); pt.push_back(p.pt);
            ey.push_back(p.py - im.shot.y); et.push_back(p.pt - im.shot.t);
            names.push_back(im.name);
            evi.push_back(p.i);
        }
    pred_vs_truth("02_중첩OOF341_예측오차.png", "Nested group OOF 341", ty, py, tt, pt, names);
    bar_chart("01_방법별_통과장수341.png", "Pass count (|eY|,|eT|<=0.09) of 341", {"nested", "raw-g", "inter-g", "inter-l", "resub"},
              {(double)Mn.passed, (double)M["raw_global|fixed_config_group_oof|0"].passed, (double)M["interaction_global|fixed_config_group_oof|0"].passed,
               (double)M["interaction_local|fixed_config_group_oof|0"].passed, (double)M["selected_joint|resub341|0"].passed},
              "images passed", 0, 341);
    bar_chart("03_20261020_60장_통과.png", "20261020 mixed 60: old-only vs calibrated", {"old raw", "old shift", "nested", "inter-g", "inter-l"},
              {(double)M["raw1260|old_only_transfer|1020"].passed, (double)M["raw_shift|old_only_transfer|1020"].passed, (double)M["selected_joint|nested_group_oof|1020"].passed,
               (double)M["interaction_global|fixed_config_group_oof|1020"].passed, (double)M["interaction_local|fixed_config_group_oof|1020"].passed},
              "images passed", 2, 60);
    {
        std::vector<double> x0, y0, x1, y1;
        for (int i = 0; i < 341; ++i) (imgs[i].m.pad > 130 ? x1 : x0).push_back(imgs[i].h), (imgs[i].m.pad > 130 ? y1 : y0).push_back(imgs[i].m.ncc_min);
        Chart c("NCC min score vs mean x shift (pad>130 = recovered)", "mean x shift of 630 points (px)", "min NCC of 3 QR");
        c.scatter(x0, y0, color(0), "pad 130").scatter(x1, y1, kRed, "recovered (pad 260+)", 5).hline(0.6, kRed);
        c.save("04_NCC탐색확장_복구.png");
    }
    // 튀는 값 그림: 세 QR 확대. 초록 = 기준 QR 상자, 빨강 = NCC 로 옮긴 상자, 점 = 630점 측정(무효는 빨강)
    auto outlier_fig = [&](int i, const std::string& file, const std::string& cap) {
        auto& im = imgs[i];
        cv::Mat col = imread_u8(im.shot, cv::IMREAD_COLOR);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            const int* b = e630::kBoxes[q];
            cv::Rect2d refb(b[0], b[1], b[2] - b[0], b[3] - b[1]);
            cv::Rect2d now(refb.x + im.m.shift[q].x, refb.y + im.m.shift[q].y, refb.width, refb.height);
            cv::Rect roi = cv::Rect(refb | now) + cv::Size(40, 40) - cv::Point(20, 20);
            std::vector<Mark> mk;
            for (int k = 0; k < 630; ++k)
                if (c630::kCorners[k].qr == q + 1 && !im.m.valid[k]) mk.push_back({im.m.cur[k], {0, 0}, "", kRed});
            tiles.push_back(zoom(col, roi, 420, mk, 0, q == 0 ? cap : "",
                                 {{refb, kGreen, ""}, {now, kRed, "dx " + std::to_string(im.m.shift[q].x) + " dy " + std::to_string(im.m.shift[q].y)}}));
        }
        save_png(file, hstack(tiles));
    };
    int wy = (int)worst(ey, 1)[0], wt = (int)worst(et, 1)[0];
    std::string fy = "05_튀는값_Y_" + names[wy] + ".png", ft = "06_튀는값_T_" + names[wt] + ".png";
    outlier_fig(evi[wy], fy, names[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(evi[wt], ft, names[wt] + " T err " + fmts(et[wt], 3) + " deg");
    // 학습 이동 범위와의 거리: 시험 사진 h 와 같은 fold 학습 사진 h 의 최소 차
    auto gap = [&](int i) {
        double g = 1e9;
        for (int j = 0; j < 341; ++j)
            if (imgs[j].fold != imgs[i].fold) g = std::min(g, std::abs(imgs[j].h - imgs[i].h));
        return g;
    };
    Table outl{{"구분", "사진", "fold", "정답 Y", "추정 Y", "정답 T", "추정 T", "평균 x 이동 (px)", "유효점", "NCC 최소", "탐색 pad", "학습 사진과 이동 차 (px)", "선택 설정"}};
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 5)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t k : idx) {
            auto& im = imgs[evi[k]];
            outl.add({lab, names[k], std::to_string(im.fold), fmt(ty[k], 2), fmt(py[k], 3), fmt(tt[k], 2), fmt(pt[k], 3), fmt(im.h, 2), std::to_string(im.nvalid),
                      fmt(im.m.ncc_min, 3), std::to_string(im.m.pad), fmt(gap(evi[k]), 2), sel_cfg_of[evi[k]]});
        }
    // 실패 사진 분포: 20261014 / 20261020
    int fail_old = 0, fail_new = 0, fail_y = 0, fail_t = 0;
    for (size_t k = 0; k < ey.size(); ++k) {
        bool fy_ = std::abs(ey[k]) > .09 + 1e-12, ft_ = std::abs(et[k]) > .09 + 1e-12;
        if (fy_ || ft_) (imgs[evi[k]].isnew ? fail_new : fail_old)++;
        fail_y += fy_;
        fail_t += ft_;
    }
    auto& Mo = M["raw1260|old_only_transfer|1020"];
    auto& Mnew = M["selected_joint|nested_group_oof|1020"];

    // ---- 원래 결과와 대조 ----
    // 원래 값 출처: summary.csv, selection.json, final_config.json, measurements/audit.json (joint_yt_generalization_20261022)
    Parity par;
    int ref_close = 0;
    for (int k = 0; k < 630; ++k)
        ref_close += std::max(std::abs(ref[k].x - c630::kCorners[k].px), std::abs(ref[k].y - c630::kCorners[k].py)) <= 1e-4;
    par.add("center 재측정 = 기준 630점 (최대 |차| px)", 0, ref_dev, 0.005, 6);
    par.add("기준 630점 중 1e-4 px 안으로 같은 점", 630, ref_close, 5, 0);
    par.add("확장 탐색으로 복구한 사진 수", 12, recovered, 0, 0);
    par.add("사진별 평균 x 변위 최대 |차| (px)", 0, max_dh, 2e-3, 5);
    par.add("사진별 평균 y 변위 최대 |차| (px)", 0, max_dhy, 2e-3, 5);
    par.add("유효점 합계 (341x630)", ref_valid_sum, valid_sum, 0.001 * ref_valid_sum, 0);
    const long long inner_ref[5][3][2] = {{{85, 14573}, {83, 13765}, {84, 14385}}, {{88, 15475}, {85, 14555}, {83, 13804}}, {{88, 15268}, {85, 14359}, {83, 13648}},
                                          {{88, 15447}, {86, 14327}, {85, 14304}}, {{115, 19573}, {114, 19643}, {112, 18754}}};
    int inner_ok = 0;
    for (int k = 0; k < 5; ++k)
        for (int j = 0; j < 3; ++j) inner_ok += sels[k].inner[j].first == inner_ref[k][j][0] && sels[k].inner[j].second == inner_ref[k][j][1];
    par.add("내부 3-fold 분할 일치 (15개 중, 장수·인덱스합)", 15, inner_ok, 0, 0);
    const char* sel_ref[5] = {"local raw_shift s0.01 w100", "global raw_shift s0.1", "local raw_shift s0.01 w20", "local raw_shift s0.01 w20", "local raw_shift s0.1 w100"};
    for (int k = 0; k < 5; ++k) {
        std::string now = configs[sels[k].best].str();
        par.add_text(k < 4 ? "fold" + std::to_string(k) + " 선택 설정" : "최종 선택 설정", sel_ref[k], now, now == sel_ref[k]);
    }
    par.add("중첩 OOF 341 통과", 249, Mn.passed, 2, 0);
    par.add("중첩 OOF 20261014 통과 (281)", 197, M["selected_joint|nested_group_oof|1014"].passed, 2, 0);
    par.add("중첩 OOF 20261020 통과 (60)", 52, Mnew.passed, 1, 0);
    par.add("중첩 OOF Y 평균 |오차| (mm)", 0.064520, Mn.ymae, 0.001);
    par.add("중첩 OOF Y 최대 |오차| (mm)", 0.328199, Mn.ymax, 0.01);
    par.add("중첩 OOF T 평균 |오차|", 0.011542, Mn.tmae, 0.0003);
    par.add("중첩 OOF T 최대 |오차|", 0.056815, Mn.tmax, 0.003);
    par.add("raw_global OOF 통과", 240, M["raw_global|fixed_config_group_oof|0"].passed, 2, 0);
    par.add("interaction_global OOF 통과", 253, M["interaction_global|fixed_config_group_oof|0"].passed, 2, 0);
    par.add("interaction_local OOF 통과", 253, M["interaction_local|fixed_config_group_oof|0"].passed, 2, 0);
    par.add("old만 학습 raw1260 -> 20261020 통과", 14, Mo.passed, 1, 0);
    par.add("old만 학습 raw1260 -> 20261020 Y 평균 |오차|", 0.330199, Mo.ymae, 0.005);
    par.add("old만 학습 raw_shift -> 20261020 통과", 10, M["raw_shift|old_only_transfer|1020"].passed, 1, 0);
    par.add("전체 학습 재대입 341 통과", 320, M["selected_joint|resub341|0"].passed, 2, 0);

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: 새 사진 일반화 - NCC 탐색 확장 + 전체점 국소 회귀\n\n"
      << "입력: `data/clean/clean_20261014` 281장 + `clean_20261020` 60장 = 341장. 기준은 10-14 `center.bmp`다.\n"
      << "측정: 고정 630 구조 모서리마다 흐린 격자 경계(erf) 6변수 모델을 scipy `least_squares(trf, soft_l1, f_scale 6, max_nfev 85)`와 같은 C++ TRF로 맞췄다. "
      << "QR 세 영역 NCC가 0.6 미만이면 탐색을 ±130에서 ±260, ±512 px로 넓힌다. 기준 좌표는 center 사진 자신을 같은 방법으로 잰 값이다.\n"
      << "특징: raw1260 = 630점 (dx, dy), raw_shift = raw1260 + raw1260 x (평균 dx / 100).\n"
      << "모델: ForwardGLS 1차(가중 최소제곱 + 잔차 공분산 shrink). local 은 시험 사진의 평균 dx 근처 학습 사진에 가중 0.01+exp(-0.5((h_i-h)/width)^2)를 줘서 사진마다 다시 맞춘다.\n"
      << "평가: 바깥 4-fold(원래 합의 분할, 20261020 같은 명령 반복 3장과 center 3장은 같은 fold), 안쪽 StratifiedGroupKFold 3겹(seed 20261022+fold)에서 "
      << "20261020 검증 사진의 실패 수 → 최악 오차 → 평균 오차로 12개 설정 중 하나를 고른다.\n\n"
      << "## 측정\n\n| 항목 | 값 |\n|---|---|\n| 측정 성공 | " << 341 - failed << " / 341 |\n| 확장 탐색(pad>130)으로 복구 | " << recovered
      << " |\n| 유효점 비율 | " << fmt(valid_sum / (341.0 * 630), 4) << " |\n| center 재측정과 원래 기준 630점 최대 차 | " << fmt(ref_dev, 6) << " px |\n"
      << "| 사진별 평균 dx 원래 측정과 최대 차 | " << fmt(max_dh, 5) << " px |\n| 원래와 유효점 수가 같은 사진 | " << same_valid << " / 341 |\n\n"
      << img_md("04_NCC탐색확장_복구.png", "NCC 복구") << "\n\n"
      << "## 방법별 요약\n\n" << sum.md() << "\n## 중첩 선택\n\n" << st.md() << "\n### 설정별 내부 점수 (실패 수 / 최악 |오차|, * = 선택)\n\n" << cs.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- `forward_screen.py`, `local_screen.py`, `robust_screen.py`, `alternatives/*`(불변량·원점 GLS·품질 선별 후보), `diagnostics/diagnose.py`: "
      << "최종 선택(evaluate.py)에 들어가지 않은 탐색 후보다.\n- `predict.py`, `verify.py`, `report.py`: 추론 CLI·해시 검사·HTML 보고서다.\n"
      << "- 바깥 4-fold 배정은 numpy PCG64 순열(joint_center_630)과 sklearn StratifiedKFold(multiref_4fold)로 정해진 합의 분할이라 `protocol341.hpp` 상수로 옮겼다. "
      << "630 구조 정의(초기 위치·축·패치)는 `corners630.hpp` 상수다.\n"
      << "\n## 파일\n\n- `predictions.csv`: 방법·평가별 사진 예측\n- `measurements_summary.csv`: 사진별 NCC·탐색 pad·유효점·평균 변위(원래 값 함께)\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& Wy = imgs[evi[wy]];
    auto& Wt = imgs[evi[wt]];
    std::ostringstream P;
    P << "# 보고서: 새 사진 일반화 - 탐색 확장 + 국소 회귀 (10-22 16:08)\n\n"
      << "## 문제\n\n20261020 혼합 자세 사진에서 Y 오차가 컸고(old만 학습 " << Mo.passed << "/60) T+2 사진 " << recovered << "장은 ±130 px 탐색에서 측정에 실패했다.\n\n"
      << "## 해결 방법\n\nNCC 탐색을 ±260 px까지 넓혀 측정을 복구하고, 20261020 조건까지 교정에 넣은 전체점 회귀 12개 설정 중 하나를 중첩 그룹 OOF로 골랐다.\n\n"
      << "## 결과\n\n20261020은 old만 학습할 때 " << Mo.passed << "/60에서 교정에 넣은 중첩 OOF " << Mnew.passed << "/60이 됐고, 341장 전체 중첩 그룹 OOF는 " << Mn.passed
      << "/341이었다.\n\n| 지표 | 값 |\n|---|---|\n"
      << "| 341장 중첩 OOF 통과 (\\|eY\\|,\\|eT\\|<=0.09) | " << Mn.passed << " / 341 |\n"
      << "| 20261014 / 20261020 통과 | " << M["selected_joint|nested_group_oof|1014"].passed << " / 281, " << Mnew.passed << " / 60 |\n"
      << "| Y 평균 / 최대 \\|오차\\| (mm) | " << fmt(Mn.ymae, 4) << " / " << fmt(Mn.ymax, 4) << " |\n"
      << "| T 평균 / 최대 \\|오차\\| (deg) | " << fmt(Mn.tmae, 4) << " / " << fmt(Mn.tmax, 4) << " |\n"
      << "| 실패 " << 341 - Mn.passed << "장 중 Y 초과 / T 초과 | " << fail_y << " / " << fail_t << " |\n"
      << "| old만 학습 → 20261020 통과 (raw1260 / raw_shift) | " << Mo.passed << " / " << M["raw_shift|old_only_transfer|1020"].passed << " (60장) |\n"
      << "| 확장 탐색으로 복구한 사진 | " << recovered << " 장 |\n| 최종 선택 설정 (전체 341) | " << configs[fs_.best].str() << " |\n\n"
      << img_md("01_방법별_통과장수341.png", "방법별 통과") << "\n\n" << img_md("03_20261020_60장_통과.png", "20261020 통과") << "\n\n"
      << img_md("02_중첩OOF341_예측오차.png", "예측 오차") << "\n\n"
      << "## 튀는 값\n\n가장 크게 틀린 사진은 Y는 " << names[wy] << "(정답 Y " << fmt(ty[wy], 2) << ", 추정 " << fmt(py[wy], 3) << " mm, 평균 x 이동 " << fmt(Wy.h, 1)
      << " px), T는 " << names[wt] << "(정답 T " << fmt(tt[wt], 2) << ", 추정 " << fmt(pt[wt], 3) << " deg, 평균 x 이동 " << fmt(Wt.h, 1)
      << " px)이고, 이 사진은 같은 fold 학습 사진과 평균 이동이 " << fmt(gap(evi[wy]), 2) << " px 떨어진 끝 자세라 회귀가 밖으로 외삽했으며, 실패 " << 341 - Mn.passed
      << "장 중 Y 초과가 " << fail_y << "장, T 초과가 " << fail_t << "장이었다.\n\n"
      << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
      << "\n\n초록 상자는 기준 사진의 QR 영역, 빨간 상자는 NCC로 찾은 이 사진의 QR 영역, 빨간 십자는 무효 판정된 모서리 점이다.\n\n"
      << "## 한줄평\n\n전체점 회귀는 학습 사진이 덮는 자세 범위 안에서만 맞는 방법이라, 20261020 조건을 교정에 넣으면 그 날짜 사진이 " << Mo.passed << "/60에서 "
      << Mnew.passed << "/60으로 올랐지만 여전히 Y 평균 오차 " << fmt(Mn.ymae, 3) << " mm가 통과 기준 0.09의 " << fmt(Mn.ymae / 0.09 * 100, 0)
      << "%라서 341장 중 " << 341 - Mn.passed << "장이 남았다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `joint_yt_generalization_20261022/summary.csv`, `selection.json`, `final_config.json`, `measurements/audit.json`, "
      << "`manifest_with_folds.csv`(유효점·평균 변위)에서 가져왔다. " << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
