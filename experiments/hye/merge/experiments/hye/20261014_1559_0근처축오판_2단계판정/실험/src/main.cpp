// 0 근처 축 오판 -> 2단계 판정 (원본: analysis_v7b/twostage.py, lib.py, exp.py 의 load/feats/M, analysis_v6 점 추적)
// center + 현재 사진 두 장: QR 점 232개 추적 -> 판 2차 변형 12값 -> 축 판정(0 근처 1차, 절편 0) + 값(전체 2차)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include "v7b_model.hpp"
#include "v7b_track.hpp"

using namespace hye;
using namespace v7b;

namespace {

const double K_TY = 80.66 / 14.39;  // Θ 1 = Y 5.6mm (같은 QR 이동)

// 촬영 구간 (v6 session_of): Y1 = 0.1 간격, Y2 = 0.05 간격 홀수 단계
std::string session_of(const Shot& s) {
    if (s.condition == "center") return "C";
    if (s.condition == "T") return s.t > 0 ? "T+" : "T-";
    int st = (int)std::lround(std::abs(s.y) / 0.05);
    return std::string(st % 2 ? "Y2" : "Y1") + (s.y > 0 ? "+" : "-");
}
// 짝홀 폴드 (v5 fold_of): 축마다 촬영 시각 순서로 번갈아. data/clean 에 촬영 시각이 없어 촬영 방향을 상수로 둔다.
// 원본 BMP 시각: 구간마다 |값| 오름차순, Y2- 구간만 내림차순 (1.95 -> 0.05). 구간 크기가 모두 짝수라 구간 안 순번의 홀짝 = 폴드.
int fold_of(const Shot& s) {
    if (s.condition == "center") return -1;
    std::string ses = session_of(s);
    int idx;
    if (s.condition == "T") idx = (int)std::lround(std::abs(s.t) * 100) - 1;
    else {
        int st = (int)std::lround(std::abs(s.y) / 0.05);
        idx = st % 2 ? (st - 1) / 2 : st / 2 - 1;
        if (ses == "Y2-") idx = 19 - idx;
    }
    return idx % 2;
}

struct Rec { size_t i; int axis; double cmd; int pa; double pred, vY, vT, llr; };

struct Score {
    double accY, accT; int wrongY, wrongT;
    double yMean, yMax, tMean, tMax, p09s, p09h, pss, psh, allok;
};
Score score(const std::vector<Rec>& R) {
    Score s{};
    int nY = 0, nT = 0, okY = 0, okT = 0;
    std::vector<double> eY, eT;
    int s09 = 0, h09 = 0, sst = 0, hst = 0;
    double allok = 0;
    for (auto& r : R) {
        bool y = r.axis == AY, ok = r.pa == r.axis;
        double e = std::abs(r.pred - r.cmd);
        (y ? nY : nT)++;
        if (ok) (y ? okY : okT)++;
        if (ok) (y ? eY : eT).push_back(e);
        for (int k = 0; k < 2; ++k) {
            double tolY = k ? 0.05 : 0.09, tolT = k ? 0.01 : 0.09;
            double tt = y ? tolY : tolT, tp = r.pa == AY ? tolY : tolT;
            bool strict = ok && e <= tt, harm = strict || (!ok && std::abs(r.cmd) <= tt && std::abs(r.pred) <= tp);
            (k ? sst : s09) += strict;
            (k ? hst : h09) += harm;
        }
        if (!ok) allok = std::max(allok, std::abs(r.cmd) * (y ? 1 : K_TY));
    }
    s.accY = 100.0 * okY / nY; s.accT = 100.0 * okT / nT;
    s.wrongY = nY - okY; s.wrongT = nT - okT;
    s.yMean = mean(eY) * 1000; s.yMax = max_abs(eY) * 1000; s.tMean = mean(eT) * 1000; s.tMax = max_abs(eT) * 1000;
    double n = (double)R.size();
    s.p09s = 100 * s09 / n; s.p09h = 100 * h09 / n; s.pss = 100 * sst / n; s.psh = 100 * hst / n; s.allok = allok;
    return s;
}

std::string short_name(const Shot& s) {
    auto nm = s.rel.substr(s.rel.find('/') + 1);
    return nm.substr(0, nm.rfind('.'));
}

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
    omp_set_num_threads(4);
    auto shots = load_1014();
    const Shot& cs = center_1014(shots);
    cv::Mat cen = to_f32(read_gray(cs));
    auto F = pick_qr(cen);
    int nreg[3] = {};
    for (auto& p : F) nreg[p.region]++;
    std::cout << "points " << F.size() << " " << nreg[0] << "/" << nreg[1] << "/" << nreg[2] << "\n";
    Tmpl T = make_templates(cen, F);

    // 점 추적 (281장)
    const int N = (int)shots.size();
    std::vector<std::vector<cv::Vec2d>> Draw(N);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) Draw[i] = track(to_f32(read_gray(shots[i])), T, F);
    int ic = 0;
    for (int i = 0; i < N; ++i) if (shots[i].condition == "center") ic = i;
    std::vector<std::vector<cv::Vec2d>> D(N, std::vector<cv::Vec2d>(F.size()));
    for (int i = 0; i < N; ++i)
        for (size_t k = 0; k < F.size(); ++k) D[i][k] = Draw[i][k] - Draw[ic][k];  // center 기준
    std::cout << "tracked\n";

    P12 p12(F);
    cv::Mat U = nuis_p12(F, p12);
    std::vector<std::vector<double>> Fe(N);
    for (int i = 0; i < N; ++i) Fe[i] = p12(D[i]);
    std::vector<double> lab(N);
    std::vector<int> AXV(N), FOLD(N);
    std::vector<std::string> SES(N), names(N);
    for (int i = 0; i < N; ++i) {
        auto& s = shots[i];
        AXV[i] = s.condition == "center" ? AC : s.condition == "Y" ? AY : AT;
        lab[i] = AXV[i] == AY ? s.y : AXV[i] == AT ? s.t : 0.0;
        FOLD[i] = fold_of(s);
        SES[i] = session_of(s);
        names[i] = short_name(s);
    }

    // 출력: 점 목록, 사진별 P12
    {
        Table pt{{"id", "x", "y", "qr"}};
        for (size_t k = 0; k < F.size(); ++k) pt.add({std::to_string(k), std::to_string(F[k].x), std::to_string(F[k].y), std::string(1, "LCR"[F[k].region])});
        pt.csv(out("points.csv"));
        Table fe{{"file", "Y", "T", "fold", "session", "mean_dx", "mean_dy", "b0x", "b0y", "b1x", "b1y", "b2x", "b2y", "b3x", "b3y", "b4x", "b4y", "b5x", "b5y"}};
        for (int i = 0; i < N; ++i) {
            double mx = 0, my = 0;
            for (auto& d : D[i]) { mx += d[0]; my += d[1]; }
            std::vector<std::string> r{shots[i].rel, fmt(shots[i].y, 2), fmt(shots[i].t, 2), std::to_string(FOLD[i]), SES[i], fmt(mx / F.size(), 5), fmt(my / F.size(), 5)};
            for (double v : Fe[i]) r.push_back(fmt(v, 6));
            fe.add(r);
        }
        fe.csv(out("p12_features.csv"));
    }

    // 실행: 교차검증 2종 x 값모델 절편 2종, 학습 라벨 = 명령값
    struct Cfg { std::string cv; bool free; };
    std::vector<Cfg> cfgs{{"짝홀", true}, {"짝홀", false}, {"구간빼기", true}, {"구간빼기", false}};
    std::map<std::string, std::vector<Rec>> RES;
    std::map<std::string, Score> SC;
    for (auto& c : cfgs) {
        std::vector<std::pair<std::string, std::vector<bool>>> folds;
        if (c.cv == "짝홀")
            for (int k = 0; k < 2; ++k) {
                std::vector<bool> te(N);
                for (int i = 0; i < N; ++i) te[i] = FOLD[i] == k;
                folds.push_back({"f" + std::to_string(k), te});
            }
        else
            for (std::string s : {"Y1+", "Y1-", "Y2+", "Y2-", "T+", "T-"}) {
                std::vector<bool> te(N);
                for (int i = 0; i < N; ++i) te[i] = SES[i] == s;
                folds.push_back({s, te});
            }
        std::vector<Rec> recs;
        for (auto& fo : folds) {
            std::vector<std::vector<double>> fe;
            std::vector<double> lb;
            std::vector<int> ax;
            for (int i = 0; i < N; ++i)
                if (!fo.second[i]) { fe.push_back(Fe[i]); lb.push_back(lab[i]); ax.push_back(AXV[i]); }
            TwoStage m(U, c.free);
            m.fit(fe, lb, ax);
            for (int i = 0; i < N; ++i) {
                if (!fo.second[i] || AXV[i] == AC) continue;
                auto p = m.predict(Fe[i]);
                int pa = p.llr < 0 ? AY : AT;
                recs.push_back({(size_t)i, AXV[i], lab[i], pa, pa == AY ? p.vY : p.vT, p.vY, p.vT, p.llr});
            }
        }
        std::string key = c.cv + (c.free ? "/free" : "/zero");
        SC[key] = score(recs);
        RES[key] = recs;
        auto& s = SC[key];
        std::cout << key << " wrong " << s.wrongY << "/" << s.wrongT << " Y " << s.yMean << "/" << s.yMax << " T " << s.tMean << "/" << s.tMax << "\n";
    }

    auto& R0 = RES["짝홀/free"];
    {
        Table pr{{"file", "axis", "cmd", "pred_axis", "pred", "vY", "vT", "llr"}};
        for (auto& r : R0)
            pr.add({shots[r.i].rel, r.axis == AY ? "Y" : "T", fmt(r.cmd, 2), r.pa == AY ? "Y" : "T", fmt(r.pred, 6), fmt(r.vY, 6), fmt(r.vT, 6), fmt(r.llr, 6)});
        pr.csv(out("predictions_evenodd_free.csv"));
    }

    // ---- 표 ----
    Table tb{{"교차검증", "값모델 절편", "틀림 Y / T", "Y 평균 / 최대 |오차| (µm)", "T 평균 / 최대 |오차| (0.001deg)", "합격 0.09 엄격 / 피해없음 (%)", "합격 한스텝 엄격 / 피해없음 (%)", "전부 맞는 시작 (Y mm 환산)"}};
    for (auto& c : cfgs) {
        auto& s = SC[c.cv + (c.free ? "/free" : "/zero")];
        tb.add({c.cv, c.free ? "자유" : "center 고정", std::to_string(s.wrongY) + " / " + std::to_string(s.wrongT), fmt(s.yMean, 1) + " / " + fmt(s.yMax, 1),
                fmt(s.tMean, 2) + " / " + fmt(s.tMax, 2), fmt(s.p09s, 1) + " / " + fmt(s.p09h, 1), fmt(s.pss, 1) + " / " + fmt(s.psh, 1), fmt(s.allok, 2)});
    }
    // 틀린 사진, 오차 큰 사진
    std::vector<double> ty, py, tt, pt, eYv, eTv;
    std::vector<std::string> nm;
    std::vector<const Rec*> wrong;
    for (auto& r : R0) {
        bool y = r.axis == AY;
        ty.push_back(y ? r.cmd : 0); tt.push_back(y ? 0 : r.cmd);
        py.push_back(r.pa == AY ? r.pred : 0); pt.push_back(r.pa == AT ? r.pred : 0);
        eYv.push_back(y && r.pa == AY ? r.pred - r.cmd : 0);
        eTv.push_back(!y && r.pa == AT ? r.pred - r.cmd : 0);
        nm.push_back(names[r.i]);
        if (r.pa != r.axis) wrong.push_back(&r);
    }
    size_t wy = worst(eYv, 1)[0], wt = worst(eTv, 1)[0];
    Table outl{{"구분", "사진", "정답 축 / 값", "판정 축", "Y 가설 값 (mm)", "T 가설 값 (deg)", "llr = log(비용Y/비용T)", "오차", "판 평균 이동 dx (px)", "b1x 가로 늘어남 (px)", "b3x 가로 휨 (px)"}};
    auto add_out = [&](const std::string& lab_, const Rec& r) {
        double mx = 0;
        for (auto& d : D[r.i]) mx += d[0];
        mx /= F.size();
        bool ok = r.pa == r.axis;
        outl.add({lab_, names[r.i], std::string(r.axis == AY ? "Y " : "T ") + fmts(r.cmd, 2), r.pa == AY ? "Y" : "T", fmt(r.vY, 4), fmt(r.vT, 4), fmts(r.llr, 4),
                  ok ? fmts(r.pred - r.cmd, 4) : "축 틀림", fmt(mx, 3), fmts(Fe[r.i][2], 4), fmts(Fe[r.i][6], 4)});
    };
    for (auto* r : wrong) add_out("축 틀림", *r);
    add_out("Y 값 오차 최대", R0[wy]);
    add_out("T 값 오차 최대", R0[wt]);

    // ---- 그림 ----
    {  // 01 점 지도
        cv::Mat c8;
        read_color(cs).copyTo(c8);
        std::vector<Mark> mk;
        for (auto& p : F) mk.push_back({cv::Point2d(p.x, p.y), {0, 0}, "", color(p.region)});
        cv::Mat z = zoom(c8, cv::Rect(X0 - 10, Y0 - 10, X1 - X0 + 20, Y1 - Y0 + 20), 1400, mk, 0,
                         "center.bmp: " + std::to_string(F.size()) + " QR points (17x17 template)");
        save_png("01_QR점_" + std::to_string(F.size()) + "개.png", z);
    }
    pred_vs_truth("02_짝홀_예측오차.png", "2-stage even/odd", ty, py, tt, pt, nm);
    {  // 03 llr vs 값
        std::vector<double> xy, ly, xt, lt;
        for (auto& r : R0) (r.axis == AY ? xy : xt).push_back(r.axis == AY ? r.cmd : r.cmd * K_TY), (r.axis == AY ? ly : lt).push_back(r.llr);
        Chart c("Axis decision: log(costY/costT), even/odd", "true value (Y mm, T as Y-equivalent mm)", "llr (<0 -> Y)");
        c.scatter(xy, ly, color(2), "Y photos").scatter(xt, lt, color(1), "T photos").hline(0, kRed).range(-0.7, 0.7, -3, 3);
        for (auto* r : wrong) c.note(r->axis == AY ? r->cmd : r->cmd * K_TY, r->llr, names[r->i]);
        c.save("03_축판정_0근처.png");
    }
    // 튀는 값 사진: 가운데 QR 확대. 초록 = center 점 위치, 빨강 = 이 사진 점 위치, 화살표 = (점 이동 - 판 평균 이동) x 200
    auto outlier_fig = [&](const Rec& r, const std::string& file, const std::string& cap) {
        cv::Mat im = read_color(shots[r.i]);
        double mx = 0, my = 0;
        for (auto& d : D[r.i]) { mx += d[0]; my += d[1]; }
        mx /= F.size(); my /= F.size();
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            std::vector<Mark> mk;
            for (size_t k = 0; k < F.size(); ++k) {
                if (F[k].region != q) continue;
                mk.push_back({cv::Point2d(F[k].x + D[r.i][k][0], F[k].y + D[r.i][k][1]), {D[r.i][k][0] - mx, D[r.i][k][1] - my}, "", kRed});
            }
            cv::Rect2d ref(kQRc[q][0] - 90, kQRc[q][1] - 95, 180, 190), now(ref.x + mx, ref.y + my, 180, 190);
            cv::Rect roi = cv::Rect(ref | now) + cv::Size(40, 40) - cv::Point(20, 20);
            tiles.push_back(zoom(im, roi, 460, mk, 200, q == 0 ? cap : "", {{ref, kGreen, "center"}, {now, kRed, ""}}));
        }
        save_png(file, hstack(tiles));
    };
    std::string fy = "04_튀는값_Y_" + names[R0[wy].i] + ".png", ft = "05_튀는값_T_" + names[R0[wt].i] + ".png";
    outlier_fig(R0[wy], fy, names[R0[wy].i] + " Y err " + fmts(eYv[wy] * 1000, 1) + " um");
    outlier_fig(R0[wt], ft, names[R0[wt].i] + " T err " + fmts(eTv[wt], 4) + " deg");
    std::vector<std::string> fw;
    for (auto* r : wrong) {
        std::string f = "06_축틀림_" + names[r->i] + ".png";
        outlier_fig(*r, f, names[r->i] + " judged " + (r->pa == AY ? "Y" : "T") + " llr " + fmts(r->llr, 3));
        fw.push_back(f);
    }

    // ---- 원래 결과와 대조 ----
    // 원래 값: analysis_v7b/out/63_v7_2단계_평가.csv (학습라벨=명령값, 정답=명령값 행), 점 수·이동은 analysis_v6/features_qr_hp8.csv, tracks_qr_hp8.npz
    Parity par;
    par.add("QR 점 수", 232, (double)F.size(), 0, 0);
    par.add("QR 좌 점 수", 81, nreg[0], 0, 0);
    par.add("QR 중 점 수", 73, nreg[1], 0, 0);
    par.add("QR 우 점 수", 78, nreg[2], 0, 0);
    auto meanD = [&](const std::string& rel, int c) {
        for (int i = 0; i < N; ++i)
            if (shots[i].rel == rel) { double m = 0; for (auto& d : D[i]) m += d[c]; return m / F.size(); }
        return kNaN;
    };
    par.add("T+1.00 평균 dx (px)", 79.48944679, meanD("Theta/T+1.00.bmp", 0), 0.001, 5);
    par.add("Y+2.00 평균 dx (px)", 28.17833953, meanD("Y/Y+2.00.bmp", 0), 0.001, 5);
    par.add("T+0.01 평균 dx (px)", 0.1594919, meanD("Theta/T+0.01.bmp", 0), 0.001, 5);
    par.add("Y-0.05 평균 dx (px)", -1.11396712, meanD("Y/Y-0.05.bmp", 0), 0.001, 5);
    par.add("T+1.00 평균 dy (px)", 0.2614074, meanD("Theta/T+1.00.bmp", 1), 0.001, 5);
    struct Orig { const char* key; int wy, wt; double ym, yx, tm, tx, p09s, p09h, pss, psh, allok; };
    const Orig orig[] = {{"짝홀/free", 2, 1, 12.006, 29.886, 1.742, 4.845, 98.929, 99.643, 98.929, 99.286, 0.25},
                         {"짝홀/zero", 2, 1, 13.294, 32.145, 2.358, 7.425, 98.929, 99.643, 98.929, 99.286, 0.25},
                         {"구간빼기/free", 2, 1, 23.238, 50.897, 8.64, 15.974, 98.929, 99.643, 65.714, 66.071, 0.2},
                         {"구간빼기/zero", 2, 1, 20.334, 42.842, 15.329, 53.037, 98.929, 99.643, 67.5, 67.857, 0.2}};
    for (auto& o : orig) {
        auto& s = SC[o.key];
        std::string k = o.key;
        par.add(k + " 틀림 Y", o.wy, s.wrongY, 0, 0);
        par.add(k + " 틀림 T", o.wt, s.wrongT, 0, 0);
        par.add(k + " Y 평균 |오차| (µm)", o.ym, s.yMean, 0.5, 3);
        par.add(k + " Y 최대 |오차| (µm)", o.yx, s.yMax, 1.0, 3);
        par.add(k + " T 평균 |오차| (0.001deg)", o.tm, s.tMean, 0.1, 3);
        par.add(k + " T 최대 |오차| (0.001deg)", o.tx, s.tMax, 0.3, 3);
        par.add(k + " 합격 0.09 엄격 (%)", o.p09s, s.p09s, 0.001, 3);
        par.add(k + " 합격 한스텝 엄격 (%)", o.pss, s.pss, 0.4, 3);
        par.add(k + " 전부맞음 시작 (mm)", o.allok, s.allok, 0.001, 3);
    }
    std::string wrong_list;
    for (auto* r : wrong) wrong_list += (wrong_list.empty() ? "" : ", ") + names[r->i];
    par.add_text("짝홀/free 축 틀린 사진", "Y+0.05, Y-0.25, T+0.01 (63_v7 리포트: 명령값 학습은 Y-0.25가 하나 더)", wrong_list,
                 wrong_list.find("Y+0.05") != std::string::npos && wrong_list.find("Y-0.25") != std::string::npos &&
                     wrong_list.find("T+0.01") != std::string::npos && wrong.size() == 3);

    auto& S0 = SC["짝홀/free"];
    int nte = (int)R0.size();
    // ---- 결과.md ----
    std::ostringstream Rm;
    Rm << "# 결과: 0 근처 축 오판 -> 2단계 판정 (v7b)\n\n"
       << "입력: `data/clean/clean_20261014` 281장. 기준은 center.bmp 한 장이다. 모든 값은 사진에서 다시 쟀다.\n\n"
       << "1. 점: 흐린 center(σ1)의 QR 판에서 goodFeaturesToTrack(품질 0.01, 간격 17, blockSize 7)으로 고르고 포화 점을 뺀 QR 점 " << F.size()
       << "개(좌 " << nreg[0] << " / 중 " << nreg[1] << " / 우 " << nreg[2] << ").\n"
       << "2. 추적: 81x81 NCC(가로 ±110, 세로 ±16) -> 17x17 NCC ±4 + 포물선 -> 흐린 사진에서 가우스-뉴턴(dx, dy, 밝기 a, b, 3차 B-spline 보간, 15회).\n"
       << "3. 특징: center 기준 점 이동을 판 2차 다항 변형 12값으로 요약.\n"
       << "4. 축 판정: |Y|<=0.6mm, |T|<=0.12 사진만으로 학습한 1차 모델(절편 0), 값: 전체 사진 2차 모델. 둘 다 12값 잔차의 전체 공분산으로 화이트닝하고, 세로 이동·화면 회전 방해 무늬를 함께 푼다. 비용비 log(Y/T) < 0 이면 Y.\n"
       << "5. 학습 라벨 = 파일명 명령값, 채점 = 명령값.\n\n"
       << "## 조건별 평가 (학습 라벨 = 명령값, 정답 = 명령값)\n\n" << tb.md()
       << "\n짝홀은 축마다 촬영 순서로 번갈아 나눈 2분할, 구간빼기는 촬영 구간 6개(Y 0.1간격 ±, Y 0.05간격 홀수 ±, T ±) 중 하나를 빼고 학습한다.\n\n"
       << "## 축이 틀리거나 오차가 큰 사진 (짝홀, 값절편 자유)\n\n" << outl.md()
       << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
       << "\n## 옮기지 않은 부분\n\n"
       << "- 보정본 라벨(실제 위치): v4 `m1_backlash_drift`(촬영 시각·QR 마커 위치로 맞춘 백래시+드리프트)가 필요한데 data/clean에 촬영 시각이 없다. 그래서 원래 대표값 `보정본/실제위치` 행(Y 1.3/9.4µm)은 다시 만들지 않았다.\n"
       << "- exp.py 단계 실험(S1 템플릿 크기, S3 라벨·절편, S4 계산 방법 비교, S4b/S4c, S5 문턱 보정), bias.py, consistency.py, multishot.py, zero_cal.py, zero_ms.py, ts33.py, 그림·ppt 스크립트.\n\n"
       << "## 파일\n\n- `points.csv`: QR 점 " << F.size() << "개\n- `p12_features.csv`: 사진별 폴드·구간·평균 이동·12값\n- `predictions_evenodd_free.csv`: 짝홀·값절편 자유 사진별 판정\n";
    write_text(out("결과.md"), Rm.str());

    // ---- 보고서.md ----
    double llrY = 0, llrT = 0; int nY = 0, nT = 0;  // 0 근처 사진의 llr 평균
    for (auto& r : R0) {
        if (r.axis == AY && std::abs(r.cmd) <= 0.3) { llrY += r.llr; ++nY; }
        if (r.axis == AT && std::abs(r.cmd) <= 0.05) { llrT += r.llr; ++nT; }
    }
    llrY /= std::max(nY, 1); llrT /= std::max(nT, 1);
    std::ostringstream P;
    P << "# 보고서: 0 근처 축 오판 -> 2단계 판정 (10-14 15:59)\n\n"
      << "## 문제\n\n0 근처 사진의 축 판정이 한쪽으로 몰려 틀리는 문제를 풀어야 했다.\n\n"
      << "## 해결 방법\n\nQR 점 " << F.size() << "개 이동을 판 2차 변형 12값으로 요약하고, 축 판정은 0 근처 사진만 쓴 1차 모델, 값은 전체 사진 2차 모델로 나눴다.\n\n"
      << "## 결과\n\n짝홀 검증 " << nte << "장 중 " << (S0.wrongY + S0.wrongT) << "장만 축이 틀렸고, 축을 맞힌 Y 사진의 명령값 대비 오차는 평균 " << fmt(S0.yMean, 1) << " µm다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 축 맞힘 Y / T | " << (80 - S0.wrongY) << "/80 / " << (200 - S0.wrongT) << "/200 |\n"
      << "| 통과 (축 맞고 \\|오차\\|<=0.09) | " << (int)std::lround(S0.p09s * nte / 100) << " / " << nte << " |\n"
      << "| Y 평균 / 최대 \\|오차\\| (µm) | " << fmt(S0.yMean, 1) << " / " << fmt(S0.yMax, 1) << " |\n"
      << "| T 평균 / 최대 \\|오차\\| (deg) | " << fmt(S0.tMean / 1000, 4) << " / " << fmt(S0.tMax / 1000, 4) << " |\n"
      << "| 구간빼기 틀림 Y / T | " << SC["구간빼기/free"].wrongY << " / " << SC["구간빼기/free"].wrongT << " |\n"
      << "| 구간빼기 Y 평균 / 최대 \\|오차\\| (µm) | " << fmt(SC["구간빼기/free"].yMean, 1) << " / " << fmt(SC["구간빼기/free"].yMax, 1) << " |\n\n"
      << img_md("01_QR점_" + std::to_string(F.size()) + "개.png", "QR 점") << "\n\n" << img_md("02_짝홀_예측오차.png", "예측 오차") << "\n\n"
      << img_md("03_축판정_0근처.png", "축 판정") << "\n\n"
      << "## 튀는 값\n\n축이 틀린 사진은 " << wrong_list << "로 모두 |값|이 0.3mm(Y 환산) 이하이고, llr이 0에서 " << fmt(max_abs([&] { std::vector<double> v; for (auto* r : wrong) v.push_back(r->llr); return v; }()), 3)
      << " 안쪽이라 판정 경계에 걸렸다.\n\n" << outl.md() << "\n"
      << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값") << "\n\n";
    for (auto& f : fw) P << img_md(f, "축 틀림") << "\n\n";
    P << "초록 상자는 center의 QR 영역, 빨간 상자는 판 평균 이동만큼 옮긴 영역이다. 빨간 십자는 이 사진에서 추적한 점, 화살표는 (점 이동 - 판 평균 이동) x 200이다.\n\n"
      << "## 한줄평\n\n축 판정 모델을 0 근처 사진으로만, 0점을 center로 맞춰 학습하니 0 근처 Y 사진 llr 평균 " << fmts(llrY, 3) << ", T 사진 " << fmts(llrT, 3)
      << "로 두 덩어리가 경계 양쪽으로 갈라졌지만, 가장 작은 한두 스텝은 신호가 잡음과 같은 크기라 " << (S0.wrongY + S0.wrongT) << "장이 남았다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `analysis_v7b/out/63_v7_2단계_평가.csv`의 학습라벨=명령값, 정답=명령값 행과 `analysis_v6/features_qr_hp8.csv`, `tracks_qr_hp8.npz`에서 가져왔다. "
      << "원래 대표값(보정본 라벨, 실제 위치 기준 Y 1.3/9.4µm, 2장 틀림)은 촬영 시각이 data/clean에 없어 다시 만들지 않았다. " << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
