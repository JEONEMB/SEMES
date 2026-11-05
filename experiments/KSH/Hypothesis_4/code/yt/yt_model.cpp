// =====================================================================================
// yt_model.cpp — 특징 → (Y, Θ) 모델 구현
// =====================================================================================
#include "yt_model.hpp"

#include <algorithm>
#include <cmath>

namespace yt {

namespace {
bool finiteAll(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return !v.empty();
}
} // namespace

bool PoseModel::fit(const std::vector<Obs>& obs) {
    std::vector<const Obs*> use;
    for (const auto& o : obs) if (finiteAll(o.f)) use.push_back(&o);
    if (use.size() < 6) return false;
    const int K = int(use[0]->f.size());
    coef_.assign(K, {});
    sig_.assign(K, 1.0);
    gain_.assign(K, 1.0);
    off_.assign(K, 0.0);
    for (int k = 0; k < K; ++k) {
        std::vector<std::vector<double>> A;
        std::vector<double> b;
        for (const Obs* o : use) { A.push_back({1.0, o->y, o->t, o->t * o->t}); b.push_back(o->f[k]); }
        coef_[k] = lstsq(A, b);
        double ss = 0;
        for (size_t i = 0; i < use.size(); ++i) {
            const double r = b[i] - (coef_[k][0] + coef_[k][1] * use[i]->y + coef_[k][2] * use[i]->t + coef_[k][3] * use[i]->t * use[i]->t);
            ss += r * r;
        }
        sig_[k] = std::max(1e-9, std::sqrt(ss / std::max<size_t>(1, use.size() - 4)));
    }
    return true;
}

std::vector<double> PoseModel::predict(double y, double t) const {
    std::vector<double> p(coef_.size());
    for (size_t k = 0; k < coef_.size(); ++k) p[k] = coef_[k][0] + coef_[k][1] * y + coef_[k][2] * t + coef_[k][3] * t * t;
    return p;
}

bool PoseModel::invert(const std::vector<double>& fRaw, double& y, double& t) const {
    if (!finiteAll(fRaw) || fRaw.size() != coef_.size()) return false;
    const std::vector<double> f = correct(fRaw);
    y = 0; t = 0;
    for (int it = 0; it < 10; ++it) {
        cv::Matx22d JtJ(0, 0, 0, 0);
        cv::Vec2d Jtr(0, 0);
        for (size_t k = 0; k < coef_.size(); ++k) {
            const auto& c = coef_[k];
            const double w = 1.0 / (sig_[k] * sig_[k]);
            const double r = f[k] - (c[0] + c[1] * y + c[2] * t + c[3] * t * t);
            const double j0 = c[1], j1 = c[2] + 2 * c[3] * t;
            JtJ(0, 0) += w * j0 * j0; JtJ(0, 1) += w * j0 * j1; JtJ(1, 1) += w * j1 * j1;
            Jtr[0] += w * j0 * r; Jtr[1] += w * j1 * r;
        }
        JtJ(1, 0) = JtJ(0, 1);
        const cv::Vec2d d = JtJ.inv() * Jtr;
        y += d[0]; t += d[1];
        if (std::abs(d[0]) < 1e-9 && std::abs(d[1]) < 1e-10) break;
    }
    return std::isfinite(y) && std::isfinite(t);
}

void PoseModel::calibrate(const std::vector<Obs>& known) {
    const int K = int(coef_.size());
    gain_.assign(K, 1.0);
    off_.assign(K, 0.0);
    for (int k = 0; k < K; ++k) {
        std::vector<std::vector<double>> A;
        std::vector<double> b;
        for (const auto& o : known) {
            if (!finiteAll(o.f)) continue;
            A.push_back({o.f[k], 1.0});
            b.push_back(predict(o.y, o.t)[k]);
        }
        if (b.size() < 2) continue;
        const auto gh = lstsq(A, b);
        gain_[k] = gh[0];
        off_[k] = gh[1];
    }
}

std::vector<double> PoseModel::correct(const std::vector<double>& f) const {
    std::vector<double> o(f.size());
    for (size_t k = 0; k < f.size(); ++k) o[k] = (k < gain_.size() ? gain_[k] : 1.0) * f[k] + (k < off_.size() ? off_[k] : 0.0);
    return o;
}

std::vector<Est> leaveOneOut(const std::vector<Obs>& set1) {
    std::vector<Est> out;
    for (size_t i = 0; i < set1.size(); ++i) {
        std::vector<Obs> tr;
        for (size_t j = 0; j < set1.size(); ++j) if (j != i) tr.push_back(set1[j]);
        PoseModel m;
        Est e;
        e.stem = set1[i].stem; e.y = set1[i].y; e.t = set1[i].t; e.set = set1[i].set; e.shot = set1[i].shot;
        e.ok = m.fit(tr) && m.invert(set1[i].f, e.ye, e.te);
        out.push_back(e);
    }
    return out;
}

std::vector<Est> estimateAll(const PoseModel& m, const std::vector<Obs>& obs) {
    std::vector<Est> out;
    for (const auto& o : obs) {
        Est e;
        e.stem = o.stem; e.y = o.y; e.t = o.t; e.set = o.set; e.shot = o.shot;
        e.ok = m.invert(o.f, e.ye, e.te);
        out.push_back(e);
    }
    return out;
}

Summary summarize(const std::vector<Est>& e, const std::function<bool(const Est&)>& sel) {
    std::vector<double> et, ey;
    Summary s;
    for (const auto& x : e) {
        if (!x.ok || !sel(x)) continue;
        et.push_back(x.te - x.t);
        ey.push_back((x.ye - x.y) * 1000.0);
        if (std::abs(x.te - x.t) <= 0.01 && std::abs((x.ye - x.y) * 1000.0) <= 90.0) ++s.both;
    }
    s.t = stat(et, 0.01);
    s.y = stat(ey, 90.0);
    s.n = s.t.n;
    return s;
}

std::string summaryRow(const Summary& s) {
    return fmt("%d장 | Θ RMS %.4f° 최대 %.4f° 평균 %+.4f° ±0.01° 통과 %d/%d | Y RMS %.1f µm 최대 %.1f ±90 통과 %d | 둘 다 %d/%d", s.n, s.t.rms,
               s.t.maxAbs, s.t.mean, s.t.pass, s.n, s.y.rms, s.y.maxAbs, s.y.pass, s.both, s.n);
}

Stat stepTest(const std::vector<Est>& e, double step, bool theta, std::vector<double>* out) {
    std::vector<const Est*> v;
    for (const auto& x : e)
        if (x.ok && x.set == 1 && (theta ? (x.y == 0 && x.t != 0) : (x.t == 0 && x.y != 0))) v.push_back(&x);
    std::sort(v.begin(), v.end(), [&](const Est* a, const Est* b) { return theta ? a->t < b->t : a->y < b->y; });
    std::vector<double> d;
    for (size_t i = 1; i < v.size(); ++i) {
        const double dl = theta ? v[i]->t - v[i - 1]->t : v[i]->y - v[i - 1]->y;
        if (std::abs(dl - step) > 1e-6) continue;
        d.push_back(theta ? v[i]->te - v[i - 1]->te : (v[i]->ye - v[i - 1]->ye));
    }
    if (out) *out = d;
    return stat(d);
}

void writeEst(const std::string& path, const std::vector<Est>& e) {
    Csv c(path);
    c << "stem" << "set" << "shot" << "y_mm" << "t_deg" << "y_est" << "t_est" << "err_y_um" << "err_t_deg" << "ok";
    c.endl();
    for (const auto& x : e) {
        c << x.stem << x.set << x.shot << x.y << x.t << x.ye << x.te << (x.ye - x.y) * 1000 << x.te - x.t << int(x.ok);
        c.endl();
    }
}

} // namespace yt
