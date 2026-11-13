#include "reference.hpp"
#include "ecc_compat.hpp"
#include "observer_internal.hpp"
#include "sha256.hpp"
#include "site.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace final00 {
namespace fs = std::filesystem;
namespace {
const int kQrPad = 120, kQrMargin = 12, kScrewHalf = 70, kBarSearch = 20, kBarHalf = 40;

cv::Mat gray_of(const cv::Mat& input) {
    if (input.empty() || input.depth() != CV_8U || input.size() != cv::Size(3840, 2160)) throw std::invalid_argument("기준 사진은 3840x2160 8비트여야 한다");
    cv::Mat g;
    if (input.channels() == 1) g = input;
    else if (input.channels() == 3) cv::cvtColor(input, g, cv::COLOR_BGR2GRAY);
    else if (input.channels() == 4) cv::cvtColor(input, g, cv::COLOR_BGRA2GRAY);
    else throw std::invalid_argument("채널 수는 1, 3, 4 중 하나여야 한다");
    return g;
}
cv::Mat shift_matrix(double x, double y) { return (cv::Mat_<double>(3, 3) << 1, 0, x, 0, 1, y, 0, 0, 1); }
cv::Point2d project(const cv::Mat& h, double x, double y) {
    const double z = h.at<double>(2, 0) * x + h.at<double>(2, 1) * y + h.at<double>(2, 2);
    return {(h.at<double>(0, 0) * x + h.at<double>(0, 1) * y + h.at<double>(0, 2)) / z, (h.at<double>(1, 0) * x + h.at<double>(1, 1) * y + h.at<double>(1, 2)) / z};
}
cv::Mat gradient_magnitude(const cv::Mat& patch8) {
    cv::Mat a, b, gx, gy, m;
    patch8.convertTo(a, CV_32F); cv::GaussianBlur(a, b, cv::Size(0, 0), 1.5);
    cv::Sobel(b, gx, CV_32F, 1, 0, 3); cv::Sobel(b, gy, CV_32F, 0, 1, 3); cv::magnitude(gx, gy, m);
    return m;
}
// 세로 81행 밝기 프로파일(64열 평균, 가우시안 σ1)의 기울기 × 부호 (numpy.gradient 와 같은 끝 처리)
std::vector<double> edge_strength(const cv::Mat& img, int x0, int x1, int y_center, int polarity) {
    cv::Mat p(2 * kBarHalf + 1, 1, CV_64F);
    for (int i = 0; i < p.rows; ++i) {
        const unsigned char* row = img.ptr<unsigned char>(y_center - kBarHalf + i) + x0; double s = 0;
        for (int x = 0; x < x1 - x0; ++x) s += row[x];
        p.at<double>(i) = s / (x1 - x0);
    }
    cv::GaussianBlur(p, p, cv::Size(0, 0), 1.0);
    std::vector<double> g(p.rows);
    for (int i = 0; i < p.rows; ++i) {
        const double d = i == 0 ? p.at<double>(1) - p.at<double>(0) : i == p.rows - 1 ? p.at<double>(i) - p.at<double>(i - 1) : .5 * (p.at<double>(i + 1) - p.at<double>(i - 1));
        g[i] = d * polarity;
    }
    return g;
}
std::vector<unsigned char> encode_bmp(const cv::Mat& g) {
    std::vector<unsigned char> buf;
    if (!cv::imencode(".bmp", g, buf)) throw std::runtime_error("센터 사진을 BMP 로 만들지 못했다");
    return buf;
}
void write_text(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary); if (!f) throw std::runtime_error("쓰지 못했다: " + p.u8string()); f << s;
}
std::string rect_json(const cv::Rect& r) { std::ostringstream s; s << '[' << r.x << ", " << r.y << ", " << r.width << ", " << r.height << ']'; return s.str(); }
}

ReferenceReport make_reference(const fs::path& base, const cv::Mat& center_input, const fs::path& out) {
    const cv::Mat now = gray_of(center_input);
    const cv::Mat old = alignment_detail::read_gray(base / "center.bmp");
    const site::Geometry g0 = site::read(base);
    ReferenceReport rep{};

    // 1. QR 상자와 QR 마다 homography
    cv::Mat H[3]; cv::Rect boxes[3]; cv::Point2d mean_shift(0, 0);
    for (int k = 0; k < 3; ++k) {
        const cv::Rect b = g0.qr[k];
        const cv::Rect window = cv::Rect(b.x - kQrPad, b.y - kQrPad, b.width + 2 * kQrPad, b.height + 2 * kQrPad) & cv::Rect(0, 0, now.cols, now.rows);
        cv::Mat score; cv::matchTemplate(now(window), old(b), score, cv::TM_CCOEFF_NORMED);
        double ncc; cv::Point loc; cv::minMaxLoc(score, nullptr, &ncc, nullptr, &loc);
        const cv::Point d = loc + window.tl() - b.tl();
        if (ncc <= .5) throw std::runtime_error("기준 다시 잡기: QR " + std::to_string(k + 1) + " 를 찾지 못했다 (NCC " + std::to_string(ncc) + ")");
        const cv::Rect tr(b.x - kQrMargin, b.y - kQrMargin, b.width + 2 * kQrMargin, b.height + 2 * kQrMargin), cr = tr + d;
        if ((cr & cv::Rect(0, 0, now.cols, now.rows)) != cr) throw std::runtime_error("기준 다시 잡기: QR 이 화면 끝에 걸렸다");
        cv::Mat warp = cv::Mat::eye(3, 3, CV_32F);
        const double cc = semes::ecc5::findTransformECCWithMask(old(tr), now(cr), cv::Mat(tr.size(), CV_8U, cv::Scalar(255)), cv::Mat(cr.size(), CV_8U, cv::Scalar(255)),
            warp, cv::MOTION_HOMOGRAPHY, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 500, 1e-9), 5);
        if (!(cc > .9)) throw std::runtime_error("기준 다시 잡기: QR " + std::to_string(k + 1) + " 호모그래피 상관이 낮다 (" + std::to_string(cc) + ")");
        cv::Mat w; warp.convertTo(w, CV_64F);
        H[k] = shift_matrix(cr.x, cr.y) * w * shift_matrix(-tr.x, -tr.y);
        boxes[k] = b + d; rep.qr_shift[k] = d; rep.qr_ncc[k] = ncc; rep.qr_ecc[k] = cc; mean_shift += cv::Point2d(d) / 3.0;
    }

    // 2. 630점 (points.json, geometry.json)
    std::ostringstream pts; pts << std::setprecision(17) << "{\"points\": [";
    {
        cv::FileStorage f((base / "points.json").u8string(), cv::FileStorage::READ); const auto rows = f["points"];
        if (!rows.isSeq() || rows.size() != 630) throw std::runtime_error("기준 다시 잡기: points.json 은 630점이어야 한다");
        int n = 0;
        for (auto it = rows.begin(); it != rows.end(); ++it, ++n) {
            const auto r = *it; const int k = int(r[2]), sel = int(r[3]);
            const cv::Point2d q = project(H[k], double(r[0]), double(r[1]));
            if (!boxes[k].contains(cv::Point(int(q.x), int(q.y)))) throw std::runtime_error("기준 다시 잡기: 옮긴 점이 QR 상자 밖이다");
            pts << (n ? ", " : "") << '[' << q.x << ", " << q.y << ", " << k << ", " << sel << ']';
        }
    }
    pts << "], \"source\": \"moved from " << base.filename().u8string() << " by per-QR ECC homography (final00 reference)\"}";
    std::ostringstream geo; geo << std::setprecision(17) << "{\"corners\": [";
    {
        cv::FileStorage f((base / "geometry.json").u8string(), cv::FileStorage::READ); int n = 0;
        for (auto it = f["corners"].begin(); it != f["corners"].end(); ++it, ++n) {
            const auto c = *it; const int q = int(c["qr"]); if (q < 1 || q > 3) throw std::runtime_error("기준 다시 잡기: geometry.json 의 QR 번호가 잘못됐다");
            const cv::Point2d p = project(H[q - 1], double(c["cx"]), double(c["cy"]));
            geo << (n ? ", " : "") << "{\"qr\": " << q << ", \"cx\": " << p.x << ", \"cy\": " << p.y << '}';
        }
        if (n != 630) throw std::runtime_error("기준 다시 잡기: geometry.json 은 630점이어야 한다");
    }
    geo << "]}";

    // 3. 나사 중심: 기울기 크기 무게중심 3회
    cv::Point screws[4];
    for (int k = 0; k < 4; ++k) {
        int gx = g0.screws[k].x, gy = g0.screws[k].y;
        for (int it = 0; it < 3; ++it) {
            const cv::Rect r(gx - kScrewHalf, gy - kScrewHalf, 2 * kScrewHalf + 1, 2 * kScrewHalf + 1);
            if ((r & cv::Rect(0, 0, now.cols, now.rows)) != r) throw std::runtime_error("기준 다시 잡기: 나사가 화면 끝에 걸렸다");
            cv::Mat m = gradient_magnitude(now(r)); double mx; cv::minMaxLoc(m, nullptr, &mx);
            double s = 0, sx = 0, sy = 0;
            for (int y = 0; y < m.rows; ++y) for (int x = 0; x < m.cols; ++x) {
                const double v = m.at<float>(y, x); if (v < .35 * mx) continue;
                s += v; sx += v * (x - kScrewHalf); sy += v * (y - kScrewHalf);
            }
            if (!(s > 0)) throw std::runtime_error("기준 다시 잡기: 나사 " + std::to_string(k + 1) + " 를 찾지 못했다");
            gx += int(std::nearbyint(sx / s)); gy += int(std::nearbyint(sy / s));
        }
        screws[k] = {gx, gy}; rep.screw_shift[k] = screws[k] - g0.screws[k];
    }

    // 4. 바 48칸: 원래 기준 행 + QR 세로 이동의 절반 주변 ±20 행
    const int prior = int(std::nearbyint(mean_shift.y / 2));
    std::ostringstream lp; lp << "{\"blocks\": [";
    rep.bar_shift_min[0] = rep.bar_shift_min[1] = 1 << 20; rep.bar_shift_max[0] = rep.bar_shift_max[1] = -(1 << 20);
    {
        cv::FileStorage f((base / "line_protocol.json").u8string(), cv::FileStorage::READ); int n = 0;
        for (auto it = f["blocks"].begin(); it != f["blocks"].end(); ++it, ++n) {
            const auto b = *it; const auto rb = b["read_box"];
            const int x0 = int(rb[0]), x1 = int(rb[2]), ref0 = int(rb[1]) + 24, pol = int(b["edge_polarity"]);
            if (pol != 1 && pol != -1) throw std::runtime_error("기준 다시 잡기: line_protocol.json 에 edge_polarity 가 없다");
            const auto gnow = edge_strength(now, x0, x1, ref0 + prior, pol), gold = edge_strength(old, x0, x1, ref0, pol);
            int best = kBarHalf - kBarSearch;
            for (int i = kBarHalf - kBarSearch; i <= kBarHalf + kBarSearch; ++i) if (gnow[i] > gnow[best]) best = i;
            if (!(gnow[best] >= .5 * gold[kBarHalf])) throw std::runtime_error("기준 다시 잡기: 바 칸 " + std::to_string(n) + " 의 가장자리를 찾지 못했다");
            const int ref = ref0 + prior - kBarHalf + best, bar = n < 24 ? 0 : 1;
            rep.bar_shift_min[bar] = std::min(rep.bar_shift_min[bar], ref - ref0); rep.bar_shift_max[bar] = std::max(rep.bar_shift_max[bar], ref - ref0);
            std::string id; b["id"] >> id;
            lp << (n ? ", " : "") << "{\"id\": \"" << id << "\", \"x0\": " << x0 << ", \"x1\": " << x1 << ", \"reference_y\": " << ref
               << ", \"read_box\": [" << x0 << ", " << ref - 24 << ", " << x1 << ", " << ref + 25 << "], \"edge_polarity\": " << pol << '}';
        }
        if (n != 48) throw std::runtime_error("기준 다시 잡기: 바 칸은 48개여야 한다");
    }
    lp << "], \"scope\": \"rows re-located on a new center by final00 reference\"}";

    // 5. 쓰기
    fs::create_directories(out);
    const auto bmp = encode_bmp(now);
    { std::ofstream f(out / "center.bmp", std::ios::binary); f.write(reinterpret_cast<const char*>(bmp.data()), std::streamsize(bmp.size())); }
    write_text(out / "points.json", pts.str()); write_text(out / "geometry.json", geo.str()); write_text(out / "line_protocol.json", lp.str());
    const cv::Point crop_shift(int(std::nearbyint(mean_shift.x)), int(std::nearbyint(mean_shift.y)));
    std::ostringstream st;
    st << "{\n \"center_sha256\": \"" << final00_sha::sha256_hex(bmp) << "\",\n \"full_crop\": " << rect_json(g0.full_crop + crop_shift)
       << ",\n \"qr\": [" << rect_json(boxes[0]) << ", " << rect_json(boxes[1]) << ", " << rect_json(boxes[2]) << "],\n \"screws\": [";
    for (int k = 0; k < 4; ++k) st << (k ? ", " : "") << '[' << screws[k].x << ", " << screws[k].y << ']';
    st << "],\n \"search_x\": " << g0.search_x << ",\n \"search_y\": " << g0.search_y << ",\n \"note\": \"final00 reference from " << base.filename().u8string() << "\"\n}\n";
    write_text(out / "site.json", st.str());
    rep.folder = out.u8string();
    return rep;
}
}
