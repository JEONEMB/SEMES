// Y·Θ 분리 검증 코드
// ------------------------------------------------------------------
// 입력  : img/center.bmp (Y=0, Θ=0 기준 이미지)
//         img/Y/Y+0.05.bmp ... Y-2.00.bmp       (Y만 변화, 0.05 mm 간격)
//         img/Theta/T+0.01.bmp ... T-2.00.bmp   (Θ만 변화, 0.01 deg 간격, 최대 ±2 deg — 있는 파일만 사용)
// 처리  : 1) 기준 이미지에서 Plate(QR 3개) 영역을 템플릿으로 저장
//         2) 각 이미지에서 중앙 QR로 대략 위치를 잡고, ECC(호모그래피)로 Plate 전체를 서브픽셀 정합
//         3) 특징 3개 계산
//              cx : 중앙 QR 중심 x (px)                 → Y, Θ 둘 다에 크게 반응
//              d  : (R-C) - (C-L) QR 간격 비대칭 (px)   → Θ에만 크게 반응
//              s  : 좌·우 QR 위치의 가로 배율 차 (%)    → Θ에만 크게 반응
//         4) 짝수 번째 이미지로 감도 행렬 A 학습:  [cx d s]ᵀ = A·[Y Θ]ᵀ + f0
//         5) 홀수 번째 이미지로 가중 최소제곱 역산 → (Y, Θ) 추정 오차 평가
// 출력  : yt_features.csv, yt_calib.yml, yt_separation.png, 화면 표시
// ------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// ============================== QR 후보 검출 ==============================
// 어두운 판 위의 밝은 QR 바탕 사각형을 찾아 윤곽 무게중심을 반환 (대략 위치용)
static vector<Point2f> qrCandidates(const Mat& gray) {
    Mat bin;
    GaussianBlur(gray, bin, Size(5, 5), 0);
    threshold(bin, bin, 130, 255, THRESH_BINARY);
    morphologyEx(bin, bin, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(9, 9)));
    vector<vector<Point>> contours;
    findContours(bin, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    vector<Point2f> out;
    for (auto& c : contours) {
        Rect r = boundingRect(c);
        double ar = (double)r.width / r.height;
        if (r.width < 120 || r.width > 300 || ar < 0.8 || ar > 1.25) continue;
        if (contourArea(c) < 0.85 * r.area()) continue;
        Moments m = moments(c);
        out.emplace_back((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
    }
    sort(out.begin(), out.end(), [](const Point2f& a, const Point2f& b) { return a.x < b.x; });
    return out;
}

// 영상 가로 중심에 가장 가까운 후보 = 중앙 QR
static bool findCenterQR(const Mat& gray, Point2f& c, int* index = nullptr) {
    vector<Point2f> cs = qrCandidates(gray);
    if (cs.empty()) return false;
    float mid = gray.cols * 0.5f;
    int best = 0;
    for (int i = 1; i < (int)cs.size(); ++i)
        if (fabs(cs[i].x - mid) < fabs(cs[best].x - mid)) best = i;
    c = cs[best];
    if (index) *index = best;
    return true;
}

// QR 내용을 해독해 이름이 'C'로 끝나는지 확인 (예: FP11C)
static bool decodeIsCenter(const Mat& gray, const Point2f& p) {
    const int h = 110, pad = 40;
    Rect r((int)p.x - h, (int)p.y - h, 2 * h, 2 * h);
    r &= Rect(0, 0, gray.cols, gray.rows);
    if (r.area() == 0) return false;
    Mat roi;
    copyMakeBorder(gray(r), roi, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(255));
    QRCodeDetector det;
    string s = det.detectAndDecode(roi);
    return !s.empty() && s.back() == 'C';
}

// 중앙 QR 선택 (Θ 최대 ±2°, Y 최대 ±2mm 처럼 크게 움직여도 동작)
// 1순위: 좌·우 모두 QR 간격(pitch)만큼 떨어진 이웃이 있는 후보
// 2순위: QR 해독 결과가 'C'로 끝나는 후보
// 3순위: 영상 가로 중심에 가장 가까운 후보
static bool findCenterQRRobust(const Mat& gray, float pitch, Point2f& c) {
    vector<Point2f> cs = qrCandidates(gray);
    if (cs.empty()) return false;
    auto hasNeighbor = [&](const Point2f& a, float dx) {
        for (auto& b : cs)
            if (fabs(b.x - (a.x + dx)) < 0.1f * pitch && fabs(b.y - a.y) < 30.f) return true;
        return false;
    };
    float mid = gray.cols * 0.5f;
    int best = -1;
    for (int i = 0; i < (int)cs.size(); ++i)
        if (hasNeighbor(cs[i], -pitch) && hasNeighbor(cs[i], pitch))
            if (best < 0 || fabs(cs[i].x - mid) < fabs(cs[best].x - mid)) best = i;
    if (best >= 0) { c = cs[best]; return true; }
    for (auto& p : cs) if (decodeIsCenter(gray, p)) { c = p; return true; }
    return findCenterQR(gray, c);
}

// ============================== Plate 정합기 ==============================
class PlateTracker {
public:
    // 기준 이미지에서 L/C/R 위치와 템플릿 저장
    bool init(const Mat& refGray) {
        vector<Point2f> cs = qrCandidates(refGray);
        int ci;
        Point2f c;
        if (!findCenterQR(refGray, c, &ci) || ci == 0 || ci == (int)cs.size() - 1) return false;
        refC = c;
        roi = Rect((int)lround(c.x) - HALF_W, (int)lround(c.y) - HALF_H, 2 * HALF_W, 2 * HALF_H);
        if ((roi & Rect(0, 0, refGray.cols, refGray.rows)) != roi) return false;

        refGray(roi).convertTo(tmpl, CV_32F);
        GaussianBlur(tmpl, tmpl, Size(5, 5), 0);
        Point2f o((float)roi.x, (float)roi.y);
        pts[0] = cs[ci - 1] - o;   // L
        pts[1] = cs[ci] - o;       // C
        pts[2] = cs[ci + 1] - o;   // R
        pitch = (cs[ci + 1].x - cs[ci - 1].x) * 0.5f;   // QR 간격 (px)
        return true;
    }

    // 특징 측정: cx(px), d(px), s(%)
    bool measure(const Mat& gray, double& cx, double& d, double& s) const {
        Point2f c;
        if (!findCenterQRRobust(gray, pitch, c)) return false;
        double dx = c.x - refC.x, dy = c.y - refC.y;
        double gx = roi.x + dx, gy = roi.y + dy;
        int ox = (int)lround(gx) - MARGIN, oy = (int)lround(gy) - MARGIN;
        Rect wr(ox, oy, roi.width + 2 * MARGIN, roi.height + 2 * MARGIN);
        if ((wr & Rect(0, 0, gray.cols, gray.rows)) != wr) return false;

        Mat win;
        gray(wr).convertTo(win, CV_32F);
        GaussianBlur(win, win, Size(5, 5), 0);

        // 초기값: 대략 위치만큼 평행이동한 호모그래피
        Mat H = Mat::eye(3, 3, CV_32F);
        H.at<float>(0, 2) = (float)(MARGIN + (gx - lround(gx)));
        H.at<float>(1, 2) = (float)(MARGIN + (gy - lround(gy)));
        try {
            // 최대 100회 또는 변화량 1e-6 이하에서 종료 (300회·1e-8 대비 약 10배 빠르고 정밀도 차이는 작음)
            findTransformECC(tmpl, win, H, MOTION_HOMOGRAPHY,
                             TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6), noArray());
        } catch (const cv::Exception&) {
            return false;   // 수렴 실패
        }
        Mat Hd;
        H.convertTo(Hd, CV_64F);

        // 템플릿 좌표 → 원본 이미지 좌표
        auto mapPt = [&](double x, double y) {
            double X = Hd.at<double>(0, 0) * x + Hd.at<double>(0, 1) * y + Hd.at<double>(0, 2);
            double Y = Hd.at<double>(1, 0) * x + Hd.at<double>(1, 1) * y + Hd.at<double>(1, 2);
            double W = Hd.at<double>(2, 0) * x + Hd.at<double>(2, 1) * y + Hd.at<double>(2, 2);
            return Point2d(X / W + ox, Y / W + oy);
        };
        // 해당 위치의 가로 배율 (템플릿 1px 이 이미지에서 몇 px 인가)
        auto xScale = [&](const Point2f& p) {
            return (mapPt(p.x + 1.0, p.y).x - mapPt(p.x - 1.0, p.y).x) / 2.0;
        };

        double xL = mapPt(pts[0].x, pts[0].y).x;
        double xC = mapPt(pts[1].x, pts[1].y).x;
        double xR = mapPt(pts[2].x, pts[2].y).x;
        cx = xC;
        d = (xR - xC) - (xC - xL);
        s = (xScale(pts[2]) - xScale(pts[0])) * 100.0;
        return true;
    }

private:
    static const int HALF_W = 400, HALF_H = 125, MARGIN = 60;
    Point2f refC, pts[3];
    float pitch = 258.f;
    Rect roi;
    Mat tmpl;
};

// ============================== 데이터 구조 ==============================
struct Sample {
    int set;              // 0 = Y 세트, 1 = Θ 세트
    double trueY, trueT;  // 파일명 기준 정답
    double f[3];          // cx, d, s
    bool train;           // 학습용(짝수 번째) / 검증용(홀수 번째)
    double estY = 0, estT = 0;
};

static string makePath(int set, int i) {
    if (i == 0) return "img/center.bmp";
    char buf[64];
    if (set == 0) snprintf(buf, sizeof(buf), "img/Y/Y%+.2f.bmp", i * 0.05);
    else          snprintf(buf, sizeof(buf), "img/Theta/T%+.2f.bmp", i * 0.01);
    return buf;
}

// ============================== 그래프 도우미 ==============================
static double niceStep(double raw) {
    double e = pow(10.0, floor(log10(raw)));
    double f = raw / e;
    return ((f < 1.5) ? 1 : (f < 3) ? 2 : (f < 7) ? 5 : 10) * e;
}

struct Axes {
    Rect a;
    double x0, x1, y0, y1;
    Point px(double x, double y) const {
        return Point(a.x + (int)lround((x - x0) / (x1 - x0) * a.width),
                     a.y + a.height - (int)lround((y - y0) / (y1 - y0) * a.height));
    }
    bool inside(double x, double y) const { return x >= x0 && x <= x1 && y >= y0 && y <= y1; }
};

static void rangeOf(const vector<vector<double>>& vs, double& lo, double& hi, double pad = 0.08) {
    lo = 1e300; hi = -1e300;
    for (auto& v : vs) for (double x : v) { lo = min(lo, x); hi = max(hi, x); }
    double span = max(hi - lo, 1e-9);
    lo -= span * pad; hi += span * pad;
}

static string fmtTick(double v, double step) {
    char buf[32];
    int dec = (step >= 1) ? 0 : (step >= 0.1) ? 1 : (step >= 0.01) ? 2 : 3;
    if (fabs(v) < step * 1e-6) v = 0.0;   // "-0" 표기 방지
    snprintf(buf, sizeof(buf), "%.*f", dec, v);
    return buf;
}

static void drawAxes(Mat& img, const Axes& A, const string& title, const string& xl, const string& yl) {
    const Scalar grid(232, 232, 232), text(60, 60, 60);
    double sx = niceStep((A.x1 - A.x0) / 6), sy = niceStep((A.y1 - A.y0) / 5);
    for (double x = ceil(A.x0 / sx) * sx; x <= A.x1; x += sx) {
        Point p = A.px(x, A.y0);
        line(img, Point(p.x, A.a.y), Point(p.x, A.a.y + A.a.height), grid, 1);
        putText(img, fmtTick(x, sx), Point(p.x - 16, A.a.y + A.a.height + 18), FONT_HERSHEY_SIMPLEX, 0.42, text, 1, LINE_AA);
    }
    for (double y = ceil(A.y0 / sy) * sy; y <= A.y1; y += sy) {
        Point p = A.px(A.x0, y);
        line(img, Point(A.a.x, p.y), Point(A.a.x + A.a.width, p.y), grid, 1);
        putText(img, fmtTick(y, sy), Point(A.a.x - 62, p.y + 5), FONT_HERSHEY_SIMPLEX, 0.42, text, 1, LINE_AA);
    }
    rectangle(img, A.a, Scalar(140, 140, 140), 1);
    putText(img, title, Point(A.a.x, A.a.y - 30), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(20, 20, 20), 1, LINE_AA);
    putText(img, yl, Point(A.a.x - 62, A.a.y - 10), FONT_HERSHEY_SIMPLEX, 0.42, text, 1, LINE_AA);
    putText(img, xl, Point(A.a.x + A.a.width / 2 - 70, A.a.y + A.a.height + 40), FONT_HERSHEY_SIMPLEX, 0.45, text, 1, LINE_AA);
}

static void scatter(Mat& img, const Axes& A, const vector<double>& x, const vector<double>& y, const Scalar& col) {
    for (size_t i = 0; i < x.size(); ++i)
        if (A.inside(x[i], y[i])) circle(img, A.px(x[i], y[i]), 3, col, FILLED, LINE_AA);
}

static void dataLine(Mat& img, const Axes& A, double xa, double ya, double xb, double yb, const Scalar& col, int th = 1) {
    line(img, A.px(xa, ya), A.px(xb, yb), col, th, LINE_AA);
}

// ============================== main ==============================
int main() {
    const double TOL_UM = 90.0;    // Y 허용 오차 (µm)
    const double TOL_DEG = 0.09;   // Θ 허용 오차 (deg)
    const Scalar BLUE(180, 110, 20), ORANGE(40, 130, 240), RED(60, 60, 220), GREEN(80, 160, 80);

    Mat ref = imread("img/center.bmp", IMREAD_GRAYSCALE);
    if (ref.empty()) { cout << "img/center.bmp 를 열 수 없습니다.\n"; return -1; }
    PlateTracker tracker;
    if (!tracker.init(ref)) { cout << "기준 이미지에서 QR 3개를 찾지 못했습니다.\n"; return -1; }

    // ---------------- 1. 특징 측정 ----------------
    vector<Sample> data;
    TickMeter tm;
    int nMeasured = 0;
    for (int set = 0; set < 2; ++set) {
        int N = (set == 0) ? 40 : 200;   // Y: ±2.00 mm (0.05 간격), Θ: ±2.00 deg (0.01 간격)
        int k = 0, missing = 0;
        for (int i = -N; i <= N; ++i) {
            string path = makePath(set, i);
            Mat g = imread(path, IMREAD_GRAYSCALE);
            if (g.empty()) { ++missing; continue; }   // 없는 파일은 건너뜀

            Sample sm;
            sm.set = set;
            sm.trueY = (set == 0) ? i * 0.05 : 0.0;
            sm.trueT = (set == 1) ? i * 0.01 : 0.0;
            tm.start();
            bool ok = tracker.measure(g, sm.f[0], sm.f[1], sm.f[2]);
            tm.stop();
            if (!ok) { cout << "[건너뜀] " << path << " 정합 실패\n"; continue; }
            ++nMeasured;
            sm.train = (k++ % 2 == 0);
            data.push_back(sm);
        }
        cout << (set == 0 ? "Y" : "Θ") << " 세트 측정 완료 (" << k << "장, 파일 없음 " << missing << "개)\n";
    }
    if (nMeasured < 10) { cout << "측정된 이미지가 너무 적습니다.\n"; return -1; }
    cout << fixed << setprecision(1) << "이미지 1장당 정합 시간(이미지 로드 제외): " << tm.getTimeMilli() / nMeasured << " ms\n";

    // ---------------- 2. 감도 행렬 학습 (짝수 번째) ----------------
    // 특징 j 마다:  f_j = a_jY * Y + a_jT * Θ + f0_j
    Mat A(3, 2, CV_64F), f0(3, 1, CV_64F), sigma(3, 1, CV_64F);
    for (int j = 0; j < 3; ++j) {
        vector<double> X, z;
        for (auto& s : data) if (s.train) {
            X.push_back(s.trueY); X.push_back(s.trueT); X.push_back(1.0);
            z.push_back(s.f[j]);
        }
        Mat Xm((int)z.size(), 3, CV_64F, X.data()), zm((int)z.size(), 1, CV_64F, z.data()), c;
        solve(Xm, zm, c, DECOMP_SVD);
        A.at<double>(j, 0) = c.at<double>(0);
        A.at<double>(j, 1) = c.at<double>(1);
        f0.at<double>(j) = c.at<double>(2);
        Mat r = zm - Xm * c;
        sigma.at<double>(j) = sqrt(r.dot(r) / r.rows);   // 특징별 잡음 (학습 잔차)
    }

    // ---------------- 3. 검증 (홀수 번째): 가중 최소제곱 역산 ----------------
    Mat Wm = Mat::zeros(3, 3, CV_64F);
    for (int j = 0; j < 3; ++j) Wm.at<double>(j, j) = 1.0 / (sigma.at<double>(j) * sigma.at<double>(j));
    Mat N = A.t() * Wm * A, Ninv = N.inv();
    for (auto& s : data) {
        Mat fv = (Mat_<double>(3, 1) << s.f[0], s.f[1], s.f[2]);
        Mat x = Ninv * A.t() * Wm * (fv - f0);
        s.estY = x.at<double>(0);
        s.estT = x.at<double>(1);
    }

    // 오차 통계
    const char* names[3] = { "cx (px)", "d  (px)", "s  (%) " };
    cout << "\n[감도 행렬]          per mm(Y)     per deg(Θ)    잡음\n";
    for (int j = 0; j < 3; ++j)
        cout << "  " << names[j] << "   " << setw(10) << setprecision(4) << A.at<double>(j, 0)
             << "   " << setw(10) << A.at<double>(j, 1) << "   " << setw(8) << sigma.at<double>(j) << "\n";
    cout << setprecision(1) << "  Θ 1° 가 만드는 cx 이동 = Y " << A.at<double>(0, 1) / A.at<double>(0, 0) << " mm 와 같음\n";

    for (int set = 0; set < 2; ++set) {
        double sY = 0, sT = 0, mY = 0, mT = 0; int n = 0, within = 0, withinT = 0;
        for (auto& s : data) if (!s.train && s.set == set) {
            double eY = (s.estY - s.trueY) * 1000.0, eT = s.estT - s.trueT;
            sY += eY * eY; sT += eT * eT; mY = max(mY, fabs(eY)); mT = max(mT, fabs(eT));
            within += (fabs(eY) <= TOL_UM); withinT += (fabs(eT) <= TOL_DEG); ++n;
        }
        if (!n) continue;
        cout << "\n[검증: " << (set == 0 ? "Y 세트" : "Θ 세트") << ", " << n << "장]\n"
             << setprecision(1) << "  Y 오차  RMS " << sqrt(sY / n) << " um, 최대 " << mY << " um, "
             << TOL_UM << " um 이내 " << within << "/" << n << "\n"
             << setprecision(4) << "  Θ 오차  RMS " << sqrt(sT / n) << " deg, 최대 " << mT << " deg, "
             << TOL_DEG << " deg 이내 " << withinT << "/" << n << "\n";
    }
    cout << setprecision(4) << "  이론 표준편차: Y " << sqrt(Ninv.at<double>(0, 0)) * 1000 << " um, Θ "
         << sqrt(Ninv.at<double>(1, 1)) << " deg\n";

    // ---------------- 4. 저장 ----------------
    ofstream csv("yt_features.csv");
    csv << "set,train,trueY_mm,trueT_deg,cx_px,d_px,s_pct,estY_mm,estT_deg\n" << setprecision(5);
    for (auto& s : data)
        csv << (s.set == 0 ? "Y" : "T") << ',' << s.train << ',' << s.trueY << ',' << s.trueT << ','
            << s.f[0] << ',' << s.f[1] << ',' << s.f[2] << ',' << s.estY << ',' << s.estT << '\n';
    csv.close();

    FileStorage fs("yt_calib.yml", FileStorage::WRITE);   // 다른 장비/세션에서 재사용할 보정값
    fs << "A" << A << "f0" << f0 << "sigma" << sigma;
    fs.release();

    // ---------------- 5. 그래프 ----------------
    Mat plot(1000, 1260, CV_8UC3, Scalar::all(255));
    vector<double> yC, yD, yS, tC, tD, tS;
    for (auto& s : data) {
        double c = s.f[0] - f0.at<double>(0);
        if (s.set == 0) { yC.push_back(c); yD.push_back(s.f[1]); yS.push_back(s.f[2]); }
        else            { tC.push_back(c); tD.push_back(s.f[1]); tS.push_back(s.f[2]); }
    }

    // (1) 특징 공간: cx 이동 vs 간격 비대칭 d
    {
        Axes ax{ Rect(90, 70, 500, 340), 0, 0, 0, 0 };
        rangeOf({ yC, tC }, ax.x0, ax.x1); rangeOf({ yD, tD }, ax.y0, ax.y1);
        drawAxes(plot, ax, "(1) feature space: cx shift vs spacing asym d", "cx - cx0 (px)", "d (px)");
        scatter(plot, ax, yC, yD, BLUE); scatter(plot, ax, tC, tD, ORANGE);
        putText(plot, "Y set", Point(ax.a.x + 12, ax.a.y + 22), FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, LINE_AA);
        putText(plot, "Theta set", Point(ax.a.x + 12, ax.a.y + 44), FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, LINE_AA);
    }
    // (2) 특징 공간: cx 이동 vs 가로 배율 비대칭 s
    {
        Axes ax{ Rect(700, 70, 500, 340), 0, 0, 0, 0 };
        rangeOf({ yC, tC }, ax.x0, ax.x1); rangeOf({ yS, tS }, ax.y0, ax.y1);
        drawAxes(plot, ax, "(2) feature space: cx shift vs x-scale asym s", "cx - cx0 (px)", "s (%)");
        scatter(plot, ax, yC, yS, BLUE); scatter(plot, ax, tC, tS, ORANGE);
    }
    // (3) Θ 추정 vs 정답 (검증용)
    {
        vector<double> x, y;
        for (auto& s : data) if (!s.train && s.set == 1) { x.push_back(s.trueT); y.push_back(s.estT); }
        double lo, hi; rangeOf({ x, y, { -1.0, 1.0 } }, lo, hi, 0.05);
        Axes ax{ Rect(90, 560, 500, 340), lo, hi, lo, hi };
        drawAxes(plot, ax, "(3) Theta: estimated vs true (test images)", "true Theta (deg)", "est (deg)");
        dataLine(plot, ax, lo, lo, hi, hi, RED, 1);
        scatter(plot, ax, x, y, ORANGE);
    }
    // (4) Y 추정 오차 (µm), 두 세트 검증 이미지 전체
    {
        vector<double> xi, ey, xj, et;
        int idx = 0;
        for (auto& s : data) if (!s.train) {
            double e = (s.estY - s.trueY) * 1000.0;
            if (s.set == 0) { xi.push_back(idx); ey.push_back(e); }
            else            { xj.push_back(idx); et.push_back(e); }
            ++idx;
        }
        double lo, hi; rangeOf({ ey, et, { -TOL_UM * 1.2, TOL_UM * 1.2 } }, lo, hi, 0.05);
        Axes ax{ Rect(700, 560, 500, 340), -1, (double)idx, lo, hi };
        drawAxes(plot, ax, "(4) Y estimation error (test images)", "test image # (Y set | Theta set)", "err (um)");
        dataLine(plot, ax, -1, TOL_UM, idx, TOL_UM, GREEN, 1);
        dataLine(plot, ax, -1, -TOL_UM, idx, -TOL_UM, GREEN, 1);
        dataLine(plot, ax, -1, 0, idx, 0, Scalar(150, 150, 150), 1);
        scatter(plot, ax, xi, ey, BLUE); scatter(plot, ax, xj, et, ORANGE);
        putText(plot, "+-90 um", Point(ax.a.x + ax.a.width - 80, ax.px(0, TOL_UM).y - 6), FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, LINE_AA);
    }

    imwrite("yt_separation.png", plot);
    cout << "\n저장: yt_features.csv, yt_calib.yml, yt_separation.png\n";

    namedWindow("Y-Theta separation", WINDOW_NORMAL);
    resizeWindow("Y-Theta separation", 1260, 1000);
    imshow("Y-Theta separation", plot);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
