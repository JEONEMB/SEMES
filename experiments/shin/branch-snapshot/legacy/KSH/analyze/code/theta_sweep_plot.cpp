// Θ축 스윕 이미지(-1.00 ~ +1.00 °, 0.01 ° 간격)에서 QR 중점을 추출해 그래프로 그리는 코드
// - 입력 : img/center.bmp (Θ = 0), img/Theta/T+0.01.bmp ... img/Theta/T-1.00.bmp
// - 출력 : theta_sweep_result.csv (수치), theta_sweep_plot.png (그래프), 화면 표시
// - 그래프 3개
//   (1) 중앙 QR 중점 x  vs Θ  : 직선 피팅 → px/° 감도 (Y 스윕 결과가 있으면 Y 환산 mm/° 도 표시)
//   (2) 좌·우 QR 간격 d = xR - xL vs Θ : 원근 때문에 Θ 에 따라 아주 조금 변함
//       ※ 윤곽선 무게중심으로 잰 d 는 잡음(약 0.15 px)이 Θ 변화량보다 커서 Θ 를 구분하기 어려움
//         → 이 그래프는 '중점만으로는 Θ 와 Y 를 구분할 수 없다'는 근거로 사용 (구분은 yt_separation.cpp 의 ECC 방식)
//   (3) 중앙 QR 중점 y  vs Θ
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// 측정 결과 한 건
struct Sample {
    double deg;      // 로봇 Θ 각도 (°)
    Point2f c;       // 중앙 QR 중점 (px)
    bool hasLR;      // 좌·우 QR 모두 찾았는지
    double d;        // 좌·우 QR 간격 xR - xL (px)
};

const double QR_PITCH = 257.0;   // QR 사이 간격 (px, center.bmp 기준)
const double PITCH_TOL = 50.0;   // 이웃 QR 판정 허용 오차 (px)

// 이미지에서 QR 바탕(밝은 사각형) 후보들의 중점을 찾음 (y_sweep_plot.cpp 와 동일)
vector<Point2f> findQRCenters(const Mat& gray) {
    Mat bin;
    GaussianBlur(gray, bin, Size(5, 5), 0);
    threshold(bin, bin, 130, 255, THRESH_BINARY);                                       // 판(어두움) vs QR 바탕(밝음)
    morphologyEx(bin, bin, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(9, 9))); // QR 내부 검은 모듈 메우기

    vector<vector<Point>> contours;
    findContours(bin, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    vector<Point2f> centers;
    for (auto& c : contours) {
        Rect r = boundingRect(c);
        double ar = (double)r.width / r.height;
        if (r.width < 120 || r.width > 300 || ar < 0.8 || ar > 1.25) continue; // QR 크기·정사각형 조건
        if (contourArea(c) < 0.85 * r.area()) continue;                         // 속이 꽉 찬 사각형만

        Moments m = moments(c);
        centers.emplace_back((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
    }
    return centers;
}

// p 에서 dx 만큼 떨어진 곳(같은 높이)에 있는 이웃 QR 의 인덱스, 없으면 -1
int findNeighbor(const vector<Point2f>& cs, const Point2f& p, double dx) {
    int best = -1;
    double bestErr = PITCH_TOL;
    for (size_t i = 0; i < cs.size(); ++i) {
        double ex = fabs((cs[i].x - p.x) - dx);
        double ey = fabs(cs[i].y - p.y);
        if (ey < PITCH_TOL && ex < bestErr) { bestErr = ex; best = (int)i; }
    }
    return best;
}

// 중앙 QR 고르기
// Θ 가 커지면 세 QR 이 모두 옆으로 밀리므로 "영상 중심에 가장 가까운 후보"만으로는 위험함
// → 좌·우에 ±피치 간격으로 이웃 QR 이 둘 다 있는 후보를 우선 선택
bool pickCenterQR(const vector<Point2f>& cs, float midX, Sample& s) {
    int bestC = -1, bestL = -1, bestR = -1;
    double bestDist = 1e9;
    for (size_t i = 0; i < cs.size(); ++i) {
        int l = findNeighbor(cs, cs[i], -QR_PITCH);
        int r = findNeighbor(cs, cs[i], +QR_PITCH);
        if (l < 0 || r < 0) continue;
        double dist = fabs(cs[i].x - midX);
        if (dist < bestDist) { bestDist = dist; bestC = (int)i; bestL = l; bestR = r; }
    }
    if (bestC >= 0) {
        s.c = cs[bestC];
        s.hasLR = true;
        s.d = cs[bestR].x - cs[bestL].x;
        return true;
    }
    // 좌·우 중 하나를 못 찾은 경우: 영상 중심에 가장 가까운 후보를 중앙으로 사용 (간격 d 는 기록하지 않음)
    if (cs.empty()) return false;
    s.c = *min_element(cs.begin(), cs.end(),
        [&](const Point2f& p, const Point2f& q) { return fabs(p.x - midX) < fabs(q.x - midX); });
    s.hasLR = false;
    s.d = 0;
    return true;
}

// 최소제곱 직선 y = k*x + b, 결정계수 R^2
void linearFit(const vector<double>& x, const vector<double>& y, double& k, double& b, double& r2) {
    size_t n = x.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < n; ++i) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; }
    k = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    b = (sy - k * sx) / n;
    double my = sy / n, ssTot = 0, ssRes = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = y[i] - (k * x[i] + b);
        ssRes += e * e;
        ssTot += (y[i] - my) * (y[i] - my);
    }
    r2 = (ssTot > 0) ? 1.0 - ssRes / ssTot : 1.0;
}

// Y 스윕 결과(y_sweep_result.csv)가 있으면 Y 기울기(px/mm)를 읽어 옴 → Θ 1° 가 Y 몇 mm 처럼 보이는지 환산
bool loadYSlope(double& kY) {
    ifstream f("y_sweep_result.csv");
    if (!f.is_open()) return false;
    string line;
    getline(f, line);                       // 헤더
    vector<double> ys, cxs;
    while (getline(f, line)) {
        stringstream ss(line);
        string a, b;
        if (!getline(ss, a, ',') || !getline(ss, b, ',')) continue;
        ys.push_back(stod(a));
        cxs.push_back(stod(b));
    }
    if (ys.size() < 2) return false;
    double b0, r2;
    linearFit(ys, cxs, kY, b0, r2);
    return true;
}

// 눈금 간격을 1, 2, 5 × 10^n 형태로 맞춤
double niceStep(double raw) {
    double e = pow(10.0, floor(log10(raw)));
    double f = raw / e;
    double n = (f < 1.5) ? 1 : (f < 3) ? 2 : (f < 7) ? 5 : 10;
    return n * e;
}

// 그래프 한 칸(패널) 그리기
// minSpan : 세로축 최소 범위(px). 값이 거의 일정한 데이터를 과장해서 보여주지 않기 위함
// drawFit : 1차 직선 피팅선과 피팅 결과 표시 여부, unit : 피팅식에 쓰는 변수 이름
// ySlope  : Y 기울기(px/mm), 0 보다 크면 Y 환산 값도 표시
void drawPanel(Mat& img, Rect a, const vector<double>& xs, const vector<double>& ys,
               const string& title, double minSpan, bool drawFit, const char* unit, double ySlope = 0) {
    const double x0 = -1.0, x1 = 1.0;
    double lo = *min_element(ys.begin(), ys.end());
    double hi = *max_element(ys.begin(), ys.end());
    double mid = (lo + hi) / 2.0;
    double span = max(hi - lo, minSpan) * 1.15;
    double y0 = mid - span / 2.0, y1 = mid + span / 2.0;

    auto px = [&](double x) { return a.x + (int)lround((x - x0) / (x1 - x0) * a.width); };
    auto py = [&](double y) { return a.y + a.height - (int)lround((y - y0) / (y1 - y0) * a.height); };

    const Scalar grid(232, 232, 232), axis(140, 140, 140), text(60, 60, 60);
    char buf[200];

    // 세로 격자 + X 눈금 (0.25 ° 간격)
    for (double x = x0; x <= x1 + 1e-9; x += 0.25) {
        int X = px(x);
        line(img, Point(X, a.y), Point(X, a.y + a.height), grid, 1);
        snprintf(buf, sizeof(buf), "%+.2f", x);
        putText(img, buf, Point(X - 22, a.y + a.height + 20), FONT_HERSHEY_SIMPLEX, 0.45, text, 1, LINE_AA);
    }
    // 가로 격자 + Y 눈금
    double st = niceStep(span / 5.0);
    for (double y = ceil(y0 / st) * st; y <= y1; y += st) {
        int Y = py(y);
        line(img, Point(a.x, Y), Point(a.x + a.width, Y), grid, 1);
        snprintf(buf, sizeof(buf), (st < 1.0) ? "%.2f" : "%.1f", y);
        putText(img, buf, Point(a.x - 80, Y + 5), FONT_HERSHEY_SIMPLEX, 0.45, text, 1, LINE_AA);
    }
    rectangle(img, a, axis, 1);
    putText(img, title, Point(a.x, a.y - 14), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(20, 20, 20), 1, LINE_AA);

    // 1차 직선 피팅
    if (drawFit) {
        double k, b, r2;
        linearFit(xs, ys, k, b, r2);
        line(img, Point(px(x0), py(k * x0 + b)), Point(px(x1), py(k * x1 + b)), Scalar(60, 60, 220), 2, LINE_AA);

        double maxRes = 0, sumSq = 0;
        for (size_t i = 0; i < xs.size(); ++i) {
            double e = ys[i] - (k * xs[i] + b);
            maxRes = max(maxRes, fabs(e));
            sumSq += e * e;
        }
        double rms = sqrt(sumSq / xs.size());
        snprintf(buf, sizeof(buf), "fit: %s = %.3f * T + %.2f   (%.3f px/deg = %.5f deg/px)", unit, k, b, k, 1.0 / k);
        putText(img, buf, Point(a.x + 12, a.y + 26), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(60, 60, 220), 1, LINE_AA);
        snprintf(buf, sizeof(buf), "R^2 = %.5f   residual rms = %.3f px (%.4f deg), max = %.3f px",
                 r2, rms, rms / fabs(k), maxRes);
        putText(img, buf, Point(a.x + 12, a.y + 50), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(60, 60, 220), 1, LINE_AA);
        if (ySlope > 0) {
            snprintf(buf, sizeof(buf), "Y-equivalent: 1 deg of T = %.2f mm of Y   (Y slope %.3f px/mm)", k / ySlope, ySlope);
            putText(img, buf, Point(a.x + 12, a.y + 74), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(20, 130, 20), 1, LINE_AA);
        }

        cout << fixed << setprecision(4)
             << "[직선 피팅] " << unit << " : 기울기 " << k << " px/deg (" << setprecision(5) << 1.0 / k << " deg/px)"
             << setprecision(3) << ", 절편 " << b << " px, R^2 " << setprecision(6) << r2
             << ", 잔차 RMS " << setprecision(3) << rms << " px (" << setprecision(4) << rms / fabs(k) << " deg)\n";
        if (ySlope > 0)
            cout << "    Y 환산 : Θ 1° = Y " << setprecision(3) << k / ySlope << " mm\n";
    }

    // 측정점
    for (size_t i = 0; i < xs.size(); ++i)
        circle(img, Point(px(xs[i]), py(ys[i])), 3, Scalar(170, 110, 20), FILLED, LINE_AA);
}

int main() {
    vector<Sample> data;

    // -1.00 ~ +1.00 °, 0.01 ° 간격 (0 ° 는 center.bmp)
    for (int i = -100; i <= 100; ++i) {
        double deg = i * 0.01;
        string path;
        if (i == 0) {
            path = "img/center.bmp";
        } else {
            char buf[64];
            snprintf(buf, sizeof(buf), "img/Theta/T%+.2f.bmp", deg);   // 예: img/Theta/T+0.35.bmp, img/Theta/T-0.07.bmp
            path = buf;
        }

        Mat gray = imread(path, IMREAD_GRAYSCALE);
        if (gray.empty()) { cout << "[건너뜀] " << path << " 로드 실패\n"; continue; }

        vector<Point2f> cs = findQRCenters(gray);
        Sample s;
        s.deg = deg;
        if (!pickCenterQR(cs, gray.cols * 0.5f, s)) { cout << "[건너뜀] " << path << " QR 후보 없음\n"; continue; }

        data.push_back(s);
        cout << showpos << fixed << setprecision(2) << deg << noshowpos << " deg -> center ("
             << s.c.x << ", " << s.c.y << ")";
        if (s.hasLR) cout << "   d = " << setprecision(2) << s.d << " px";
        else         cout << "   (좌·우 QR 중 일부 미검출)";
        cout << "   후보 " << cs.size() << "개\n";
    }

    if (data.size() < 2) { cout << "그래프를 그릴 데이터가 부족합니다.\n"; return -1; }

    // CSV 저장 (d 를 못 구한 경우 빈 칸)
    ofstream csv("theta_sweep_result.csv");
    csv << "Theta_deg,center_x_px,center_y_px,spacing_d_px\n" << fixed << setprecision(3);
    vector<double> xs, cxs, cys, dxs, ds;
    for (auto& s : data) {
        csv << s.deg << ',' << s.c.x << ',' << s.c.y << ',';
        if (s.hasLR) { csv << s.d; dxs.push_back(s.deg); ds.push_back(s.d); }
        csv << '\n';
        xs.push_back(s.deg); cxs.push_back(s.c.x); cys.push_back(s.c.y);
    }
    csv.close();

    double kY = 0;
    if (loadYSlope(kY)) cout << "\n[참고] y_sweep_result.csv 에서 Y 기울기 " << fixed << setprecision(3) << kY << " px/mm 읽음\n";
    else                cout << "\n[참고] y_sweep_result.csv 가 없어 Y 환산 값은 표시하지 않습니다.\n";
    cout << "\n";

    // 그래프: 위 = 중앙 QR 중점 x, 가운데 = 좌·우 QR 간격 d, 아래 = 중앙 QR 중점 y
    Mat plot(1240, 1240, CV_8UC3, Scalar::all(255));
    drawPanel(plot, Rect(120, 60, 1070, 430), xs, cxs, "Center QR  -  center x (px)  vs  robot Theta (deg)", 0, true, "cx", kY);
    if (ds.size() >= 2)
        drawPanel(plot, Rect(120, 580, 1070, 300), dxs, ds, "Left-Right QR spacing  d = xR - xL (px)  vs  robot Theta (deg)", 1, true, "d");
    drawPanel(plot, Rect(120, 970, 1070, 200), xs, cys, "Center QR  -  center y (px)  vs  robot Theta (deg)", 10, false, "cy");
    putText(plot, "robot Theta (deg)", Point(560, 1222), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(20, 20, 20), 1, LINE_AA);

    imwrite("theta_sweep_plot.png", plot);
    cout << "\n저장: theta_sweep_result.csv, theta_sweep_plot.png (" << data.size() << "개 이미지, 간격 d "
         << ds.size() << "개)\n";

    namedWindow("Theta sweep", WINDOW_NORMAL);
    resizeWindow("Theta sweep", 1000, 1000);
    imshow("Theta sweep", plot);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
