// Y축 스윕 이미지(-2.00 ~ +2.00 mm, 0.05 mm 간격)에서 중앙 QR 중점을 추출해 그래프로 그리는 코드
// - 입력 : img/center.bmp (Y = 0), img/Y/Y+0.05.bmp ... img/Y/Y-2.00.bmp
// - 출력 : y_sweep_result.csv (수치), y_sweep_plot.png (그래프), 화면 표시
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

// 측정 결과 한 건
struct Sample {
    double ymm;      // 로봇 Y 위치 (mm)
    Point2f c;       // 중앙 QR 중점 (px)
};

// 이미지에서 QR 바탕(밝은 사각형) 후보들의 중점을 찾음
// 중점은 QR 검출기의 꼭짓점 대신 윤곽선 무게중심(moments)을 사용 → 프레임 간 흔들림이 훨씬 작음
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

// 눈금 간격을 1, 2, 5 × 10^n 형태로 맞춤
double niceStep(double raw) {
    double e = pow(10.0, floor(log10(raw)));
    double f = raw / e;
    double n = (f < 1.5) ? 1 : (f < 3) ? 2 : (f < 7) ? 5 : 10;
    return n * e;
}

// 그래프 한 칸(패널) 그리기
// minSpan : 세로축 최소 범위(px). 값이 거의 일정한 데이터를 과장해서 보여주지 않기 위함
void drawPanel(Mat& img, Rect a, const vector<double>& xs, const vector<double>& ys,
               const string& title, double minSpan, bool drawFit) {
    const double x0 = -2.0, x1 = 2.0;
    double lo = *min_element(ys.begin(), ys.end());
    double hi = *max_element(ys.begin(), ys.end());
    double mid = (lo + hi) / 2.0;
    double span = max(hi - lo, minSpan) * 1.15;
    double y0 = mid - span / 2.0, y1 = mid + span / 2.0;

    auto px = [&](double x) { return a.x + (int)lround((x - x0) / (x1 - x0) * a.width); };
    auto py = [&](double y) { return a.y + a.height - (int)lround((y - y0) / (y1 - y0) * a.height); };

    const Scalar grid(232, 232, 232), axis(140, 140, 140), text(60, 60, 60);
    char buf[160];

    // 세로 격자 + X 눈금 (0.5 mm 간격)
    for (double x = x0; x <= x1 + 1e-9; x += 0.5) {
        int X = px(x);
        line(img, Point(X, a.y), Point(X, a.y + a.height), grid, 1);
        snprintf(buf, sizeof(buf), "%+.1f", x);
        putText(img, buf, Point(X - 18, a.y + a.height + 20), FONT_HERSHEY_SIMPLEX, 0.45, text, 1, LINE_AA);
    }
    // 가로 격자 + Y 눈금
    double st = niceStep(span / 5.0);
    for (double y = ceil(y0 / st) * st; y <= y1; y += st) {
        int Y = py(y);
        line(img, Point(a.x, Y), Point(a.x + a.width, Y), grid, 1);
        snprintf(buf, sizeof(buf), "%.1f", y);
        putText(img, buf, Point(a.x - 72, Y + 5), FONT_HERSHEY_SIMPLEX, 0.45, text, 1, LINE_AA);
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
        snprintf(buf, sizeof(buf), "fit: cx = %.3f * Y + %.2f   (%.2f px/mm = %.4f mm/px)", k, b, k, 1.0 / k);
        putText(img, buf, Point(a.x + 12, a.y + 26), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(60, 60, 220), 1, LINE_AA);
        snprintf(buf, sizeof(buf), "R^2 = %.5f   residual rms = %.3f px, max = %.3f px", r2, rms, maxRes);
        putText(img, buf, Point(a.x + 12, a.y + 50), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(60, 60, 220), 1, LINE_AA);

        cout << fixed << setprecision(4)
             << "\n[직선 피팅] 기울기 " << k << " px/mm (" << 1.0 / k << " mm/px), 절편 " << b
             << " px, R^2 " << setprecision(6) << r2
             << ", 잔차 RMS " << setprecision(3) << rms << " px\n";
    }

    // 측정점
    for (size_t i = 0; i < xs.size(); ++i)
        circle(img, Point(px(xs[i]), py(ys[i])), 4, Scalar(170, 110, 20), FILLED, LINE_AA);
}

int main() {
    vector<Sample> data;

    // -2.00 ~ +2.00 mm, 0.05 mm 간격 (0 mm 는 center.bmp)
    for (int i = -40; i <= 40; ++i) {
        double ymm = i * 0.05;
        string path;
        if (i == 0) {
            path = "img/center.bmp";
        } else {
            char buf[64];
            snprintf(buf, sizeof(buf), "img/Y/Y%+.2f.bmp", ymm);   // 예: img/Y/Y+0.20.bmp, img/Y/Y-1.55.bmp
            path = buf;
        }

        Mat gray = imread(path, IMREAD_GRAYSCALE);
        if (gray.empty()) { cout << "[건너뜀] " << path << " 로드 실패\n"; continue; }

        vector<Point2f> cs = findQRCenters(gray);
        if (cs.empty()) { cout << "[건너뜀] " << path << " QR 후보 없음\n"; continue; }

        // 중앙 QR = 영상 가로 중심에 가장 가까운 후보 (Y만 바뀌는 데이터에서는 좌·우 QR이 250px 이상 떨어져 있음)
        float midX = gray.cols * 0.5f;
        Point2f c = *min_element(cs.begin(), cs.end(),
            [&](const Point2f& p, const Point2f& q) { return fabs(p.x - midX) < fabs(q.x - midX); });

        data.push_back({ ymm, c });
        cout << showpos << fixed << setprecision(2) << ymm << noshowpos << " mm -> center ("
             << c.x << ", " << c.y << ")   후보 " << cs.size() << "개\n";
    }

    if (data.size() < 2) { cout << "그래프를 그릴 데이터가 부족합니다.\n"; return -1; }

    // CSV 저장
    ofstream csv("y_sweep_result.csv");
    csv << "Y_mm,center_x_px,center_y_px\n" << fixed << setprecision(3);
    vector<double> xs, cxs, cys;
    for (auto& s : data) {
        csv << s.ymm << ',' << s.c.x << ',' << s.c.y << '\n';
        xs.push_back(s.ymm); cxs.push_back(s.c.x); cys.push_back(s.c.y);
    }
    csv.close();

    // 그래프: 위 = 중점 x(px), 아래 = 중점 y(px)
    Mat plot(920, 1240, CV_8UC3, Scalar::all(255));
    drawPanel(plot, Rect(110, 60, 1080, 470), xs, cxs, "Center QR  -  center x (px)  vs  robot Y (mm)", 0, true);
    drawPanel(plot, Rect(110, 640, 1080, 200), xs, cys, "Center QR  -  center y (px)  vs  robot Y (mm)", 10, false);
    putText(plot, "robot Y position (mm)", Point(560, 895), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(20, 20, 20), 1, LINE_AA);

    imwrite("y_sweep_plot.png", plot);
    cout << "\n저장: y_sweep_result.csv, y_sweep_plot.png (" << data.size() << "개 이미지)\n";

    namedWindow("Y sweep", WINDOW_NORMAL);
    resizeWindow("Y sweep", 1240, 920);
    imshow("Y sweep", plot);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
