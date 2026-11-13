#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


// QR 정보들 구조체
struct QRResult { 
    string text;                // 해석한 문자열
    vector<Point2f> corners;    // 코너 좌표를 저장할 벡터
    Point2f center;             // 센터 좌표
};


// 이미지에서 QR를 찾는 커스텀 함수
vector<QRResult> findQRs(const Mat& bgr) {
    Mat gray, bin;

    cvtColor(bgr, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, bin, Size(5, 5), 0); // Qorl

    threshold(bin, bin, 130, 255, THRESH_BINARY);              // 판(어두움) vs QR 바탕(밝음)

    morphologyEx(bin, bin, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(9, 9))); // QR 내부 검은 모듈 메우기


    vector<vector<Point>> contours;

    findContours(bin, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    QRCodeDetector det;
    vector<QRResult> out;

    const int PAD = 40;


    for (auto& c : contours) {
        Rect r = boundingRect(c);
        double ar = (double)r.width / r.height;

        if (r.width < 120 || r.width > 300 || ar < 0.8 || ar > 1.25) continue;  // QR 크기 후보만

        Mat roi;

        copyMakeBorder(gray(r), roi, PAD, PAD, PAD, PAD, BORDER_CONSTANT, Scalar(255)); // 여백 추가

        vector<Point2f> pts;
        string s = det.detectAndDecode(roi, pts);

        if (s.empty() || pts.size() != 4) continue;

        QRResult q; q.text = s; q.center = Point2f(0, 0);
        for (auto& p : pts) {
            Point2f g(p.x + r.x - PAD, p.y + r.y - PAD);           // 원본 좌표로 되돌림
            q.corners.push_back(g); q.center += g;
        }
        q.center *= 0.25f;
        out.push_back(q);
    }
    sort(out.begin(), out.end(), [](const QRResult& a, const QRResult& b) { return a.center.x < b.center.x; });
    return out;
}


// 메인 함수
int main() {

    // 이미지 로드
    Mat img = imread("./img/center.bmp", IMREAD_COLOR);
    // 불러올 수 없는 경우 예외 처리
    if (img.empty()) { cout << "이미지를 로드할 수 없습니다." << endl; return -1; }


    // 커스텀 함수로 QR들의 정보를 저장
    vector<QRResult> qrs = findQRs(img);
    cout << "검출 QR 수 : " << qrs.size() << '\n';


    // 원본 이미지 보호를 위해 이미지 복제
    Mat view = img.clone();

    // 찾은 QR마다 반복
    for (size_t i = 0; i < qrs.size(); ++i) {
        // QR 해석값 : QR center 위치
        cout << '[' << i << "] " << qrs[i].text << " center= " << qrs[i].center << '\n';

        // 4개의 꼭짓점 좌표(정수)
        vector<Point> quad(qrs[i].corners.begin(), qrs[i].corners.end());
        for (int d = 0; d < quad.size(); d++) {
            cout << d+1 << "번: " << quad[d].y << ", " << quad[d].x << '\n';
        }

        // QR 박스 표시
        polylines(view, quad, true, Scalar(0, 255, 0), 3);
        // QR 중점 원 표시
        circle(view, qrs[i].center, 6, Scalar(0, 0, 255), FILLED);
        // QR 해석값 출력
        putText(view, qrs[i].text, qrs[i].center + Point2f(-60, -110), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 255), 2);
    }

    // 이미지를 보여줄 윈도우
    namedWindow("Window", WINDOW_NORMAL);
    // 사이즈 조절
    resizeWindow("Window", 1280, 1280 * view.rows / view.cols);
    // 이미지 보여줌
    imshow("Window", view);
    // 키가 눌릴때 까지 대기 - 0이면 아무키 눌릴때 까지 대기
    waitKey(0);
    // 모든 창을 닫음
    destroyAllWindows();
    // 종료
    return 0;
}