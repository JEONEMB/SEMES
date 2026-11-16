// 확인 앱용 알고리즘 실행기: alignment(cv::Mat&) 를 별도 프로세스로 돌린다.
// (알고리즘이 죽거나 멈춰도 웹 서버는 살아 있고, 다음 요청 때 다시 띄운다.)
//
// 통신 (stdin/stdout, 바이너리 입력 + 한 줄 텍스트 출력)
//   시작:  "READY[ 기능...]\n"   기능: ref = 영점 등록(ALGR) 가능, roi = QR·나사 위치를 같이 돌려줌
//   추론:  "ALGN" + uint32 width + uint32 height (리틀엔디언) + width*height 바이트 (8bit 흑백)
//          → "OK <y> <t> <ms>[ | 이름 x y w h cx cy; ...]\n"  또는  "ERR <메시지>\n" (측정 실패 = NaN 도 ERR)
//          ROI 는 ALGO_HAS_REFERENCE 일 때만 (QR_L/QR_C/QR_R 사각형, 나사 S_LB/S_RT)
//   영점:  "ALGR" + 같은 형식의 센터 사진 → alignment_set_reference
//          → "OK <영점 y> <영점 t>[ | ROI...]\n"  또는  "ERR <메시지>\n"
//
// 옵션: --gray         흑백 1채널 그대로 넘김 (기본은 SEMES BMP 를 imread 한 것과 같은 BGR 3채널)
//       --config PATH  알고리즘 설정 파일 (alignment_init) 또는 자료 폴더 (initialize_joint)
//
// ALGO_HAS_REFERENCE: 김성현 A507 버전처럼 alignment_init / alignment_set_reference /
//   alignment_last_info / alignment_reference_info 가 있는 알고리즘 (CMake -DALGO_DIR=... 이 켬)
// ALGO_HAS_JOINT: 이혜원 값 7개 모듈처럼 initialize_joint(자료 폴더) 를 먼저 부르고 alignment() 를 쓰는 알고리즘
//   (CMake -DHYE_DIR=... -DHYE_MODULE=... 이 켬). 기본은 자료의 절대 영점. ALGO_JOINT_ZERO(joint_reference.hpp 가 있는
//   버전)면 ALGR = set_joint_zero (측정이 안 될 만큼 화면이 밀렸으면 set_joint_reference 뒤 다시)
// ALGO_HYE2: 이혜원 SSAFY 패키지 (CMake -DHYE2_DIR=... 이 켬). --config = data/model 폴더.
//   alignment::initialize → ALGR = set_zero_from_photos → measure_alignment. correction_allowed 가 아니면 ERR <코드> <메시지>.
//   학습 범위 밖 경고(W_OUT_OF_RANGE)만 있으면 보정 허용 (OK 뒤에 ' | 경고코드' 를 붙임)
// ALGO_HAS_QR_RIG: 신윤석 QR 전용 Rig 모델 qr::Cancellation (CMake -DQR_RIG_DIR=.../qr_only_multiple_exe/native 가 켬).
//   --config 는 rig-cancellation.xml.gz. 센터 기준 사진으로 모델을 만들고(ALGR) 그 사진 대비 Y·Θ 를 낸다.
//   QR_RIG_JOINT(979c1d75~) 이고 --config 의 kind 가 qr_affine_joint 면(예: rig-reverse-fixed.xml.gz)
//   qr::JointVisualModel: 기준 사진 대비 QR 3개 아핀 값 18개 → Y·Θ
// ALGO_HAS_QR_POSE: 신윤석 QR 전용 단일 프레임 추정 qr_pose::estimate (CMake -DQR_POSE_DIR=... 이 켬).
//   --config 는 보정 파일(yml, 환경변수 QR_POSE_MODEL 로 넘김). 영점은 그 파일의 센터 템플릿에 고정
//
// 알고리즘 코드가 printf/std::cout 로 찍는 내용이 통신을 깨지 않도록
// 표준 출력(fd 1)은 표준 에러로 돌리고, 통신은 따로 복제한 fd 로 한다.
#include "alignment.hpp"
#include <cmath>
#include <cstdio>
#include <string>

// 화면 윤곽용 위치 한 개: "이름 x y w h cx cy" (알고리즘 좌표). 여러 개는 " | a; b; ..." 로 붙인다
static void add_roi(std::string &out, const char *name, double cx, double cy, int w, int h)
{
    if (!std::isfinite(cx) || !std::isfinite(cy))
        return;
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s%s %d %d %d %d %.2f %.2f", out.empty() ? " | " : "; ", name,
                  static_cast<int>(std::lround(cx - w / 2.0)), static_cast<int>(std::lround(cy - h / 2.0)), w, h, cx, cy);
    out += buf;
}

#ifdef ALGO_HYE2
#include "common/reference.hpp"
#include "setup/find_new_reference.hpp"
struct Alignment_t { double y; double t; };   // 이 패키지 헤더는 namespace alignment 라 공용 구조체가 없음
static std::string g_hye2_note;              // 마지막 측정의 경고 코드 (화면 기록용)
// 화면 윤곽용 (알고리즘 계산에는 쓰지 않음): 모듈 안의 기준과 같은 사본, 영점 사진에서 찾은 QR·나사 위치와 그때의 이동량
static alignment::Reference g_hye_ref;
static bool g_hye_ref_ok = false;
static cv::Point2d g_hye_qr0[3], g_hye_screw0[4];
static double g_hye_qr_dx0 = NAN, g_hye_screw_dx0[4] = {NAN, NAN, NAN, NAN};
static bool g_hye_zero_pos = false;
static std::string g_hye2_rois;
static const char *kHyeQr[3] = {"QR_L", "QR_C", "QR_R"};
static const char *kHyeScrew[4] = {"S1", "S2", "S3", "S4"};

// 영점 사진에서 기준 사진 조각(가운데 c, 크기 w×h)을 찾는다 (가로 ±sx, 세로 ±sy). 못 찾으면 NaN
static cv::Point2d hye_locate(const cv::Mat &gray, cv::Point c, int w, int h, int sx, int sy)
{
    const cv::Rect patch(c.x - w / 2, c.y - h / 2, w, h), win(patch.x - sx, patch.y - sy, w + 2 * sx, h + 2 * sy);
    const cv::Rect all(0, 0, gray.cols, gray.rows);
    if ((patch & all) != patch || (win & all) != win || g_hye_ref.photo.empty())
        return {NAN, NAN};
    cv::Mat r;
    cv::matchTemplate(gray(win), g_hye_ref.photo(patch), r, cv::TM_CCOEFF_NORMED);
    double best;
    cv::Point at;
    cv::minMaxLoc(r, nullptr, &best, nullptr, &at);
    if (best < 0.6)
        return {NAN, NAN};
    return {static_cast<double>(c.x + at.x - sx), static_cast<double>(c.y + at.y - sy)};
}

// 영점 사진: QR·나사 실제 위치와 그때의 가로 이동량을 기억하고 목표 윤곽 줄을 만든다
static std::string hye_zero_rois(const cv::Mat &gray)
{
    std::string out;
    g_hye_zero_pos = false;
    if (!g_hye_ref_ok)
        return out;
    const alignment::Result d = alignment::measure_diagnostic(gray);
    g_hye_qr_dx0 = d.qr_shift_px;
    for (int k = 0; k < 4; ++k)
        g_hye_screw_dx0[k] = d.screw_shift_px[k];
    for (int i = 0; i < 3; ++i) {
        const cv::Rect q = g_hye_ref.qr[i];
        g_hye_qr0[i] = hye_locate(gray, {q.x + q.width / 2, q.y + q.height / 2}, q.width, q.height, 150, 40);
        add_roi(out, kHyeQr[i], g_hye_qr0[i].x, g_hye_qr0[i].y, q.width, q.height);
    }
    for (int k = 0; k < 4; ++k) {
        g_hye_screw0[k] = hye_locate(gray, g_hye_ref.screw[k], 100, 100, 80, 40);
        add_roi(out, kHyeScrew[k], g_hye_screw0[k].x, g_hye_screw0[k].y, 160, 160);
    }
    g_hye_zero_pos = true;
    return out;
}

// 측정한 사진: 영점 사진 위치 + (지금 가로 이동 − 영점 때 가로 이동). 세로는 이 알고리즘이 재지 않음
static std::string hye_now_rois(const alignment::Result &m)
{
    std::string out;
    if (!g_hye_zero_pos)
        return out;
    const double dq = m.qr_shift_px - g_hye_qr_dx0;
    for (int i = 0; i < 3; ++i)
        add_roi(out, kHyeQr[i], g_hye_qr0[i].x + dq, g_hye_qr0[i].y, g_hye_ref.qr[i].width, g_hye_ref.qr[i].height);
    for (int k = 0; k < 4; ++k)
        add_roi(out, kHyeScrew[k], g_hye_screw0[k].x + (m.screw_shift_px[k] - g_hye_screw_dx0[k]), g_hye_screw0[k].y, 160, 160);
    return out;
}
#endif
#ifdef ALGO_JOINT_ZERO
#include "joint_reference.hpp"
#endif
#ifdef ALGO_HAS_QR_POSE
#include "qr_pose.hpp"
#include <cstdlib>
#include <fstream>
#endif
#ifdef ALGO_HAS_QR_RIG
#include "qr_pose.hpp"
#include <fstream>
#include <memory>
static std::string g_rig_model;                 // rig-cancellation.xml.gz
static std::unique_ptr<qr::Cancellation> g_rig; // 센터 기준 사진으로 만든 모델 (ALGR 전에는 없음)
#ifdef QR_RIG_JOINT
static std::unique_ptr<qr::JointVisualModel> g_joint;   // 비전 상대 보정 계수 (시작 때 읽음)
static std::unique_ptr<qr::Geometry> g_joint_geo;       // 센터 기준 사진의 QR 아핀 값 (ALGR)
static std::string g_joint_rois;                        // 화면 윤곽: QR 3개 중심 (기준 + 아핀 이동)
static const char *kRigQr[3] = {"QR_L", "QR_C", "QR_R"};
static std::string joint_rois(const qr::AffineFeatures *f)
{
    std::string out;
    const auto &c = g_joint->qr_centers();
    for (int i = 0; i < 3 && i < static_cast<int>(c.size()); ++i)
        add_roi(out, kRigQr[i], c[i].x + (f ? (*f)[6 * i] : 0.0), c[i].y + (f ? (*f)[6 * i + 1] : 0.0), 159, 173);
    return out;
}
#endif
#endif

#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

static bool read_all(void *buf, size_t n)
{
    return std::fread(buf, 1, n, stdin) == n;
}

static std::string one_line(std::string s)
{
    for (char &c : s)
        if (c == '\n' || c == '\r')
            c = ' ';
    return s;
}

#ifdef ALGO_HAS_REFERENCE
// 측정한 QR·나사 위치 (알고리즘 좌표): " | 이름 x y w h cx cy; ..." — 화면에 무엇을 보고 계산했는지 그리기 위해
static std::string rois_text(const std::vector<AlignmentRoi> &rois)
{
    std::string out;
    char buf[160];
    for (const AlignmentRoi &r : rois) {
        if (!r.ok)
            continue;
        std::snprintf(buf, sizeof buf, "%s%s %d %d %d %d %.2f %.2f", out.empty() ? " | " : "; ", r.name.c_str(),
                      r.ref_rect_px.x, r.ref_rect_px.y, r.ref_rect_px.width, r.ref_rect_px.height,
                      r.center_px.x, r.center_px.y);
        out += buf;
    }
    return out;
}
#endif

// 사진 한 장 → (y mm, t °). 알고리즘마다 진입 함수가 다르다
static Alignment_t run_algorithm(cv::Mat &src)
{
#ifdef ALGO_HYE2
    const alignment::Result m = alignment::measure_alignment(src);
    if (!m.correction_allowed)
        throw std::runtime_error((m.code.empty() ? std::string("보정 불가") : m.code) + " " + m.message);
    g_hye2_note = m.code;
    g_hye2_rois = hye_now_rois(m);
    Alignment_t r;
    r.y = m.y_mm;
    r.t = m.t_deg;
    return r;
#elif defined(ALGO_HAS_QR_RIG)
    cv::Mat gray;
    if (src.channels() == 1)
        gray = src;
    else
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
#ifdef QR_RIG_JOINT
    if (g_joint) {
        if (!g_joint_geo)
            throw std::runtime_error("센터 기준 사진을 먼저 등록해야 함 (이 알고리즘은 기준 사진 대비로만 잰다)");
        const qr::AffineFeatures f = g_joint_geo->affine(gray);
        g_joint_rois = joint_rois(&f);
        const qr::Pose p = g_joint->predict(f);
        Alignment_t r;
        r.y = p[0];
        r.t = p[1];
        return r;
    }
#endif
    if (!g_rig)
        throw std::runtime_error("센터 기준 사진을 먼저 등록해야 함 (이 알고리즘은 기준 사진 대비로만 잰다)");
    const qr::Pose p = g_rig->predict(gray);
    Alignment_t r;
    r.y = p[0];
    r.t = p[1];
    return r;
#elif defined(ALGO_HAS_QR_POSE)
    const qr_pose::Pose p = qr_pose::estimate(src);
    Alignment_t r;
    r.y = p.y_mm;
    r.t = p.theta_deg;
    return r;
#else
    return alignment(src);
#endif
}

static std::string last_message()
{
#ifdef ALGO_HAS_REFERENCE
    return one_line(alignment_last_info().message);
#else
    return "";
#endif
}

int main(int argc, char **argv)
{
    bool color = true;
    std::string config;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gray") == 0)
            color = false;
        else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc)
            config = argv[++i];
    }

    std::fflush(stdout);
    const int proto_fd = dup(STDOUT_FILENO);
    dup2(STDERR_FILENO, STDOUT_FILENO);
    FILE *proto = fdopen(proto_fd, "w");
    if (!proto)
        return 3;

#ifdef ALGO_HAS_REFERENCE
    if (!(config.empty() ? alignment_init() : alignment_init(config))) {
        std::fprintf(proto, "ERR 설정 파일을 읽지 못함: %s\n", config.empty() ? "(기본)" : config.c_str());
        std::fflush(proto);
        return 4;
    }
    std::fprintf(proto, "READY ref roi\n");
#elif defined(ALGO_HAS_JOINT)
    try {
        if (config.empty())
            throw std::runtime_error("--config 로 자료(assets) 폴더를 주세요");
        initialize_joint(config);
    } catch (const std::exception &e) {
        std::fprintf(proto, "ERR 초기화 실패: %s\n", one_line(e.what()).c_str());
        std::fflush(proto);
        return 4;
    }
#ifdef ALGO_JOINT_ZERO
    std::fprintf(proto, "READY ref\n");
#else
    std::fprintf(proto, "READY\n");
#endif
#elif defined(ALGO_HYE2)
    {
        const std::string folder = config.empty() ? std::string("data/model") : config;
        const alignment::Result r = alignment::initialize(folder);
        if (r.status == alignment::Status::Failed) {
            std::fprintf(proto, "ERR 초기화 실패: %s %s\n", r.code.c_str(), one_line(r.message).c_str());
            std::fflush(proto);
            return 4;
        }
        std::string err;
        g_hye_ref_ok = alignment::load_reference(folder, g_hye_ref, err);   // 윤곽용 사본 (모듈과 같은 자료)
        if (!g_hye_ref_ok)
            std::fprintf(stderr, "[hye2] 윤곽용 기준 읽기 실패: %s\n", err.c_str());
    }
    std::fprintf(proto, "READY ref\n");
#elif defined(ALGO_HAS_QR_RIG)
    if (config.empty() || !std::ifstream(config)) {
        std::fprintf(proto, "ERR --config 로 rig-cancellation.xml.gz 를 주세요: %s\n", config.c_str());
        std::fflush(proto);
        return 4;
    }
    g_rig_model = config;
#ifdef QR_RIG_JOINT
    try {
        cv::FileStorage f(config, cv::FileStorage::READ);
        if (f.isOpened() && std::string(f["kind"]) == "qr_affine_joint")
            g_joint.reset(new qr::JointVisualModel(config));
    } catch (const std::exception &e) {
        std::fprintf(proto, "ERR 모델을 읽지 못함: %s\n", one_line(e.what()).c_str());
        std::fflush(proto);
        return 4;
    }
#endif
    std::fprintf(proto, "READY ref\n");
#elif defined(ALGO_HAS_QR_POSE)
    if (!config.empty()) {
        if (!std::ifstream(config)) {
            std::fprintf(proto, "ERR 보정 파일을 열 수 없음: %s\n", config.c_str());
            std::fflush(proto);
            return 4;
        }
        setenv("QR_POSE_MODEL", config.c_str(), 1);   // 첫 estimate() 때 이 파일을 읽는다
    }
    std::fprintf(proto, "READY\n");
#else
    if (!config.empty())
        std::fprintf(stderr, "--config 무시: 이 알고리즘은 설정 파일 API 가 없음\n");
    std::fprintf(proto, "READY\n");
#endif
    std::fflush(proto);

    for (;;) {
        char magic[4];
        uint32_t w = 0, h = 0;
        if (!read_all(magic, 4))
            return 0;  // 입력 끝 = 정상 종료
        const bool is_ref = std::memcmp(magic, "ALGR", 4) == 0;
        if ((!is_ref && std::memcmp(magic, "ALGN", 4) != 0) || !read_all(&w, 4) || !read_all(&h, 4)) {
            std::fprintf(proto, "ERR bad header\n");
            std::fflush(proto);
            return 2;
        }
        if (w == 0 || h == 0 || w > 10000 || h > 10000) {
            std::fprintf(proto, "ERR bad size %ux%u\n", w, h);
            std::fflush(proto);
            return 2;
        }
        cv::Mat gray(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
        if (!read_all(gray.data, static_cast<size_t>(w) * h))
            return 0;

        cv::Mat src;
        if (color)
            cv::cvtColor(gray, src, cv::COLOR_GRAY2BGR);
        else
            src = gray;

        try {
            if (is_ref) {
#ifdef ALGO_HAS_REFERENCE
                if (alignment_set_reference(src, false)) {
                    const ReferenceInfo &ri = alignment_reference_info();
                    std::fprintf(proto, "OK %.9g %.9g%s\n", ri.zero_y_mm, ri.zero_t_deg, rois_text(ri.rois).c_str());
                } else {
                    std::fprintf(proto, "ERR %s\n", one_line(alignment_reference_info().message).c_str());
                }
#elif defined(ALGO_HYE2)
                alignment::Result z = alignment::set_zero_from_photos({gray});
                std::string how;
                if (z.status == alignment::Status::Failed) {
                    // 영점 사진 점검 실패 (예: QR 판과 나사 관계가 학습 때와 달라짐) → 그 사진으로 QR·나사 기준을 다시 등록하고 영점 재설정
                    const alignment::Result ref = alignment::set_reference_from_photos({gray});
                    std::fprintf(stderr, "[hye2] 영점 실패(%s) → 기준 다시 등록: %s %s\n", z.code.c_str(), ref.code.c_str(),
                                 one_line(ref.message).c_str());
                    if (ref.status != alignment::Status::Failed) {
                        z = alignment::set_zero_from_photos({gray});
                        how = "(기준 다시 등록)";
                        if (g_hye_ref_ok) {   // 모듈 안의 기준과 같게 사본도 갱신 (같은 함수, 같은 사진)
                            alignment::Reference fresh;
                            std::string err;
                            g_hye_ref_ok = alignment::find_new_reference(g_hye_ref, gray, fresh, err);
                            if (g_hye_ref_ok)
                                g_hye_ref = fresh;
                        }
                    }
                }
                if (z.status == alignment::Status::Failed)
                    std::fprintf(proto, "ERR %s %s\n", z.code.c_str(), one_line(z.message).c_str());
                else {   // 돌려주는 값 = 그 사진의 자세 (기준 사진 대비) + 목표 윤곽
                    if (!how.empty())
                        std::fprintf(stderr, "[hye2] %s\n", how.c_str());
                    std::fprintf(proto, "OK %.9g %.9g%s\n", z.y_mm, z.t_deg, hye_zero_rois(gray).c_str());
                }
#elif defined(ALGO_HAS_QR_RIG)
#ifdef QR_RIG_JOINT
                if (g_joint) {
                    std::unique_ptr<qr::Geometry> geo(new qr::Geometry(gray, g_joint->qr_centers()));
                    g_joint_geo = std::move(geo);
                    g_joint_rois.clear();
                } else
#endif
                {
                    std::unique_ptr<qr::Cancellation> m(new qr::Cancellation(g_rig_model, gray));   // 실패하면 예전 모델 그대로
                    g_rig = std::move(m);
                }
                // 이 모델은 기준 사진 대비로만 재서 '학습 영점 대비' 값이 없다 → 0 0
#ifdef QR_RIG_JOINT
                std::fprintf(proto, "OK 0 0%s\n", g_joint ? joint_rois(nullptr).c_str() : "");
#else
                std::fprintf(proto, "OK 0 0\n");
#endif
#elif defined(ALGO_JOINT_ZERO)
                Alignment_t z;
                try {
                    z = set_joint_zero(src);                 // 보통: 측정 기준은 그대로, 영점만 그 사진으로
                } catch (const std::exception &) {
                    set_joint_reference(src);               // 측정이 안 될 만큼 화면이 밀렸으면 측정 기준부터 다시
                    z = set_joint_zero(src);
                }
                // 돌려주는 값 = 그 센터 사진의 절대 영점 기준 자세 (성현 alignment_set_reference 의 '학습 영점 대비' 와 같은 뜻)
                std::fprintf(proto, "OK %.9g %.9g\n", static_cast<double>(z.y), static_cast<double>(z.t));
#else
                std::fprintf(proto, "ERR 이 알고리즘은 영점 등록을 지원하지 않음\n");
#endif
            } else {
                const auto t0 = std::chrono::steady_clock::now();
                const Alignment_t r = run_algorithm(src);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (std::isnan(static_cast<double>(r.y)) || std::isnan(static_cast<double>(r.t)))
                    std::fprintf(proto, "ERR 측정 실패 %s\n", last_message().c_str());
                else
#ifdef ALGO_HAS_REFERENCE
                    std::fprintf(proto, "OK %.9g %.9g %.3f%s\n", static_cast<double>(r.y), static_cast<double>(r.t), ms,
                                 rois_text(alignment_last_info().rois).c_str());
#elif defined(ALGO_HYE2)
                    std::fprintf(proto, "OK %.9g %.9g %.3f%s\n", static_cast<double>(r.y), static_cast<double>(r.t), ms,
                                 g_hye2_rois.c_str());
#elif defined(ALGO_HAS_QR_RIG) && defined(QR_RIG_JOINT)
                    std::fprintf(proto, "OK %.9g %.9g %.3f%s\n", static_cast<double>(r.y), static_cast<double>(r.t), ms,
                                 g_joint ? g_joint_rois.c_str() : "");
#else
                    std::fprintf(proto, "OK %.9g %.9g %.3f\n", static_cast<double>(r.y), static_cast<double>(r.t), ms);
#endif
            }
        } catch (const std::exception &e) {
            std::fprintf(proto, "ERR %s\n", one_line(e.what()).c_str());
        } catch (...) {
            std::fprintf(proto, "ERR unknown exception\n");
        }
        std::fflush(proto);
    }
}
