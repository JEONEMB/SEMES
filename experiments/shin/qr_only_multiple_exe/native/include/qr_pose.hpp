#pragma once
#include <array>
#include <functional>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace qr {
using Pose = cv::Vec2d; // mm, degree
using Position = cv::Vec2i; // micrometre, millidegree
using Features = cv::Vec4d;
cv::FileStorage open_model(const std::string& path);
cv::Mat matrix(const cv::FileStorage& file,const std::string& key);
void check_frame(const cv::Mat& gray);
cv::Point2d match(const cv::Mat& frame, const cv::Mat& patch, cv::Point center,
                  int sx, int sy, double minimum, bool subpixel=true);
std::vector<cv::Point> centers(const cv::Mat& coordinates);

class FeaturePose {
public:
    explicit FeaturePose(const std::string& path);
    Pose predict(const cv::Mat& frame) const;
private:
    cv::Mat coarse_, zero_, anchors_, beta_, rotate_, horizontal_;
    std::vector<cv::Point> centers_;
    std::array<cv::Mat,3> patches_, inverse_, polynomial_;
    std::array<std::vector<cv::Point2f>,3> points_;
    std::vector<double> anchor_sq_, anchor_zero_;
    double width_, wide_;
    bool anchored_;
    std::array<double,4> weight_;
    double kernel(double distance) const;
};

class Geometry {
public:
    Geometry() = default;
    Geometry(const cv::Mat& reference,const std::vector<cv::Point>& centers);
    Geometry(const cv::FileStorage& model,const std::vector<cv::Point>& centers);
    Features predict(const cv::Mat& frame) const;
    cv::Point seed(const cv::Mat& frame) const;
private:
    Features measure(const cv::Mat& frame) const;
    std::vector<cv::Point> centers_;
    std::array<cv::Mat,3> templates_;
    Features zero_{0.,0.,0.,0.};
};

class Cancellation {
public:
    Cancellation(const std::string& path,const cv::Mat& reference);
    Pose predict(const cv::Mat& frame) const;
    double horizontal(const cv::Mat& frame) const;
private:
    Geometry geometry_;
    cv::Mat coefficients_;
    std::vector<cv::Point> centers_;
    std::array<cv::Mat,3> patches_;
    double zero_dx_;
};

enum class SsafyAlignmentAction { center_candidate, theta_alignment, two_axis_alignment };
const char* name(SsafyAlignmentAction action);
struct SsafyAlignmentStep {
    SsafyAlignmentAction action;
    Pose residual;
    double horizontal_px;
    Position relative_move;
};

class SsafyAligner {
public:
    explicit SsafyAligner(const std::string& model_path);
    Features observe(const cv::Mat& frame) const;
    Pose residual(Features observation, Pose target) const;
    double rig_horizontal(Features observation, Pose target) const;
    SsafyAlignmentStep step(Features observation, Pose target = {0.,0.}) const;
private:
    double horizontal_error(Features observation, Pose target) const;
    Geometry geometry_;
    cv::Mat coefficients_;
    Pose horizontal_response_;
};

struct SsafyAlignmentResult { const char* reason; int moves; int two_axis_moves; };
SsafyAlignmentResult run_ssafy_alignment(
    const SsafyAligner& aligner, Pose target, const std::function<Features()>& capture,
    const std::function<bool(Position)>& move_relative, int max_moves = 20);

struct Fit { Pose pose; double pixel_rmse; int iterations; };
class DirectPose {
public:
    DirectPose(const std::string& path,const std::string& domain,const cv::Mat& reference={});
    Pose initial(const cv::Mat& frame) const;
    Fit refine(const cv::Mat& frame,Pose seed) const;
    static bool ssafy_supported(Pose initial);
private:
    Geometry geometry_;
    cv::Mat coefficients_, initializer_, intercept_, intensity_;
    Pose limit_;
    std::vector<cv::Point> centers_;
    std::vector<cv::Point2d> normalized_;
    cv::Point2d origin_;
    cv::Vec3d energy_;
};

// The robot adapter owns capture, rectification and blocking stage I/O.
struct Stage {
    virtual ~Stage() = default;
    virtual Position position() = 0;
    virtual std::array<int,4> limits() = 0;
    virtual Position move_absolute(Position target) = 0;
};
struct Settings { Position center{0,0}; int max_moves=20; int cancellation_limit=3; };
struct ControlResult { std::string reason; Position reported; int moves; int corrections; };
using Event = std::function<void(const char*,Position,Pose)>;
constexpr double rig_horizontal_slope_px_per_degree = -85.;
ControlResult run_correction(Stage& stage,const Settings& settings,
    const std::function<double()>& horizontal,const std::function<Pose()>& estimate,
    const Event& event={});
ControlResult run_ssafy_rig_alignment(Stage& stage, const Settings& settings,
    const SsafyAligner& model, Pose visual_target, const std::function<Features()>& capture,
    const Event& event={});

cv::Mat read_frame(const std::string& path);
}
