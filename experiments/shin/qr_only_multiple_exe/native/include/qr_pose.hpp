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
ControlResult run_correction(Stage& stage,const Settings& settings,
    const std::function<double()>& horizontal,const std::function<Pose()>& estimate,
    const Event& event={});
cv::Mat read_frame(const std::string& path);
}
