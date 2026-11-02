// 03: Canny -> standard Hough transform (S03) -> select the target boundary -> refit on Canny pixels.
// Hough theta is the angle of the line NORMAL; the image line angle is theta - 90 deg.
#include "common.hpp"
#include "methods.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>

namespace lm {

MethodResult method_canny_hough(MethodContext& c) {
    MethodResult r;
    const TargetSpec& t = c.target;
    const json& p = c.params;
    const double kDeg = 3.14159265358979323846 / 180.0;
    cv::Rect reg = t.detector_region(c.gray.size());
    cv::Mat src = c.gray(reg), blur;
    cv::GaussianBlur(src, blur, cv::Size(0, 0), p.at("blur_sigma").get<double>(), 0, cv::BORDER_REFLECT101);
    cv::Mat edges;
    cv::Canny(blur, edges, p.at("canny_low").get<double>(), p.at("canny_high").get<double>(),
              p.at("canny_aperture").get<int>(), p.at("l2gradient").get<bool>());
    cv::Mat gx, gy;
    cv::Sobel(blur, gx, CV_32F, 1, 0, 3);
    cv::Sobel(blur, gy, CV_32F, 0, 1, 3);
    // Vote only with Canny pixels of the target polarity, inside band / x range / visible mask.
    cv::Mat filt = cv::Mat::zeros(edges.size(), CV_8U);
    int n_canny = 0, n_filt = 0;
    std::vector<cv::Point2d> edge_px;
    for (int y = 0; y < edges.rows; ++y)
        for (int x = 0; x < edges.cols; ++x) {
            if (!edges.at<uchar>(y, x)) continue;
            ++n_canny;
            double X = x + reg.x, Y = y + reg.y;
            float vx = gx.at<float>(y, x), vy = gy.at<float>(y, x);
            if (!t.in_band(Y) || !t.in_x(X) || t.excluded(X)) continue;
            if (t.polarity * vy <= 0 || std::fabs(vy) <= std::fabs(vx)) continue;
            filt.at<uchar>(y, x) = 255;
            edge_px.emplace_back(X, Y);
            ++n_filt;
        }
    c.sink.save_png("canny_edges", edges, reg.tl(), r.images);
    c.sink.save_png("canny_polarity_band_filtered", filt, reg.tl(), r.images);

    std::vector<cv::Vec3f> lines;
    cv::HoughLines(filt, lines, p.at("hough_rho").get<double>(), p.at("hough_theta_deg").get<double>() * kDeg,
                   p.at("hough_threshold").get<int>(), 0, 0, p.at("theta_min_deg").get<double>() * kDeg,
                   p.at("theta_max_deg").get<double>() * kDeg);
    r.n_candidates = (int)lines.size();
    int max_saved = p.at("max_candidates_saved").get<int>();
    json cand = json::array();
    int sel = -1;
    Line hline;
    for (size_t i = 0; i < lines.size(); ++i) {
        double rho = lines[i][0], th = lines[i][1], votes = lines[i][2];
        double ct = std::cos(th), st = std::sin(th);
        double rho_o = rho + reg.x * ct + reg.y * st;  // back to original coordinates
        Line l = normalize_line(rho_o * ct, rho_o * st, -st, ct);
        double ya = (rho_o - t.x0 * ct) / st, yb = (rho_o - t.x1 * ct) / st;
        std::string state;
        if (!t.in_band(ya) || !t.in_band(yb)) state = "excluded:outside_band";
        else if (sel < 0) { state = "selected"; sel = (int)i; hline = l; }
        else state = "not_selected:fewer_votes";
        if ((int)i < max_saved || state == "selected")
            cand.push_back({{"rank", (int)i}, {"rho_original_px", rho_o}, {"theta_normal_deg", th / kDeg}, {"votes", votes},
                            {"image_angle_deg", line_angle_deg(l)}, {"y_at_x0", ya}, {"y_at_x1", yb}, {"state", state}});
    }
    r.intermediates["hough_candidates"] = cand;
    r.diagnostics["canny_pixels_in_region"] = n_canny;
    r.diagnostics["voting_pixels_after_polarity_band_mask"] = n_filt;
    r.diagnostics["hough_lines_returned"] = (int)lines.size();
    r.diagnostics["accumulator_note"] = "OpenCV HoughLines does not expose the accumulator; votes per returned line are saved.";
    if (sel < 0) {
        r.fail("detection_failed", lines.empty() ? "no Hough line above vote threshold" : "no Hough line inside the target band");
        return r;
    }
    r.diagnostics["hough_selected_angle_deg"] = line_angle_deg(hline);
    r.diagnostics["hough_selected_line"] = line_json(hline);
    // Refit: Canny pixels (integer) within refit distance of the selected Hough line.
    double dmax = p.at("refit_dist_px").get<double>();
    std::vector<cv::Point2d> near;
    json pts = json::array();
    for (auto& q : edge_px) {
        bool keep = std::fabs(signed_distance(hline, q.x, q.y)) <= dmax;
        if (keep) near.push_back(q);
        pts.push_back({(int)q.x, (int)q.y, keep ? 1 : 0});
    }
    r.intermediates["voting_pixels_fields"] = {"x", "y", "within_refit_dist"};
    r.intermediates["voting_pixels"] = pts;
    finish_with_points(r, near, c.fit, "canny_integer_pixels within refit_dist of the selected Hough line");
    return r;
}

}  // namespace lm
