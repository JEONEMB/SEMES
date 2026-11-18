// Segment detectors 04 LSD (S04), 07 EDLines (S07), 08 FLD (S08), 09 ELSED (S09) and the common
// rule that turns their segments into ONE target line (no per-side angle averaging).
#include "common.hpp"
#include "methods.hpp"
#include "segment_support.hpp"
#include "effective_settings.hpp"

extern "C" {
#include "lsd.h"
}
#include "EDLines.h"
#include "ELSED.h"
#include "fld_port.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>

namespace lm {

using Seg = DetectedSegment;
using Piece = SegmentPiece;

MethodResult select_and_fit_segments(MethodContext& c, std::vector<DetectedSegment> segs, const std::string& detector) {
    MethodResult r;
    const TargetSpec& t = c.target;
    const json& sp = c.common.at("segment_selection").at(t.kind);
    const double max_ang = sp.at("max_abs_angle_deg").get<double>();
    const double min_len = sp.at("min_visible_len_px").get<double>();
    const double off = sp.at("side_offset_px").get<double>();
    const double min_con = sp.at("min_contrast").get<double>();
    const double tol = sp.at("consensus_inlier_px").get<double>();
    const double step = sp.at("sample_step_px").get<double>();
    cv::Mat g; c.gray.convertTo(g, CV_64F);  // bilinear sampling on the raw image

    struct Info { Line l; double angle, vis_len = 0, contrast = 0; std::vector<Piece> pieces; std::string state; };
    std::vector<Info> info(segs.size());
    std::vector<int> cand;
    for (size_t i = 0; i < segs.size(); ++i) {
        Seg& s = segs[i];
        if (s.x2 < s.x1) { std::swap(s.x1, s.x2); std::swap(s.y1, s.y2); }
        Info& in = info[i];
        in.l = normalize_line(s.x1, s.y1, s.x2 - s.x1, s.y2 - s.y1);
        in.angle = line_angle_deg(in.l);
        if (std::fabs(in.angle) > max_ang) { in.state = "excluded:angle"; continue; }
        if (std::fabs(s.x2 - s.x1) < 1e-9) { in.state = "excluded:angle"; continue; }
        in.pieces = clip_segment_support(t, s);
        for (auto& pc : in.pieces) in.vis_len += (pc.xb - pc.xa) / std::fabs(in.l.dx);
        if (in.vis_len < min_len) { in.state = "excluded:occluded_outside_x_or_short"; continue; }
        // Polarity: mean intensity on the +normal side (toward +y for dx>0) minus the -normal side.
        double sp_sum = 0, sn_sum = 0; int n = 0;
        double nx = -in.l.dy, ny = in.l.dx;
        for (auto& pc : in.pieces)
            for (double x = pc.xa; x <= pc.xb; x += 2.0) {
                double y = segment_y_at(s,x);
                if (!allowed_support_point(t,x,y)) continue;
                sp_sum += bilinear(g, x + off * nx, y + off * ny);
                sn_sum += bilinear(g, x - off * nx, y - off * ny);
                ++n;
            }
        in.contrast = n ? (sp_sum - sn_sum) / n : 0;
        if (t.polarity * in.contrast < min_con) { in.state = "excluded:wrong_polarity_or_low_contrast"; continue; }
        in.state = "candidate";
        cand.push_back((int)i);
    }
    r.n_candidates = (int)segs.size();
    r.diagnostics["segments_detected_in_region"] = (int)segs.size();
    r.diagnostics["segments_passing_filters"] = (int)cand.size();

    // Consensus line: hypotheses from each candidate and each candidate pair (midpoint to midpoint).
    // Score = total visible length of candidates whose visible end points lie within tol of the hypothesis.
    auto ends_of = [&](int i, double& xa, double& ya, double& xb, double& yb) {
        const Seg& s = segs[i]; const Info& in = info[i];
        xa = in.pieces.front().xa; xb = in.pieces.back().xb;
        ya = segment_y_at(s,xa); yb = segment_y_at(s,xb);
    };
    auto score_of = [&](const Line& h, std::vector<int>* members) {
        double sc = 0;
        for (int i : cand) {
            double xa, ya, xb, yb; ends_of(i, xa, ya, xb, yb);
            if (std::fabs(signed_distance(h, xa, ya)) <= tol && std::fabs(signed_distance(h, xb, yb)) <= tol) {
                sc += info[i].vis_len;
                if (members) members->push_back(i);
            }
        }
        return sc;
    };
    std::vector<int> order = cand;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return info[a].vis_len > info[b].vis_len; });
    if (order.size() > 300) order.resize(300);
    double best = -1; Line best_h; std::string best_desc;
    for (size_t a = 0; a < order.size(); ++a) {
        int i = order[a];
        double sc = score_of(info[i].l, nullptr);
        if (sc > best) { best = sc; best_h = info[i].l; best_desc = "segment " + std::to_string(i); }
        for (size_t b = a + 1; b < order.size(); ++b) {
            int j = order[b];
            cv::Point2d mi = visible_support_midpoint(segs[i],info[i].pieces), mj = visible_support_midpoint(segs[j],info[j].pieces);
            double mxi=mi.x,myi=mi.y,mxj=mj.x,myj=mj.y;
            if (std::fabs(mxj - mxi) < 1e-6) continue;
            Line h = normalize_line(mxi, myi, mxj - mxi, myj - myi);
            if (std::fabs(line_angle_deg(h)) > max_ang) continue;
            double s2 = score_of(h, nullptr);
            if (s2 > best) { best = s2; best_h = h; best_desc = "pair " + std::to_string(i) + "-" + std::to_string(j); }
        }
    }
    std::vector<int> members;
    if (best > 0) score_of(best_h, &members);
    for (int i : cand) info[i].state = "excluded:not_in_consensus_line";
    for (int i : members) info[i].state = "selected";

    json arr = json::array();
    for (size_t i = 0; i < segs.size(); ++i) {
        const Seg& s = segs[i]; const Info& in = info[i];
        arr.push_back({rnd(s.x1, 4), rnd(s.y1, 4), rnd(s.x2, 4), rnd(s.y2, 4), rnd(in.angle, 5), rnd(in.vis_len, 2),
                       rnd(in.contrast, 2), rnd(s.score, 3), in.state});
    }
    r.intermediates["segments_fields"] = {"x1", "y1", "x2", "y2", "angle_deg", "visible_len_px", "side_contrast", "detector_score", "state"};
    r.intermediates["segments"] = arr;
    json clipped = json::array();
    for (size_t i=0;i<segs.size();++i) for (const auto& pc:info[i].pieces)
        clipped.push_back({{"segment_index",i},{"endpoints_xy",{{pc.xa,segment_y_at(segs[i],pc.xa)},{pc.xb,segment_y_at(segs[i],pc.xb)}}},
                           {"state",info[i].state}});
    r.intermediates["visible_segment_pieces"] = clipped;
    r.diagnostics["segment_support_bounds"] = "x range AND y band, minus closed occlusion intervals; original detected endpoints are separately retained";
    r.diagnostics["sample_weighting"] = {{"axis","image_x"},{"step_px",step},
        {"initial_weight","equal per generated sample; Huber subsequently reweights residuals"},
        {"length_relation","sample count is approximately proportional to projected x length, not exactly to arc length"}};
    r.diagnostics["consensus_hypothesis"] = best_desc;
    r.diagnostics["consensus_score_visible_len_px"] = best;
    r.diagnostics["segments_selected"] = (int)members.size();
    std::map<std::string, int> state_count;
    for (const auto& in : info) state_count[in.state]++;
    json sc = json::object();
    std::string counts;
    for (auto& kv : state_count) { sc[kv.first] = kv.second; counts += " " + kv.first + "=" + std::to_string(kv.second); }
    r.diagnostics["segment_state_counts"] = sc;
    if (members.empty()) {
        r.fail(cand.empty() ? "detection_failed" : "fit_failed",
               (cand.empty() ? detector + ": no segment passed angle/band/visibility/polarity filters;" : std::string("no consensus line;")) + counts);
        return r;
    }
    std::vector<cv::Point2d> pts;
    for (int i : members) {
        auto sampled = sample_segment_support(t,segs[i],info[i].pieces,step);
        pts.insert(pts.end(),sampled.begin(),sampled.end());
    }
    finish_with_points(r, pts, c.fit,
                       "segment_samples: equal initial weights at configured image-x step; approximately projected-x-length proportional, not exact arc-length weighting; not original edge pixels");
    if (r.has_rms) r.diagnostics["segment_sample_fit_rms_px"] = r.fit_rms;
    if (r.has_rms && members.size() == 1) {
        // Samples of one segment lie on that segment, so their residual is ~0 by construction.
        r.has_rms = false;
        r.rms_null_reason = "single selected segment: samples lie on the segment, residual ~0 by construction (see support_pixels_rms_to_final_line_px)";
    }
    // Diagnostic: the detector's own integer support pixels against the final line.
    if (r.has_line) {
        std::vector<cv::Point2d> sup;
        json supj = json::array();
        for (int i : members)
            for (auto& q : segs[i].support) {
                if (!allowed_support_point(t,q.x,q.y)) continue;
                bool within_piece = false;
                for (const auto& pc:info[i].pieces) if(q.x>=pc.xa&&q.x<=pc.xb){within_piece=true;break;}
                if (!within_piece) continue;
                sup.emplace_back(q.x, q.y);
                supj.push_back({q.x, q.y});
            }
        r.intermediates["selected_support_pixels"] = supj;
        if (!sup.empty()) {
            double s2 = 0;
            for (auto& q : sup) { double d = signed_distance(r.line, q.x, q.y); s2 += d * d; }
            r.diagnostics["support_pixels_count"] = (int)sup.size();
            r.diagnostics["support_pixels_rms_to_final_line_px"] = std::sqrt(s2 / sup.size());
            FitResult fr = robust_fit(sup, c.fit);
            if (fr.ok) r.diagnostics["support_pixels_refit_angle_deg_diagnostic"] = line_angle_deg(fr.line);
            else r.diagnostics["support_pixels_refit_failed"] = fr.reason;
        } else {
            r.diagnostics["support_pixels_count"] = 0;
        }
    }
    return r;
}

cv::Mat region_u8(MethodContext& c, cv::Rect& reg) {
    reg = c.target.detector_region(c.gray.size());
    return c.gray(reg).clone();
}

MethodResult method_lsd(MethodContext& c) {
    validate_method_parameters(c.target.kind=="qr"?"Q04":"E04",c.params);
    const json& p = c.params;
    cv::Rect reg;
    cv::Mat u8 = region_u8(c, reg), f;
    u8.convertTo(f, CV_64F);
    int n = 0, *reg_img = nullptr, rx = 0, ry = 0;
    double* out = LineSegmentDetection(&n, f.ptr<double>(), f.cols, f.rows, p.at("scale").get<double>(),
                                       p.at("sigma_scale").get<double>(), p.at("quant").get<double>(),
                                       p.at("ang_th").get<double>(), p.at("log_eps").get<double>(),
                                       p.at("density_th").get<double>(), p.at("n_bins").get<int>(), &reg_img, &rx, &ry);
    std::vector<Seg> segs(n);
    for (int i = 0; i < n; ++i) {
        // LSD output: origin at the centre of pixel (0,0) (lsd.c adds the 2x2-gradient 0.5 offset).
        segs[i] = {out[7 * i] + reg.x, out[7 * i + 1] + reg.y, out[7 * i + 2] + reg.x, out[7 * i + 3] + reg.y, out[7 * i + 6], {}};
    }
    cv::Mat mask = cv::Mat::zeros(ry, rx, CV_8U);
    for (int y = 0; y < ry; ++y)
        for (int x = 0; x < rx; ++x) {
            int lab = reg_img[x + y * rx];
            if (lab > 0 && lab <= n) { segs[lab - 1].support.emplace_back(x + reg.x, y + reg.y); mask.at<uchar>(y, x) = 255; }
        }
    std::free(out); std::free(reg_img);
    MethodResult r = select_and_fit_segments(c, segs, "LSD");
    c.sink.save_png("lsd_line_support_regions", mask, reg.tl(), r.images);
    r.diagnostics["detector_region_xywh"] = {reg.x, reg.y, reg.width, reg.height};
    r.diagnostics["support_pixel_source"] = "LSD reg_img line-support regions (integer pixels)";
    return r;
}

MethodResult method_edlines(MethodContext& c) {
    validate_method_parameters(c.target.kind=="qr"?"Q07":"E07",c.params);
    const json& p = c.params;
    cv::Rect reg;
    cv::Mat u8 = region_u8(c, reg);
    std::string op = p.at("ed_operator").get<std::string>();
    GradientOperator gop = op == "PREWITT" ? PREWITT_OPERATOR : op == "SCHARR" ? SCHARR_OPERATOR : op == "LSD" ? LSD_OPERATOR : SOBEL_OPERATOR;
    ED ed(u8, gop, p.at("ed_grad_thresh").get<int>(), p.at("ed_anchor_thresh").get<int>(), p.at("ed_scan_interval").get<int>(),
          p.at("ed_min_path_len").get<int>(), p.at("ed_sigma").get<double>(), p.at("ed_sum_flag").get<bool>());
    EDLines edl(ed, p.at("line_error").get<double>(), p.at("min_line_len").get<int>(),
                p.at("max_distance_between_two_lines").get<double>(), p.at("max_error").get<double>());
    std::vector<LS> lines = edl.getLines();
    const std::vector<LineSegment>& info = edl.getLineSegments();
    std::vector<std::vector<cv::Point>> chains = edl.getSegments();
    std::vector<Seg> segs;
    for (size_t i = 0; i < lines.size(); ++i) {
        Seg s{lines[i].start.x + reg.x, lines[i].start.y + reg.y, lines[i].end.x + reg.x, lines[i].end.y + reg.y, 0, {}};
        if (i < info.size() && info[i].segmentNo >= 0 && info[i].segmentNo < (int)chains.size()) {
            const auto& ch = chains[info[i].segmentNo];
            int a = std::max(0, info[i].firstPixelIndex), b = std::min((int)ch.size(), info[i].firstPixelIndex + info[i].len);
            for (int k = a; k < b; ++k) s.support.emplace_back(ch[k].x + reg.x, ch[k].y + reg.y);
            s.score = info[i].len;
        }
        segs.push_back(s);
    }
    MethodResult r = select_and_fit_segments(c, segs, "EDLines");
    c.sink.save_png("ed_edge_map", edl.getEdgeImage(), reg.tl(), r.images);
    r.diagnostics["detector_region_xywh"] = {reg.x, reg.y, reg.width, reg.height};
    r.diagnostics["support_pixel_source"] = "ED edge-segment pixels [firstPixelIndex, firstPixelIndex+len) of each EDLines line (collinear joins may not extend this range)";
    r.diagnostics["edlines_line_info_count"] = (int)info.size();
    return r;
}

MethodResult method_fld(MethodContext& c) {
    validate_method_parameters(c.target.kind=="qr"?"Q08":"E08",c.params);
    const json& p = c.params;
    cv::Rect reg;
    cv::Mat u8 = region_u8(c, reg);
    auto fld = lsj_fld::createFastLineDetectorImpl(p.at("length_threshold").get<int>(), p.at("distance_threshold").get<float>(),
                                                   p.at("canny_th1").get<double>(), p.at("canny_th2").get<double>(),
                                                   p.at("canny_aperture_size").get<int>(), p.at("do_merge").get<bool>());
    std::vector<cv::Vec4f> lines;
    std::vector<std::vector<cv::Point2i>> support;
    lsj_fld::detectWithSupport(fld, u8, lines, support);
    std::vector<Seg> segs;
    for (size_t i = 0; i < lines.size(); ++i) {
        Seg s{lines[i][0] + (double)reg.x, lines[i][1] + (double)reg.y, lines[i][2] + (double)reg.x, lines[i][3] + (double)reg.y, 0, {}};
        if (i < support.size()) for (auto& q : support[i]) s.support.emplace_back(q.x + reg.x, q.y + reg.y);
        segs.push_back(s);
    }
    MethodResult r = select_and_fit_segments(c, segs, "FLD");
    cv::Mat canny;  // same Canny call FLD makes internally (for display only)
    cv::Canny(u8, canny, p.at("canny_th1").get<double>(), p.at("canny_th2").get<double>(), p.at("canny_aperture_size").get<int>());
    c.sink.save_png("fld_internal_canny_recomputed", canny, reg.tl(), r.images);
    r.diagnostics["detector_region_xywh"] = {reg.x, reg.y, reg.width, reg.height};
    r.diagnostics["support_pixel_source"] = "FLD chain pixels (l_points) used in each segment's fitLine";
    r.diagnostics["support_list_aligned"] = support.size() == lines.size();
    return r;
}

MethodResult method_elsed(MethodContext& c) {
    validate_method_parameters(c.target.kind=="qr"?"Q09":"E09",c.params);
    const json& p = c.params;
    cv::Rect reg;
    cv::Mat u8 = region_u8(c, reg);
    upm::ELSEDParams ep;
    ep.ksize = p.at("ksize").get<int>();
    ep.sigma = p.at("sigma").get<float>();
    ep.gradientThreshold = p.at("gradientThreshold").get<float>();
    ep.anchorThreshold = (uint8_t)p.at("anchorThreshold").get<int>();
    ep.scanIntervals = p.at("scanIntervals").get<unsigned int>();
    ep.minLineLen = p.at("minLineLen").get<int>();
    ep.lineFitErrThreshold = p.at("lineFitErrThreshold").get<double>();
    ep.pxToSegmentDistTh = p.at("pxToSegmentDistTh").get<double>();
    ep.junctionEigenvalsTh = p.at("junctionEigenvalsTh").get<double>();
    ep.junctionAngleTh = p.at("junctionAngleTh_deg").get<double>() * (3.14159265358979323846 / 180.0);
    ep.validationTh = p.at("validationTh").get<double>();
    ep.validate = p.at("validate").get<bool>();
    ep.treatJunctions = p.at("treatJunctions").get<bool>();
    upm::ELSED elsed(ep);
    upm::Segments out = elsed.detect(u8);
    const upm::ImageEdges& pix = elsed.getValidSegmentPixels();
    std::vector<Seg> segs;
    for (size_t i = 0; i < out.size(); ++i) {
        Seg s{out[i][0] + (double)reg.x, out[i][1] + (double)reg.y, out[i][2] + (double)reg.x, out[i][3] + (double)reg.y, 0, {}};
        if (i < pix.size()) for (auto& q : pix[i]) s.support.emplace_back(q.x + reg.x, q.y + reg.y);
        segs.push_back(s);
    }
    MethodResult r = select_and_fit_segments(c, segs, "ELSED");
    r.diagnostics["detector_region_xywh"] = {reg.x, reg.y, reg.width, reg.height};
    r.diagnostics["support_pixel_source"] = "ELSED FullSegmentInfo pixels of each returned segment";
    r.diagnostics["support_list_aligned"] = pix.size() == out.size();
    return r;
}

}  // namespace lm
