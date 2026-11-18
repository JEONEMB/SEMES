// Method entry points. One function per method family; the target spec (external / QR) and the
// per-kind parameters select the external (Exx) or QR (Qxx) variant.
#pragma once

#include "common.hpp"

namespace lm {

MethodResult method_profile_gaussian(MethodContext& c);  // E01 / Q01  methods_points.cpp
MethodResult method_canny_devernay(MethodContext& c);    // E02 / Q02  methods_points.cpp
MethodResult method_canny_hough(MethodContext& c);       // E03 / Q03  methods_hough.cpp
MethodResult method_lsd(MethodContext& c);               // E04 / Q04  methods_segments.cpp
MethodResult method_zernike(MethodContext& c);           // E05 / Q05  methods_points.cpp
MethodResult method_pae(MethodContext& c);               // E06 / Q06  methods_points.cpp
MethodResult method_edlines(MethodContext& c);           // E07 / Q07  methods_segments.cpp
MethodResult method_fld(MethodContext& c);               // E08 / Q08  methods_segments.cpp
MethodResult method_elsed(MethodContext& c);             // E09 / Q09  methods_segments.cpp
MethodResult method_qr_contour(MethodContext& c);        // Q10        methods_qr.cpp
MethodResult method_qr_corners(MethodContext& c);        // Q11        methods_qr.cpp
MethodResult method_qr_finders(MethodContext& c);        // Q12        methods_qr.cpp

}  // namespace lm
