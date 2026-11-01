// =====================================================================================
// report_media.hpp — 보고서용 설명 영상 프레임 생성 (실제 사진으로)
// =====================================================================================
#ifndef YT_REPORT_MEDIA_HPP
#define YT_REPORT_MEDIA_HPP

#include "../yt/yt_common.hpp"

namespace yt {
namespace media {
// outDir/v1_parallax, v2_phasecorr, v3_subpixel, v4_gaussnewton 에 PNG 프레임
bool makeAll(const std::vector<Sample>& ds, const std::string& outDir);
} // namespace media
} // namespace yt

#endif
