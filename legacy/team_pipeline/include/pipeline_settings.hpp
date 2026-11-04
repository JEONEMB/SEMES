#ifndef PIPELINE_SETTINGS_HPP
#define PIPELINE_SETTINGS_HPP
namespace alignment_detail
{
namespace settings
{
// Selected on the development split; see docs/qr-subpixel-comparison/README.md.
constexpr int FourierRefinementRadius = 5; // 11x11 original-image samples.
constexpr double FourierRefinementAlpha = 2.4;
constexpr bool FourierRefinementDemean = true;
constexpr int DecodeScale = 2;
constexpr int DecodeBorder = 32;
constexpr int DecodeKernel = 3;
constexpr double DecodeSigma = 0.6;
constexpr double MergeDistance = 3.0;
constexpr double HorizontalMergeGap = 60.0;
constexpr double VerticalMergeGap = 30.0;
constexpr double MinimumHorizontalLength = 300.0;
} // namespace settings
} // namespace alignment_detail
#endif
