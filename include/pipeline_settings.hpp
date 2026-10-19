#ifndef PIPELINE_SETTINGS_HPP
#define PIPELINE_SETTINGS_HPP
namespace alignment_detail
{
namespace settings
{
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
