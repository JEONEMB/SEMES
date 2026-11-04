#include "pipeline_internal.hpp"
#include "third_party/qrcodegen.hpp"
#include <algorithm>
namespace alignment_detail
{
bool regenerateQrModel(const std::string &payload, const cv::Mat &observed, cv::Mat &best,
                       double &mismatch)
{
    using namespace qrcodegen;
    int n = observed.rows, version = (n - 17) / 4;
    if (observed.type() != CV_8UC1 || observed.cols != n || version < 1 || version > 40 ||
        n != 17 + 4 * version)
    {
        return false;
    }
    std::vector<std::vector<QrSegment>> modes{
        QrSegment::makeSegments(payload.c_str()),
        {QrSegment::makeBytes(std::vector<std::uint8_t>(payload.begin(), payload.end()))}};
    int bestErrors = n * n + 1, secondErrors = n * n + 1;
    // Content alone is insufficient: enumerate ECC, mask and common encoding modes.
    // Mixed segmentation/ECI not covered here: unmatched symbols fail explicitly.
    for (const auto &segments : modes)
    {
        for (int ec = 0; ec < 4; ++ec)
        {
            for (int mask = 0; mask < 8; ++mask)
            {
                try
                {
                    auto code = QrCode::encodeSegments(segments, static_cast<QrCode::Ecc>(ec),
                                                       version, version, mask, false);
                    cv::Mat candidate(n, n, CV_8U);
                    int errors = 0;
                    for (int y = 0; y < n; ++y)
                    {
                        for (int x = 0; x < n; ++x)
                        {
                            candidate.at<uchar>(y, x) = code.getModule(x, y) ? 0 : 255;
                            errors += (candidate.at<uchar>(y, x) < 128) !=
                                      (observed.at<uchar>(y, x) < 128);
                        }
                    }
                    if (!best.empty() && cv::countNonZero(candidate != best) == 0)
                    {
                        continue;
                    }
                    if (errors < bestErrors)
                    {
                        secondErrors = bestErrors;
                        bestErrors = errors;
                        best = candidate;
                    }
                    else
                    {
                        secondErrors = std::min(secondErrors, errors);
                    }
                }
                catch (const qrcodegen::data_too_long &)
                { /* Try another encoding/ECC. */
                }
            }
        }
    }
    mismatch = double(bestErrors) / (n * n);
    return !best.empty() && mismatch <= 0.03 && double(secondErrors - bestErrors) / (n * n) > 0.02;
}
} // namespace alignment_detail
