#include "alignment.hpp"

Alignment_t alignment(cv::Mat &src)
{
    // Compatibility entry: the first stage always uses the joint observation.
    return joint(src);
}
