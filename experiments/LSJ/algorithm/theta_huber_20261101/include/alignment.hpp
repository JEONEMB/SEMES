#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <opencv2/opencv.hpp>
#include <string>

typedef double error_t;
typedef struct _alignment_t {
    error_t y;
    error_t t; // theta, degrees
} Alignment_t;

// Original dev skeleton signature. This module returns theta only.
// y == 0 means "no Y command"; it does not declare Y aligned.
// Configure once before calling; errors are exceptions, never {0,0} fallback.
Alignment_t alignment(cv::Mat &src);
void configureThetaAlignment(const std::string& config_utf8_path);

#endif
