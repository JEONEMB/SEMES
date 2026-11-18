#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <opencv2/opencv.hpp>

typedef double error_t;

typedef struct _alignment_t {
    error_t y;
    error_t t; // theta
} Alignment_t;

Alignment_t alignment(cv::Mat &src);

#endif // __ALIGNMENT_HPP__