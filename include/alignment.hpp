#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include "alignment_error.hpp"

typedef double error_t;
typedef struct _alignment_t {
    error_t y;
    error_t t; // robot theta correction, degrees
} Alignment_t;

// Legacy theta-only initializer. Reference must be first center.bmp.
void configureAlignment(const cv::Mat& reference, const std::string& configPath);

// Theta -> Y initializer. Reloading preserves movement counts and pending/failure state.
// These global adapters are not thread-safe.
void configureAlignment(const cv::Mat& reference, const std::string& thetaConfigPath,
                        const std::string& yConfigPath);

// Call only after the external controller confirms that the returned nonzero move completed.
void acknowledgeAlignmentMoveExecuted();
int completedAlignmentMoves();
int usedAlignmentMoves(); // completed moves plus partial/unknown movement attempts
void reportAlignmentMoveFailure(semes::MoveFailure failure);
void resumeAlignmentAfterInspection(); // after external confirmation and a new camera frame
void startNewAlignmentSession(); // explicit NEW positioning task; never recovery within the same task

// Preserved dev interface and units: Y mm, theta degrees in supplied nominal label coordinates.
// Returns one axis at a time; combined mode returns (0,0) only after both actual-image tests pass.
// Throws on failed measurement, infeasible tick, pending unacknowledged command or 20-move limit.
Alignment_t alignment(cv::Mat &src);

#endif
