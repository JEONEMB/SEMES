#ifndef __HARDWARE_HPP__
#define __HARDWARE_HPP__

#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

// Device contract. The service flow (src/sequence.cpp) only knows these declarations.
// hardware/sim implements them with dataset photos; hardware/device is for the hardware team.

// Pose in the same coordinates as the photo filename commands (Y, T).
struct Pose {
    double y = 0;
    double t = 0;
};

enum class HardwareFault { camera, motion, emergency_stop };

class HardwareError : public std::runtime_error {
public:
    HardwareError(HardwareFault fault, const std::string& message) : std::runtime_error(message), fault_(fault) {}
    HardwareFault fault() const { return fault_; }
private:
    HardwareFault fault_;
};

struct Frame {
    cv::Mat image;          // 3840x2160 8-bit grayscale
    std::string source;     // file path or device name
    bool simulated = false; // true when an existing photo stands in for a capture
    std::string note;
};

class ICamera {
public:
    virtual ~ICamera() = default;
    virtual Frame capture() = 0; // throws HardwareError(camera)
};

class IRobot {
public:
    virtual ~IRobot() = default;
    // Relative move. Units and sign are fixed by config/service.yml per profile.
    virtual void move(double dy, double dt) = 0; // throws HardwareError(motion|emergency_stop)
    // Only a simulator knows the true pose. Real devices return false.
    virtual bool knows_pose() const = 0;
    virtual Pose pose() const = 0;
};

#endif // __HARDWARE_HPP__
