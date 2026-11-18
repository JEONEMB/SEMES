#include "device_camera.hpp"
#include <utility>

// Placeholder until the camera is connected. Each function keeps the contract shape
// (exceptions, Frame fields) so the service can be written and tested against it now.

DeviceCamera::DeviceCamera(std::string device) : device_(std::move(device)) {}

DeviceCamera::~DeviceCamera() { close(); }

void DeviceCamera::open() {
    // TODO(hardware): open the device (MIL10 MdigAlloc / cv::VideoCapture), set 3840x2160 mono8.
    throw HardwareError(HardwareFault::camera, "Camera not connected: " + device_ + " (see hardware/device/README.md)");
}

void DeviceCamera::close() {
    // TODO(hardware): release the device handles.
    connected_ = false;
}

bool DeviceCamera::is_connected() const { return connected_; }

Frame DeviceCamera::capture() {
    // TODO(hardware): grab one frame, convert to 8-bit grayscale, check 3840x2160.
    // Fill Frame{image, source = device_, simulated = false, note = capture time}.
    throw HardwareError(HardwareFault::camera, "Camera not connected: " + device_ + " (see hardware/device/README.md)");
}
