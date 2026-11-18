#ifndef __DEVICE_CAMERA_HPP__
#define __DEVICE_CAMERA_HPP__

#include "hardware.hpp"
#include <string>

// Real camera behind the ICamera contract. NOT CONNECTED YET: every call reports
// "camera not connected". See hardware/device/README.md for what to implement.
class DeviceCamera : public ICamera {
public:
    // device: camera identifier (MIL system/digitizer name, or OpenCV device index as text).
    explicit DeviceCamera(std::string device);
    ~DeviceCamera() override;

    void open();               // TODO(hardware): connect to the device
    void close();              // TODO(hardware): release the device
    bool is_connected() const;
    Frame capture() override;  // TODO(hardware): grab one 3840x2160 8-bit grayscale frame

private:
    std::string device_;
    bool connected_ = false;
};

#endif // __DEVICE_CAMERA_HPP__
