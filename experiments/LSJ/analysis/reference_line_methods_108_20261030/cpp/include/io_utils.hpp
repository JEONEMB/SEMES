// UTF-8 path I/O for paths with Korean text, spaces and brackets (Windows).
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace lm {

std::vector<uint8_t> read_bytes(const std::string& utf8_path);
void write_bytes(const std::string& utf8_path, const std::vector<uint8_t>& data);
void write_text(const std::string& utf8_path, const std::string& text);
std::string read_text(const std::string& utf8_path);
void make_dirs(const std::string& utf8_path);
bool file_exists(const std::string& utf8_path);
std::string join_path(const std::string& a, const std::string& b);

std::string sha256_hex(const std::vector<uint8_t>& data);

struct BmpHeader {
    bool valid = false;
    int width = 0, height = 0, bits_per_pixel = 0;
};
BmpHeader parse_bmp_header(const std::vector<uint8_t>& data);

cv::Mat decode_gray(const std::vector<uint8_t>& data);
void write_png(const std::string& utf8_path, const cv::Mat& image);

std::string wide_to_utf8(const std::wstring& w);

}  // namespace lm
