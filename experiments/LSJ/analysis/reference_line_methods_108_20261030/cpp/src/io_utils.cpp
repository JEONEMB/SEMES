#include "io_utils.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace lm {

static fs::path upath(const std::string& p) { return fs::u8path(p); }

std::vector<uint8_t> read_bytes(const std::string& utf8_path) {
    std::ifstream in(upath(utf8_path), std::ios::binary);
    if (!in) throw std::runtime_error("cannot open for reading: " + utf8_path);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_bytes(const std::string& utf8_path, const std::vector<uint8_t>& data) {
    fs::path p = upath(utf8_path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open for writing: " + utf8_path);
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

void write_text(const std::string& utf8_path, const std::string& text) {
    write_bytes(utf8_path, std::vector<uint8_t>(text.begin(), text.end()));
}

std::string read_text(const std::string& utf8_path) {
    auto b = read_bytes(utf8_path);
    std::string s(b.begin(), b.end());
    if (s.size() >= 3 && (uint8_t)s[0] == 0xEF && (uint8_t)s[1] == 0xBB && (uint8_t)s[2] == 0xBF) s = s.substr(3);
    return s;
}

void make_dirs(const std::string& utf8_path) { fs::create_directories(upath(utf8_path)); }

bool file_exists(const std::string& utf8_path) { return fs::exists(upath(utf8_path)); }

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

// ---- SHA-256 (FIPS 180-4), written for this tool ----
namespace {
const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
}  // namespace

std::string sha256_hex(const std::vector<uint8_t>& data) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> msg(data);
    uint64_t bit_len = static_cast<uint64_t>(data.size()) * 8ULL;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<uint8_t>(bit_len >> (8 * i)));
    uint32_t w[64];
    for (size_t off = 0; off < msg.size(); off += 64) {
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(msg[off + 4 * i]) << 24) | (uint32_t(msg[off + 4 * i + 1]) << 16) |
                   (uint32_t(msg[off + 4 * i + 2]) << 8) | uint32_t(msg[off + 4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (uint32_t v : h)
        for (int i = 7; i >= 0; --i) out.push_back(hex[(v >> (4 * i)) & 0xF]);
    return out;
}

BmpHeader parse_bmp_header(const std::vector<uint8_t>& d) {
    BmpHeader h;
    if (d.size() < 54 || d[0] != 'B' || d[1] != 'M') return h;
    auto i32 = [&](size_t o) { return int32_t(d[o] | (d[o + 1] << 8) | (d[o + 2] << 16) | (d[o + 3] << 24)); };
    auto u16 = [&](size_t o) { return int(d[o] | (d[o + 1] << 8)); };
    h.width = i32(18);
    h.height = i32(22);
    if (h.height < 0) h.height = -h.height;
    h.bits_per_pixel = u16(28);
    h.valid = true;
    return h;
}

cv::Mat decode_gray(const std::vector<uint8_t>& data) {
    cv::Mat buf(1, static_cast<int>(data.size()), CV_8UC1, const_cast<uint8_t*>(data.data()));
    return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

void write_png(const std::string& utf8_path, const cv::Mat& image) {
    std::vector<uint8_t> buf;
    if (!cv::imencode(".png", image, buf)) throw std::runtime_error("png encode failed: " + utf8_path);
    write_bytes(utf8_path, buf);
}

std::string wide_to_utf8(const std::wstring& w) {
#ifdef _WIN32
    if (w.empty()) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
#else
    return std::string(w.begin(), w.end());
#endif
}

}  // namespace lm
