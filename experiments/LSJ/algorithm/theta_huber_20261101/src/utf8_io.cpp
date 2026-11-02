#include "utf8_io.hpp"
#include <fstream>
#include <iterator>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace theta {
#ifdef _WIN32
static std::wstring toWide(const std::string& value) {
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                    static_cast<int>(value.size()), nullptr, 0);
    if (n <= 0) throw std::runtime_error("invalid UTF-8 path");
    std::wstring out(n, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                        static_cast<int>(value.size()), &out[0], n);
    return out;
}
std::string toUtf8(const wchar_t* value) {
    std::wstring s(value);
    const int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, s.data(),
                                    static_cast<int>(s.size()), nullptr, 0, nullptr, nullptr);
    if (n <= 0) throw std::runtime_error("invalid command line path");
    std::string out(n, '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, s.data(),
                       static_cast<int>(s.size()), &out[0], n, nullptr, nullptr);
    return out;
}
#endif
std::vector<unsigned char> readBytes(const std::string& path) {
#ifdef _WIN32
    std::ifstream in(toWide(path), std::ios::binary);
#else
    std::ifstream in(path, std::ios::binary);
#endif
    if (!in) throw std::runtime_error("cannot open input: " + path);
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(in)), {});
    if (in.bad()) throw std::runtime_error("cannot read input: " + path);
    return data;
}
std::string readText(const std::string& path) {
    auto bytes = readBytes(path);
    return std::string(bytes.begin(), bytes.end());
}
}
