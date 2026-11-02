#pragma once
#include <string>
#include <vector>
namespace theta {
std::vector<unsigned char> readBytes(const std::string& utf8_path);
std::string readText(const std::string& utf8_path);
#ifdef _WIN32
std::string toUtf8(const wchar_t* value);
#endif
}
