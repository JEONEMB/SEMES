#pragma once
#include <string>
#include <vector>

// 운영체제 라이브러리를 쓰지 않는 SHA-256 (FIPS 180-4). Windows BCrypt, macOS CommonCrypto 대신 쓴다(리눅스에서도 빌드되게).
namespace final00_sha {
std::string sha256_hex(const std::vector<unsigned char>& data);
}
