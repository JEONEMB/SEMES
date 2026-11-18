// data/clean 의 원본 zip(provenance.json 의 source_archive) 안에서 사진별 파일 수정 시각을 읽는다.
// 바깥 zip 안에 분할 zip(.z01.. + .zip)이 들어 있고, 마지막 조각(.zip) 끝의 중앙 디렉터리에
// 사진마다 NTFS 수정 시각(0x000a, 100 ns, UTC)이 있다. 원래 EDA 의 taken_at 은 이 시각으로 풀린 BMP 의
// 파일 수정 시각을 Asia/Seoul 로 읽어 초 아래를 버린 값이다.
#pragma once
#include <zlib.h>
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ziptimes {

namespace fs = std::filesystem;

inline uint16_t u16(const unsigned char* p) { return uint16_t(p[0] | (p[1] << 8)); }
inline uint32_t u32(const unsigned char* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
inline uint64_t u64(const unsigned char* p) { return uint64_t(u32(p)) | (uint64_t(u32(p + 4)) << 32); }

inline std::string cp949_to_utf8(const std::string& s) {
    int n = MultiByteToWideChar(949, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(949, 0, s.data(), (int)s.size(), &w[0], n);
    int m = WideCharToMultiByte(CP_UTF8, 0, w.data(), n, nullptr, 0, nullptr, nullptr);
    std::string o(m, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), n, &o[0], m, nullptr, nullptr);
    return o;
}

struct Entry {
    std::string name;     // UTF-8
    int64_t ntfs = -1;    // FILETIME (100 ns since 1601 UTC), 없으면 -1
    uint16_t dos_time = 0, dos_date = 0;
    uint32_t comp_size = 0, local_offset = 0;
    uint16_t method = 0;
};

// 중앙 디렉터리 해석. buf 는 zip(또는 분할 zip 의 마지막 조각) 전체
inline std::vector<Entry> central_directory(const std::vector<unsigned char>& buf) {
    size_t e = std::string::npos;
    for (size_t i = buf.size() >= 22 ? buf.size() - 22 : 0;; --i) {
        if (u32(&buf[i]) == 0x06054b50) { e = i; break; }
        if (i == 0) break;
    }
    if (e == std::string::npos) throw std::runtime_error("EOCD 없음");
    const uint16_t total = u16(&buf[e + 10]);
    const uint32_t size = u32(&buf[e + 12]), off = u32(&buf[e + 16]);
    size_t p = (off + 4 <= buf.size() && u32(&buf[off]) == 0x02014b50) ? off : e - size;
    std::vector<Entry> out;
    for (int k = 0; k < total; ++k) {
        if (u32(&buf[p]) != 0x02014b50) throw std::runtime_error("중앙 디렉터리 손상");
        Entry en;
        const uint16_t flags = u16(&buf[p + 8]);
        en.method = u16(&buf[p + 10]);
        en.dos_time = u16(&buf[p + 12]);
        en.dos_date = u16(&buf[p + 14]);
        en.comp_size = u32(&buf[p + 20]);
        const uint16_t fn = u16(&buf[p + 28]), ex = u16(&buf[p + 30]), co = u16(&buf[p + 32]);
        en.local_offset = u32(&buf[p + 42]);
        std::string raw((const char*)&buf[p + 46], fn);
        std::string uni;
        const unsigned char* x = &buf[p + 46 + fn];
        for (size_t i = 0; i + 4 <= ex;) {
            const uint16_t tag = u16(x + i), sz = u16(x + i + 2);
            if (tag == 0x000a) {  // NTFS: reserved 4, 그 뒤 (tag 1, size 24, mtime, atime, ctime)
                for (size_t j = i + 8; j + 4 <= i + 4 + sz;) {
                    const uint16_t at = u16(x + j), as = u16(x + j + 2);
                    if (at == 1 && as >= 8) en.ntfs = (int64_t)u64(x + j + 4);
                    j += 4 + as;
                }
            }
            if (tag == 0x7075 && sz > 5) uni.assign((const char*)x + i + 9, sz - 5);  // Info-ZIP Unicode Path
            i += 4 + sz;
        }
        en.name = (flags & 0x800) ? raw : !uni.empty() ? uni : cp949_to_utf8(raw);
        out.push_back(en);
        p += 46 + fn + ex + co;
    }
    return out;
}

inline std::vector<unsigned char> read_range(const fs::path& path, uint64_t off, size_t n) {
    std::ifstream f(path, std::ios::binary);
    f.seekg((std::streamoff)off);
    std::vector<unsigned char> b(n);
    f.read((char*)b.data(), (std::streamsize)n);
    if ((size_t)f.gcount() != n) throw std::runtime_error("zip 읽기 실패");
    return b;
}

// 바깥 zip 에서 이름이 suffix 로 끝나는 멤버 하나를 풀어 돌려준다
inline std::vector<unsigned char> extract_member(const fs::path& zip, const std::string& suffix, std::string* member_name = nullptr) {
    const uint64_t len = fs::file_size(zip);
    const size_t tail = (size_t)std::min<uint64_t>(len, 1 << 20);
    auto end = read_range(zip, len - tail, tail);
    // 바깥 zip 은 중앙 디렉터리가 꼬리 안에 있다 (offset 은 파일 기준이라 꼬리 기준으로 옮긴다)
    size_t e = std::string::npos;
    for (size_t i = end.size() - 22;; --i) {
        if (u32(&end[i]) == 0x06054b50) { e = i; break; }
        if (i == 0) break;
    }
    if (e == std::string::npos) throw std::runtime_error("바깥 EOCD 없음");
    const uint32_t size = u32(&end[e + 12]), off = u32(&end[e + 16]);
    std::vector<unsigned char> cd(end.begin() + (std::ptrdiff_t)(off - (len - tail)), end.begin() + (std::ptrdiff_t)(off - (len - tail) + size));
    std::vector<unsigned char> fake(cd);
    // central_directory 가 EOCD 를 찾도록 붙인다 (offset 0)
    std::vector<unsigned char> eocd(end.begin() + (std::ptrdiff_t)e, end.begin() + (std::ptrdiff_t)e + 22);
    eocd[16] = eocd[17] = eocd[18] = eocd[19] = 0;
    fake.insert(fake.end(), eocd.begin(), eocd.end());
    for (auto& en : central_directory(fake)) {
        if (en.name.size() < suffix.size() || en.name.compare(en.name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        auto lh = read_range(zip, en.local_offset, 30);
        const uint64_t data = en.local_offset + 30 + u16(&lh[26]) + u16(&lh[28]);
        auto comp = read_range(zip, data, en.comp_size);
        if (member_name) *member_name = en.name;
        if (en.method == 0) return comp;
        if (en.method != 8) throw std::runtime_error("지원하지 않는 압축");
        std::vector<unsigned char> out;
        z_stream zs{};
        if (inflateInit2(&zs, -15) != Z_OK) throw std::runtime_error("inflateInit");
        zs.next_in = comp.data();
        zs.avail_in = (uInt)comp.size();
        std::vector<unsigned char> chunk(1 << 20);
        int rc;
        do {
            zs.next_out = chunk.data();
            zs.avail_out = (uInt)chunk.size();
            rc = inflate(&zs, Z_NO_FLUSH);
            if (rc != Z_OK && rc != Z_STREAM_END) { inflateEnd(&zs); throw std::runtime_error("inflate 실패"); }
            out.insert(out.end(), chunk.data(), chunk.data() + (chunk.size() - zs.avail_out));
        } while (rc != Z_STREAM_END);
        inflateEnd(&zs);
        return out;
    }
    throw std::runtime_error("멤버 없음: " + suffix);
}

// FILETIME -> 한국 시각(UTC+9) 초. 초 아래는 버린다 (datetime.fromtimestamp(...).strftime 과 같음)
inline int64_t kst_seconds(int64_t filetime) { return (filetime - 116444736000000000LL) / 10000000LL + 9 * 3600; }
inline int64_t dos_seconds(uint16_t t, uint16_t d) {  // DOS 시각(지역 시각, 2초 단위) -> 같은 기준 초
    int y = 1980 + (d >> 9), m = (d >> 5) & 15, dd = d & 31;
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + dd - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const int64_t days = era * 146097 + (int64_t)doe - 719468;
    return days * 86400 + (t >> 11) * 3600 + ((t >> 5) & 63) * 60 + (t & 31) * 2;
}
inline std::string format(int64_t s) {
    int64_t days = s >= 0 ? s / 86400 : (s - 86399) / 86400, rem = s - days * 86400;
    days += 719468;
    const int64_t era = (days >= 0 ? days : days - 146096) / 146097;
    const unsigned doe = (unsigned)(days - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t y = (int64_t)yoe + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    const unsigned d = doy - (153 * mp + 2) / 5 + 1;
    const unsigned m = mp < 10 ? mp + 3 : mp - 9;
    y += m <= 2;
    char b[32];
    std::snprintf(b, sizeof b, "%04lld-%02u-%02u %02lld:%02lld:%02lld", (long long)y, m, d, (long long)(rem / 3600), (long long)(rem / 60 % 60), (long long)(rem % 60));
    return b;
}

}  // namespace ziptimes
