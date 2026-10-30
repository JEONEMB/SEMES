// 촬영 시각 읽기. data/clean 사본은 파일 수정 시각이 복사 시각으로 바뀌어 있어서,
// 원래 스크립트가 쓰던 os.path.getmtime 값(압축 해제 때 복원된 NTFS 수정 시각)을 data/raw 의 원본 zip 에서 읽는다.
// zip_20261014.zip 안에 분할 zip(SSAFY Test Image Set.z01~.z19 + .zip)이 들어 있고, 파일 목록(중앙 디렉터리)은 마지막 .zip 조각에 있다.
// 그 조각을 inflate 해서 중앙 디렉터리의 NTFS extra(0x000a) 수정 시각을 꺼낸다. 사진 바이트는 data/clean 과 같다(manifest sha256).
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace zipm {

// 최소 inflate (RFC 1951, zlib puff.c 와 같은 정준 허프만 해석)
struct Inflate {
    const uint8_t* s;
    size_t n, pos = 0;
    uint32_t buf = 0;
    int cnt = 0;
    std::vector<uint8_t> out;
    struct Huff { uint16_t count[16]; uint16_t sym[320]; };
    inline int bit() {
        if (!cnt) { buf = pos < n ? s[pos++] : 0; cnt = 8; }
        int b = buf & 1; buf >>= 1; --cnt; return b;
    }
    int bits(int k) { int v = 0; for (int i = 0; i < k; ++i) v |= bit() << i; return v; }
    static void build(Huff& h, const uint8_t* len, int num) {
        std::memset(h.count, 0, sizeof h.count);
        for (int i = 0; i < num; ++i) h.count[len[i]]++;
        h.count[0] = 0;
        uint16_t offs[16]; offs[1] = 0;
        for (int i = 1; i < 15; ++i) offs[i + 1] = offs[i] + h.count[i];
        for (int i = 0; i < num; ++i) if (len[i]) h.sym[offs[len[i]]++] = (uint16_t)i;
    }
    int decode(const Huff& h) {
        int code = 0, first = 0, index = 0;
        for (int l = 1; l < 16; ++l) {
            code |= bit();
            int c = h.count[l];
            if (code - c < first) return h.sym[index + (code - first)];
            index += c; first += c; first <<= 1; code <<= 1;
        }
        throw std::runtime_error("inflate: bad code");
    }
    void codes(const Huff& lc, const Huff& dc) {
        static const uint16_t lb[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
        static const uint8_t le[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
        static const uint16_t db[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
        static const uint8_t de[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
        for (;;) {
            int sym = decode(lc);
            if (sym < 256) out.push_back((uint8_t)sym);
            else if (sym == 256) return;
            else {
                sym -= 257;
                if (sym >= 29) throw std::runtime_error("inflate: bad length");
                int len = lb[sym] + bits(le[sym]);
                int ds = decode(dc);
                if (ds >= 30) throw std::runtime_error("inflate: bad dist");
                size_t dist = db[ds] + bits(de[ds]);
                if (dist > out.size()) throw std::runtime_error("inflate: dist too far");
                size_t from = out.size() - dist;
                for (int i = 0; i < len; ++i) out.push_back(out[from + i]);
            }
        }
    }
    std::vector<uint8_t> run(const uint8_t* src, size_t len, size_t expect) {
        s = src; n = len; out.reserve(expect);
        int last;
        do {
            last = bit();
            int type = bits(2);
            if (type == 0) {
                cnt = 0;
                if (pos + 4 > n) throw std::runtime_error("inflate: stored");
                unsigned L = s[pos] | (s[pos + 1] << 8);
                pos += 4;
                out.insert(out.end(), s + pos, s + pos + L);
                pos += L;
            } else if (type == 1) {
                static Huff fl, fd;
                static bool init = false;
                if (!init) {
                    uint8_t l[288];
                    for (int i = 0; i < 144; ++i) l[i] = 8;
                    for (int i = 144; i < 256; ++i) l[i] = 9;
                    for (int i = 256; i < 280; ++i) l[i] = 7;
                    for (int i = 280; i < 288; ++i) l[i] = 8;
                    build(fl, l, 288);
                    for (int i = 0; i < 30; ++i) l[i] = 5;
                    build(fd, l, 30);
                    init = true;
                }
                codes(fl, fd);
            } else if (type == 2) {
                static const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
                int nlen = bits(5) + 257, ndist = bits(5) + 1, ncode = bits(4) + 4;
                uint8_t l[320] = {0};
                for (int i = 0; i < ncode; ++i) l[order[i]] = (uint8_t)bits(3);
                Huff lc, dc;
                build(lc, l, 19);
                int i = 0;
                while (i < nlen + ndist) {
                    int sym = decode(lc);
                    if (sym < 16) l[i++] = (uint8_t)sym;
                    else {
                        int v = 0, rep;
                        if (sym == 16) { if (!i) throw std::runtime_error("inflate: repeat"); v = l[i - 1]; rep = 3 + bits(2); }
                        else if (sym == 17) rep = 3 + bits(3);
                        else rep = 11 + bits(7);
                        while (rep--) l[i++] = (uint8_t)v;
                    }
                }
                std::memset(l + i, 0, sizeof(l) - i);
                build(lc, l, nlen);
                build(dc, l + nlen, ndist);
                codes(lc, dc);
            } else throw std::runtime_error("inflate: bad block");
        } while (!last);
        return std::move(out);
    }
};

inline uint16_t u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline uint32_t u32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
inline uint64_t u64(const uint8_t* p) { return u32(p) | ((uint64_t)u32(p + 4) << 32); }

struct Entry { std::string name; uint32_t csize, usize, local; uint16_t method; uint64_t ntfs_mtime = 0; };

// 중앙 디렉터리 목록 (buf 는 zip 파일 끝부분을 포함하는 바이트, base = buf[0] 의 파일 위치)
inline std::vector<Entry> central(const std::vector<uint8_t>& b) {
    size_t e = std::string::npos;
    for (size_t i = b.size() - 22 + 1; i-- > 0;) if (u32(&b[i]) == 0x06054b50) { e = i; break; }
    if (e == std::string::npos) throw std::runtime_error("zip: EOCD 없음");
    uint32_t cdsize = u32(&b[e + 12]);
    size_t end = e;  // 중앙 디렉터리 끝. 분할 zip 은 그 뒤에 zip64 EOCD 레코드(56) + 로케이터(20)가 붙는다
    if (e >= 20 && u32(&b[e - 20]) == 0x07064b50)
        for (size_t i = e - 20; i-- > 0;) if (u32(&b[i]) == 0x06064b50) { end = i; break; }
    size_t p = end - cdsize;
    std::vector<Entry> v;
    while (p + 46 <= e && u32(&b[p]) == 0x02014b50) {
        Entry en;
        en.method = u16(&b[p + 10]);
        en.csize = u32(&b[p + 20]);
        en.usize = u32(&b[p + 24]);
        int nl = u16(&b[p + 28]), el = u16(&b[p + 30]), cl = u16(&b[p + 32]);
        en.local = u32(&b[p + 42]);
        en.name.assign((const char*)&b[p + 46], nl);
        const uint8_t* x = &b[p + 46 + nl];
        for (int k = 0; k + 4 <= el;) {
            int tag = u16(x + k), sz = u16(x + k + 2);
            if (tag == 0x000a && sz >= 32 && u16(x + k + 8) == 1) en.ntfs_mtime = u64(x + k + 12);
            k += 4 + sz;
        }
        v.push_back(en);
        p += 46 + nl + el + cl;
    }
    return v;
}

// 바깥 zip 에서 이름이 suffix 로 끝나는 항목을 풀어 바이트로 돌려준다
inline std::vector<uint8_t> extract(const std::filesystem::path& zip, const std::string& suffix) {
    std::ifstream f(zip, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("zip 열기 실패");
    const uint64_t size = (uint64_t)f.tellg();
    const uint64_t tail = std::min<uint64_t>(size, 1 << 20);
    std::vector<uint8_t> t(tail);
    f.seekg(size - tail);
    f.read((char*)t.data(), tail);
    for (auto& en : central(t)) {
        if (en.name.size() < suffix.size() || en.name.compare(en.name.size() - suffix.size(), suffix.size(), suffix)) continue;
        uint8_t lh[30];
        f.seekg(en.local);
        f.read((char*)lh, 30);
        if (u32(lh) != 0x04034b50) throw std::runtime_error("zip: local header");
        f.seekg((uint64_t)en.local + 30 + u16(lh + 26) + u16(lh + 28));
        std::vector<uint8_t> c(en.csize);
        f.read((char*)c.data(), en.csize);
        if (en.method == 0) return c;
        if (en.method != 8) throw std::runtime_error("zip: 압축 방식");
        Inflate inf;
        auto o = inf.run(c.data(), c.size(), en.usize);
        if (o.size() != en.usize) throw std::runtime_error("zip: 풀린 크기 다름");
        return o;
    }
    throw std::runtime_error("zip: 항목 없음 " + suffix);
}

// 원본 상대 경로(manifest source_member, 예 "SSAFY Test Image Set/Theta/T+0.01.bmp") -> 수정 시각(초, 1601 기준)
inline std::map<std::string, double> capture_times(const std::filesystem::path& outer_zip, const std::string& inner_suffix) {
    auto inner = extract(outer_zip, inner_suffix);
    std::map<std::string, double> m;
    for (auto& en : central(inner))
        if (en.ntfs_mtime) m[en.name] = en.ntfs_mtime / 1e7;
    return m;
}

}  // namespace zipm
