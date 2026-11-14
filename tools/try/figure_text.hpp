#pragma once
// 문서 그림용 한글 표시. 글자 모양을 그려 사진에 덧붙이기

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <stdexcept>
#include <string>

// text_mask: 맑은 고딕 글자 모양 그리기
//   입력: text (UTF-8), height (글자 높이 px)
//   출력: 글자 부분의 밝기를 표시한 영상. 실패 시 예외
inline cv::Mat text_mask(const std::string& text, int height) {
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), nullptr, 0);
    if (length == 0) throw std::runtime_error("글자를 UTF-8로 읽지 못했습니다");
    std::wstring wide(length, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), wide.data(), length);
    HDC dc = CreateCompatibleDC(nullptr);
    HFONT font = CreateFontW(-height, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                             DEFAULT_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
                             ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Malgun Gothic");
    HGDIOBJ old_font = SelectObject(dc, font);
    SIZE size{};
    GetTextExtentPoint32W(dc, wide.data(), length, &size);
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = size.cx + 4;
    info.bmiHeader.biHeight = -size.cy;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    void* pixels = nullptr;
    HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &pixels, nullptr, 0);
    cv::Mat mask;
    if (bitmap) {
        HGDIOBJ old_bitmap = SelectObject(dc, bitmap);
        std::memset(pixels, 0, size_t(size.cx + 4) * size.cy * 4);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(255, 255, 255));
        TextOutW(dc, 2, 0, wide.data(), length);
        GdiFlush();
        cv::extractChannel(cv::Mat(size.cy, size.cx + 4, CV_8UC4, pixels), mask, 1);
        SelectObject(dc, old_bitmap);
        DeleteObject(bitmap);
    }
    SelectObject(dc, old_font);
    DeleteObject(font);
    DeleteDC(dc);
    if (mask.empty()) throw std::runtime_error("글자 그림을 만들지 못했습니다");
    return mask;
}

// write_text: 영상에 한글 합성
//   입력: picture (사진), text (표시할 글자), top (글자 위치), height (글자 높이), color (글자 색), right (오른쪽 정렬 여부)
//   출력: 없음 (picture 수정)
inline void write_text(cv::Mat& picture, const std::string& text, cv::Point top, int height,
                       const cv::Scalar& color, bool right = false) {
    if (text.empty()) return;
    const cv::Mat mask = text_mask(text, height);
    if (right) top.x -= mask.cols;
    const cv::Rect area = cv::Rect(top, mask.size()) & cv::Rect(0, 0, picture.cols, picture.rows);
    for (int y = area.y; y < area.br().y; ++y) {
        for (int x = area.x; x < area.br().x; ++x) {
            const double alpha = mask.at<uchar>(y - top.y, x - top.x) / 255.0;
            cv::Vec3b& pixel = picture.at<cv::Vec3b>(y, x);
            for (int ch = 0; ch < 3; ++ch)
                pixel[ch] = cv::saturate_cast<uchar>((1 - alpha) * pixel[ch] + alpha * color[ch]);
        }
    }
}
