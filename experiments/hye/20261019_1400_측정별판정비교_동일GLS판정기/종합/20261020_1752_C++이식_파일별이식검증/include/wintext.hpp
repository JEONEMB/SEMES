#pragma once
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
namespace semes {
    inline void wintext(cv::Mat& image,const std::string& utf8,cv::Point origin,int size,COLORREF color,bool bold=false){
        int length=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,utf8.data(),int(utf8.size()),nullptr,0);
        if(!length)throw std::runtime_error("Invalid text UTF8");
        std::wstring value(length,L' ');
        MultiByteToWideChar(CP_UTF8,0,utf8.data(),int(utf8.size()),value.data(),length);
        BITMAPINFO info{
        };
        info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
        info.bmiHeader.biWidth=image.cols;
        info.bmiHeader.biHeight=-image.rows;
        info.bmiHeader.biPlanes=1;
        info.bmiHeader.biBitCount=32;
        info.bmiHeader.biCompression=BI_RGB;
        HDC dc=CreateCompatibleDC(nullptr);
        void* data=nullptr;
        HBITMAP bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&data,nullptr,0);
        if(!dc||!bitmap){
            if(dc)DeleteDC(dc);
            throw std::runtime_error("GDI text allocation failed");
        }
        cv::Mat surface(image.rows,image.cols,CV_8UC4,data);
        cv::cvtColor(image,surface,cv::COLOR_BGR2BGRA);
        auto old_bitmap=SelectObject(dc,bitmap);
        HFONT font=CreateFontW(-size,0,0,0,bold?FW_BOLD:FW_NORMAL,FALSE,FALSE,FALSE,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,ANTIALIASED_QUALITY,DEFAULT_PITCH,L"Malgun Gothic");
        auto old_font=SelectObject(dc,font);
        SetBkMode(dc,TRANSPARENT);
        SetTextColor(dc,color);
        TextOutW(dc,origin.x,origin.y,value.data(),length);
        GdiFlush();
        cv::cvtColor(surface,image,cv::COLOR_BGRA2BGR);
        SelectObject(dc,old_font);
        SelectObject(dc,old_bitmap);
        DeleteObject(font);
        DeleteObject(bitmap);
        DeleteDC(dc);
    }
}
