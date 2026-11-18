// =====================================================================================
// MatView.cpp — cv::Mat 표시 컨트롤 구현 (더블 버퍼 + StretchDIBits)
// =====================================================================================
#include "pch.h"
#include "MatView.h"

BEGIN_MESSAGE_MAP(CMatView, CStatic)
    ON_WM_PAINT()
    ON_WM_ERASEBKGND()
END_MESSAGE_MAP()

void CMatView::SetImage(const cv::Mat& img)
{
    if (img.empty()) { Clear(); return; }
    if (img.channels() == 1)      cv::cvtColor(img, m_bgra, cv::COLOR_GRAY2BGRA);
    else if (img.channels() == 3) cv::cvtColor(img, m_bgra, cv::COLOR_BGR2BGRA);
    else                          img.copyTo(m_bgra);
    Invalidate(FALSE);
}

void CMatView::Clear()
{
    m_bgra.release();
    Invalidate(FALSE);
}

BOOL CMatView::OnEraseBkgnd(CDC* /*pDC*/)
{
    return TRUE;   // 배경은 OnPaint 에서 한 번에 그림 (깜빡임 방지)
}

void CMatView::OnPaint()
{
    CPaintDC dc(this);
    CRect rc;
    GetClientRect(&rc);
    if (rc.Width() <= 0 || rc.Height() <= 0) return;

    // 메모리 DC 에 그린 뒤 한 번에 복사 (더블 버퍼)
    CDC mem;
    mem.CreateCompatibleDC(&dc);
    CBitmap bmp;
    bmp.CreateCompatibleBitmap(&dc, rc.Width(), rc.Height());
    CBitmap* oldBmp = mem.SelectObject(&bmp);
    mem.FillSolidRect(&rc, RGB(40, 40, 40));

    if (!m_bgra.empty()) {
        // 비율 유지 맞춤
        const double sx = (double)rc.Width() / m_bgra.cols, sy = (double)rc.Height() / m_bgra.rows;
        const double s = (sx < sy) ? sx : sy;
        const int w = (int)(m_bgra.cols * s), h = (int)(m_bgra.rows * s);
        const int x = (rc.Width() - w) / 2, y = (rc.Height() - h) / 2;

        BITMAPINFO bmi = {};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = m_bgra.cols;
        bmi.bmiHeader.biHeight = -m_bgra.rows;          // 음수 = 위에서 아래로 저장된 영상
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        cv::Mat cont = m_bgra.isContinuous() ? m_bgra : m_bgra.clone();
        mem.SetStretchBltMode(HALFTONE);
        SetBrushOrgEx(mem.GetSafeHdc(), 0, 0, nullptr);
        StretchDIBits(mem.GetSafeHdc(), x, y, w, h, 0, 0, cont.cols, cont.rows,
                      cont.data, &bmi, DIB_RGB_COLORS, SRCCOPY);
    }
    else {
        mem.SetBkMode(TRANSPARENT);
        mem.SetTextColor(RGB(200, 200, 200));
        CRect rt = rc;
        rt.top = rc.Height() / 2 - 20;
        mem.DrawText(_T("이미지가 없습니다.\n[기준 이미지] → [이미지 열기] 순서로 불러오세요."),
                     &rt, DT_CENTER | DT_WORDBREAK);
    }

    dc.BitBlt(0, 0, rc.Width(), rc.Height(), &mem, 0, 0, SRCCOPY);
    mem.SelectObject(oldBmp);
}
