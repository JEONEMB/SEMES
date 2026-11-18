// =====================================================================================
// MatView.h — cv::Mat 을 비율을 유지해 그려 주는 Static 컨트롤
//   대화 상자의 Picture Control(IDC_VIEW)을 DDX_Control 로 이 클래스에 연결해서 사용
// =====================================================================================
#pragma once
#include "cv_inc.h"

class CMatView : public CStatic
{
public:
    // 표시할 영상 지정 (8UC1 또는 8UC3). 내부에서 BGRA 로 바꿔 보관 (DIB 행 길이 4바이트 정렬)
    void SetImage(const cv::Mat& img);
    void Clear();
    bool HasImage() const { return !m_bgra.empty(); }

protected:
    cv::Mat m_bgra;

    afx_msg void OnPaint();
    afx_msg BOOL OnEraseBkgnd(CDC* pDC);
    DECLARE_MESSAGE_MAP()
};
