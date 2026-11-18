
// QR_OpenCV_CheckDlg.h: 헤더 파일
//   QR Y-Θ Viewer — 이미지를 등록하면 위상(Gabor) 기반 측정 엔진으로 Y·Θ 이동량을 계산해 보여 준다.
//

#pragma once
#include <vector>
#include <atomic>
#include "MatView.h"
#include "cv_inc.h"

#define WM_BATCH_ITEM (WM_APP + 1)   // 일괄 처리 : 사진 한 장 끝 (wParam = 행 번호)
#define WM_BATCH_DONE (WM_APP + 2)   // 일괄 처리 : 전체 끝

// 사진 한 장의 분석 결과 (목록 한 줄)
struct ResultRow {
    CString      path;               // 전체 경로
    CString      name;               // 파일 이름
    bool         loaded   = false;   // 영상을 읽었는지
    bool         hasTruth = false;   // 파일 이름에서 정답(Y±x.xx / T±x.xx)을 읽었는지
    double       trueY = 0, trueT = 0;
    pa::Features feat;
    pa::Estimate est;
    double       ms = 0;             // 측정 + 추정 시간
};

// CQROpenCVCheckDlg 대화 상자
class CQROpenCVCheckDlg : public CDialogEx
{
// 생성입니다.
public:
	CQROpenCVCheckDlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_QR_OPENCV_CHECK_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// ---------------------------------------------------------------- 컨트롤
	CMatView  m_view;        // 이미지 + 오버레이
	CEdit     m_info;        // 오른쪽 수치 패널
	CListCtrl m_list;        // 일괄 결과
	CButton   m_chkZoom;     // QR 영역 확대
	CFont     m_monoFont;    // 수치 패널용 고정폭 글꼴
	bool      m_ready = false;

	// ---------------------------------------------------------------- 측정 엔진·상태
	pa::PhaseAligner m_aligner;
	bool      m_hasRef = false, m_hasCalib = false;
	bool      m_calibUseMag = false;   // 보정 모델이 3변수인지 (표시용)
	CString   m_refPath, m_calibPath;
	CString   m_imageDir;    // 교보재 사진 폴더 (절대 경로, 없으면 빈 문자열)
	CString   m_resultDir;   // 결과 CSV·PNG 폴더 (절대 경로)
	CString   m_lastCsv;     // 마지막으로 저장한 CSV
	cv::Mat   m_refImage;    // 기준 영상 (8UC1, 원본 크기)
	cv::Mat   m_curImage;    // 지금 보고 있는 영상 (8UC1, 원본 크기)
	ResultRow m_cur;         // 지금 보고 있는 영상의 결과
	bool      m_hasCur = false;

	// ---------------------------------------------------------------- 일괄 처리
	std::vector<ResultRow> m_rows;
	CWinThread*       m_worker = nullptr;
	std::atomic<bool> m_stop{ false };
	bool              m_busy = false;
	bool              m_autoReport = false;   // [전체 분석] : 끝나면 CSV 자동 저장 + 그래프
	size_t            m_batchBegin = 0;   // 이번 일괄 처리가 시작되는 m_rows 위치 (목록에 추가할 때)

	// ---------------------------------------------------------------- 동작
	bool LoadReference(const CString& path, bool quiet = false);     // quiet : 실패해도 메시지 상자를 띄우지 않음
	bool LoadCalibration(const CString& path, bool quiet = false);
	void AnalyzeImage(const cv::Mat& gray, ResultRow& row) const;   // 측정 + 추정 (스레드에서도 호출)
	void ShowCurrent();                                            // 오버레이 + 수치 패널 갱신
	void RenderView();
	void UpdateInfo();
	void UpdateStatus(const CString& msg);
	void AddListRow(int i);
	void SetBusy(bool busy);
	void FitListColumns();
	void StartBatch(const std::vector<CString>& files, bool append);
	bool WriteCsv(const CString& path) const;
	void ShowReport(const CString& csvPath);
	void LayoutControls(int cx, int cy);
	static UINT BatchProc(LPVOID param);

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	virtual void OnOK();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnGetMinMaxInfo(MINMAXINFO* lpMMI);
	afx_msg void OnDestroy();
	afx_msg void OnBnClickedRef();
	afx_msg void OnBnClickedShowRef();
	afx_msg void OnBnClickedCalib();
	afx_msg void OnBnClickedOpen();
	afx_msg void OnBnClickedFolder();
	afx_msg void OnBnClickedSave();
	afx_msg void OnBnClickedAnalyze();
	afx_msg void OnBnClickedGraph();
	afx_msg void OnBnClickedZoom();
	afx_msg void OnLvnItemchangedList(NMHDR* pNMHDR, LRESULT* pResult);
	afx_msg LRESULT OnBatchItem(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnBatchDone(WPARAM wParam, LPARAM lParam);
	DECLARE_MESSAGE_MAP()
};
