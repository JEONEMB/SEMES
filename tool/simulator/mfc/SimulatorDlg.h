// =====================================================================================
// SimulatorDlg.h — 영점 맞춤 시뮬레이터 대화 상자
//
//  화면 : 왼쪽 위 = 카메라(합성 사진 + 모델 ROI) / 왼쪽 아래 = 파노라마(현재 시야 위치)
//         오른쪽 = (Y, Θ) 궤적 · 수렴 그래프 · 수치 패널 / 아래 = 단계(또는 시험·검증) 목록
//  계산은 모두 작업 스레드에서 하고, 그림은 작업 스레드가 만들어 두면 화면 스레드는 붙이기만 한다.
//  글자 크기 : config/simulator.yml 의 display.ui_font_pt (화면), display.plot_font_scale (그림)
//  추정기     : 윗줄 [추정기] 상자 — ksh(내장) 또는 plugins 폴더의 DLL (팀원 알고리즘)
//  영점 사진  : 윗줄 [영점 사진] 상자 — 참 자세 (0, 0) 로 볼 사진 (center, center1 …). 바꾸면 가상 장면을 다시 만듦
// =====================================================================================
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include "MatView.h"
#include "cv_inc.h"

#define WM_SIM_UPDATE (WM_APP + 1)   // 작업 중 화면 갱신 (그림 · 목록)
#define WM_SIM_DONE   (WM_APP + 2)   // 작업 끝 (wParam = 작업 종류)

class CSimulatorDlg : public CDialogEx
{
public:
	CSimulatorDlg(CWnd* pParent = nullptr);

#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_SIMULATOR_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);

	// ---------------------------------------------------------------- 작업 종류
	enum Job { JOB_NONE, JOB_STEP, JOB_RUN, JOB_MONTE, JOB_VALIDATE, JOB_BUILD };
	enum ListMode { LIST_STEPS, LIST_TRIALS, LIST_VALID };

	HICON m_hIcon;
	bool  m_ready = false;

	// ---------------------------------------------------------------- 컨트롤
	CMatView  m_viewCam, m_viewPano, m_viewPlane, m_viewConv;
	CEdit     m_info;
	CListCtrl m_list;
	CFont     m_monoFont, m_uiFont;
	CComboBox m_cmbEst, m_cmbRef;      // 추정기, 영점 사진
	double    m_uiScale = 1.0;        // (글자 크기 / 9 pt) × (화면 DPI / 96) — 줄 높이·목록 열 폭에 곱함
	std::vector<std::string> m_estItems;   // 상자 항목 : "ksh" 또는 플러그인 경로
	std::atomic<int>  m_camW{ 1600 }, m_panoW{ 1600 };              // 그림을 그릴 크기 = 화면 칸 크기 (글자가 줄어들지 않게)
	std::atomic<int>  m_planeW{ 1000 }, m_planeH{ 560 }, m_convW{ 1000 }, m_convH{ 360 };
	ListMode  m_listMode = LIST_STEPS;

	// ---------------------------------------------------------------- 시뮬레이터 (작업 스레드에서만 계산)
	sim::Simulator    m_sim;
	cv::RNG           m_rng;
	bool              m_simOk = false;
	sim::TrialResult  m_trial;        // 지금 진행 중인 시험 (작업 스레드가 고침 → 갱신 메시지 때 복사본을 화면에)
	bool              m_trialTaught = false;
	int               m_monteN = 20;

	// ---------------------------------------------------------------- 작업 스레드
	CWinThread*       m_worker = nullptr;
	std::atomic<bool> m_cancel{ false };
	Job               m_job = JOB_NONE;

	// 작업 스레드 → 화면 스레드로 넘기는 값 (m_mx 로 보호)
	std::mutex        m_mx;
	cv::Mat           m_visCam, m_visPano, m_visPlane, m_visConv;
	sim::TrialResult  m_trialShow;
	sim::MonteSummary m_monteShow;
	sim::ValidationSummary m_validShow;
	std::string       m_progressText;
	std::string       m_resultText;   // 작업이 끝난 뒤 수치 패널에 붙일 글
	bool              m_newTrialRows = false;

	// ---------------------------------------------------------------- 동작
	void StartJob(Job job);
	static UINT WorkerProc(LPVOID param);
	void RunJob(Job job);                         // 작업 스레드
	void PublishTrial(const cv::Mat& img);        // 작업 스레드 : 그림 만들어 넘김
	bool ReadStart(sim::Pose& p);
	void ResetTrial(const sim::Pose& start);
	void SetBusy(bool busy);
	void UpdateStatus(const CString& msg);
	void SetListMode(ListMode mode);
	void FillStepRows(const sim::TrialResult& t);
	void FillTrialRows(const sim::MonteSummary& m);
	void FillValidRows(const sim::ValidationSummary& v);
	void UpdateInfo(const CString& extra);
	void LayoutControls(int cx, int cy);
	void ApplyFonts();
	void FillEstimators();
	void FillReferences();
	int  S(int px) const { return (int)(px * m_uiScale + 0.5); }

	virtual BOOL OnInitDialog();
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	virtual void OnOK();
	virtual void OnCancel();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnGetMinMaxInfo(MINMAXINFO* lpMMI);
	afx_msg void OnBnClickedRandom();
	afx_msg void OnBnClickedReset();
	afx_msg void OnBnClickedStep();
	afx_msg void OnBnClickedRun();
	afx_msg void OnBnClickedStop();
	afx_msg void OnBnClickedMonte();
	afx_msg void OnBnClickedValidate();
	afx_msg void OnBnClickedBuild();
	afx_msg void OnBnClickedFolder();
	afx_msg void OnCbnSelchangeEst();
	afx_msg void OnCbnSelchangeRef();
	afx_msg LRESULT OnSimUpdate(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnSimDone(WPARAM wParam, LPARAM lParam);
	DECLARE_MESSAGE_MAP()
};
