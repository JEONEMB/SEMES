
// QR_OpenCV_CheckDlg.cpp: 구현 파일
//   QR Y-Θ Viewer
//     [기준 이미지] center.bmp 등록 → QR 3개 위치를 찾고 기준 위상을 계산
//     [보정 모델]   phase_calib.yml (특징 → Y·Θ 모델) 불러오기
//     [이미지 열기] 사진 한 장 측정 → 오버레이(기준 위치 / 측정 위치)와 Y·Θ·QR 별 이동 표시
//     [폴더 일괄]   폴더(하위 폴더 포함)의 사진을 작업 스레드에서 모두 측정 → 목록, 정답과 비교
//     [CSV 저장]    일괄 결과 저장
//     [전체 분석]   image 폴더 전체를 한 번에 측정 → results\ 에 CSV·PNG 자동 저장 → 그래프 창
//     [결과 그래프] 저장된 CSV 를 골라 모델 적합도 그래프 (추정 vs 정답, 오차, 분포, Θ→Y 누설)
//     [원본 보기] 측정 중에도 언제든 기준(center) 영상으로 돌아가기
//   파일 이름에서 정답을 읽는다 : "Y + 1.00 T + 1.00 (3).bmp", "Y+1.05.bmp", "T-0.35.bmp", "center.bmp"
//

#include "pch.h"
#include "framework.h"
#include "QR_OpenCV_Check.h"
#include "QR_OpenCV_CheckDlg.h"
#include "afxdialogex.h"
#include "ResultPlot.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <string>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// ---------------------------------------------------------------------------------
// 설정값
// ---------------------------------------------------------------------------------
static const double TOL_Y_MM  = 0.09;    // Y 허용 오차 (mm)
static const double TOL_T_DEG = 0.01;    // Θ 허용 오차 (deg) — 2026-10-20 요구사항 (사진마다 ±0.01°)
static const int    DISP_MAX_W = 1600;   // 오버레이를 그릴 표시 영상의 최대 가로 (px)
static const LPCTSTR DEFAULT_CALIB = _T("model\\phase_calib.yml");   // 시작할 때 자동으로 찾는 파일 (작업 디렉터리 기준)
static const LPCTSTR IMAGE_DIR     = _T("image");                     // 교보재 사진 폴더 (center.bmp, Y 폴더, Theta 폴더)
static const LPCTSTR DEFAULT_REF   = _T("image\\center.bmp");
static const LPCTSTR RESULT_DIR    = _T("results");                   // [전체 분석] 결과 CSV·PNG 저장 폴더

// ---------------------------------------------------------------------------------
// 도움 함수
// ---------------------------------------------------------------------------------
namespace {

// 한글 경로도 읽을 수 있게 파일을 바이트로 읽은 뒤 imdecode (cv::imread 는 유니코드 경로 미지원)
cv::Mat ReadGray(const CString& path)
{
	CFile f;
	if (!f.Open(path, CFile::modeRead | CFile::shareDenyWrite)) return cv::Mat();
	const ULONGLONG len = f.GetLength();
	if (len == 0 || len > 512ull * 1024 * 1024) return cv::Mat();
	std::vector<uchar> buf((size_t)len);
	UINT got = f.Read(buf.data(), (UINT)len);
	if (got != (UINT)len) return cv::Mat();
	return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

// 상대 경로 → 절대 경로 (파일 대화 상자 시작 폴더용)
CString FullPathOf(LPCTSTR rel)
{
	TCHAR buf[MAX_PATH] = {};
	DWORD n = GetFullPathName(rel, MAX_PATH, buf, nullptr);
	return (n > 0 && n < MAX_PATH) ? CString(buf) : CString(rel);
}

CString FileNameOf(const CString& path)
{
	int k = path.ReverseFind(_T('\\'));
	return k >= 0 ? path.Mid(k + 1) : path;
}

// 파일 이름 → 정답 (Y mm, Θ deg)
//   "Y + 1.00 T + 1.00 (3).bmp" → Y +1.00, Θ +1.00   (부호·숫자 사이 공백 허용, 끝의 "(n)" 무시)
//   "Y+1.05.bmp" → Y +1.05, Θ 0       "T-0.35.bmp" → Y 0, Θ −0.35       "center.bmp" → 0, 0
//   Y 또는 T 중 하나만 있으면 나머지는 0. 둘 다 없으면 정답 없음.
bool ParseTruth(const CString& name, double& y, double& t)
{
	CString stem = name;
	int dot = stem.ReverseFind(_T('.'));
	if (dot > 0) stem = stem.Left(dot);
	stem.Trim();
	if (stem.Left(6).CompareNoCase(_T("center")) == 0) { y = 0; t = 0; return true; }

	// 앞이 영문자가 아닌 위치의 Y / T(Theta, Θ) + 부호 + 숫자
	static const std::wregex reY(L"(?:^|[^A-Za-z])[Yy]\\s*([+-])\\s*(\\d+(?:\\.\\d+)?)");
	static const std::wregex reT(L"(?:^|[^A-Za-z])(?:[Tt]heta|THETA|[Tt]|\u0398|\u03B8)\\s*([+-])\\s*(\\d+(?:\\.\\d+)?)");
	const std::wstring s((LPCTSTR)stem);
	std::wsmatch m;
	bool hasY = false, hasT = false;
	y = 0; t = 0;
	if (std::regex_search(s, m, reY)) {
		y = std::stod(m[2].str()) * (m[1].str() == L"-" ? -1.0 : 1.0);
		hasY = true;
	}
	if (std::regex_search(s, m, reT)) {
		t = std::stod(m[2].str()) * (m[1].str() == L"-" ? -1.0 : 1.0);
		hasT = true;
	}
	return hasY || hasT;
}

// 하위 폴더까지 이미지 파일 모으기
void CollectImages(const CString& dir, std::vector<CString>& out)
{
	CFileFind ff;
	BOOL more = ff.FindFile(dir + _T("\\*.*"));
	while (more) {
		more = ff.FindNextFile();
		if (ff.IsDots()) continue;
		if (ff.IsDirectory()) { CollectImages(ff.GetFilePath(), out); continue; }
		CString ext = ff.GetFileName();
		int dot = ext.ReverseFind(_T('.'));
		if (dot < 0) continue;
		ext = ext.Mid(dot + 1).MakeLower();
		if (ext == _T("bmp") || ext == _T("png") || ext == _T("jpg") || ext == _T("jpeg") ||
			ext == _T("tif") || ext == _T("tiff"))
			out.push_back(ff.GetFilePath());
	}
}

// 점선 사각형 (기준 위치 표시용)
void DashedRect(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, int thick)
{
	const int dash = 10, gap = 6;
	cv::Point p[4] = { r.tl(), cv::Point(r.x + r.width, r.y), r.br(), cv::Point(r.x, r.y + r.height) };
	for (int e = 0; e < 4; ++e) {
		cv::Point a = p[e], b = p[(e + 1) % 4];
		double len = std::hypot((double)(b.x - a.x), (double)(b.y - a.y));
		if (len < 1) continue;
		for (double s = 0; s < len; s += dash + gap) {
			double s1 = (std::min)(len, s + dash);
			cv::Point q0(a.x + (int)((b.x - a.x) * s / len), a.y + (int)((b.y - a.y) * s / len));
			cv::Point q1(a.x + (int)((b.x - a.x) * s1 / len), a.y + (int)((b.y - a.y) * s1 / len));
			cv::line(img, q0, q1, color, thick, cv::LINE_AA);
		}
	}
}

bool PassT(const ResultRow& r) { return std::fabs(r.est.t - r.trueT) <= TOL_T_DEG; }
bool PassY(const ResultRow& r) { return std::fabs(r.est.y - r.trueY) <= TOL_Y_MM; }
LPCTSTR JudgeText(const ResultRow& r)
{
	const bool py = PassY(r), pt = PassT(r);
	return (py && pt) ? _T("OK") : (!py && !pt) ? _T("Y·Θ NG") : (!py ? _T("Y NG") : _T("Θ NG"));
}

// CSV 한 줄을 쉼표로 나눔 (큰따옴표 안의 쉼표는 무시)
std::vector<std::string> SplitCsv(const std::string& line)
{
	std::vector<std::string> out;
	std::string cur;
	bool q = false;
	for (char ch : line) {
		if (ch == '"') q = !q;
		else if (ch == ',' && !q) { out.push_back(cur); cur.clear(); }
		else cur += ch;
	}
	out.push_back(cur);
	return out;
}

// 결과 CSV 읽기 : 열 이름(file, trueY_mm, estY_mm, trueT_deg, estT_deg)으로 찾으므로 열 순서와 무관
bool ReadResultCsv(const CString& path, std::vector<rp::Rec>& recs, CString& err)
{
	recs.clear();
	FILE* fp = nullptr;
	if (_wfopen_s(&fp, path, L"rb") != 0 || fp == nullptr) { err = _T("파일을 열 수 없습니다."); return false; }
	std::string text;
	char buf[65536];
	size_t got;
	while ((got = fread(buf, 1, sizeof(buf), fp)) > 0) text.append(buf, got);
	fclose(fp);
	if (text.size() >= 3 && (unsigned char)text[0] == 0xEF && (unsigned char)text[1] == 0xBB && (unsigned char)text[2] == 0xBF)
		text.erase(0, 3);

	std::vector<std::string> lines;
	size_t a = 0;
	while (a < text.size()) {
		size_t b = text.find('\n', a);
		if (b == std::string::npos) b = text.size();
		std::string ln = text.substr(a, b - a);
		if (!ln.empty() && ln.back() == '\r') ln.pop_back();
		if (!ln.empty()) lines.push_back(ln);
		a = b + 1;
	}
	if (lines.size() < 2) { err = _T("내용이 없습니다."); return false; }

	std::vector<std::string> head = SplitCsv(lines[0]);
	auto col = [&](const char* n) { for (size_t i = 0; i < head.size(); ++i) if (head[i] == n) return (int)i; return -1; };
	int cName = col("file"); if (cName < 0) cName = col("path");
	const int cTy = col("trueY_mm"), cEy = col("estY_mm"), cTt = col("trueT_deg"), cEt = col("estT_deg");
	if (cTy < 0 || cEy < 0 || cTt < 0 || cEt < 0) { err = _T("필요한 열(trueY_mm, estY_mm, trueT_deg, estT_deg)이 없습니다."); return false; }

	auto num = [](const std::vector<std::string>& f, int c, double& v) {
		if (c < 0 || c >= (int)f.size() || f[c].empty()) return false;
		char* end = nullptr;
		v = std::strtod(f[c].c_str(), &end);
		return end != f[c].c_str() && std::isfinite(v);
	};
	for (size_t i = 1; i < lines.size(); ++i) {
		std::vector<std::string> f = SplitCsv(lines[i]);
		rp::Rec r;
		if (cName >= 0 && cName < (int)f.size()) r.name = f[cName];
		r.hasTruth = num(f, cTy, r.trueY) && num(f, cTt, r.trueT);
		r.hasEst = num(f, cEy, r.estY) && num(f, cEt, r.estT);
		recs.push_back(r);
	}
	return true;
}

// 그래프 창 : 그림 한 장을 창 크기에 맞춰 보여 주는 단순한 프레임 (닫으면 스스로 삭제)
class CGraphFrame : public CFrameWnd
{
public:
	explicit CGraphFrame(const cv::Mat& img) : m_img(img.clone()) {}
protected:
	cv::Mat  m_img;
	CMatView m_view;
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct)
	{
		if (CFrameWnd::OnCreate(lpCreateStruct) == -1) return -1;
		m_view.Create(_T(""), WS_CHILD | WS_VISIBLE | SS_NOTIFY, CRect(0, 0, 10, 10), this, AFX_IDW_PANE_FIRST);
		m_view.SetImage(m_img);
		return 0;
	}
	afx_msg void OnSize(UINT nType, int cx, int cy)
	{
		CFrameWnd::OnSize(nType, cx, cy);
		if (m_view.GetSafeHwnd()) m_view.MoveWindow(0, 0, cx, cy);
	}
	DECLARE_MESSAGE_MAP()
};
BEGIN_MESSAGE_MAP(CGraphFrame, CFrameWnd)
	ON_WM_CREATE()
	ON_WM_SIZE()
END_MESSAGE_MAP()

} // namespace


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CQROpenCVCheckDlg 대화 상자



CQROpenCVCheckDlg::CQROpenCVCheckDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_QR_OPENCV_CHECK_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CQROpenCVCheckDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_VIEW, m_view);
	DDX_Control(pDX, IDC_INFO, m_info);
	DDX_Control(pDX, IDC_LIST, m_list);
	DDX_Control(pDX, IDC_CHK_ZOOM, m_chkZoom);
}

BEGIN_MESSAGE_MAP(CQROpenCVCheckDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_SIZE()
	ON_WM_GETMINMAXINFO()
	ON_WM_DESTROY()
	ON_BN_CLICKED(IDC_BTN_REF, &CQROpenCVCheckDlg::OnBnClickedRef)
	ON_BN_CLICKED(IDC_BTN_SHOWREF, &CQROpenCVCheckDlg::OnBnClickedShowRef)
	ON_BN_CLICKED(IDC_BTN_CALIB, &CQROpenCVCheckDlg::OnBnClickedCalib)
	ON_BN_CLICKED(IDC_BTN_OPEN, &CQROpenCVCheckDlg::OnBnClickedOpen)
	ON_BN_CLICKED(IDC_BTN_FOLDER, &CQROpenCVCheckDlg::OnBnClickedFolder)
	ON_BN_CLICKED(IDC_BTN_SAVE, &CQROpenCVCheckDlg::OnBnClickedSave)
	ON_BN_CLICKED(IDC_BTN_ANALYZE, &CQROpenCVCheckDlg::OnBnClickedAnalyze)
	ON_BN_CLICKED(IDC_BTN_GRAPH, &CQROpenCVCheckDlg::OnBnClickedGraph)
	ON_BN_CLICKED(IDC_CHK_ZOOM, &CQROpenCVCheckDlg::OnBnClickedZoom)
	ON_NOTIFY(LVN_ITEMCHANGED, IDC_LIST, &CQROpenCVCheckDlg::OnLvnItemchangedList)
	ON_MESSAGE(WM_BATCH_ITEM, &CQROpenCVCheckDlg::OnBatchItem)
	ON_MESSAGE(WM_BATCH_DONE, &CQROpenCVCheckDlg::OnBatchDone)
END_MESSAGE_MAP()


// CQROpenCVCheckDlg 메시지 처리기

BOOL CQROpenCVCheckDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// ---------------------------------------------------------------- 컨트롤 초기화
	m_monoFont.CreatePointFont(100, _T("Consolas"));
	m_info.SetFont(&m_monoFont);

	m_list.SetExtendedStyle(LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES | LVS_EX_DOUBLEBUFFER);
	struct Col { LPCTSTR name; int w; int fmt; };
	const Col cols[] = {
		{ _T("파일"), 190, LVCFMT_LEFT },
		{ _T("정답 Y (mm)"), 90, LVCFMT_RIGHT },  { _T("추정 Y (mm)"), 95, LVCFMT_RIGHT },  { _T("Y 오차 (µm)"), 95, LVCFMT_RIGHT },
		{ _T("정답 Θ (°)"), 85, LVCFMT_RIGHT },   { _T("추정 Θ (°)"), 95, LVCFMT_RIGHT },   { _T("Θ 오차 (°)"), 95, LVCFMT_RIGHT },
		{ _T("판정"), 75, LVCFMT_CENTER },        { _T("유효 QR"), 65, LVCFMT_CENTER },     { _T("시간 (ms)"), 75, LVCFMT_RIGHT } };
	for (int i = 0; i < (int)(sizeof(cols) / sizeof(cols[0])); ++i)
		m_list.InsertColumn(i, cols[i].name, cols[i].fmt, cols[i].w);
	FitListColumns();

	// 처음 창 크기 : 2560 × 1440. 작업 영역(작업 표시줄 제외)보다 크면 작업 영역에 맞춤
	CRect work;
	SystemParametersInfo(SPI_GETWORKAREA, 0, &work, 0);
	const int winW = (std::min)(2560, work.Width()), winH = (std::min)(1440, work.Height());
	SetWindowPos(nullptr, 0, 0, winW, winH, SWP_NOMOVE | SWP_NOZORDER);
	CenterWindow();
	m_ready = true;
	CRect rc;
	GetClientRect(&rc);
	LayoutControls(rc.Width(), rc.Height());

	// ---------------------------------------------------------------- 기본 파일 자동 불러오기 (있을 때만)
	//   작업 디렉터리(VS 디버깅 기본값 = 프로젝트 폴더) 기준 model\phase_calib.yml, image\center.bmp
	m_resultDir = FullPathOf(RESULT_DIR);
	m_imageDir = FullPathOf(IMAGE_DIR);
	if (GetFileAttributes(m_imageDir) == INVALID_FILE_ATTRIBUTES) m_imageDir.Empty();   // 폴더가 없으면 대화 상자 기본 위치 사용
	if (GetFileAttributes(DEFAULT_CALIB) != INVALID_FILE_ATTRIBUTES) LoadCalibration(FullPathOf(DEFAULT_CALIB), true);
	if (GetFileAttributes(DEFAULT_REF) != INVALID_FILE_ATTRIBUTES)   LoadReference(FullPathOf(DEFAULT_REF), true);
	ShowCurrent();
	CString st;
	st.Format(_T("기준 이미지 %s, 보정 모델 %s — %s"), m_hasRef ? _T("등록됨") : _T("없음"), m_hasCalib ? _T("불러옴") : _T("없음"),
	          m_hasRef ? _T("[이미지 열기] 또는 [폴더 일괄] (image 폴더 선택 시 전체 검증)") : _T("[기준 이미지] 로 center 사진을 먼저 등록하세요."));
	UpdateStatus(st);

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CQROpenCVCheckDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CQROpenCVCheckDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CQROpenCVCheckDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

// Enter / Esc 로 대화 상자가 닫히지 않게 함 (닫기는 X 버튼)
BOOL CQROpenCVCheckDlg::PreTranslateMessage(MSG* pMsg)
{
	if (pMsg->message == WM_KEYDOWN && (pMsg->wParam == VK_RETURN || pMsg->wParam == VK_ESCAPE))
		return TRUE;
	return CDialogEx::PreTranslateMessage(pMsg);
}

void CQROpenCVCheckDlg::OnOK()
{
	// Enter 로 닫히지 않게 비워 둠
}

// ---------------------------------------------------------------------------------
// 배치 (창 크기에 맞춰 컨트롤 위치 조정)
// ---------------------------------------------------------------------------------
void CQROpenCVCheckDlg::OnSize(UINT nType, int cx, int cy)
{
	CDialogEx::OnSize(nType, cx, cy);
	if (m_ready && nType != SIZE_MINIMIZED) LayoutControls(cx, cy);
}

void CQROpenCVCheckDlg::OnGetMinMaxInfo(MINMAXINFO* lpMMI)
{
	lpMMI->ptMinTrackSize.x = 1180;
	lpMMI->ptMinTrackSize.y = 640;
	CDialogEx::OnGetMinMaxInfo(lpMMI);
}

void CQROpenCVCheckDlg::LayoutControls(int cx, int cy)
{
	const int M = 8, BTN_W = 112, BTN_H = 28, LIST_H = 230, STATUS_H = 20;
	const int PANEL_W = (std::max)(460, cx * 36 / 100);       // 수치 패널 : 한 줄이 줄바꿈 없이 보이도록 넓게
	const UINT btns[] = { IDC_BTN_REF, IDC_BTN_SHOWREF, IDC_BTN_CALIB, IDC_BTN_OPEN, IDC_BTN_FOLDER,
	                      IDC_BTN_ANALYZE, IDC_BTN_SAVE, IDC_BTN_GRAPH };
	int x = M;
	for (UINT id : btns) {
		if (CWnd* w = GetDlgItem(id)) w->MoveWindow(x, M, BTN_W, BTN_H);
		x += BTN_W + 6;
	}
	if (CWnd* w = GetDlgItem(IDC_CHK_ZOOM)) w->MoveWindow(x + 10, M, 160, BTN_H);

	const int top = M + BTN_H + M;
	const int statusY = cy - M - STATUS_H;
	const int listY = statusY - 4 - LIST_H;
	const int viewH = (std::max)(80, listY - M - top);
	const int viewW = (std::max)(200, cx - 3 * M - PANEL_W);
	m_view.MoveWindow(M, top, viewW, viewH);
	m_info.MoveWindow(M + viewW + M, top, PANEL_W, viewH);
	m_list.MoveWindow(M, listY, cx - 2 * M, LIST_H);
	if (CWnd* w = GetDlgItem(IDC_STATUS)) w->MoveWindow(M, statusY, cx - 2 * M, STATUS_H);
	m_view.Invalidate(FALSE);
}

// ---------------------------------------------------------------------------------
// 불러오기
// ---------------------------------------------------------------------------------
bool CQROpenCVCheckDlg::LoadReference(const CString& path, bool quiet)
{
	cv::Mat g = ReadGray(path);
	if (g.empty()) {
		if (!quiet) AfxMessageBox(_T("이미지를 읽을 수 없습니다.\n") + path);
		return false;
	}
	CWaitCursor wait;
	if (!m_aligner.setReference(g)) {
		m_hasRef = false;
		if (!quiet) AfxMessageBox(_T("기준 이미지에서 QR 3개를 찾지 못했습니다.\n(Y=0, Θ=0 인 center 사진을 선택하세요)"));
		ShowCurrent();
		return false;
	}
	m_refImage = g;
	m_refPath = path;
	m_hasRef = true;

	if (m_hasCur && !m_curImage.empty()) AnalyzeImage(m_curImage, m_cur);   // 기준이 바뀌면 다시 측정
	ShowCurrent();
	UpdateStatus(_T("기준 이미지 등록 : ") + FileNameOf(path));
	return true;
}

bool CQROpenCVCheckDlg::LoadCalibration(const CString& path, bool quiet)
{
	pa::Calibration cal;
	pa::PhaseParams prm = m_aligner.params();
	// cv::FileStorage 는 ANSI 경로만 받으므로 시스템 코드 페이지(한글 Windows = CP949)로 변환
	CW2A pathA(path, CP_ACP);
	std::string sp((LPCSTR)pathA);
	if (!cal.load(sp, &prm)) {
		if (!quiet) AfxMessageBox(_T("보정 모델을 읽을 수 없습니다.\n") + path);
		return false;
	}
	CWaitCursor wait;
	m_aligner.setParams(prm);                      // 필터 매개변수 적용 (기준이 있으면 기준 위상도 다시 계산)
	if (m_hasRef) m_hasRef = m_aligner.hasReference();
	m_aligner.setCalibration(cal);
	m_hasCalib = true;
	m_calibUseMag = cal.useMag;
	m_calibPath = path;

	if (m_hasCur && !m_curImage.empty()) AnalyzeImage(m_curImage, m_cur);
	// 이미 측정한 일괄 결과는 특징을 그대로 두고 추정만 다시 (같은 필터 매개변수일 때 유효)
	if (!m_busy) {
		for (auto& r : m_rows) if (r.loaded) r.est = m_aligner.estimate(r.feat);
		m_list.DeleteAllItems();
		for (int i = 0; i < (int)m_rows.size(); ++i) if (m_rows[i].loaded) AddListRow(i);
	}
	ShowCurrent();
	UpdateStatus(_T("보정 모델 불러옴 : ") + FileNameOf(path));
	return true;
}

// 측정 + 추정 (작업 스레드에서도 호출 — m_aligner 의 const 함수만 사용)
void CQROpenCVCheckDlg::AnalyzeImage(const cv::Mat& gray, ResultRow& row) const
{
	auto t0 = std::chrono::steady_clock::now();
	row.feat = m_aligner.measure(gray);
	row.est = m_hasCalib ? m_aligner.estimate(row.feat) : pa::Estimate();
	row.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	row.loaded = true;
}

// ---------------------------------------------------------------------------------
// 표시
// ---------------------------------------------------------------------------------
void CQROpenCVCheckDlg::ShowCurrent()
{
	RenderView();
	UpdateInfo();
}

void CQROpenCVCheckDlg::RenderView()
{
	const cv::Mat& src = !m_curImage.empty() ? m_curImage : m_refImage;
	if (src.empty()) { m_view.Clear(); return; }

	// 보여 줄 영역 : 전체 또는 QR 띠 주변 (Θ ±2° ≈ ±160 px 이동까지 보이도록 여유)
	cv::Rect crop(0, 0, src.cols, src.rows);
	if (m_hasRef && m_chkZoom.GetCheck() == BST_CHECKED) {
		cv::Rect u = m_aligner.roi(0) | m_aligner.roi(1) | m_aligner.roi(2);
		u.x -= 220; u.width += 440; u.y -= 120; u.height += 240;
		crop = u & cv::Rect(0, 0, src.cols, src.rows);
	}
	const double scale = (std::min)(1.0, (double)DISP_MAX_W / crop.width);
	cv::Mat small, disp;
	cv::resize(src(crop), small, cv::Size(), scale, scale, cv::INTER_AREA);
	cv::cvtColor(small, disp, cv::COLOR_GRAY2BGR);

	if (m_hasRef) {
		auto toDisp = [&](const cv::Rect& r) {
			return cv::Rect(cvRound((r.x - crop.x) * scale), cvRound((r.y - crop.y) * scale),
			                cvRound(r.width * scale), cvRound(r.height * scale));
		};
		const char* qn[pa::NUM_QR] = { "L", "C", "R" };
		const cv::Scalar red(40, 40, 230), green(60, 200, 60), orange(0, 150, 255), yellow(0, 220, 255);
		for (int q = 0; q < pa::NUM_QR; ++q) {
			cv::Rect r0 = toDisp(m_aligner.roi(q));
			DashedRect(disp, r0, red, 2);                                         // 기준 위치 (빨간 점선)
			if (!(m_hasCur && m_cur.loaded && m_cur.feat.qr[q].nValid > 0)) continue;

			const pa::QRMeasure& m = m_cur.feat.qr[q];
			double dx = m.shift * scale, dy = m_cur.feat.coarseDy * scale;
			cv::Rect r1(r0.x + cvRound(dx), r0.y + cvRound(dy), r0.width, r0.height);
			cv::Scalar col = m.valid ? green : orange;
			cv::rectangle(disp, r1, col, 2, cv::LINE_AA);                        // 측정 위치 (실선)
			cv::Point c0(r0.x + r0.width / 2, r0.y + r0.height / 2);
			cv::Point c1(c0.x + cvRound(dx), c0.y + cvRound(dy));
			if (std::abs(c1.x - c0.x) + std::abs(c1.y - c0.y) >= 2)
				cv::arrowedLine(disp, c0, c1, yellow, 2, cv::LINE_AA, 0, 0.25);    // 이동 화살표
			char buf[64];
			std::snprintf(buf, sizeof(buf), "%s  t %+.2f px", qn[q], m.shift);
			cv::putText(disp, buf, cv::Point(r1.x, (std::max)(18, r1.y - 10)), cv::FONT_HERSHEY_SIMPLEX,
			            0.7, cv::Scalar(255, 255, 255), 4, cv::LINE_AA);
			cv::putText(disp, buf, cv::Point(r1.x, (std::max)(18, r1.y - 10)), cv::FONT_HERSHEY_SIMPLEX,
			            0.7, col, 2, cv::LINE_AA);
		}
	}
	m_view.SetImage(disp);
}

void CQROpenCVCheckDlg::UpdateInfo()
{
	CString s, l;
	const CString bar = _T("──────────────────────────────\r\n");

	l.Format(_T("기준 이미지 : %s\r\n"), m_hasRef ? (LPCTSTR)FileNameOf(m_refPath) : _T("(없음)"));  s += l;
	l.Format(_T("보정 모델   : %s\r\n"), m_hasCalib ? (LPCTSTR)(FileNameOf(m_calibPath) + (m_calibUseMag ? _T(" (3변수)") : _T(" (2변수)")))
	                                           : _T("(없음 — QR 이동만 표시)"));  s += l;
	s += bar;

	if (!m_hasCur || !m_cur.loaded) {
		if (m_hasRef) {
			l.Format(_T("[기준 이미지 표시 중] %s  (%d × %d)\r\n"), (LPCTSTR)FileNameOf(m_refPath), m_refImage.cols, m_refImage.rows);
			s += l;
			const TCHAR* qn0[pa::NUM_QR] = { _T("L"), _T("C"), _T("R") };
			for (int q = 0; q < pa::NUM_QR; ++q) {
				cv::Rect r0 = m_aligner.roi(q);
				l.Format(_T("  QR %s 측정 영역 : x %d ~ %d, y %d ~ %d\r\n"), qn0[q], r0.x, r0.x + r0.width - 1, r0.y, r0.y + r0.height - 1);
				s += l;
			}
			s += _T("\r\n");
		}
		else s += _T("측정한 이미지가 없습니다.\r\n\r\n");
		s += _T("사용 순서\r\n");
		s += _T(" 1) [기준 이미지] : Y=0, Θ=0 인 center 사진\r\n");
		s += _T(" 2) [보정 모델]   : phase_calib.yml\r\n");
		s += _T(" 3) [이미지 열기] (여러 장 선택 가능) 또는 [폴더 일괄]\r\n");
		s += _T(" 4) 목록에서 행 선택 → 그 사진, [원본 보기] → 기준으로 복귀\r\n\r\n");
		s += _T("오버레이\r\n");
		s += _T(" 빨간 점선 : 기준 위치 (측정 영역)\r\n");
		s += _T(" 초록 실선 : 측정 위치 (주황 = 품질 미달)\r\n");
		s += _T(" 노란 화살표 : 이동 방향·크기\r\n");
		m_info.SetWindowText(s);
		return;
	}

	const ResultRow& r = m_cur;
	l.Format(_T("파일 : %s\r\n처리 시간 : %.1f ms\r\n"), (LPCTSTR)r.name, r.ms);  s += l;
	s += bar;

	if (r.est.valid) {
		l.Format(_T("Y = %+9.4f mm   (σ %.4f)\r\n"), r.est.y, r.est.sdY);  s += l;
		l.Format(_T("Θ = %+9.4f °    (σ %.4f)\r\n"), r.est.t, r.est.sdT);  s += l;
		l.Format(_T("로봇 보정값 : Y %+.4f mm, Θ %+.4f °\r\n"), -r.est.y, -r.est.t);  s += l;
		if (r.hasTruth) {
			double eY = (r.est.y - r.trueY) * 1000, eT = r.est.t - r.trueT;
			l.Format(_T("정답 : Y %+.2f mm, Θ %+.2f °\r\n"), r.trueY, r.trueT);  s += l;
			l.Format(_T("오차 : Y %+7.1f µm  [%s ±%.0f]\r\n"), eY, PassY(r) ? _T("OK") : _T("NG"), TOL_Y_MM * 1000);  s += l;
			l.Format(_T("       Θ %+8.4f °  [%s ±%.3f]\r\n"), eT, PassT(r) ? _T("OK") : _T("NG"), TOL_T_DEG);  s += l;
		}
	}
	else if (!m_hasCalib) {
		s += _T("Y·Θ : 보정 모델이 없어 계산하지 않음\r\n");
	}
	else {
		s += _T("Y·Θ : 추정 실패 (QR 측정 실패)\r\n");
	}
	s += bar;

	s += _T("QR   t(px)    s(px/100px)  유효%  잔차px  판정\r\n");
	const TCHAR* qn[pa::NUM_QR] = { _T("L"), _T("C"), _T("R") };
	for (int q = 0; q < pa::NUM_QR; ++q) {
		const pa::QRMeasure& m = r.feat.qr[q];
		l.Format(_T("%s  %+9.3f  %+9.4f   %5.1f  %6.3f  %s\r\n"), qn[q], m.shift, m.scale,
		         m.validRatio * 100, m.fitStd, m.valid ? _T("OK") : _T("NG"));
		s += l;
	}
	l.Format(_T("\r\n거친 이동 : dx %+.2f, dy %+.2f px\r\n유효 QR   : %d / 3\r\n"),
	         r.feat.coarseDx, r.feat.coarseDy, r.feat.numValidQR);
	s += l;
	m_info.SetWindowText(s);
}

void CQROpenCVCheckDlg::UpdateStatus(const CString& msg)
{
	SetDlgItemText(IDC_STATUS, msg);
}

void CQROpenCVCheckDlg::OnBnClickedZoom()
{
	RenderView();
}

// ---------------------------------------------------------------------------------
// 버튼
// ---------------------------------------------------------------------------------
static const LPCTSTR IMG_FILTER =
	_T("이미지 (*.bmp;*.png;*.jpg;*.tif)|*.bmp;*.png;*.jpg;*.jpeg;*.tif;*.tiff|모든 파일 (*.*)|*.*||");

void CQROpenCVCheckDlg::OnBnClickedRef()
{
	CFileDialog dlg(TRUE, nullptr, nullptr, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY, IMG_FILTER, this);
	dlg.m_ofn.lpstrTitle = _T("기준 이미지 (center) 선택");
	if (!m_imageDir.IsEmpty()) dlg.m_ofn.lpstrInitialDir = m_imageDir;
	if (dlg.DoModal() == IDOK) LoadReference(dlg.GetPathName());
}

void CQROpenCVCheckDlg::OnBnClickedCalib()
{
	CFileDialog dlg(TRUE, _T("yml"), nullptr, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY,
	                _T("보정 모델 (*.yml)|*.yml;*.yaml;*.xml|모든 파일 (*.*)|*.*||"), this);
	dlg.m_ofn.lpstrTitle = _T("보정 모델 (phase_calib.yml) 선택");
	if (dlg.DoModal() == IDOK) LoadCalibration(dlg.GetPathName());
}

void CQROpenCVCheckDlg::OnBnClickedOpen()
{
	if (m_busy) return;
	if (!m_hasRef) { AfxMessageBox(_T("먼저 [기준 이미지] 를 등록하세요.")); return; }
	CFileDialog dlg(TRUE, nullptr, nullptr, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY | OFN_ALLOWMULTISELECT | OFN_EXPLORER,
	                IMG_FILTER, this);
	dlg.m_ofn.lpstrTitle = _T("측정할 이미지 선택 (Ctrl / Shift 로 여러 장)");
	if (!m_imageDir.IsEmpty()) dlg.m_ofn.lpstrInitialDir = m_imageDir;
	std::vector<TCHAR> nameBuf(64 * 1024, 0);           // 여러 장 선택용 큰 버퍼
	dlg.m_ofn.lpstrFile = nameBuf.data();
	dlg.m_ofn.nMaxFile = (DWORD)nameBuf.size();
	if (dlg.DoModal() != IDOK) return;

	std::vector<CString> files;
	POSITION pos = dlg.GetStartPosition();
	while (pos) files.push_back(dlg.GetNextPathName(pos));
	if (files.empty()) return;
	std::sort(files.begin(), files.end(), [](const CString& a, const CString& b) { return a.CompareNoCase(b) < 0; });

	if (files.size() > 1) { StartBatch(files, true); return; }   // 여러 장은 작업 스레드로 (목록에 추가)

	// 한 장 : 바로 측정해서 화면에 표시하고 목록에도 추가
	const CString& path = files[0];
	cv::Mat g = ReadGray(path);
	if (g.empty()) { AfxMessageBox(_T("이미지를 읽을 수 없습니다.\n") + path); return; }
	if (g.size() != m_refImage.size()) { AfxMessageBox(_T("기준 이미지와 크기가 다릅니다.")); return; }

	CWaitCursor wait;
	ResultRow row;
	row.path = path;
	row.name = FileNameOf(path);
	row.hasTruth = ParseTruth(row.name, row.trueY, row.trueT);
	AnalyzeImage(g, row);
	m_rows.push_back(row);
	m_list.SetItemState(-1, 0, LVIS_SELECTED);               // 이전 선택 해제
	AddListRow((int)m_rows.size() - 1);

	m_curImage = g;
	m_cur = row;
	m_hasCur = true;
	ShowCurrent();
	CString msg;
	msg.Format(_T("측정 완료 : %s (%.1f ms) — [원본 보기] 로 기준 영상으로 돌아갈 수 있습니다."), (LPCTSTR)row.name, row.ms);
	UpdateStatus(msg);
}

// 기준 영상으로 돌아가기 (측정 결과·목록은 그대로)
void CQROpenCVCheckDlg::OnBnClickedShowRef()
{
	if (!m_hasRef) { AfxMessageBox(_T("등록된 기준 이미지가 없습니다.")); return; }
	m_list.SetItemState(-1, 0, LVIS_SELECTED);
	m_curImage.release();
	m_hasCur = false;
	ShowCurrent();
	UpdateStatus(_T("기준 이미지 표시 : ") + FileNameOf(m_refPath) + _T("  (목록에서 행을 고르면 그 사진으로 이동)"));
}

// 작업 스레드로 여러 장 측정. append = false 면 목록을 비우고 새로 시작
void CQROpenCVCheckDlg::StartBatch(const std::vector<CString>& files, bool append)
{
	if (!append) { m_rows.clear(); m_list.DeleteAllItems(); }
	m_batchBegin = m_rows.size();
	for (const CString& f : files) {
		ResultRow r;
		r.path = f;
		r.name = FileNameOf(f);
		r.hasTruth = ParseTruth(r.name, r.trueY, r.trueT);
		m_rows.push_back(r);
	}
	SetBusy(true);
	m_stop = false;
	m_worker = AfxBeginThread(BatchProc, this, THREAD_PRIORITY_BELOW_NORMAL, 0, CREATE_SUSPENDED);
	if (m_worker == nullptr) { SetBusy(false); AfxMessageBox(_T("작업 스레드를 만들 수 없습니다.")); return; }
	m_worker->m_bAutoDelete = FALSE;   // 끝날 때 핸들을 기다리기 위해 직접 삭제
	m_worker->ResumeThread();
}

void CQROpenCVCheckDlg::OnBnClickedFolder()
{
	if (m_busy) { m_stop = true; UpdateStatus(_T("중지하는 중...")); return; }   // 진행 중이면 "중지" 버튼
	if (!m_hasRef) { AfxMessageBox(_T("먼저 [기준 이미지] 를 등록하세요.")); return; }

	CFolderPickerDialog dlg(m_imageDir.IsEmpty() ? nullptr : (LPCTSTR)m_imageDir, 0, this);
	dlg.m_ofn.lpstrTitle = _T("이미지 폴더 선택 (하위 폴더 포함)");
	if (dlg.DoModal() != IDOK) return;

	std::vector<CString> files;
	CollectImages(dlg.GetPathName(), files);
	if (files.empty()) { AfxMessageBox(_T("폴더에 이미지가 없습니다.")); return; }
	std::sort(files.begin(), files.end(), [](const CString& a, const CString& b) { return a.CompareNoCase(b) < 0; });
	StartBatch(files, false);
}

// 작업 스레드 : 사진을 하나씩 읽어 측정 → 결과를 m_rows[i] 에 저장하고 대화 상자에 알림
UINT CQROpenCVCheckDlg::BatchProc(LPVOID param)
{
	CQROpenCVCheckDlg* self = reinterpret_cast<CQROpenCVCheckDlg*>(param);
	const HWND hwnd = self->GetSafeHwnd();
	for (size_t i = self->m_batchBegin; i < self->m_rows.size(); ++i) {
		if (self->m_stop) break;
		ResultRow& r = self->m_rows[i];
		cv::Mat g = ReadGray(r.path);
		if (!g.empty() && g.size() == self->m_refImage.size()) self->AnalyzeImage(g, r);
		::PostMessage(hwnd, WM_BATCH_ITEM, (WPARAM)i, 0);
	}
	::PostMessage(hwnd, WM_BATCH_DONE, 0, 0);
	return 0;
}

LRESULT CQROpenCVCheckDlg::OnBatchItem(WPARAM wParam, LPARAM /*lParam*/)
{
	int i = (int)wParam;
	if (i < 0 || i >= (int)m_rows.size()) return 0;
	AddListRow(i);
	CString msg;
	msg.Format(_T("일괄 처리 중... %d / %d  (%s)"), i + 1 - (int)m_batchBegin, (int)(m_rows.size() - m_batchBegin), (LPCTSTR)m_rows[i].name);
	UpdateStatus(msg);
	return 0;
}

LRESULT CQROpenCVCheckDlg::OnBatchDone(WPARAM /*wParam*/, LPARAM /*lParam*/)
{
	if (m_worker) {
		WaitForSingleObject(m_worker->m_hThread, INFINITE);
		delete m_worker;
		m_worker = nullptr;
	}
	SetBusy(false);

	// 요약 : 정답이 있고 추정에 성공한 사진만
	int n = 0, nT = 0, nY = 0, nDone = 0;
	double ssT = 0, ssY = 0, maxT = 0, maxY = 0, sumMs = 0;
	for (const auto& r : m_rows) {
		if (!r.loaded) continue;
		nDone++; sumMs += r.ms;
		if (!r.hasTruth || !r.est.valid) continue;
		double eT = r.est.t - r.trueT, eY = (r.est.y - r.trueY) * 1000;
		n++; ssT += eT * eT; ssY += eY * eY;
		maxT = (std::max)(maxT, std::fabs(eT)); maxY = (std::max)(maxY, std::fabs(eY));
		if (PassT(r)) nT++;
		if (PassY(r)) nY++;
	}
	CString msg;
	if (n > 0)
		msg.Format(_T("완료 %d장 (평균 %.0f ms) | 정답 비교 %d장 : Θ RMS %.4f° 최대 %.4f° 통과 %d/%d (±%.3f°) | Y RMS %.1f µm 최대 %.1f µm 통과 %d/%d (±%.0f µm)"),
		           nDone, sumMs / (std::max)(1, nDone), n, std::sqrt(ssT / n), maxT, nT, n, TOL_T_DEG,
		           std::sqrt(ssY / n), maxY, nY, n, TOL_Y_MM * 1000);
	else
		msg.Format(_T("완료 %d장 (평균 %.0f ms) | 정답과 비교할 수 있는 사진 없음 (보정 모델 또는 파일 이름 확인)"),
		           nDone, sumMs / (std::max)(1, nDone));
	if (m_stop) msg = _T("[중지됨] ") + msg;
	UpdateStatus(msg);

	// [전체 분석] 으로 시작했으면 results 폴더에 CSV 자동 저장 → 그래프
	if (m_autoReport) {
		m_autoReport = false;
		if (!m_stop && !m_resultDir.IsEmpty()) {
			CreateDirectory(m_resultDir, nullptr);
			CString csv = m_resultDir + _T("\\qr_yt_") + CTime::GetCurrentTime().Format(_T("%Y%m%d_%H%M%S")) + _T(".csv");
			if (WriteCsv(csv)) { m_lastCsv = csv; ShowReport(csv); UpdateStatus(msg + _T("  →  ") + FileNameOf(csv)); }
			else AfxMessageBox(_T("결과 CSV 를 저장할 수 없습니다.\n") + csv);
		}
	}
	return 0;
}

void CQROpenCVCheckDlg::AddListRow(int i)
{
	// 열 : 0 파일 | 1 정답 Y | 2 추정 Y | 3 Y 오차 | 4 정답 Θ | 5 추정 Θ | 6 Θ 오차 | 7 판정 | 8 유효 QR | 9 시간
	const ResultRow& r = m_rows[i];
	int item = m_list.InsertItem(m_list.GetItemCount(), r.name);
	m_list.SetItemData(item, (DWORD_PTR)i);
	CString t;
	if (r.hasTruth) {
		t.Format(_T("%+.2f"), r.trueY); m_list.SetItemText(item, 1, t);
		t.Format(_T("%+.2f"), r.trueT); m_list.SetItemText(item, 4, t);
	}
	if (!r.loaded) { m_list.SetItemText(item, 7, _T("읽기 실패")); return; }
	if (r.est.valid) {
		t.Format(_T("%+.4f"), r.est.y); m_list.SetItemText(item, 2, t);
		t.Format(_T("%+.4f"), r.est.t); m_list.SetItemText(item, 5, t);
		if (r.hasTruth) {
			t.Format(_T("%+.1f"), (r.est.y - r.trueY) * 1000); m_list.SetItemText(item, 3, t);
			t.Format(_T("%+.4f"), r.est.t - r.trueT);          m_list.SetItemText(item, 6, t);
			m_list.SetItemText(item, 7, JudgeText(r));
		}
	}
	else m_list.SetItemText(item, 7, m_hasCalib ? _T("추정 실패") : _T("모델 없음"));
	t.Format(_T("%d/3"), r.feat.numValidQR); m_list.SetItemText(item, 8, t);
	t.Format(_T("%.0f"), r.ms);              m_list.SetItemText(item, 9, t);
	m_list.EnsureVisible(item, FALSE);
}

// 목록 열 너비 : 제목 글자가 모두 보이는 너비와 기본 너비 중 큰 값 (마지막 열은 남는 폭까지)
void CQROpenCVCheckDlg::FitListColumns()
{
	CHeaderCtrl* hdr = m_list.GetHeaderCtrl();
	if (!hdr) return;
	const int n = hdr->GetItemCount();
	CClientDC dc(&m_list);
	CFont* old = dc.SelectObject(m_list.GetFont());
	for (int i = 0; i < n; ++i) {
		TCHAR text[128] = {};
		HDITEM it = {};
		it.mask = HDI_TEXT;
		it.pszText = text;
		it.cchTextMax = 127;
		hdr->GetItem(i, &it);
		const int need = dc.GetTextExtent(text).cx + 28;     // 제목 글자 폭 + 여백(정렬 표시·구분선)
		if (m_list.GetColumnWidth(i) < need) m_list.SetColumnWidth(i, need);
	}
	dc.SelectObject(old);
}

void CQROpenCVCheckDlg::SetBusy(bool busy)
{
	m_busy = busy;
	GetDlgItem(IDC_BTN_REF)->EnableWindow(!busy);
	GetDlgItem(IDC_BTN_CALIB)->EnableWindow(!busy);
	GetDlgItem(IDC_BTN_OPEN)->EnableWindow(!busy);
	GetDlgItem(IDC_BTN_SAVE)->EnableWindow(!busy);
	GetDlgItem(IDC_BTN_ANALYZE)->EnableWindow(!busy);
	GetDlgItem(IDC_BTN_GRAPH)->EnableWindow(!busy);
	SetDlgItemText(IDC_BTN_FOLDER, busy ? _T("중지") : _T("폴더 일괄"));
}

// 목록에서 선택한 사진을 화면에 표시 (저장된 결과 사용, 다시 측정하지 않음)
void CQROpenCVCheckDlg::OnLvnItemchangedList(NMHDR* pNMHDR, LRESULT* pResult)
{
	LPNMLISTVIEW p = reinterpret_cast<LPNMLISTVIEW>(pNMHDR);
	*pResult = 0;
	if (!(p->uChanged & LVIF_STATE) || !(p->uNewState & LVIS_SELECTED) || (p->uOldState & LVIS_SELECTED)) return;
	int idx = (int)m_list.GetItemData(p->iItem);
	if (idx < 0 || idx >= (int)m_rows.size() || !m_rows[idx].loaded) return;

	CWaitCursor wait;
	cv::Mat g = ReadGray(m_rows[idx].path);
	if (g.empty()) return;
	m_curImage = g;
	m_cur = m_rows[idx];
	m_hasCur = true;
	ShowCurrent();
}

void CQROpenCVCheckDlg::OnBnClickedSave()
{
	if (m_rows.empty()) { AfxMessageBox(_T("저장할 일괄 결과가 없습니다. [폴더 일괄] 또는 [전체 분석] 을 먼저 실행하세요.")); return; }
	CFileDialog dlg(FALSE, _T("csv"), _T("qr_yt_results.csv"), OFN_OVERWRITEPROMPT | OFN_HIDEREADONLY,
	                _T("CSV (*.csv)|*.csv||"), this);
	if (!m_resultDir.IsEmpty()) dlg.m_ofn.lpstrInitialDir = m_resultDir;
	if (dlg.DoModal() != IDOK) return;
	if (!WriteCsv(dlg.GetPathName())) { AfxMessageBox(_T("파일을 저장할 수 없습니다.")); return; }
	m_lastCsv = dlg.GetPathName();
	UpdateStatus(_T("CSV 저장 : ") + dlg.GetPathName());
}

// 일괄 결과 → CSV (UTF-8 BOM). 열 : file, trueY_mm, estY_mm, errY_um, trueT_deg, estT_deg, errT_deg, judge, validQR, 특징 6개, 거친 이동, ms
bool CQROpenCVCheckDlg::WriteCsv(const CString& path) const
{
	CString text = _T("file,trueY_mm,estY_mm,errY_um,trueT_deg,estT_deg,errT_deg,judge,validQR,")
	               _T("tL,sL,tC,sC,tR,sR,coarseDx,coarseDy,ms\r\n");
	CString l;
	for (const auto& r : m_rows) {
		if (!r.loaded) continue;
		// 이름에 쉼표가 있을 수 있어 따옴표로 감쌈
		l.Format(_T("\"%s\","), (LPCTSTR)r.name); text += l;
		const bool both = r.hasTruth && r.est.valid;
		if (r.hasTruth) { l.Format(_T("%.4f,"), r.trueY); text += l; } else text += _T(",");
		if (r.est.valid) { l.Format(_T("%.6f,"), r.est.y); text += l; } else text += _T(",");
		if (both) { l.Format(_T("%.2f,"), (r.est.y - r.trueY) * 1000); text += l; } else text += _T(",");
		if (r.hasTruth) { l.Format(_T("%.4f,"), r.trueT); text += l; } else text += _T(",");
		if (r.est.valid) { l.Format(_T("%.6f,"), r.est.t); text += l; } else text += _T(",");
		if (both) { l.Format(_T("%.6f,%s,"), r.est.t - r.trueT, JudgeText(r)); text += l; } else text += _T(",,");
		l.Format(_T("%d"), r.feat.numValidQR); text += l;
		for (int k = 0; k < pa::NUM_FEAT; ++k) { l.Format(_T(",%.6f"), r.feat.f[k]); text += l; }
		l.Format(_T(",%.3f,%.3f,%.1f\r\n"), r.feat.coarseDx, r.feat.coarseDy, r.ms); text += l;
	}

	FILE* fp = nullptr;
	if (_wfopen_s(&fp, path, L"wb") != 0 || fp == nullptr) return false;
	const unsigned char bom[3] = { 0xEF, 0xBB, 0xBF };   // 엑셀에서 한글이 깨지지 않게 UTF-8 BOM
	fwrite(bom, 1, 3, fp);
	CW2A utf8(text, CP_UTF8);
	fwrite((const char*)utf8, 1, strlen(utf8), fp);
	fclose(fp);
	return true;
}

// [전체 분석] : image 폴더 전체를 한 번에 → 끝나면 CSV·PNG 자동 저장 + 그래프 창
void CQROpenCVCheckDlg::OnBnClickedAnalyze()
{
	if (m_busy) return;
	if (!m_hasRef) { AfxMessageBox(_T("먼저 [기준 이미지] 를 등록하세요.")); return; }
	if (!m_hasCalib) { AfxMessageBox(_T("보정 모델이 없어 Y·Θ 를 계산할 수 없습니다. [보정 모델] 을 먼저 불러오세요.")); return; }
	if (m_imageDir.IsEmpty()) { AfxMessageBox(_T("image 폴더를 찾을 수 없습니다. [폴더 일괄] 로 폴더를 직접 고르세요.")); return; }

	std::vector<CString> files;
	CollectImages(m_imageDir, files);
	if (files.empty()) { AfxMessageBox(_T("image 폴더에 이미지가 없습니다.")); return; }
	std::sort(files.begin(), files.end(), [](const CString& a, const CString& b) { return a.CompareNoCase(b) < 0; });
	m_autoReport = true;
	StartBatch(files, false);
}

// [결과 그래프] : CSV 를 골라 그래프 창 (기본 위치 = results 폴더, 마지막 CSV)
void CQROpenCVCheckDlg::OnBnClickedGraph()
{
	CFileDialog dlg(TRUE, _T("csv"), m_lastCsv.IsEmpty() ? nullptr : (LPCTSTR)m_lastCsv, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY,
	                _T("결과 CSV (*.csv)|*.csv|모든 파일 (*.*)|*.*||"), this);
	dlg.m_ofn.lpstrTitle = _T("그래프로 볼 결과 CSV 선택");
	if (!m_resultDir.IsEmpty() && GetFileAttributes(m_resultDir) != INVALID_FILE_ATTRIBUTES) dlg.m_ofn.lpstrInitialDir = m_resultDir;
	if (dlg.DoModal() != IDOK) return;
	ShowReport(dlg.GetPathName());
}

// CSV → 그래프 그림 → 같은 이름의 PNG 저장 → 그래프 창
void CQROpenCVCheckDlg::ShowReport(const CString& csvPath)
{
	std::vector<rp::Rec> recs;
	CString err;
	if (!ReadResultCsv(csvPath, recs, err)) { AfxMessageBox(_T("CSV 를 읽을 수 없습니다 : ") + err + _T("\n") + csvPath); return; }

	CWaitCursor wait;
	CString title = FileNameOf(csvPath);
	std::string titleA = "QR Y-Theta model check  |  ";
	titleA += (LPCSTR)CW2A(title, CP_UTF8);
	cv::Mat img = rp::drawReport(recs, TOL_Y_MM, TOL_T_DEG, titleA);

	// PNG 저장 (한글 경로 대비 imencode → 파일 쓰기)
	CString png = csvPath;
	int dot = png.ReverseFind(_T('.'));
	if (dot > 0) png = png.Left(dot);
	png += _T(".png");
	std::vector<uchar> buf;
	if (cv::imencode(".png", img, buf)) {
		FILE* fp = nullptr;
		if (_wfopen_s(&fp, png, L"wb") == 0 && fp) { fwrite(buf.data(), 1, buf.size(), fp); fclose(fp); }
	}

	CGraphFrame* f = new CGraphFrame(img);
	// 그래프 창 : 작업 영역의 95% 크기 (그래프 원본 1820 × 1180 에 가깝게 크게)
	CRect work;
	SystemParametersInfo(SPI_GETWORKAREA, 0, &work, 0);
	CRect rc(0, 0, work.Width() * 95 / 100, work.Height() * 95 / 100);
	if (!f->Create(nullptr, _T("결과 그래프 — ") + title, WS_OVERLAPPEDWINDOW, rc, this)) return;   // 실패 시 프레임이 스스로 정리
	f->CenterWindow();
	f->ShowWindow(SW_SHOW);
	f->UpdateWindow();
	UpdateStatus(_T("그래프 : ") + csvPath + _T("  (PNG 도 같은 폴더에 저장)"));
}

void CQROpenCVCheckDlg::OnDestroy()
{
	if (m_worker) {                     // 일괄 처리 중에 창을 닫으면 스레드를 멈추고 기다림
		m_stop = true;
		WaitForSingleObject(m_worker->m_hThread, 10000);
		delete m_worker;
		m_worker = nullptr;
	}
	CDialogEx::OnDestroy();
}
