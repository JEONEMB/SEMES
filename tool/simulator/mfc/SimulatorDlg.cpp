// =====================================================================================
// SimulatorDlg.cpp — 영점 맞춤 시뮬레이터 대화 상자 구현
// =====================================================================================
#include "pch.h"
#include "framework.h"
#include "SimulatorApp.h"
#include "SimulatorDlg.h"
#include "afxdialogex.h"

#include <algorithm>   // std::min, std::max — Windows 헤더의 min/max 매크로와 겹치지 않게 (std::min)(a, b) 로 부름
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

namespace {

const char* kConfigPath = sim::kConfigRelPath;   // 시뮬레이터 루트 기준 (루트는 실행할 때 자동으로 찾음)

// UTF-8 std::string → CString
CString U(const std::string& s)
{
	if (s.empty()) return CString();
	const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
	CString out;
	MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), out.GetBuffer(n), n);
	out.ReleaseBuffer(n);
	return out;
}

// 마크다운·콘솔 글의 줄바꿈 → 편집 컨트롤 줄바꿈
CString Lines(const std::string& s)
{
	CString c = U(s);
	c.Replace(_T("\r\n"), _T("\n"));
	c.Replace(_T("\n"), _T("\r\n"));
	return c;
}

std::string Stamp()
{
	const std::time_t t = std::time(nullptr);
	std::tm tm{};
	localtime_s(&tm, &t);
	char b[32];
	std::strftime(b, sizeof(b), "%Y%m%d_%H%M%S", &tm);
	return b;
}

CString Num(double v, int prec)
{
	if (!std::isfinite(v)) return _T("-");
	CString s;
	s.Format(_T("%+.*f"), prec, v);
	return s;
}

} // namespace


// 응용 프로그램 정보 대화 상자
class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg() : CDialogEx(IDD_ABOUTBOX) {}
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif
protected:
	DECLARE_MESSAGE_MAP()
};
BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// =====================================================================================
CSimulatorDlg::CSimulatorDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_SIMULATOR_DIALOG, pParent), m_rng((unsigned)std::time(nullptr))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CSimulatorDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_VIEW_CAM, m_viewCam);
	DDX_Control(pDX, IDC_VIEW_PANO, m_viewPano);
	DDX_Control(pDX, IDC_VIEW_PLANE, m_viewPlane);
	DDX_Control(pDX, IDC_VIEW_CONV, m_viewConv);
	DDX_Control(pDX, IDC_INFO, m_info);
	DDX_Control(pDX, IDC_LIST, m_list);
	DDX_Control(pDX, IDC_CMB_EST, m_cmbEst);
	DDX_Control(pDX, IDC_CMB_REF, m_cmbRef);
}

BEGIN_MESSAGE_MAP(CSimulatorDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_SIZE()
	ON_WM_GETMINMAXINFO()
	ON_BN_CLICKED(IDC_BTN_RANDOM, &CSimulatorDlg::OnBnClickedRandom)
	ON_BN_CLICKED(IDC_BTN_RESET, &CSimulatorDlg::OnBnClickedReset)
	ON_BN_CLICKED(IDC_BTN_STEP, &CSimulatorDlg::OnBnClickedStep)
	ON_BN_CLICKED(IDC_BTN_RUN, &CSimulatorDlg::OnBnClickedRun)
	ON_BN_CLICKED(IDC_BTN_STOP, &CSimulatorDlg::OnBnClickedStop)
	ON_BN_CLICKED(IDC_BTN_MONTE, &CSimulatorDlg::OnBnClickedMonte)
	ON_BN_CLICKED(IDC_BTN_VALIDATE, &CSimulatorDlg::OnBnClickedValidate)
	ON_BN_CLICKED(IDC_BTN_BUILD, &CSimulatorDlg::OnBnClickedBuild)
	ON_BN_CLICKED(IDC_BTN_FOLDER, &CSimulatorDlg::OnBnClickedFolder)
	ON_CBN_SELCHANGE(IDC_CMB_EST, &CSimulatorDlg::OnCbnSelchangeEst)
	ON_CBN_SELCHANGE(IDC_CMB_REF, &CSimulatorDlg::OnCbnSelchangeRef)
	ON_MESSAGE(WM_SIM_UPDATE, &CSimulatorDlg::OnSimUpdate)
	ON_MESSAGE(WM_SIM_DONE, &CSimulatorDlg::OnSimDone)
END_MESSAGE_MAP()


// =====================================================================================
// 초기화
// =====================================================================================
BOOL CSimulatorDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);
	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		CString strAboutMenu;
		if (strAboutMenu.LoadString(IDS_ABOUTBOX) && !strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}
	SetIcon(m_hIcon, TRUE);
	SetIcon(m_hIcon, FALSE);

	// ---------------------------------------------------------------- 루트 찾기 (exe 폴더·상위 폴더에서 config/simulator.yml) → 작업 폴더로
	std::filesystem::path root;
	std::string rootMsg;
	const bool rootOk = sim::enterRoot("", root, rootMsg);
	// 설정을 먼저 읽어 글자 크기를 정함 (시뮬레이터 준비는 아래에서)
	{
		sim::SimConfig pre;
		if (rootOk) pre.load(kConfigPath);
		CClientDC dc(this);
		m_uiScale = (std::max)(6, pre.ui_font_pt) / 9.0 * dc.GetDeviceCaps(LOGPIXELSY) / 96.0;
		sim::setPlotFontScale(pre.plot_font_scale);
		m_uiFont.CreatePointFont((std::max)(6, pre.ui_font_pt) * 10, _T("맑은 고딕"));
		m_monoFont.CreatePointFont((std::max)(6, pre.ui_font_pt) * 10, _T("Consolas"));
	}
	ApplyFonts();
	m_list.SetExtendedStyle(LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES | LVS_EX_DOUBLEBUFFER);
	SetListMode(LIST_STEPS);
	m_viewCam.SetEmptyText(_T("카메라 화면\n[1 단계] 또는 [자동 실행] 을 누르세요."));
	m_viewPano.SetEmptyText(_T("파노라마 없음\n[파노라마 생성] 을 먼저 실행하세요 (수 분)."));
	m_viewPlane.SetEmptyText(_T("(Y, Θ) 궤적"));
	m_viewConv.SetEmptyText(_T("수렴 그래프"));

	// 처음 창 크기 : 2560 × 1440 (작업 영역보다 크면 맞춤)
	CRect work;
	SystemParametersInfo(SPI_GETWORKAREA, 0, &work, 0);
	SetWindowPos(nullptr, 0, 0, (std::min)(2560, (int)work.Width()), (std::min)(1440, (int)work.Height()), SWP_NOMOVE | SWP_NOZORDER);
	CenterWindow();
	m_ready = true;
	CRect rc;
	GetClientRect(&rc);
	LayoutControls(rc.Width(), rc.Height());

	// ---------------------------------------------------------------- 시뮬레이터 준비
	std::string msg = rootMsg + "\n";
	m_simOk = rootOk && m_sim.init(kConfigPath, msg);
	FillEstimators();
	FillReferences();
	m_monteN = m_sim.config().monte_carlo_trials;
	CString n;
	n.Format(_T("%d"), m_monteN);
	SetDlgItemText(IDC_EDIT_N, n);
	OnBnClickedRandom();
	if (m_sim.worldReady()) m_viewPano.SetImage(sim::drawPanorama(m_sim.world(), nullptr, m_panoW));
	UpdateInfo(Lines(msg));
	UpdateStatus(m_simOk ? (m_sim.worldReady() ? _T("준비 — [1 단계] 또는 [자동 실행]") : _T("[파노라마 생성] 을 먼저 실행하세요."))
	                     : _T("설정을 읽지 못했습니다. exe 옆(또는 상위 폴더)에 config\\simulator.yml 이 있는지 확인하세요."));
	SetBusy(false);
	return TRUE;
}

void CSimulatorDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX) { CAboutDlg dlg; dlg.DoModal(); }
	else CDialogEx::OnSysCommand(nID, lParam);
}

void CSimulatorDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this);
		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);
		int cxIcon = GetSystemMetrics(SM_CXICON), cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		dc.DrawIcon((rect.Width() - cxIcon + 1) / 2, (rect.Height() - cyIcon + 1) / 2, m_hIcon);
	}
	else CDialogEx::OnPaint();
}

HCURSOR CSimulatorDlg::OnQueryDragIcon() { return static_cast<HCURSOR>(m_hIcon); }

BOOL CSimulatorDlg::PreTranslateMessage(MSG* pMsg)
{
	// Enter / Esc 로 닫히지 않게 (닫기는 X 버튼)
	if (pMsg->message == WM_KEYDOWN && (pMsg->wParam == VK_RETURN || pMsg->wParam == VK_ESCAPE)) return TRUE;
	return CDialogEx::PreTranslateMessage(pMsg);
}

void CSimulatorDlg::OnOK() {}

void CSimulatorDlg::OnCancel()
{
	// X 버튼 : 작업 중이면 멈춘 뒤 닫음 (파노라마 생성은 중간에 멈출 수 없어 끝날 때까지 기다리게 함)
	if (m_worker) {
		if (m_job == JOB_BUILD) { AfxMessageBox(_T("파노라마 생성 중입니다. 끝난 뒤 닫아 주세요.")); return; }
		m_cancel = true;
		WaitForSingleObject(m_worker->m_hThread, INFINITE);
		delete m_worker;
		m_worker = nullptr;
	}
	CDialogEx::OnCancel();
}

// =====================================================================================
// 배치
// =====================================================================================
void CSimulatorDlg::OnSize(UINT nType, int cx, int cy)
{
	CDialogEx::OnSize(nType, cx, cy);
	if (m_ready && nType != SIZE_MINIMIZED) LayoutControls(cx, cy);
}

void CSimulatorDlg::OnGetMinMaxInfo(MINMAXINFO* lpMMI)
{
	// 최소 크기 : 화면 배율만 반영하고 작업 영역을 넘지 않게 (글자 배율까지 곱하면 화면보다 커짐)
	CRect work;
	SystemParametersInfo(SPI_GETWORKAREA, 0, &work, 0);
	HDC hdc = ::GetDC(nullptr);
	const int dpi = (std::max)(96, GetDeviceCaps(hdc, LOGPIXELSY));
	::ReleaseDC(nullptr, hdc);
	lpMMI->ptMinTrackSize.x = (std::min)((int)work.Width(), MulDiv(960, dpi, 96));
	lpMMI->ptMinTrackSize.y = (std::min)((int)work.Height(), MulDiv(600, dpi, 96));
	CDialogEx::OnGetMinMaxInfo(lpMMI);
}

void CSimulatorDlg::LayoutControls(int cx, int cy)
{
	// 높이는 글자 높이에서, 버튼·글 폭은 글자 폭에서 계산 (글자 크기·화면 배율과 무관하게 잘리지 않음)
	CClientDC dc(this);
	CFont* oldFont = dc.SelectObject(&m_uiFont);
	TEXTMETRIC tm{};
	dc.GetTextMetrics(&tm);
	const int charW = (int)tm.tmAveCharWidth, charH = (int)tm.tmHeight;   // LONG → int (std::min/max 에 같은 형식으로 넘기기 위해)
	const int M = 8, H = charH * 17 / 10, STATUS_H = charH * 13 / 10, PAD = charW * 2;
	const auto textW = [&](int id) {
		CString s;
		GetDlgItemText(id, s);
		return (int)dc.GetTextExtent(s).cx;
	};
	// 윗부분 2줄 (컨트롤 ID, 입력 칸·상자면 글자 수). id 0 = 묶음 사이 간격
	//   1줄 = 시험 조작 : 시작 자세 · 단계 실행 · 다회 시험
	//   2줄 = 설정·도구 : 추정기 · 영점 사진 · 합성 검증 · 파노라마 생성 · 결과 폴더
	struct Item { int id; int chars; };
	const Item row1[] = {
		{ IDC_LBL_Y, 0 }, { IDC_EDIT_Y, 7 }, { IDC_LBL_T, 0 }, { IDC_EDIT_T, 7 }, { IDC_BTN_RANDOM, 0 }, { IDC_BTN_RESET, 0 }, { 0, 0 },
		{ IDC_BTN_STEP, 0 }, { IDC_BTN_RUN, 0 }, { IDC_BTN_STOP, 0 }, { 0, 0 },
		{ IDC_BTN_MONTE, 0 }, { IDC_LBL_N, 0 }, { IDC_EDIT_N, 5 } };
	const Item row2[] = {
		{ IDC_LBL_EST, 0 }, { IDC_CMB_EST, 40 }, { IDC_LBL_REF, 0 }, { IDC_CMB_REF, 11 }, { 0, 0 },
		{ IDC_BTN_VALIDATE, 0 }, { IDC_BTN_BUILD, 0 }, { IDC_BTN_FOLDER, 0 } };
	const auto isLabel = [](int id) { return id == IDC_LBL_Y || id == IDC_LBL_T || id == IDC_LBL_N || id == IDC_LBL_EST || id == IDC_LBL_REF; };
	const auto placeRow = [&](const Item* items, int n, int y) {
		int x = M;
		for (int k = 0; k < n; ++k) {
			const Item& t = items[k];
			if (t.id == 0) { x += 3 * PAD; continue; }   // 묶음 사이 간격
			const bool label = isLabel(t.id);
			int w = t.chars > 0 ? charW * (t.chars + 2) : textW(t.id) + (label ? charW : 2 * PAD);
			if (t.id == IDC_CMB_EST) w = (std::max)(charW * 16, (std::min)(w, cx - x - M));   // 창이 좁으면 줄임
			if (CWnd* c = GetDlgItem(t.id)) {
				if (t.id == IDC_CMB_EST || t.id == IDC_CMB_REF) c->MoveWindow(x, y, w, H * 10);   // 콤보 상자 높이 = 펼친 목록 높이
				else c->MoveWindow(x, y + (label ? (H - charH) / 2 : 0), w, label ? charH + 2 : H);
			}
			x += w + (label ? 2 : 6);
		}
	};
	placeRow(row1, int(sizeof(row1) / sizeof(row1[0])), M);
	placeRow(row2, int(sizeof(row2) / sizeof(row2[0])), M + H + M / 2);
	dc.SelectObject(oldFont);
	const int topY = M + 2 * H + M / 2 + M;
	const int listH = (std::max)(S(140), cy / 5);
	const int listY = cy - M - STATUS_H - M - listH;
	const int bodyH = listY - M - topY;
	const int leftW = (cx - 3 * M) * 62 / 100, rightW = cx - 3 * M - leftW;
	const int camH = (std::min)(bodyH * 64 / 100, leftW * 9 / 16);
	m_viewCam.MoveWindow(M, topY, leftW, camH);
	m_viewPano.MoveWindow(M, topY + camH + M, leftW, bodyH - camH - M);
	const int planeH = bodyH * 42 / 100, convH = bodyH * 26 / 100;
	m_viewPlane.MoveWindow(2 * M + leftW, topY, rightW, planeH);
	m_viewConv.MoveWindow(2 * M + leftW, topY + planeH + M, rightW, convH);
	m_info.MoveWindow(2 * M + leftW, topY + planeH + convH + 2 * M, rightW, bodyH - planeH - convH - 2 * M);
	// 그림은 칸 크기 그대로 그림 (작업 스레드가 읽음)
	m_camW = (std::max)(640, leftW); m_panoW = (std::max)(640, leftW);
	m_planeW = (std::max)(560, rightW); m_planeH = (std::max)(320, planeH);   // 너무 작으면 크게 그려 줄여 보임
	m_convW = (std::max)(560, rightW); m_convH = (std::max)(220, convH);
	m_list.MoveWindow(M, listY, cx - 2 * M, listH);
	if (CWnd* w = GetDlgItem(IDC_STATUS)) w->MoveWindow(M, cy - M - STATUS_H, cx - 2 * M, STATUS_H);
	Invalidate(FALSE);
}

// =====================================================================================
// 글꼴 · 추정기 선택
// =====================================================================================
void CSimulatorDlg::ApplyFonts()
{
	// 모든 자식 컨트롤 = 화면 글꼴, 수치 패널 = 고정폭 글꼴 (크기는 display.ui_font_pt)
	for (CWnd* w = GetWindow(GW_CHILD); w != nullptr; w = w->GetNextWindow()) w->SetFont(&m_uiFont, FALSE);
	m_info.SetFont(&m_uiFont, FALSE);   // Θ·µm 글자가 있는 글꼴 (Consolas 는 Θ 가 0 처럼 보임)
}

void CSimulatorDlg::FillEstimators()
{
	m_cmbEst.ResetContent();
	m_estItems.clear();
	m_estItems.push_back("ksh");
	m_cmbEst.AddString(_T("ksh (내장 Alignment)"));
	for (const auto& p : sim::listPlugins()) {
		m_estItems.push_back(p);
		m_cmbEst.AddString(U(p));
	}
	// 지금 쓰는 추정기를 고름 (플러그인은 파일 이름으로 비교)
	const sim::SimConfig& c = m_sim.config();
	int sel = 0;
	if (c.estimator_type == "plugin") {
		const std::string want = sim::u8s(sim::u8p(c.estimator_plugin).stem());
		for (int i = 1; i < (int)m_estItems.size(); ++i)
			if (sim::u8s(sim::u8p(m_estItems[i]).stem()) == want) { sel = i; break; }
	}
	m_cmbEst.SetCurSel(sel);
}

void CSimulatorDlg::FillReferences()
{
	m_cmbRef.ResetContent();
	const std::vector<std::string> v = m_sim.referenceCandidates();
	const std::string cur = m_sim.worldReady() ? m_sim.world().referencePhoto() : m_sim.config().reference_photo;
	int sel = -1;
	for (int i = 0; i < (int)v.size(); ++i) {
		m_cmbRef.AddString(U(v[i]));
		if (v[i] == cur) sel = i;
	}
	m_cmbRef.SetCurSel(sel);
}

void CSimulatorDlg::OnCbnSelchangeRef()
{
	if (m_worker) return;
	const int i = m_cmbRef.GetCurSel();
	if (i < 0) return;
	CString name;
	m_cmbRef.GetLBText(i, name);
	CT2CA utf8(name, CP_UTF8);
	const std::string stem(utf8);
	const std::string cur = m_sim.worldReady() ? m_sim.world().referencePhoto() : m_sim.config().reference_photo;
	if (stem == cur && m_sim.referenceMatchesWorld()) return;
	CString q;
	q.Format(_T("영점 사진을 %s → %s 로 바꿉니다.\n\n참 자세 (0, 0) 이 이 사진이 되도록 가상 장면을 다시 만듭니다 (약 2~4분, 사진 데이터 필요).\n")
	         _T("기존 장면은 world\\_backup\\날짜 폴더에 보관합니다. 진행할까요?"), (LPCTSTR)U(cur), (LPCTSTR)name);
	if (AfxMessageBox(q, MB_YESNO | MB_ICONQUESTION) != IDYES) { FillReferences(); return; }
	std::string msg;
	if (!m_sim.setReferencePhoto(stem, msg)) { AfxMessageBox(U(msg)); FillReferences(); return; }
	UpdateInfo(Lines(msg + "\n"));
	StartJob(JOB_BUILD);
}

void CSimulatorDlg::OnCbnSelchangeEst()
{
	if (m_worker) return;
	const int i = m_cmbEst.GetCurSel();
	if (i < 0 || i >= (int)m_estItems.size()) return;
	std::string msg;
	const bool builtin = m_estItems[i] == "ksh";
	const bool ok = m_sim.setEstimator(builtin ? "ksh" : "plugin", m_estItems[i], msg);
	if (!ok) { AfxMessageBox(U(msg)); FillEstimators(); }
	m_trialTaught = false;   // 새 추정기에 기준 사진을 다시 등록
	m_simOk = m_sim.estimatorReady();
	UpdateInfo(Lines(msg + "\n"));
	UpdateStatus(ok ? _T("추정기 : ") + U(m_sim.estimatorName()) : CString(_T("추정기를 바꾸지 못함 — 이전 추정기 유지")));
	SetBusy(false);
}

// =====================================================================================
// 표시 도구
// =====================================================================================
void CSimulatorDlg::UpdateStatus(const CString& msg) { SetDlgItemText(IDC_STATUS, msg); }

void CSimulatorDlg::SetBusy(bool busy)
{
	const int ids[] = { IDC_BTN_RANDOM, IDC_BTN_RESET, IDC_BTN_STEP, IDC_BTN_RUN, IDC_BTN_MONTE, IDC_BTN_VALIDATE, IDC_BTN_BUILD, IDC_EDIT_Y, IDC_EDIT_T, IDC_EDIT_N };
	const bool world = m_simOk && m_sim.worldReady();
	for (int id : ids) {
		bool en = !busy;
		if (id != IDC_BTN_BUILD && id != IDC_EDIT_Y && id != IDC_EDIT_T && id != IDC_EDIT_N) en = en && world;
		if (id == IDC_BTN_BUILD) en = en && m_simOk;
		if (CWnd* w = GetDlgItem(id)) w->EnableWindow(en);
	}
	if (CWnd* w = GetDlgItem(IDC_BTN_STOP)) w->EnableWindow(busy && m_job != JOB_BUILD);
	m_cmbEst.EnableWindow(!busy && m_simOk);
	m_cmbRef.EnableWindow(!busy && m_simOk);
}

void CSimulatorDlg::SetListMode(ListMode mode)
{
	m_listMode = mode;
	m_list.DeleteAllItems();
	while (m_list.DeleteColumn(0)) {}
	struct Col { LPCTSTR name; int w; int fmt; };
	std::vector<Col> cols;
	if (mode == LIST_STEPS)
		cols = { { _T("단계"), 50, LVCFMT_RIGHT }, { _T("참 Y (mm)"), 95, LVCFMT_RIGHT }, { _T("참 Θ (°)"), 95, LVCFMT_RIGHT },
		         { _T("추정 Y (mm)"), 95, LVCFMT_RIGHT }, { _T("추정 Θ (°)"), 95, LVCFMT_RIGHT },
		         { _T("추정 오차 Y (µm)"), 115, LVCFMT_RIGHT }, { _T("추정 오차 Θ (°)"), 115, LVCFMT_RIGHT },
		         { _T("실제 이동 Y (mm)"), 120, LVCFMT_RIGHT }, { _T("실제 이동 Θ (°)"), 120, LVCFMT_RIGHT },
		         { _T("참값 판정"), 90, LVCFMT_CENTER }, { _T("다음 동작"), 110, LVCFMT_CENTER }, { _T("시간 (ms)"), 80, LVCFMT_RIGHT } };
	else if (mode == LIST_TRIALS)
		cols = { { _T("시험"), 50, LVCFMT_RIGHT }, { _T("시작 Y (mm)"), 100, LVCFMT_RIGHT }, { _T("시작 Θ (°)"), 100, LVCFMT_RIGHT },
		         { _T("이동 횟수"), 80, LVCFMT_RIGHT }, { _T("결과"), 70, LVCFMT_CENTER },
		         { _T("최종 Y 오차 (µm)"), 120, LVCFMT_RIGHT }, { _T("최종 Θ 오차 (°)"), 120, LVCFMT_RIGHT }, { _T("사유"), 320, LVCFMT_LEFT } };
	else
		cols = { { _T("사진"), 190, LVCFMT_LEFT }, { _T("묶음"), 150, LVCFMT_LEFT }, { _T("라벨 Y"), 70, LVCFMT_RIGHT }, { _T("라벨 Θ"), 70, LVCFMT_RIGHT },
		         { _T("실제 사진 추정 Y"), 120, LVCFMT_RIGHT }, { _T("실제 사진 추정 Θ"), 120, LVCFMT_RIGHT },
		         { _T("합성 사진 추정 Y"), 120, LVCFMT_RIGHT }, { _T("합성 사진 추정 Θ"), 120, LVCFMT_RIGHT },
		         { _T("합성−실제 Y (µm)"), 120, LVCFMT_RIGHT }, { _T("합성−실제 Θ (°)"), 120, LVCFMT_RIGHT } };
	for (int i = 0; i < (int)cols.size(); ++i) m_list.InsertColumn(i, cols[i].name, cols[i].fmt, S(cols[i].w));
}

void CSimulatorDlg::FillStepRows(const sim::TrialResult& t)
{
	if (m_listMode != LIST_STEPS) SetListMode(LIST_STEPS);
	m_list.SetRedraw(FALSE);
	m_list.DeleteAllItems();
	for (int i = 0; i < (int)t.steps.size(); ++i) {
		const sim::StepRecord& s = t.steps[i];
		CString k;
		k.Format(_T("%d"), s.k);
		m_list.InsertItem(i, k);
		m_list.SetItemText(i, 1, Num(s.truth.y_mm, 4));
		m_list.SetItemText(i, 2, Num(s.truth.t_deg, 4));
		m_list.SetItemText(i, 3, s.est_ok ? Num(s.est.y_mm, 4) : CString(_T("실패")));
		m_list.SetItemText(i, 4, s.est_ok ? Num(s.est.t_deg, 4) : CString(_T("실패")));
		m_list.SetItemText(i, 5, s.est_ok ? Num((s.est.y_mm - s.truth.y_mm) * 1000.0, 1) : CString(_T("-")));
		m_list.SetItemText(i, 6, s.est_ok ? Num(s.est.t_deg - s.truth.t_deg, 4) : CString(_T("-")));
		const bool moved = s.est_ok && !s.stop && (i + 1 < (int)t.steps.size() || !sim::Simulator::finished(t));
		m_list.SetItemText(i, 7, moved ? Num(s.applied.y_mm, 4) : CString(_T("-")));
		m_list.SetItemText(i, 8, moved ? Num(s.applied.t_deg, 4) : CString(_T("-")));
		m_list.SetItemText(i, 9, s.truth_in_tol ? _T("허용 안") : _T("허용 밖"));
		m_list.SetItemText(i, 10, s.stop ? _T("멈춤") : (moved ? _T("보정 이동") : _T("끝")));
		CString ms;
		ms.Format(_T("%.0f"), s.ms_render + s.ms_align);
		m_list.SetItemText(i, 11, ms);
	}
	if (!t.steps.empty()) m_list.EnsureVisible((int)t.steps.size() - 1, FALSE);
	m_list.SetRedraw(TRUE);
}

void CSimulatorDlg::FillTrialRows(const sim::MonteSummary& m)
{
	if (m_listMode != LIST_TRIALS) SetListMode(LIST_TRIALS);
	m_list.SetRedraw(FALSE);
	m_list.DeleteAllItems();
	for (int i = 0; i < (int)m.trials.size(); ++i) {
		const sim::TrialResult& t = m.trials[i];
		CString a;
		a.Format(_T("%d"), i + 1);
		m_list.InsertItem(i, a);
		m_list.SetItemText(i, 1, Num(t.start.y_mm, 3));
		m_list.SetItemText(i, 2, Num(t.start.t_deg, 3));
		a.Format(_T("%d"), t.moves);
		m_list.SetItemText(i, 3, a);
		m_list.SetItemText(i, 4, t.success ? _T("성공") : _T("실패"));
		m_list.SetItemText(i, 5, Num(t.final_truth.y_mm * 1000.0, 1));
		m_list.SetItemText(i, 6, Num(t.final_truth.t_deg, 4));
		m_list.SetItemText(i, 7, U(t.reason));
	}
	if (!m.trials.empty()) m_list.EnsureVisible((int)m.trials.size() - 1, FALSE);
	m_list.SetRedraw(TRUE);
}

void CSimulatorDlg::FillValidRows(const sim::ValidationSummary& v)
{
	if (m_listMode != LIST_VALID) SetListMode(LIST_VALID);
	m_list.SetRedraw(FALSE);
	m_list.DeleteAllItems();
	for (int i = 0; i < (int)v.rows.size(); ++i) {
		const sim::ValidationRow& r = v.rows[i];
		m_list.InsertItem(i, U(r.stem));
		m_list.SetItemText(i, 1, U(r.group));
		m_list.SetItemText(i, 2, Num(r.label.y_mm, 2));
		m_list.SetItemText(i, 3, Num(r.label.t_deg, 2));
		m_list.SetItemText(i, 4, r.ok_real ? Num(r.est_real.y_mm, 4) : CString(_T("실패")));
		m_list.SetItemText(i, 5, r.ok_real ? Num(r.est_real.t_deg, 4) : CString(_T("실패")));
		m_list.SetItemText(i, 6, r.ok_sim ? Num(r.est_sim.y_mm, 4) : CString(_T("실패")));
		m_list.SetItemText(i, 7, r.ok_sim ? Num(r.est_sim.t_deg, 4) : CString(_T("실패")));
		const bool both = r.ok_real && r.ok_sim;
		m_list.SetItemText(i, 8, both ? Num((r.est_sim.y_mm - r.est_real.y_mm) * 1000.0, 1) : CString(_T("-")));
		m_list.SetItemText(i, 9, both ? Num(r.est_sim.t_deg - r.est_real.t_deg, 4) : CString(_T("-")));
	}
	m_list.SetRedraw(TRUE);
}

void CSimulatorDlg::UpdateInfo(const CString& extra)
{
	const sim::SimConfig& c = m_sim.config();
	CString s;
	s.Format(_T("[추정기] %s\r\n[조건]\r\n허용 범위 : Y ±%.0f µm, Θ ±%.3f°\r\n멈춤 : 추정값이 허용 × %.2f 안\r\n이동 한도 %d회, 한 번 최대 %.1f mm / %.1f°\r\n")
	         _T("로봇 오차 : 반복 정밀도 Y σ %.2f µm, Θ σ %.5f° (실측), 배율 σ %.2f %% (가정)\r\n촬영 잡음 σ %.2f, 기준 사진 = %s\r\n"),
	         (LPCTSTR)U(m_sim.estimatorName()),
	         c.tol_y_um, c.tol_t_deg, c.control_stop_ratio, c.control_max_moves, c.robot_max_step_y_mm, c.robot_max_step_t_deg,
	         c.robot_repeat_y_um, c.robot_repeat_t_deg, c.robot_gain_error * 100.0, c.render_noise_sigma,
	         c.teach_center_photo ? _T("실제 center 사진 + 새 잡음") : _T("자세 (0,0) 합성 사진"));
	if (m_sim.worldReady()) {
		const sim::Pose rp = m_sim.world().referencePose();
		CString z;
		z.Format(_T("영점 사진 : %s — 참 (0,0) = 이 사진 (1차 지령 좌표 Y %+.1f µm, Θ %+.4f°)%s\r\n"), (LPCTSTR)U(m_sim.world().referencePhoto()),
		         rp.y_mm * 1000.0, rp.t_deg, m_sim.referenceMatchesWorld() ? _T("") : _T(" ※ 설정과 다름 → [파노라마 생성]"));
		s += z;
	}
	s += c.zero_calibration == 0 ? _T("영점 맞춤 : 안 함\r\n\r\n")
	   : c.zero_calibration == 1 ? _T("영점 맞춤 : 항상 (시험마다 (0,0) 사진 추정값을 뺌)\r\n\r\n")
	                             : _T("영점 맞춤 : 자동 (기준 사진 교체 미지원 추정기만)\r\n\r\n");
	m_info.SetWindowText(s + extra);
}

// =====================================================================================
// 버튼
// =====================================================================================
bool CSimulatorDlg::ReadStart(sim::Pose& p)
{
	CString y, t;
	GetDlgItemText(IDC_EDIT_Y, y);
	GetDlgItemText(IDC_EDIT_T, t);
	wchar_t* e1 = nullptr; wchar_t* e2 = nullptr;
	p.y_mm = wcstod(y, &e1);
	p.t_deg = wcstod(t, &e2);
	if (e1 == (LPCTSTR)y || e2 == (LPCTSTR)t) { AfxMessageBox(_T("시작 Y, Θ 에 숫자를 넣으세요.")); return false; }
	return true;
}

void CSimulatorDlg::ResetTrial(const sim::Pose& start)
{
	m_trial = sim::TrialResult();
	m_trial.start = start;
	m_trialTaught = false;
	FillStepRows(m_trial);
	m_viewCam.Clear();
	m_viewPlane.Clear();
	m_viewConv.Clear();
	if (m_sim.worldReady()) {
		sim::TrialResult t;
		sim::StepRecord s;
		s.truth = start;
		t.steps.push_back(s);
		m_viewPano.SetImage(sim::drawPanorama(m_sim.world(), &t, m_panoW));
	}
}

void CSimulatorDlg::OnBnClickedRandom()
{
	const sim::Pose p = m_sim.randomStart(m_rng);
	CString y, t;
	y.Format(_T("%.3f"), p.y_mm);
	t.Format(_T("%.3f"), p.t_deg);
	SetDlgItemText(IDC_EDIT_Y, y);
	SetDlgItemText(IDC_EDIT_T, t);
	ResetTrial(p);
	UpdateStatus(_T("새 시작 자세 : ") + U(sim::poseText(p)));
}

void CSimulatorDlg::OnBnClickedReset()
{
	sim::Pose p;
	if (!ReadStart(p)) return;
	ResetTrial(p);
	UpdateStatus(_T("시작 자세 : ") + U(sim::poseText(p)));
}

void CSimulatorDlg::OnBnClickedStep()
{
	if (sim::Simulator::finished(m_trial)) { sim::Pose p; if (!ReadStart(p)) return; ResetTrial(p); }
	StartJob(JOB_STEP);
}

void CSimulatorDlg::OnBnClickedRun()
{
	if (sim::Simulator::finished(m_trial)) { sim::Pose p; if (!ReadStart(p)) return; ResetTrial(p); }
	StartJob(JOB_RUN);
}

void CSimulatorDlg::OnBnClickedStop()
{
	m_cancel = true;
	UpdateStatus(_T("멈추는 중..."));
}

void CSimulatorDlg::OnBnClickedMonte()
{
	CString n;
	GetDlgItemText(IDC_EDIT_N, n);
	m_monteN = _ttoi(n);
	if (m_monteN <= 0) { AfxMessageBox(_T("횟수를 1 이상으로 넣으세요.")); return; }
	{
		std::lock_guard<std::mutex> lk(m_mx);
		m_monteShow = sim::MonteSummary();
	}
	SetListMode(LIST_TRIALS);
	StartJob(JOB_MONTE);
}

void CSimulatorDlg::OnBnClickedValidate()
{
	if (AfxMessageBox(_T("실제 사진 전체(약 340장)를 실제 / 합성 두 번씩 추정합니다.\nRelease 에서 수 분, Debug 에서는 훨씬 오래 걸립니다. 진행할까요?"),
	                  MB_YESNO | MB_ICONQUESTION) != IDYES) return;
	SetListMode(LIST_VALID);
	StartJob(JOB_VALIDATE);
}

void CSimulatorDlg::OnBnClickedBuild()
{
	if (m_sim.worldReady() &&
	    AfxMessageBox(_T("이미 만든 파노라마가 있습니다. 실제 사진으로 다시 만들까요? (수 분)"), MB_YESNO | MB_ICONQUESTION) != IDYES) return;
	StartJob(JOB_BUILD);
}

void CSimulatorDlg::OnBnClickedFolder()
{
	std::error_code ec;
	std::filesystem::create_directories(m_sim.config().result_dir, ec);
	const std::filesystem::path p = std::filesystem::absolute(m_sim.config().result_dir, ec);
	ShellExecute(GetSafeHwnd(), _T("open"), p.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

// =====================================================================================
// 작업 스레드
// =====================================================================================
void CSimulatorDlg::StartJob(Job job)
{
	if (m_worker) return;
	m_job = job;
	m_cancel = false;
	SetBusy(true);
	m_worker = AfxBeginThread(WorkerProc, this, THREAD_PRIORITY_BELOW_NORMAL, 0, CREATE_SUSPENDED);
	if (m_worker == nullptr) { m_job = JOB_NONE; SetBusy(false); AfxMessageBox(_T("작업 스레드를 만들 수 없습니다.")); return; }
	m_worker->m_bAutoDelete = FALSE;   // 끝날 때 핸들을 기다리기 위해 직접 삭제
	m_worker->ResumeThread();
}

UINT CSimulatorDlg::WorkerProc(LPVOID param)
{
	CSimulatorDlg* self = reinterpret_cast<CSimulatorDlg*>(param);
	const Job job = self->m_job;
	self->RunJob(job);
	::PostMessage(self->GetSafeHwnd(), WM_SIM_DONE, (WPARAM)job, 0);
	return 0;
}

// 작업 스레드 : 현재 시험 그림을 만들어 화면 스레드에 넘김
void CSimulatorDlg::PublishTrial(const cv::Mat& img)
{
	const sim::StepRecord* st = m_trial.steps.empty() ? nullptr : &m_trial.steps.back();
	cv::Mat cam = sim::drawCamera(img, st, m_sim.lastInfo(), m_camW);
	cv::Mat pano = sim::drawPanorama(m_sim.world(), &m_trial, m_panoW);
	cv::Mat plane = sim::drawPlane(m_trial, m_sim.config(), cv::Size(m_planeW, m_planeH));
	cv::Mat conv = sim::drawConvergence(m_trial, m_sim.config(), cv::Size(m_convW, m_convH));
	{
		std::lock_guard<std::mutex> lk(m_mx);
		m_visCam = cam; m_visPano = pano; m_visPlane = plane; m_visConv = conv;
		m_trialShow = m_trial;
		m_newTrialRows = true;
	}
	::PostMessage(GetSafeHwnd(), WM_SIM_UPDATE, 0, 0);
}

void CSimulatorDlg::RunJob(Job job)
{
	const sim::SimConfig& c = m_sim.config();
	std::error_code ec;
	std::filesystem::create_directories(c.result_dir, ec);

	if (job == JOB_STEP || job == JOB_RUN) {
		if (!m_trialTaught) {
			m_sim.teach(m_rng);
			if (!m_sim.referenceNote().empty()) { std::lock_guard<std::mutex> lk(m_mx); m_resultText = "[기준 사진] " + m_sim.referenceNote() + "\n"; }
			m_trialTaught = true;
		}
		cv::Mat img;
		for (;;) {
			if (m_cancel) break;
			const bool more = m_sim.stepTrial(m_trial, m_rng, &img);
			PublishTrial(img);
			if (!more || job == JOB_STEP) break;
		}
		if (sim::Simulator::finished(m_trial)) {
			const std::string dir = c.result_dir + "/mfc_trial_" + Stamp();
			std::filesystem::create_directories(dir, ec);
			m_sim.writeTrialCsv(m_trial, dir + "/trial.csv");
			cv::imwrite(dir + "/plane.png", sim::drawPlane(m_trial, c, cv::Size(1200, 600)));
			cv::imwrite(dir + "/convergence.png", sim::drawConvergence(m_trial, c, cv::Size(900, 400)));
			std::string txt = std::string(m_trial.success ? "성공" : "실패") + " : 이동 " + std::to_string(m_trial.moves) + "회\n멈춘 뒤 참 오차 : " +
			                  sim::poseText(m_trial.final_truth) + (m_trial.reason.empty() ? "" : "\n사유 : " + m_trial.reason) + "\n저장 : " + dir + "\n";
			if (m_trial.zero_applied) {
				char z[160];
				std::snprintf(z, sizeof(z), "영점 맞춤 : 영점 차이 Y %+.1f µm, Θ %+.4f° 를 추정값에서 뺌\n", m_trial.zero.y_mm * 1000.0, m_trial.zero.t_deg);
				txt = z + txt;
			}
			std::lock_guard<std::mutex> lk(m_mx);
			m_resultText = txt;
		}
	}
	else if (job == JOB_MONTE) {
		const sim::MonteSummary m = m_sim.monteCarlo(m_monteN, c.seed, [&](int i, const sim::TrialResult& t) {
			{
				std::lock_guard<std::mutex> lk(m_mx);
				m_monteShow.trials.push_back(t);
				m_progressText = "다회 시험 " + std::to_string(i + 1) + " / " + std::to_string(m_monteN);
			}
			::PostMessage(GetSafeHwnd(), WM_SIM_UPDATE, 1, 0);
		}, &m_cancel);
		const std::string base = c.result_dir + "/mfc_monte_" + Stamp();
		m_sim.writeMonteCsv(m, base + ".csv");
		const cv::Mat plot = sim::drawMonte(m, c, cv::Size(1200, 520));
		cv::imwrite(base + ".png", plot);
		const std::string txt = m_sim.monteText(m);
		std::ofstream(base + ".md") << "# 영점 맞춤 다회 시험\n\n```\n" << txt << "```\n";
		std::lock_guard<std::mutex> lk(m_mx);
		m_monteShow = m;
		m_visPlane = sim::drawMonte(m, c, cv::Size(m_planeW, m_planeH));   // 화면용 (파일은 1200 × 520)
		m_resultText = txt + "저장 : " + base + ".csv / .png / .md\n";
	}
	else if (job == JOB_VALIDATE) {
		const sim::ValidationSummary v = m_sim.validate([&](double f, const std::string& s) {
			{
				std::lock_guard<std::mutex> lk(m_mx);
				m_progressText = "합성 검증 " + std::to_string(int(f * 100)) + " % : " + s;
			}
			::PostMessage(GetSafeHwnd(), WM_SIM_UPDATE, 2, 0);
		}, &m_cancel);
		std::ofstream md(c.result_dir + "/validation_mfc.md");
		md << "# 합성 사진 검증\n\n" << v.table;
		std::lock_guard<std::mutex> lk(m_mx);
		m_validShow = v;
		m_resultText = "[합성 검증] 같은 라벨에서 실제 사진 vs 합성 사진의 모델 추정\n" + v.table + "\n저장 : " + c.result_dir + "/validation_mfc.md\n";
	}
	else if (job == JOB_BUILD) {
		std::string msg;
		const bool ok = m_sim.buildWorld([&](double f, const std::string& s) {
			{
				std::lock_guard<std::mutex> lk(m_mx);
				m_progressText = "파노라마 생성 " + std::to_string(int(f * 100)) + " % : " + s;
			}
			::PostMessage(GetSafeHwnd(), WM_SIM_UPDATE, 2, 0);
		}, msg);
		cv::Mat pano = ok ? sim::drawPanorama(m_sim.world(), nullptr, m_panoW) : cv::Mat();
		cv::Mat field = ok ? sim::drawField(m_sim.world(), 1000) : cv::Mat();
		if (ok) {
			cv::imwrite(c.result_dir + "/panorama_view.png", pano);
			cv::imwrite(c.result_dir + "/field_map.png", field);
		}
		std::lock_guard<std::mutex> lk(m_mx);
		m_visPano = pano;
		m_visPlane = field;
		m_resultText = msg + "\n" + (ok ? m_sim.world().report() : std::string());
	}
}

LRESULT CSimulatorDlg::OnSimUpdate(WPARAM wParam, LPARAM /*lParam*/)
{
	std::lock_guard<std::mutex> lk(m_mx);
	if (wParam == 0) {   // 시험 단계
		m_viewCam.SetImage(m_visCam);
		m_viewPano.SetImage(m_visPano);
		m_viewPlane.SetImage(m_visPlane);
		m_viewConv.SetImage(m_visConv);
		if (m_newTrialRows) { FillStepRows(m_trialShow); m_newTrialRows = false; }
		const sim::StepRecord& s = m_trialShow.steps.back();
		CString st;
		st.Format(_T("단계 %d : 참 %s → 추정 %s"), s.k, (LPCTSTR)U(sim::poseText(s.truth)),
		          s.est_ok ? (LPCTSTR)U(sim::poseText(s.est)) : _T("실패"));
		UpdateStatus(st);
	}
	else if (wParam == 1) {   // 다회 시험 한 회
		FillTrialRows(m_monteShow);
		UpdateStatus(U(m_progressText));
	}
	else UpdateStatus(U(m_progressText));
	return 0;
}

LRESULT CSimulatorDlg::OnSimDone(WPARAM wParam, LPARAM /*lParam*/)
{
	if (m_worker) {
		WaitForSingleObject(m_worker->m_hThread, INFINITE);
		delete m_worker;
		m_worker = nullptr;
	}
	const Job job = (Job)wParam;
	m_job = JOB_NONE;
	std::string txt;
	{
		std::lock_guard<std::mutex> lk(m_mx);
		txt = m_resultText;
		m_resultText.clear();
		if (job == JOB_MONTE) { FillTrialRows(m_monteShow); m_viewPlane.SetImage(m_visPlane); }
		if (job == JOB_VALIDATE) FillValidRows(m_validShow);
		if (job == JOB_BUILD) { if (!m_visPano.empty()) m_viewPano.SetImage(m_visPano); if (!m_visPlane.empty()) m_viewPlane.SetImage(m_visPlane); }
	}
	if (!txt.empty()) UpdateInfo(Lines(txt));
	SetBusy(false);
	CString st;
	switch (job) {
	case JOB_STEP: st = sim::Simulator::finished(m_trial) ? _T("시험 끝 — 결과는 오른쪽 패널") : _T("[1 단계] 로 다음 단계, [자동 실행] 으로 끝까지"); break;
	case JOB_RUN: st = m_cancel ? _T("중단함 — [자동 실행] 으로 이어서") : _T("시험 끝 — 결과는 오른쪽 패널"); break;
	case JOB_MONTE: st = _T("다회 시험 끝"); break;
	case JOB_VALIDATE: st = _T("합성 검증 끝"); break;
	case JOB_BUILD: FillReferences(); ResetTrial(m_trial.start);
		st = m_sim.worldReady() ? _T("파노라마 생성 끝 — 이제 시험할 수 있습니다.") : _T("파노라마 생성 실패 — 오른쪽 패널 참고"); break;
	default: break;
	}
	UpdateStatus(st);
	return 0;
}
