// SimulatorApp.h: 영점 맞춤 시뮬레이터 (MFC) 애플리케이션 주 헤더
#pragma once

#ifndef __AFXWIN_H__
	#error "PCH에 대해 이 파일을 포함하기 전에 'pch.h'를 포함합니다."
#endif

#include "resource.h"		// 주 기호입니다.

class CSimulatorApp : public CWinApp
{
public:
	CSimulatorApp();
	virtual BOOL InitInstance();
	DECLARE_MESSAGE_MAP()
};

extern CSimulatorApp theApp;
