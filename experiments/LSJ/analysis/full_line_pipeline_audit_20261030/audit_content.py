"""Manually reviewed findings. This module contains prose, not measurement algorithms."""
SOURCES = {
 'S01': ('Gaussian 원논문, III.A–C', 'https://annals-csis.org/proceedings/2014/pliks/136.pdf', '이번 점검에서 원문 III.A–C 대조. Gaussian/Nelder–Mead/초기값은 근거 있음. 전체 처리의 동등성이나 우리 설정의 성능 근거는 아님.'),
 'S02': ('Canny–Devernay IPOL·저자 소스', 'https://www.ipol.im/pub/art/2017/216/', '논문 소개와 저자 코드 1.0 대조. 전문 PDF는 도구 접근 시간 초과로 미열람. 로컬 공식 ZIP과 실제 c/h가 바이트 단위로 같음.'),
 'S03': ('OpenCV 특징 검출 공식 문서', 'https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html', 'Canny/Hough/cornerSubPix/goodFeaturesToTrack API의 정의 확인. 온라인 주소는 3.4.20으로 이동하므로 실행 버전 3.4.15의 수치적 동일성을 입증하는 자료로 쓰지 않음.'),
 'S04': ('LSD 1.6 원논문·저자 구현', 'https://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf', '원문의 Gaussian subsampling, 선분과 지지 영역, 영상 크기 영향 및 로컬 소스 scale 분기 확인. 공식 ZIP과 실제 c/h 일치.'),
 'S05': ('Zernike ramp edge 원논문', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8321028/', '식 1, 15, 18–25, 31, 33, 40–42와 구현 대조. 7×7 마스크 자체는 허용되는 형태. 흐림 폭 가정과 수치적분 정확성은 별도 문제.'),
 'S06': ('PAE 구현자 Method.md', 'https://github.com/juanse77/EdgeLocator/blob/master/Method.md', '기본 3×5, 2차 열 합 공식, 모서리 A/B 추정 설명 대조. 2013 원논문 전문은 이번에도 직접 확인하지 못함. 원문 재현 인증을 하지 않음.'),
 'S07': ('EDLines 고정 버전 저자 코드', 'https://raw.githubusercontent.com/CihanTopal/ED_Lib/69b8d081bd6d28192d816ec0ed02aff9186d73c1/EDLines.cpp', 'EDLines 생성자 기본 ED 36/8, 분할·병합·검증 흐름, 병합 시 지지 구간 처리 대조. 논문 전문의 모든 절을 검토한 것은 아님.'),
 'S08': ('OpenCV contrib FLD 원본', 'https://raw.githubusercontent.com/opencv/opencv_contrib/3.4/modules/ximgproc/src/fast_line_detector.cpp', '공식 소스와 보관 원본/이식본의 실제 diff 확인. 온라인 3.4는 이동 브랜치이므로 재현 기준은 보관 다운로드와 해시. 출력용 코드 추가와 계산 변경을 구분함.'),
 'S09': ('ELSED 고정 버전 파라미터', 'https://raw.githubusercontent.com/iago-suarez/ELSED/1878213b2f5f06a9261d8b1838f53d48e5fd128d/src/ELSED.h', '저자 기본값과 실제 호출 값 대조. 원문 전체의 증명 재검토가 아니라 고정 소스와 추가 코드의 정적 대조.'),
 'S10': ('OpenCV 윤곽·직선 피팅 공식 문서', 'https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html', 'findContours/approxPolyDP/fitLine의 정의 확인. 3.4.20 문서로 이동. 각 API의 존재는 우리 조합 규칙이 검증됐다는 뜻이 아님.'),
 'S11': ('표준화된 Huber 가중의 공식 예', 'https://www.mathworks.com/help/stats/robustfit.html', '1.345와 scale을 함께 쓰는 설명 확인. MATLAB 수직잔차 회귀를 우리 직교 피팅과 같은 알고리즘이라고 주장하지 않음. 95%는 정확도나 포함 확률이 아닌 특정 조건의 통계적 효율.'),
}

FINDINGS = []
def finding(id, category, priority, title, methods, current, judgement, effect, proposal, evidence, sources=(), observed='정적 검토. 현재 결과에서의 영향 크기는 측정하지 않음.'):
    FINDINGS.append(dict(id=id, category=category, priority=priority, title=title, methods=methods,
        current=current, judgement=judgement, effect=effect, proposal=proposal,
        evidence=evidence, sources=list(sources), observed=observed))

FITTED = [f'E{i:02}' for i in range(1,10)] + [f'Q{i:02}' for i in range(1,11)]
SEG = ['E04','E07','E08','E09','Q04','Q07','Q08','Q09']
ALL = [f'E{i:02}' for i in range(1,10)] + [f'Q{i:02}' for i in range(1,13)]

finding('F01','설정 근거·검증 부족','우선','Huber 1.345px의 적용 근거',FITTED,
 '픽셀 단위 직교잔차 r에 w=min(1,1.345/|r|)를 적용한다. 잔차 scale 추정은 없다.',
 '1.345라는 상수의 출처는 있지만 이 영상에서 1.345px가 적정하다는 근거는 없다. 고정 scale Huber 자체를 수학적 오류라고 단정할 수는 없다.',
 '방법별 점 산포와 입력 종류가 달라도 같은 px 경계로 가중치가 바뀐다. 표준화된 Huber의 효율 설명을 그대로 붙일 수 없다.',
 '고정 px 모델을 유지할지, 실제 경계점의 scale을 추정할지 먼저 명시하고 비교 근거를 확보한다. 선분 위 인공 표본까지 일괄 MAD 표준화하지 않는다.',
 [('cpp/src/line_fit.cpp','w[i] = r <= p.huber_c'),('configs/methods.json','"huber_c_px"')],('S10','S11'))

finding('F02','설정 근거·검증 부족','우선','2px/1px hard trimming과 2회 재피팅',FITTED,
 '외부 2px, QR 1px 밖의 점을 제외한 뒤 두 번 재피팅한다. 최소 점 수는 50/20이다.',
 '설정 설명은 경계 혼동점 제거용 초기값이다. 이 거리 밖의 점이 잘못 검출된 점이라는 근거나 임계값 비교 기록은 없다.',
 '진짜 경계의 휨·계단·방법 편향도 잘라낼 수 있다. 잔차가 줄었다는 사실만으로 추출이 정확해졌다고 판단할 수 없다.',
 '제외 전후 점·각도·범위를 분리 기록하고, 점을 제외할 이유와 임계값을 검증한 뒤 확정한다. 지금은 더 작은 임계값으로 바꾸지 않는다.',
 [('cpp/src/line_fit.cpp','for (int it = 0; it < p.trim_iterations'),('configs/methods.json','"trim_iterations"')])

finding('F03','설정 근거·검증 부족','우선','수렴 조건과 종료 사유 미기록',FITTED,
 '각도 변화 1e-13rad, 중심 x/y 변화 각각 1e-9px를 모두 만족하면 종료한다. 최대 100회이며 종료 사유를 반환하지 않는다.',
 '현재 수치에 대한 문헌/안정성 근거가 없다. 과거 Python의 cv::fitLine(reps=1e-6, aeps=1e-8)과 다른 기준이다.',
 '100회 제한으로 종료해도 정상 수렴과 구분되지 않는다. 소수 자릿수나 엄격한 수렴값이 측정 정확도를 보장하지 않는다.',
 '각 반복의 종료 사유·횟수·최종 변화량을 저장하도록 수정안을 마련한다. 허용 계산 오차와 실제 출력 안정성에 근거해 종료값을 정한다.',
 [('cpp/src/line_fit.cpp','bool converged ='),('cpp/include/common.hpp','struct FitResult')],observed='기존 결과에 Huber 반복 횟수와 수렴 플래그가 없어 과거 수렴 여부는 확정할 수 없음.')

finding('F04','조건부 코드 위험','우선','마지막 피팅에 사용한 점과 최종 표시 점이 달라질 수 있음',FITTED,
 '마지막 Huber 호출 뒤 전체 점의 inlier 여부를 다시 계산하지만 그 집합으로 다시 피팅하지 않는다. 그 뒤 최소 점 수도 재검사하지 않는다.',
 '최종 flags와 n_selected는 마지막 직선에 가까운 점 목록이다. 마지막 피팅에 실제 사용한 집합과 같다는 보장이 없다.',
 '보고서의 “피팅 사용점” 표현이 엄밀히 맞지 않을 수 있다. 마지막 분류에서 점이 크게 줄어도 f.ok=true가 될 수 있다.',
 '실제 fit 입력 마스크와 최종 진단 마스크를 따로 보존하고 마지막 유효성 검사를 추가한다. 집합 안정화를 요구할지는 별도 설계한다.',
 [('cpp/src/line_fit.cpp','double s_in = 0'),('cpp/src/line_fit.cpp','f.ok = true;'),('tools/make_figures.py','선택 지지점(피팅 사용)')],
 observed='저장된 성공 결과에서 최종 점 개수 50/20 미달은 0건. 과거 마지막 fit 입력 마스크는 저장되지 않아 두 집합의 일치 여부는 확인 불가.')

finding('F05','조건부 코드 위험','후속','유한값·퇴화 직선 방어가 충분하지 않음',ALL,
 '피팅에서 NaN/Inf, 분산이 없는 점 집합, 너무 짧은 기준선에 대한 공통 검사가 없다. 길이 0인 normalize_line은 수평 방향을 반환한다.',
 '현재 고정 입력에서 오류가 관측됐다는 뜻은 아니다. 입력이 잘못되었을 때 정상 직선처럼 반환될 수 있는 방어 누락이다.',
 '이후 모듈을 다른 데이터에 재사용할 때 예외 상황을 성공으로 해석할 위험이 있다.',
 '유한값·서로 다른 점·기하적 길이·행렬 퇴화를 확인하고 실패 원인을 반환한다. 새로운 임계값은 목적과 단위를 명시한다.',
 [('cpp/src/line_fit.cpp','if (n <= 0)'),('cpp/src/line_fit.cpp','double cx = sx / sw'),('cpp/src/methods_qr.cpp','void two_point_result')],
 observed='216개 저장 기록의 성공 각도/기울기/절편 및 저장 fit 좌표에서 비정상 숫자는 발견되지 않음.')

finding('F06','설정 근거·검증 부족','우선','입력 종류가 달라도 동일한 직교 피팅을 선택함',FITTED,
 '고정 열에서 y만 추정한 점, x/y 모두 보정한 점, 정수 경계점, 검출 선분을 샘플링한 점을 같은 직교 Huber에 넘긴다.',
 '공통 인터페이스는 유용하지만 오차의 성격까지 같아지는 것은 아니다. 다른 손실함수보다 적합하다는 비교가 없다.',
 '방법 고유 성능과 공통 후처리 효과가 결합된다. 현재 순위를 원래 논문 방법의 순위로 해석할 수 없다.',
 '각 입력의 측정 정의를 유지하고 기본 추정치와 후처리 추정치를 구분한다. OLS/TLS/Huber 선택은 데이터 정의에 맞춰 근거를 마련한다.',
 [('cpp/src/line_fit.cpp','void finish_with_points'),('cpp/src/methods_segments.cpp','segment_samples: points')])

finding('F07','설정 근거·검증 부족','우선','ROI는 대상 근거가 있으나 경계 선택의 안정성은 미검증',ALL,
 'R01과 기존 지정 영역으로 외부 두 띠·가림 영역·QR 세 영역을 고정한다. 점 방식에는 열별 가장 강한 전이를 주로 사용한다.',
 '영역 선정이 무근거한 것은 아니다. 그러나 가장 강한 전이가 항상 같은 물리적 경계라는 검증, ROI 이동 민감도 검증은 없다.',
 '반사·다중 경계·국소 휨이 있으면 방법별로 다른 부분을 대표할 수 있다. 기준 4장용 고정 범위를 이동 사진 전체의 탐색 범위로 확장할 수도 없다.',
 '현재 ROI와 가림 의미는 보존한다. “동일 물리 경계”의 판단 근거와 검출 실패/혼동 기록을 보강한 뒤 일반화 범위를 정한다.',
 [('configs/targets.json','"selection_basis"'),('cpp/src/methods_points.cpp','if (v > best)')],
 observed='원본 확대·프로파일 그림을 확인. 저장 성공 외부 결과는 모두 좌우 가시 구간의 지지를 가지며, 좌우 별도 선의 평균을 낸 구현은 아님.')

finding('F08','조건부 코드 위험','후속','선분의 band 검사는 중점만 검사함',SEG,
 '선분 양 끝 평균 y가 band 안에 있으면 통과한다. 이후 visible x만 잘라 표본을 만들고 y band로 다시 자르지 않는다.',
 '“최종 선택은 band 안에서만”이라는 설명을 일반적으로 보장하는 코드가 아니다.',
 '다른 입력에서는 끝부분이 band 밖인 선분이 피팅에 들어갈 수 있다.',
 'band가 검출 후보 중점 조건인지 최종 지지 범위인지 명시하고, 후자라면 교차 구간을 정확히 자른다.',
 [('cpp/src/methods_segments.cpp','double my = 0.5'),('cpp/src/methods_segments.cpp','pts.emplace_back(x, s.y1')],
 observed='현재 저장 성공 결과의 최종 fit 점에서 band 이탈은 0건. 현재 값이 이 이유로 틀렸다고 주장하지 않음.')

finding('F09','설정 근거·검증 부족','우선','선분 consensus는 별도 설계된 측정 단계',SEG,
 '각도 3°/5°, 최소 가시 길이 20/10px, 양쪽 밝기차 3/20, 거리 2/1px로 선분을 걸러 길이 합이 가장 큰 가설을 선택한다. 가설 생성은 긴 선분 최대 300개다.',
 'LSD/EDLines/FLD/ELSED 원논문이 이 공통 선택 규칙을 검증해 준 것은 아니다. 최대 길이 합이 정답 경계라는 보장도 없다.',
 '겹치는 검출 선분의 지지가 중복 계산될 수 있고 동점은 탐색 순서에 의존한다. 원 검출기가 같은데도 선택기 때문에 결과가 달라질 수 있다.',
 '검출과 선택을 별도 모듈로 명시하고 후보/선택 이유를 유지한다. 길이 중복·동점·300개 제한의 영향을 별도로 검증한다.',
 [('cpp/src/methods_segments.cpp','auto score_of'),('cpp/src/methods_segments.cpp','if (order.size() > 300)'),('configs/methods.json','"segment_selection"')])

finding('F10','기록·설명 문제','우선','“1px 샘플링 = 정확한 길이 가중”은 부정확',SEG,
 '선분을 호 길이 기준이 아니라 x 좌표 1px 간격으로 샘플링한다. 이를 length weighting과 동일하다고 기록했다.',
 '표본 수는 대략 가로 투영 길이에 비례한다. 실제 선분 길이와 정확히 같지 않고 시작점·끝점 포함 방식과 중복 선분에도 의존한다.',
 '현재처럼 작은 각도에서는 차이가 작을 수 있으나 정확히 동등하다고 설명할 수 없다. 이 점들이 독립된 새 관측값이 되는 것도 아니다.',
 '실제 샘플링 정의를 그대로 기록한다. 길이 가중을 의도했다면 호 길이/명시적 가중 방식 중 선택하여 검증한다.',
 [('cpp/src/methods_segments.cpp','for (double x = pc.xa; x <= pc.xb + 1e-9'),('cpp/src/methods_segments.cpp','segment_samples: points')])

finding('F11','설정 근거·검증 부족','우선','원 지지점이 있어도 최종선은 선분 표본으로 계산',SEG,
 '선분 검출기의 원 지지 화소를 보관하지만 최종선은 선분 위 인공 표본으로 피팅한다. 원 지지 화소는 뒤에서 진단값만 계산한다.',
 '원래 검출 선분을 유지하려는 선택은 가능하다. 다만 최초 작업서는 원 지지점 사용을 우선 검토하도록 했고, 이 선택의 방법별 근거는 부족하다.',
 '인공 표본의 잔차는 경계 검출 오차가 아니다. 원 지지점 재피팅도 원 검출기의 고유 추정치를 바꾸므로 무조건 더 좋은 대안은 아니다.',
 'native 선분 통합과 실제 경계점 재피팅 중 무엇을 비교할지 분리 명명한다. 단일 선분 잔차 null 처리는 유지한다.',
 [('cpp/src/methods_segments.cpp','// Diagnostic: the detector'),('cpp/src/methods_segments.cpp','if (r.has_rms && members.size() == 1)')])

finding('F12','레퍼런스 대비 변형','우선','Gaussian 방식은 원문 전체 재현이 아님',['E01','Q01'],
 '원문의 Sobel gradient magnitude·ISODATA·골격화 대신 부호 있는 세로 Sobel의 열별 최대를 쓰고, 마지막 spline 대신 Huber 직선을 쓴다.',
 'Gaussian 모델, Nelder–Mead, A/μ/σ 초기값의 기본 아이디어는 원문에 있다. 거친 경계·표본 구성·최종 연결은 변형이다.',
 '논문의 성능 결론을 현재 전체 처리에 적용할 수 없다. 수직 탐색은 거의 수평인 현재 대상에 맞춘 제한이다.',
 '“논문 기반 Gaussian 경계점 + 자체 선택/직선 피팅”으로 명시하고 변형의 필요성과 영향을 검증한다.',
 [('cpp/src/methods_points.cpp','MethodResult method_profile_gaussian'),('cpp/src/methods_points.cpp','cv::Sobel(bp.f, gy')],('S01',))

finding('F13','설정 근거·검증 부족','우선','Gaussian 창·최적화·채택 조건의 검증 부족',['E01','Q01'],
 '11개 표본, peak 8/40, μ 이동 제한 2px, 최대 800회, 함수값 spread 상대 1e-10을 사용한다. 초기 simplex 크기 0.2A/1/5.5도 코드에서 정한다.',
 '11표본은 기존 ERF 창과 맞춘 이유다. 현재 Sobel 프로파일에서 최선이라는 근거가 아니다. 최적화 종료가 Gaussian 모델 적합성을 의미하지 않는다.',
 '잘못된 peak나 Gaussian과 다른 음영에도 계산이 수렴할 수 있다. A와 μ 제한만으로 경계의 정확성은 판단할 수 없다.',
 '창 크기·초기화·종료·채택을 따로 평가한다. 프로파일 잔차 및 파라미터 이상 상태를 보존하되 새 통과 임계는 임의로 정하지 않는다.',
 [('cpp/src/methods_points.cpp','std::fabs(v[3] - v[0])'),('cpp/src/methods_points.cpp','std::array<double, 3> st =')],('S01',))

finding('F14','레퍼런스 대비 변형','후속','Devernay 이후의 선택은 자체 구현',['E02','Q02'],
 '원 검출기 뒤에서 극성·|gy|>|gx|·열 구간별 최강점을 적용한다. 검출기 평활은 대칭 경계, 선택용 OpenCV 평활은 BORDER_REPLICATE다.',
 '원 검출기 c/h는 수정되지 않았다. 그러나 선택은 원 논문의 성능 보장 대상이 아니며 “same Gaussian-smoothed patch”라는 주석도 영상 경계에서는 엄밀히 같지 않다.',
 '서로 다른 곡선의 최강점이 같은 최종선에 섞일 수 있다. 경계 처리 차이는 현재 여백 내부에서의 실질 영향이 확인된 오류는 아니다.',
 '곡선 ID·선택 이유를 계속 보존하고 선택용 gradient의 정의를 정확히 기록한다. core와 wrapper를 별도로 검증한다.',
 [('cpp/src/methods_points.cpp','Gradient of the same Gaussian'),('cpp/src/methods_points.cpp','best_in_bin')],('S02',))

finding('F15','설정 근거·검증 부족','우선','Hough의 세밀한 격자가 서브픽셀 정확도를 보장하지 않음',['E03','Q03'],
 'Canny 정수 화소에 Hough rho=1px, theta=0.01°를 쓰고 최다 득표선 주변 1.5/1px 점을 Huber에 넘긴다.',
 '0.01°는 검색 격자이다. 최종 영상 각도의 정확도나 장비 최소 이동량과 같지 않다. 임계 득표 150/40 및 재피팅 거리의 데이터 검증도 없다.',
 '후보선이 고른 좁은 점 집합에 결과가 의존한다. 동일 행의 정수 점은 결과 0°를 만들 수 있다.',
 'Hough 후보 각도와 최종 재피팅 각도를 유지하고 격자·후보 선택·양자화 영향을 분리 검토한다.',
 [('cpp/src/methods_hough.cpp','cv::HoughLines'),('cpp/src/methods_hough.cpp','double dmax')],('S03',))

finding('F16','레퍼런스 대비 변형','우선','LSD scale=1은 기본 전처리를 바꿈',['E04','Q04'],
 '저자 기본 scale=0.8 대신 1.0을 사용한다. 이 분기에서는 Gaussian subsampling이 실행되지 않아 sigma_scale=0.6도 적용되지 않는다.',
 '원본 해상도 유지라는 작업서 이유가 있으므로 무근거 변경으로 단정하면 안 된다. 다만 기본 LSD와 같은 전처리·성능이라고 할 수 없다.',
 '논문이 설명하는 계단 양자화 완화 단계가 빠진다. 현재 실패나 편차의 원인이라고 확정하려면 별도 비교가 필요하다.',
 'scale=1 변형을 정확히 명명하고 sigma_scale 비활성 상태를 기록한다. 해상도 요구 해석을 정한 뒤 기본 설정 비교 여부를 결정한다.',
 [('cpp/src/methods_segments.cpp','LineSegmentDetection(&n'),('configs/methods.json','scale만 0.8')],('S04',))

finding('F17','설정 근거·검증 부족','우선','Zernike의 흐림 폭은 측정값이 아님',['E05','Q05'],
 'blur_sigma_px를 외부 1.5, QR 1.0으로 가정하고 w=2×1.8×sigma/7을 계산한다.',
 '원문 근사식은 폭 w를 알고 있다는 조건을 둔다. 식이 맞아도 입력 폭이 타당하다는 근거가 없으면 결과 위치의 타당성은 별도이다.',
 '경계점의 이동량에 직접 영향을 준다. coarse 검출용 Gaussian sigma=1과 광학 흐림 sigma를 혼동하면 안 된다.',
 '폭을 어떻게 정할지와 적용 가능한 영상 조건을 먼저 명시한다. 지금 값으로 얻은 결과는 가정한 폭에 조건부인 결과로 다룬다.',
 [('cpp/src/methods_points.cpp','const double wbar'),('configs/methods.json','"blur_sigma_px"')],('S05',))

finding('F18','조건부 코드 위험','우선','Zernike 수치 근사와 수식의 유효 조건 검사',['E05','Q05'],
 '7×7 마스크를 화소당 64×64 적분으로 만들고 창 평균을 뺀다. A20/A11p 나눗셈 전에 분모·유한값 검사가 없다.',
 '논문은 화소 내부 적분 마스크를 정의하므로 직접 수치적분 자체가 잘못은 아니다. 64의 충분성, 이산 마스크에서 평균 제거 영향, 근사식의 적용 범위는 미검증이다.',
 '작은 A11p나 퇴화 영상에서는 비정상 결과 경로가 있다. 판별식이 음수가 아니라는 것만으로 모델의 모든 전제가 만족되는 것은 아니다.',
 '분모/유한값·파라미터 범위 방어를 추가하고 마스크·근사식은 원문 조건과 분리해 확인한다. 결과가 좋게 나오도록 sigma를 조정하지 않는다.',
 [('cpp/src/methods_points.cpp','Q = a20 / a11p'),('cpp/src/methods_points.cpp','double mean = cv::mean(win)')],('S05',),
 observed='현재 저장 성공 결과에서 NaN/Inf는 발견되지 않음. 근사 오차의 크기는 이번에 새로 계산하지 않음.')

finding('F19','레퍼런스 대비 변형','우선','PAE는 3×9·A/B 추정이 바뀐 기본형',['E06','Q06'],
 '3×5 기본 설명을 3×9로 일반화하고, 모서리 세 점 평균 대신 맨 위/아래 행 세 점 평균으로 A/B를 구한다. 평활·floating window는 쓰지 않는다.',
 '열 합으로 a,b,c를 구하는 대수는 일관된다. 그러나 창의 두 영역이 일정한 밝기이고 경계가 창을 가로지른다는 영상 가정은 별도로 확인해야 한다. 원논문 전문은 미확인이다.',
 '변형한 A/B와 창 크기 때문에 원문 PAE의 성능을 그대로 주장할 수 없다. 하단처럼 넓은 전이에서는 끝 행이 평탄한 밝기인지도 쟁점이다.',
 '현재 버전을 “PAE 2차 기본형의 3×9 변형”으로 유지 표기한다. 원형과 변형의 비교 근거를 마련한 뒤 채택한다.',
 [('cpp/src/methods_points.cpp','MethodResult method_pae'),('cpp/src/methods_points.cpp','A = (I(x - 1')],('S06',))

finding('F20','레퍼런스 대비 변형','후속','EDLines 임계 변경과 지지 화소의 불완전성',['E07','Q07'],
 'QR은 저자 EDLines 기본 ED 36/8, 외부는 12/2다. getLineSegments 접근자를 추가해 체인 구간을 재구성한다.',
 '낮춘 이유는 약한 하단 경계라는 관찰이다. 적정값이라는 비교는 없다. 저자 병합 코드는 일부 경우 더 긴 원 구간만 남겨 전체 병합 선의 지지를 완전히 담지 못한다.',
 'support_pixels 진단 잔차는 전체 병합 선의 모든 근거 화소 잔차라고 할 수 없다. 최종선은 별도 선분 표본이므로 이것이 곧 최종선 계산 오류는 아니다.',
 '변경 임계와 지지점 범위를 분리 기록한다. 원 지지점 기반 재피팅을 채택할 경우 병합 provenance부터 해결한다.',
 [('cpp/src/methods_segments.cpp','MethodResult method_edlines'),('cpp/src/methods_segments.cpp','collinear joins may not extend')],('S07',))

finding('F21','레퍼런스 대비 변형','후속','FLD 검출 코어와 자체 적용 설정 구분',['E08','Q08'],
 '공식 3.4 소스를 이식하고 지지점 반환을 추가했다. QR Canny 50/50은 기본값, 외부 10/20은 변경값이다. do_merge=false이다.',
 '실제 diff에서 검출 계산식 변경은 확인되지 않았다. 외부 임계 변경과 뒤의 consensus/Huber는 별도 검증 대상이다.',
 'E08 하단 실패는 현재 설정의 전체 파이프라인 결과이지 FLD 자체가 원천적으로 불가능하다는 뜻은 아니다. merge를 켜면 지지점 목록 정렬을 그대로 신뢰할 수 없다.',
 '이식 변경은 보존하고 원본 검출과 wrapper 성능을 구분한다. merge 설정을 확장한다면 지원 범위를 명시적으로 검사한다.',
 [('cpp/src/methods_segments.cpp','MethodResult method_fld'),('cpp/src/methods_segments.cpp','support_list_aligned')],('S08',))

finding('F22','레퍼런스 대비 변형','후속','ELSED 설정과 숨은 기본값',['E09','Q09'],
 'QR은 기본 gradient/anchor 30/8, 외부는 10/4다. 나머지 주요 값은 기본값이다. listJunctionSizes={5,7,9}는 저자 구조체 기본값으로만 존재한다.',
 '실제 변경은 반환 선분별 픽셀 보관과 접근자 추가다. 원 검출식 변경은 확인되지 않았다. 임계 변경의 데이터 근거와 모든 유효 설정의 출력은 부족하다.',
 '현재 해시로 소스를 추적할 수 있지만 methods.json만으로는 유효 설정을 전부 알 수 없다.',
 '저자 기본값을 이유 없이 바꾸지 말고, 변경한 두 임계와 미출력 기본값을 명시한다.',
 [('cpp/src/methods_segments.cpp','upm::ELSEDParams ep;')],('S09',))

finding('F23','설정 근거·검증 부족','후속','Q10 윤곽 선택과 2%·8% 기준',['Q10'],
 'Otsu 이진화 후 ROI 경계에 닿지 않는 가장 큰 밝은 윤곽을 고른다. 둘레의 2%로 다각형 근사, 네 꼭짓점 요구, 상단 양끝 8%를 제외한다.',
 'API는 표준이지만 가장 큰 윤곽=QR 외곽이라는 규칙과 2%·8% 선택은 프로젝트 자체 설정이다.',
 '조명/연결 상태가 바뀌면 다른 윤곽을 선택하거나 꼭짓점이 사라질 수 있다. 정수 윤곽점의 계단이 기울기에 영향을 줄 수 있다.',
 '물리적 외곽 식별과 파라미터 민감도를 검토한다. Otsu 임계 자체는 영상에서 자동 계산되는 값임을 구분한다.',
 [('cpp/src/methods_qr.cpp','if (!border && area > best_area)'),('cpp/src/methods_qr.cpp','double trim =')],('S10',))

finding('F24','기록·설명 문제','우선','Q10의 “항상 약 0.5px 위, 기울기 영향 없음” 설명',['Q10'],
 'diagnostics에 physical boundary lies about 0.5 px above (constant offset, no slope effect)라고 저장한다.',
 '실제 밝기 전이·Otsu 임계·계단 윤곽으로부터 항상 일정한 0.5px 차이가 나온다고 보장할 수 없다. 이것은 좌표 규약만으로 결론낼 수 없다.',
 '정수 윤곽의 위치/기울기 편향을 없다고 오해하게 만든다. 이 문구는 계산에 0.5를 빼는 코드가 아니라 설명 오류다.',
 '“밝은 영역의 외곽 화소 중심이며 실제 경계와의 차이는 별도 검증 대상”으로 설명을 수정한다. 수치를 임의로 0.5px 이동하지 않는다.',
 [('cpp/src/methods_qr.cpp','contour_offset_note')],('S10',),observed='Q10의 저장 진단값에 해당 문구 존재. 기존 Q10 그림에서 계단 윤곽도 확인.')

finding('F25','설정 근거·검증 부족','후속','Q11 코너 식별·정밀화 기준',['Q11'],
 'Shi–Tomasi 후보 300개, quality=.01, minDistance=4, block=5. 사분면 밝기차 40, 가장 위 코너, 폭≥ROI 절반·높이차≤15px·refine 이동≤3px 조건을 쓴다.',
 'cornerSubPix 원리는 근거가 있으나 외곽 코너를 고르는 규칙은 자체 설계다. 100회/1e-5px 종료를 쓴다는 사실만으로 성공 수렴·정확도를 알 수 없다.',
 '두 코너가 잘못 식별되거나 둥근 경계의 서로 다른 위치로 보정되면 직선은 계산돼도 대상 방향과 어긋날 수 있다.',
 '식별과 정밀화를 별도 상태로 관리하고 코너 품질/이동 정보를 유지한다. 두 점의 잔차 null은 그대로 유지한다.',
 [('cpp/src/methods_qr.cpp','MethodResult method_qr_corners'),('cpp/src/methods_qr.cpp','b0.x - a0.x < 0.5')],('S03',))

finding('F26','설정 근거·검증 부족','후속','Q12는 별도 특징이며 파인더 식별 기준이 경험값',['Q12'],
 'Otsu 역이진화·중첩 윤곽·면적비·볼록 4각형·변 길이비로 3개 파인더를 고른다. 코너를 정밀화하고 대각선 교점을 연결한다.',
 '기존 Python을 이식했다는 것은 출처이지 면적 200, 비율 .25~.85/.1~.8, 꼭짓점 근사 .03, 변비 .65의 검증은 아니다. 실제 결과는 다른 feature_class로 이미 구분한다.',
 '바깥 테두리와 내부 파인더 중심선이 같은 물리 방향이라고 자동 가정할 수 없다. refine 후 이동량 검사는 Q11과 달리 없고 대각선 교차 검사가 주 방어다.',
 'Q12를 다른 특징으로 계속 분리하고 파인더 배치·refine 유효성·조건 적용 범위를 검토한다. Q01~Q11과 무조건 통합하지 않는다.',
 [('cpp/src/methods_qr.cpp','MethodResult method_qr_finders'),('cpp/src/main.cpp','std::string q12_definition')],('S03','S10'))

finding('F27','기록·설명 문제','우선','기록된 설정과 실제 적용되는 설정이 다름',ALL,
 'Q10~Q12는 t.roi를 쓰지만 공통 band/x_range/margin/column_step을 검색 제한으로 쓰지 않는다. Q11/12에도 사용하지 않는 common_fit 해시가 붙는다. E01 gradient/init_sigma, Q10 threshold/contour_mode 문자열은 실행 분기를 바꾸지 않는다.',
 '현재 값과 하드코딩된 동작은 대체로 일치하지만 설정 파일만 바꾸면 반영될 것처럼 읽힌다. 보고서의 Q01~Q11 모두 동일 band 검색이라는 설명은 정확하지 않다.',
 '방법 비교에서 실제 탐색 영역 차이가 가려지고, 향후 설정 변경이 조용히 무시될 수 있다.',
 '유효 설정과 설명용 메타데이터를 구분해 저장한다. 지원하지 않는 값을 넣으면 명시적으로 실패하도록 수정안을 마련한다.',
 [('cpp/src/methods_qr.cpp','cv::Mat crop = c.gray(t.roi)'),('cpp/src/main.cpp','json hashed ='),('tools/make_report.py','Q01~Q11 주 선')])

finding('F28','코드 논리 문제','우선','재실행 비교의 identical이 누락 ID를 무시함',ALL,
 'compare_runs는 두 파일의 공통 measurement_id만 비교한다. only_in_a/b는 따로 계산하지만 identical=not diffs이다.',
 '공통 결과가 같으면 일부 기록이 빠져도 identical=true가 된다. 전체 비교와 부분 비교의 성공 조건이 섞였다.',
 '향후 누락된 전체 실행을 완전히 동일하다고 보고할 수 있다. 이 검사는 통계적 재현성이나 측정 정확도 검증도 아니다.',
 '전체 비교는 ID 집합까지 동일해야 하고, 부분 비교는 의도한 부분집합과 비교 개수가 맞아야 한다. 두 모드를 분리한다.',
 [('tools/run_pipeline.py','"identical": not diffs')],
 observed='기존 full/repeat는 실제 216개 ID 집합이 같아 이번 로그의 동일 결론은 뒤집히지 않음. 단독 실행은 의도적으로 5개만 비교한 기록.')

finding('F29','기록·설명 문제','후속','그림 생성 스크립트의 실행 기록 해시가 현재와 다름',ALL,
 'logs/hashes.json과 현재 파일을 대조하면 tools/make_figures.py 하나가 다르다.',
 '측정 C++ 코드·설정·기록된 검출 소스 해시는 일치한다. 그림 코드 변경 자체를 측정값 변조로 볼 수 없다. 다만 어떤 그림 코드 버전이 어느 산출물을 만들었는지 추적이 불완전하다.',
 '측정과 보고서 생성의 provenance를 하나의 시점으로 취급하면 재현 설명이 어긋난다.',
 '측정 실행과 그림/보고서 생성을 별도 버전·해시로 기록한다. 기존 로그는 덮어쓰지 않고 변경 이력을 보충한다.',
 [('tools/run_pipeline.py','"tools": tree_hashes')],observed='artifact_checks.json에 파일별 대조 결과 보관. 본 점검 시작 이후 원본 변경 여부와는 별개인 기존 기록 차이.')

finding('F30','기록·설명 문제','후속','반올림 중간 좌표와 피팅 입력의 정밀도가 다름',ALL,
 '최종선은 C++ 원래 좌표로 계산하지만 저장 fit_points는 4~5자리(정수법은 1자리), HTML 표시점은 2자리로 저장한다.',
 '표시용 반올림은 가능하다. 다만 저장 중간 좌표를 다시 피팅해서 원래 결과를 비트 단위로 복원할 수 있다고 주장하면 안 된다.',
 '후속 공통 피팅 수정 시 원래 검출 결과를 정확히 재사용하려면 현재 중간 파일 정밀도에 한계가 있다.',
 '재계산용 원정밀 좌표와 표시용 좌표를 분리한다. 기존 수치를 보존하고 복원 오차를 숨기지 않는다.',
 [('cpp/src/methods_points.cpp','round_pt(q.x, q.y, 5)'),('cpp/src/methods_segments.cpp','round_pt(q.x, q.y, 4)'),('tools/make_report.py','return [[round(a, 2)')])

finding('F31','설정 근거·검증 부족','우선','현재 선별 순위는 정확도 순위가 아님',ALL,
 '기준 4장 모두 성공한 방법에서 대상별 최대−최소 각도 폭의 최댓값으로 외부 상위 3개를 남겼다. QR은 사용자 지시로 Q02/Q04만 남겼다.',
 '기준 4장의 실제 영상 선 각도 정답이 없으며 서로 같은 자세라고 검증된 반복 실험도 아니다. 이 폭에는 촬영 차이와 방법 효과가 함께 들어갈 수 있다.',
 '작은 폭은 참고 관찰값일 뿐 정확한 선이나 θ 민감도를 보장하지 않는다. 공통 선택/피팅을 바꾸면 순위도 달라질 수 있다.',
 '기존 후보 선택은 기록으로 보존한다. 정확도 합격 또는 보류 방법의 영구 탈락으로 승격하지 말고, 수정 후 필요한 평가 범위를 별도로 결정한다.',
 [('selection/select_existing_results.py','item[\'score_deg\']'),('selection/select_existing_results.py','QR_METHODS =')],
 observed='현재 active 후보는 E01/E06/E02 + Q02/Q04. 사용자가 상세 확인을 요청한 E05는 active 후보에 자동 복귀시키지 않았음.')

finding('F32','설정 근거·검증 부족','우선','기존 시험이 확인한 범위가 좁음',ALL,
 '자체 시험은 각도 규약·한 직선과 이상점·SHA를 확인한다. 재실행 비교는 같은 프로그램이 같은 값을 내는지 확인한다.',
 '점 추출 수식, 후보 선택, 단위 변화에 대한 Huber 동작, 각 방법의 설정 민감도, 실제 경계 정확도까지 검증한 시험은 아니다.',
 '11개 자체 시험 통과와 216개 재실행 일치를 방법 타당성의 근거로 사용할 수 없다.',
 '수정 범위를 정한 후 알려진 기하/광학 조건의 검증과 실제 영상 경계 검토를 단계별로 설계한다. 이번 점검에서는 새 시험이나 통계 검정을 실행하지 않았다.',
 [('cpp/tests/line_selftest.cpp','// Exact points on y'),('tools/run_pipeline.py','def numeric_fields')])

METHODS = {
 '01': dict(name='Gaussian 프로파일', pipeline='세로 Sobel → 열별 최대 → Gaussian μ → Huber·trim',
            kept='Gaussian 모델·Nelder–Mead·기본 초기값은 원문 근거 있음.', decision='핵심 아이디어는 유지 가능. 거친 경계와 표본·피팅의 변형을 검증해야 함.', ids=['F12','F13'],src=['S01']),
 '02': dict(name='Canny–Devernay', pipeline='저자 서브픽셀 검출 → 극성/방향/열별 최강점 → Huber·trim',
            kept='원본 c/h는 공식 보관 ZIP과 동일.',decision='검출 코어를 폐기할 근거 없음. wrapper와 Huber 적용 근거를 먼저 보완.',ids=['F14'],src=['S02']),
 '03': dict(name='Canny + Hough',pipeline='Canny 정수 점 → 제한 Hough → 인접 점 선택 → Huber·trim',
            kept='법선각·원본 좌표 복원 식에 명백한 대수 오류는 발견하지 못함.',decision='검색 해상도와 측정 정확도를 분리하고 정수 화소·선택 임계 영향을 확인.',ids=['F15'],src=['S03']),
 '04': dict(name='LSD',pipeline='저자 LSD(scale=1) → 자체 consensus → 선분 표본 → Huber·trim',
            kept='저자 c/h 변경 없음. 단일 선분 잔차를 정확도처럼 쓰지 않고 null 처리.',decision='원본 해상도 요구를 따른 변형. 기본 LSD 성능과 동일시 금지.',ids=['F16'],src=['S04']),
 '05': dict(name='Zernike ramp edge',pipeline='거친 peak → 7×7 모멘트 → 가정한 폭으로 위치 복원 → Huber·trim',
            kept='정규화 좌표·모멘트 비·위치 복원 식은 원문 대응 확인.',decision='흐림 폭 가정과 수치 마스크/유효조건이 우선 검토 대상.',ids=['F17','F18'],src=['S05']),
 '06': dict(name='PAE 2차 3×9 변형',pipeline='국소 경사 최대 → 3열 합·A/B → 중앙 a → Huber·trim',
            kept='열 합에서 2차 계수를 얻는 대수는 일관. 근본 수식 오류로 판정하지 않음.',decision='원형 재현으로 부르면 안 됨. A/B와 창 가정의 실제 영상 적합성 검증 필요.',ids=['F19'],src=['S06']),
 '07': dict(name='EDLines',pipeline='ED 체인 → 저자 선분 → 자체 consensus → 선분 표본 → Huber·trim',
            kept='주 계산 소스는 보관 공식 원본과 같고 헤더 접근자만 추가.',decision='변경 임계와 병합 지지점의 범위를 구분해 평가.',ids=['F20'],src=['S07']),
 '08': dict(name='FLD',pipeline='Canny 체인 → 저자 선분 → 자체 consensus → 선분 표본 → Huber·trim',
            kept='공식 소스에서 빌드 연결/지지점 출력 부분을 이식. 계산식 변경은 diff에서 확인되지 않음.',decision='현재 E08 실패를 방법 전체 불가능 판정으로 쓰지 않음.',ids=['F21'],src=['S08']),
 '09': dict(name='ELSED',pipeline='저자 앵커·선분 확장 → 자체 consensus → 선분 표본 → Huber·trim',
            kept='주 검출식 변경 없이 반환 선분의 지지점 보관 추가.',decision='저자 기본값과 외부용 변경값을 나눠 기록.',ids=['F22'],src=['S09']),
 '10': dict(name='QR Otsu·윤곽',pipeline='밝은 최대 윤곽 → 네 꼭짓점 → 상단 윤곽 양끝 제외 → Huber·trim',
            kept='Otsu·윤곽·다각형 API 자체의 근거는 있음.',decision='0.5px 고정 오프셋 설명은 정정 대상. 자체 선택 기준은 미검증.',ids=['F23','F24','F27'],src=['S10']),
 '11': dict(name='QR 외곽 코너',pipeline='Shi–Tomasi → 사분면 규칙 → cornerSubPix → 두 코너 연결',
            kept='Huber 미사용. 두 점 잔차 null 처리. 초기/정밀 코너를 보관함.',decision='코너 식별 규칙과 정밀화 유효성 보완. 공통 Huber 수정 대상에 포함하지 않음.',ids=['F25','F27'],src=['S03']),
 '12': dict(name='QR 파인더 중심선',pipeline='중첩 윤곽 → 3개 파인더 → 코너 정밀화 → 대각선 교점 → 위쪽 두 중심 연결',
            kept='Huber 미사용. 다른 feature_class로 구분. 중심의 기하학적 정의는 명확함.',decision='다른 QR 경계와 섞지 않음. 기존 Python 이식 사실과 타당성 검증은 구분.',ids=['F26','F27'],src=['S03','S10']),
}
