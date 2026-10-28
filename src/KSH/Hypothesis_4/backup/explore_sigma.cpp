// 사전 필터 σ 선택용 탐색 (검증 기준을 정하기 전에 실행 — 결과는 explore_sigma.txt 로 보관)
#include "../src/yt/m2_icgn.hpp"
#include <cstdio>
#include <cmath>
using namespace yt; using namespace yt::m2;
namespace yt { namespace m2 { cv::Mat stretchXpub(const cv::Mat&, double, double); } }
int main(){
  auto ds=scanDataset("/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/data/img");
  cv::Mat ref=loadGray32(findStem(ds,"center")->path);
  for (int q : {0,2}) {
  const Region& QL=regions()[q]; int hw=QL.hw-6, P=64; cv::Rect B(QL.cx-hw-P,QL.cy-hw-P,2*(hw+P),2*(hw+P)); cv::Mat big=ref(B).clone(); int c0=hw+P;
  for (double sg : {0.0,0.5,0.7,1.0,1.4}) for (Interp it : {Interp::Linear, Interp::BSpline}) {
    std::vector<double> e,x; for(int i=0;i<=20;++i){double dx=0.05*i; cv::Mat sh=fourierShift(big,dx,0.25); Affine in; in.u=std::round(dx); Result r=icgn(big,sh,c0,c0,hw,in,it,sg); e.push_back(r.p.u-dx); x.push_back(r.p.ux);}
    Stat a=stat(e), b=stat(x);
    std::printf("%s sigma %.1f %-8s u_rms %.4f u_max %.4f  ux_max %.2e\n", QL.name.c_str(), sg, interpName(it), a.rms, a.maxAbs, b.maxAbs);
  }}
}
