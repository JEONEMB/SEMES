#include "../src/yt/yt_common.hpp"
#include <cstdio>
int main(){ auto ds=yt::scanDataset("/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/data/img"); cv::Mat r=yt::loadGray32(yt::findStem(ds,"center")->path);
 cv::Mat b=r(cv::Rect(1500,960,336,200)).clone(); cv::Mat a=yt::fourierStretchX(b,0,0.3), f=yt::fourierShift(b,0.3,0); cv::Mat d=cv::abs(a-f); double mx; cv::minMaxLoc(d,0,&mx); std::printf("stretch(0,0.3) vs shift(0.3): max diff %.2e\n",mx);
 cv::Mat z=yt::fourierStretchX(b,0,0); cv::minMaxLoc(cv::abs(z-b),0,&mx); std::printf("identity max diff %.2e\n",mx);}
