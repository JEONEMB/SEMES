#include "../src/yt/yt_common.hpp"
namespace yt{namespace trackA{bool verify(const std::vector<Sample>&,const std::string&,const std::string&,const std::string&,const std::string&);}}
int main(){ auto ds=yt::scanDataset("/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/data/img"); return yt::trackA::verify(ds,"out/TrackA","out/M1_phase_ecc/m1_shifts.csv","out/M2_icgn/m2_affine.csv","out/M4_gabor/m4_slopes.csv")?0:2; }
