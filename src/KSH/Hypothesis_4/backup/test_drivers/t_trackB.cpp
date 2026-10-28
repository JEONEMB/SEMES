#include "../src/yt/yt_common.hpp"
namespace yt{namespace trackB{bool verify(const std::vector<Sample>&,const std::string&,const std::string&);}}
int main(){ auto ds=yt::scanDataset("/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/data/img"); return yt::trackB::verify(ds,"out/TrackB","out/M1_phase_ecc/m1_shifts.csv")?0:2; }
