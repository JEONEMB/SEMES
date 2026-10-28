#include "../src/yt/m2_icgn.hpp"
int main(){ auto ds=yt::scanDataset("/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/data/img"); return yt::m2::verify(ds,"out/M2_icgn","out/M1_phase_ecc/m1_shifts.csv")?0:2; }
