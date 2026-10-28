#include "../src/yt/m4_gabor.hpp"
int main(){ auto ds=yt::scanDataset("/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/data/img"); return yt::m4::verify(ds,"out/M4_gabor","out/M1_phase_ecc/m1_shifts.csv")?0:2; }
