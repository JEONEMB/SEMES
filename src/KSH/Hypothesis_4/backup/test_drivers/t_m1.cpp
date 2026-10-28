#include "../src/yt/m1_phase_ecc.hpp"
int main(){ auto ds=yt::scanDataset("/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/data/img"); return yt::m1::verify(ds,"out/M1_phase_ecc")?0:2; }
