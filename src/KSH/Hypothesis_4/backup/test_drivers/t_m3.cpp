#include "../src/yt/m3_grid.hpp"
int main(){ auto ds=yt::scanDataset("/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/data/img"); return yt::m3::verify(ds,"out/M3_grid")?0:2; }
