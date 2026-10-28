#include "../src/report/report_media.hpp"
int main(){ auto ds=yt::scanDataset("/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/data/img"); return yt::media::makeAll(ds,"out/H5_media")?0:2; }
