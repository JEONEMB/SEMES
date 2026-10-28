#include "../src/roi/roi_opt.hpp"
int main(){ auto ds=yt::scanDataset("/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad/data/img"); return yt::roi::verify(ds,"out/H5_roi","config/roi_search.yml")?0:2; }
