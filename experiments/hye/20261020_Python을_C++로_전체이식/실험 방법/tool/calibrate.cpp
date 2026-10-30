#include "calibration.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{
    if(argc!=3&&argc!=4)throw std::runtime_error("calibrate training.json new-output [--trace]");if(argc==4&&std::string(argv[3])!="--trace")throw std::runtime_error("Unknown calibration option");fs::path out=fs::u8path(argv[2]);if(fs::exists(out))throw std::runtime_error("New output required");auto input=Json::parse(read(fs::u8path(argv[1])));std::vector<std::string> dates;for(const auto& v:input.at("dates").array)dates.push_back(v.str());Json trace;auto cal=Calibration::fit(matrix(input.at("X")),matrix(input.at("commands")),dates,argc==4?&trace:nullptr);write(out/"calibration.json",cal.json().dump());if(argc==4)write(out/"trace.json",trace.dump());std::cout<<cal.diagnostics.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
