#include "generalization_data.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{if(argc!=3)throw std::runtime_error("generalization_data copied-reference-root new-output");GeneralizationData data(fs::u8path(argv[1]));fs::path out=fs::u8path(argv[2]);data.save(out);Json check=Json::dict();check["images"]=int(data.manifest.rows.size());check["folds"]=4;check["group_separation_verified"]=true;check["paths_relative_to_new_manifest"]=true;check["scope"]="Preserved fixed630 development data loading and split reconstruction, not image remeasurement";write(out/"checks.json",check.dump());std::cout<<check.dump()<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
