#include "tolerance_metrics.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{if(argc!=3)throw std::runtime_error("tolerance_metrics truth-and-predictions.json new-output");fs::path out=fs::u8path(argv[2]);if(fs::exists(out))throw std::runtime_error("New output required");auto input=Json::parse(read(fs::u8path(argv[1])));auto result=tolerance_metrics(nullable_pairs(input.at("truth")),nullable_pairs(input.at("prediction")));write(out/"metrics.json",result.dump());std::cout<<result.dump()<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
