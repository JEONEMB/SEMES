#include "observation_model.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{
    if(argc!=3)throw std::runtime_error("fit_observation training.json new-output");fs::path out=fs::u8path(argv[2]);if(fs::exists(out))throw std::runtime_error("New output required");auto input=Json::parse(read(fs::u8path(argv[1])));auto X=matrix(input.at("X"));std::vector<std::string> dates;for(const auto& date:input.at("dates").array)dates.push_back(date.str());std::vector<int> indices;if(input.has("indices"))for(const auto& id:input.at("indices").array)indices.push_back(int(id.num()));ObservationModel model(int(input.at("degree").num()),input.at("shrink").num(),input.at("scope").str());model.fit(X,matrix(input.at("commands")),dates,indices);auto result=model.json();if(input.has("condition"))result["condition"]=input.at("condition");result["predictions"]=jsonmatrix(model.predict(input.has("test_X")?matrix(input.at("test_X")):X));write(out/"model.json",result.dump());std::cout<<"Native observation model trained\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
