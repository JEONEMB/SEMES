#pragma once
#include "fixed_feature_maps.hpp"
#include "table.hpp"
#include "workspace.hpp"
#include "tolerance_metrics.hpp"
namespace semes {
struct GeneralizationData {
    CsvTable manifest;std::map<std::string,Npy> arrays;
    explicit GeneralizationData(const fs::path& reference):manifest(reference/"codex_experiments_20261014/joint_yt_generalization_20261022/measurements/manifest.csv"),arrays(npz(reference/"codex_experiments_20261014/joint_yt_generalization_20261022/measurements/features.npz")){
        CsvTable original(reference/"joint_center_630_20261022/results/manifest.csv");auto frozen=Json::parse(read(reference/"joint_center_630_20261022/results/expanded_features/splits.json"));std::map<std::string,int> fold_by_name;for(size_t f=0;f<frozen.size();++f)for(const auto& name:frozen.at(f).at("test").array)fold_by_name[name.str()]=int(f);
        auto ensure=[&](const std::string& name){if(!manifest.column.count(name)){manifest.column[name]=manifest.header.size();manifest.header.push_back(name);for(auto& row:manifest.rows)row.push_back("");}};for(auto name:{"relative","fold","cv_group"})ensure(name);std::map<std::string,int> group_fold;
        for(size_t i=0;i<manifest.rows.size();++i){int source=int(manifest.number(i,"expanded_cache_index"));auto relative=original.text(source,"relative");int fold=fold_by_name.at(relative);bool mixed=manifest.text(i,"dataset")=="20261020";if(mixed&&manifest.text(i,"scope")=="center")fold=0;std::string group=mixed?"new:"+manifest.text(i,"condition"):"old:"+relative;auto found=group_fold.emplace(group,fold);if(!found.second&&found.first->second!=fold)throw std::runtime_error("Repeated capture group straddles folds");manifest.rows[i][manifest.column.at("relative")]=relative;manifest.rows[i][manifest.column.at("fold")]=std::to_string(fold);manifest.rows[i][manifest.column.at("cv_group")]=group;auto success=manifest.text(i,"success");if(success!="True"&&success!="true"&&success!="1")throw std::runtime_error("Original successful-measurement contract not met");}
        const auto& d=arrays.at("D");if(manifest.rows.size()!=341||d.shape!=std::vector<size_t>{341,630,2})throw std::runtime_error("Expected preserved341x630x2 data");
    }
    Json splits()const{Json result=Json::list();for(int fold=0;fold<4;++fold){Json r=Json::dict();r["fold"]=fold;for(auto key:{"train_indices","test_indices","train","test"})r[key]=Json::list();for(size_t i=0;i<manifest.rows.size();++i){bool test=int(manifest.number(i,"fold"))==fold;r[test?"test_indices":"train_indices"].push(int(i));r[test?"test":"train"].push(manifest.text(i,"relative"));}result.push(r);}return result;}
    std::map<std::string,cv::Mat> features()const{return generalization_feature_bank(arrays.at("D"),arrays.at("points"),arrays.at("regions"));}
    void save(const fs::path& out)const{
        if(fs::exists(out))throw std::runtime_error("New generalization output required");fs::create_directories(out);write(out/"splits.json",splits().dump());std::ofstream csvfile(out/"manifest_with_folds.csv");for(size_t k=0;k<manifest.header.size();++k)csvfile<<(k?",":"")<<csv(manifest.header[k]);csvfile<<'\n';auto raw=workspace_root()/"data/raw/hw/bmp";
        for(size_t i=0;i<manifest.rows.size();++i){auto values=manifest.rows[i];auto file=raw/fs::u8path(manifest.text(i,"relative"));if(!fs::is_regular_file(file))throw std::runtime_error("Copied raw image is missing");values[manifest.column.at("path")]=file.lexically_relative(fs::absolute(out)).generic_u8string();for(size_t k=0;k<values.size();++k)csvfile<<(k?",":"")<<csv(values[k]);csvfile<<'\n';}
    }
};
}
