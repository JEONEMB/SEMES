#include "io.hpp"
#include <fstream>
#include <iostream>
using namespace semes;
static std::map<std::string,Json> load(const fs::path& root) {
    auto index=Json::parse(read(root/"index.json"));
    std::map<std::string,Json> result;
    for(auto& g:index.at("groups").array) {
        auto group=Json::parse(read(root/"experiments"/fs::u8path(g.at("experiment").str())/"inventory.json"));
        for(auto& file:group.at("files").array)result[file.at("path").str()]=file;
    }
    return result;
}
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try {
        if(argc!=4)throw std::runtime_error("inventory_diff before-inventory after-inventory new-output");
        auto before=load(fs::u8path(argv[1])),after=load(fs::u8path(argv[2]));
        fs::path out=fs::u8path(argv[3]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        fs::create_directories(out);
        std::ofstream table(out/"changes.csv");
        table<<"path,status,category,before_sha256,after_sha256,before_modified_kst,after_modified_kst\n";
        int added=0,changed=0,removed=0,source_added=0,source_changed=0,data_changed=0,data_compared=0;
        for(auto& kv:after) {
            auto& file=kv.second;
            auto category=file.at("category").str();
            std::string status;
            if(!before.count(kv.first)) {
                status="added";
                ++added;
                source_added+=category=="project_source";
            }
            else {
                auto& old=before.at(kv.first);
                if(category=="data_keep_in_place")++data_compared;
                if(old.at("sha256").str()!=file.at("sha256").str()) {
                    status="content_changed";
                    ++changed;
                    source_changed+=category=="project_source";
                    data_changed+=category=="data_keep_in_place";
                }
                else if(old.at("modified_kst").str()!=file.at("modified_kst").str())status="timestamp_changed_only";
            }
            if(!status.empty())table<<csv(kv.first)<<','<<status<<','<<category<<','<<csv(before.count(kv.first)?before.at(kv.first).at("sha256").str():"")<<','<<csv(file.at("sha256").str())<<','<<csv(before.count(kv.first)?before.at(kv.first).at("modified_kst").str():"")<<','<<csv(file.at("modified_kst").str())<<'\n';
        }
        for(auto& kv:before)if(!after.count(kv.first)) {
            ++removed;
            table<<csv(kv.first)<<",absent_from_later_inventory,"<<kv.second.at("category").str()<<','<<csv(kv.second.at("sha256").str())<<",,"<<csv(kv.second.at("modified_kst").str())<<",\n";
        }
        Json summary=Json::dict();
        summary["added_files"]=added;
        summary["content_changed_files"]=changed;
        summary["absent_files"]=removed;
        summary["added_project_sources"]=source_added;
        summary["changed_project_sources"]=source_changed;
        summary["data_files_compared"]=data_compared;
        summary["changed_data_files"]=data_changed;
        summary["created_kst"]=now();
        summary["interpretation"]="Observed differences between two inventories. These records do not identify who caused a change. Locked files first recorded in the later version are not necessarily newly created.";
        write(out/"summary.json",summary.dump());
        std::cout<<summary.dump()<<'\n';
        return 0;
    }
    catch(const std::exception&e) {
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
