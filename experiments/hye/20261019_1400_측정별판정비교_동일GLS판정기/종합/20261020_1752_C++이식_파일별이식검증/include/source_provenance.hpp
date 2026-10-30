#pragma once
#include "table.hpp"
#include "workspace.hpp"
namespace semes {
inline std::string original_source_hash(const fs::path& historical_root,const std::string& relative){
    auto path=historical_root/fs::u8path(relative);if(fs::is_regular_file(path))return sha256(path);
    CsvTable inventory(workspace_root()/"config/original_source_inventory.csv");for(size_t i=0;i<inventory.rows.size();++i)if(inventory.text(i,"original")==relative)return inventory.text(i,"sha256");throw std::runtime_error("Original source provenance missing: "+relative);
}
}
