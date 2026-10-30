#pragma once
#include "io.hpp"
#include <fstream>
namespace semes {
    struct CsvTable {
        std::vector<std::string> header;
        std::map<std::string,size_t> column;
        std::vector<std::vector<std::string>> rows;
        explicit CsvTable(const fs::path& path) {
            std::ifstream input(path);
            if(!input)throw std::runtime_error("Cannot read CSV: "+path.u8string());
            std::string line;
            if(!std::getline(input,line))throw std::runtime_error("Empty CSV");
            header=csvrow(line);
            if(header[0].size()>=3&&(unsigned char)header[0][0]==239)header[0]=header[0].substr(3);
            for(size_t i=0; i<header.size(); ++i)column[header[i]]=i;
            while(std::getline(input,line)) {
                if(line.empty())continue;
                auto r=csvrow(line);
                if(r.size()!=header.size())throw std::runtime_error("CSV column count mismatch: "+path.u8string());
                rows.push_back(std::move(r));
            }
        }
        std::string text(size_t row,const std::string& field)const{
            return rows.at(row).at(column.at(field));
        }
        double number(size_t row,const std::string& field)const{
            return std::stod(text(row,field));
        }
    };
}
