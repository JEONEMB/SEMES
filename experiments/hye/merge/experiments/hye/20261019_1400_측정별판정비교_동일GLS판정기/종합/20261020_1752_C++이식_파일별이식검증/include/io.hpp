#pragma once
#include "json.hpp"
#include <filesystem>
#include <vector>
#include <cstdint>
namespace semes {
    namespace fs=std::filesystem;
    struct Utf8Arguments {
        std::vector<std::string> values;
        std::vector<char*> pointers;
        Utf8Arguments(int argc,wchar_t** argv) {
            values.reserve(argc);
            pointers.reserve(argc);
            for(int i=0; i<argc; ++i)values.push_back(fs::path(argv[i]).u8string());
            for(auto& value:values)pointers.push_back(value.data());
        }
        char** data(){
            return pointers.data();
        }
    };
    std::vector<unsigned char> bytes(const fs::path& p);
    std::string read(const fs::path& p);
    void write(const fs::path& p,const std::string& s);
    std::string sha256(const fs::path& p);
    std::string sha256(const std::vector<unsigned char>& data);
    std::string csv(const std::string& s);
    std::vector<std::string> csvrow(const std::string& s);
    std::string timestamp(const fs::path& p,bool creation,bool korea);
    std::string now();
    struct Npy {
        std::string dtype;
        std::vector<size_t> shape;
        std::vector<unsigned char> data;
        double number(size_t i)const;
        std::string text(size_t i)const;
        size_t count()const;
    };
    Npy npy(const std::vector<unsigned char>& data);
    std::map<std::string,std::vector<unsigned char>> read_zip(const fs::path& p);
    void write_zip(const fs::path& p,const std::map<std::string,std::string>& entries);
    std::map<std::string,Npy> npz(const fs::path& p);
}
