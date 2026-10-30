#include "table.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide){Utf8Arguments args(argc,wide);auto argv=args.data();try{
    if(argc!=4)throw std::runtime_error("relocate_manifest original.csv copied-raw-root new-manifest.csv");CsvTable table(fs::u8path(argv[1]));auto raw=fs::weakly_canonical(fs::u8path(argv[2]));auto out=fs::absolute(fs::u8path(argv[3]));if(fs::exists(out))throw std::runtime_error("New manifest required");if(!table.column.count("relative")||!table.column.count("path"))throw std::runtime_error("Manifest relative/path columns required");
    std::ostringstream stream;for(size_t k=0;k<table.header.size();++k)stream<<(k?",":"")<<csv(table.header[k]);stream<<'\n';
    for(size_t i=0;i<table.rows.size();++i){auto rel=fs::u8path(table.text(i,"relative"));if(rel.is_absolute()||rel.empty()||*rel.begin()=="..")throw std::runtime_error("Unsafe relative image path");auto file=fs::weakly_canonical(raw/rel);auto within=file.lexically_relative(raw);if(within.empty()||*within.begin()=="..")throw std::runtime_error("Image escapes copied raw root");if(!fs::is_regular_file(file))throw std::runtime_error("Copied image missing");if(table.column.count("sha256")&&sha256(file)!=table.text(i,"sha256"))throw std::runtime_error("Copied image hash differs");auto row=table.rows[i];row[table.column.at("path")]=file.lexically_relative(out.parent_path()).generic_u8string();for(size_t k=0;k<row.size();++k)stream<<(k?",":"")<<csv(row[k]);stream<<'\n';}
    write(out,stream.str());std::cout<<table.rows.size()<<" image paths relocated and verified\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
