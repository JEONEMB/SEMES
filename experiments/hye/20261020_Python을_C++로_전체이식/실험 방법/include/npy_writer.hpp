#pragma once
#include "io.hpp"
#include <cstring>
#include <limits>
namespace semes {
inline std::string encode_npy(const Npy& array){
    std::string shape="(";for(size_t i=0;i<array.shape.size();++i){shape+=std::to_string(array.shape[i]);if(i+1<array.shape.size()||array.shape.size()==1)shape+=",";}shape+=")";
    std::string header="{'descr': '"+array.dtype+"', 'fortran_order': False, 'shape': "+shape+", }";header.append((64-(10+header.size()+1)%64)%64,' ');header+='\n';if(header.size()>65535)throw std::runtime_error("NPY header too large");
    std::string result="\x93NUMPY";result+=char(1);result+=char(0);result+=char(header.size()&255);result+=char(header.size()>>8);result+=header;result.append(reinterpret_cast<const char*>(array.data.data()),array.data.size());return result;
}
inline void write_npz(const fs::path& destination,const std::map<std::string,Npy>& arrays){std::map<std::string,std::string> entries;for(const auto& item:arrays)entries[item.first+".npy"]=encode_npy(item.second);write_zip(destination,entries);}
inline Npy numeric_npy(const std::vector<double>& values,const std::vector<size_t>& shape){Npy a;a.dtype="<f8";a.shape=shape;if(a.count()!=values.size())throw std::runtime_error("NPY shape mismatch");a.data.resize(values.size()*sizeof(double));std::memcpy(a.data.data(),values.data(),a.data.size());return a;}
inline Npy boolean_npy(const std::vector<unsigned char>& values,const std::vector<size_t>& shape){Npy a;a.dtype="|b1";a.shape=shape;a.data=values;if(a.count()!=values.size())throw std::runtime_error("NPY boolean shape mismatch");return a;}
}
