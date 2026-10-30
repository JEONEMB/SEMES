#pragma once
#include <windows.h>
#include <filesystem>
#include <stdexcept>
namespace semes {
inline std::filesystem::path workspace_root(){
    wchar_t buffer[32768];auto n=GetModuleFileNameW(nullptr,buffer,32768);if(!n||n==32768)throw std::runtime_error("Cannot locate native executable");
    auto p=std::filesystem::path(std::wstring(buffer,n)).parent_path();
    for(;;){if(std::filesystem::is_regular_file(p/L"config/data.json")&&std::filesystem::is_directory(p/L"models"))return p;auto parent=p.parent_path();if(parent==p||parent.empty())break;p=parent;}
    throw std::runtime_error("Native executable must be inside the self-contained final workspace");
}
}
