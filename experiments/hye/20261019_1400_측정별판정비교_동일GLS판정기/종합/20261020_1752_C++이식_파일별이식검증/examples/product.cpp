#include "alignment.hpp"
#include <iostream>
// Product integration uses the same byte-only inference engine as the CLI.
// The calling product supplies one fixed center at initialization and image bytes at inference.
// Filename commands and trial IDs are never passed into Engine::infer.
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try{
        if(argc!=4)throw std::runtime_error("product_example assets fixed-center current-image");
        semes::Engine engine(semes::fs::u8path(argv[1]),semes::fs::u8path(argv[2]));
        auto prediction=engine.infer(semes::bytes(semes::fs::u8path(argv[3])));
        std::cout<<prediction.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
