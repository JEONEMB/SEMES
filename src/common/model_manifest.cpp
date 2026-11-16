#include "model_manifest.hpp"
#include "sha256.hpp"
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
namespace alignment {
// verify_model_manifest: 필수 파일 목록과 SHA-256 비교
//   입력: folder (배포 모델), error (실패 사유 출력)
//   출력: 모든 필수 파일이 검증되면 true
bool verify_model_manifest(const std::string& folder,std::string& error){
 const auto root=std::filesystem::u8path(folder);std::ifstream list(root/"model_manifest.tsv");
 if(!list){error="model_manifest.tsv missing";return false;}
 std::set<std::string> expected{"center.bmp","reference.json"};
 for(const char* n:{"1234","234","134","124","123","12","13","14","23","24","34","1","2","3","4","none"})expected.insert(std::string("coefficients/screws_")+n+".tsv");
 for(const char* n:{"1234","234","134","124","123","12","13","14","23","24","34"})expected.insert(std::string("checks/screws_")+n+".tsv");
 std::string line;while(std::getline(list,line)){
  if(!line.empty()&&line.back()=='\r')line.pop_back();std::istringstream row(line);std::string name,hash,extra;
  if(!(row>>name>>hash)||row>>extra||!expected.erase(name)||hash.size()!=64){error="Invalid model manifest row";return false;}
  std::ifstream file(root/std::filesystem::u8path(name),std::ios::binary|std::ios::ate);if(!file){error="Missing model file: "+name;return false;}
  const auto size=file.tellg();if(size<0){error="Cannot read model file";return false;}
  std::vector<unsigned char> bytes(static_cast<size_t>(size));file.seekg(0);
  if(!file.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size()))||final00_sha::sha256_hex(bytes)!=hash){error="Model SHA-256 mismatch: "+name;return false;}
 }
 if(!expected.empty()){error="Incomplete model manifest";return false;}return true;
}
}
