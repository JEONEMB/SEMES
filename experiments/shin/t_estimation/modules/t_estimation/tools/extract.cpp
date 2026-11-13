#include "t_features.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <set>
#include <regex>
namespace fs=std::filesystem;
int wmain(int argc,wchar_t** argv){try{
 if(argc!=4)throw std::invalid_argument("Usage: t_extract IMAGE_ROOT QR_MODULE OUTPUT_DIRECTORY");cv::setNumThreads(1);
 fs::path root=argv[1],module=argv[2],out=argv[3];fs::create_directories(out);
 std::set<std::string> oldmixed;std::ifstream manifest(module/"tests/fixtures/manifest.csv");std::string line;std::getline(manifest,line);while(std::getline(manifest,line))oldmixed.insert(line.substr(0,line.find(',')));
 auto reference=qrwithbar::detail::read_gray(module/"assets/reference.png");auto rw=qrwithbar::detail::align_qrs(reference,module/"assets");auto ro=qrwithbar::detail::analytic_observe(reference,rw);
 std::vector<fs::path> paths;for(auto& e:fs::recursive_directory_iterator(root))if(e.path().extension()==L".bmp")paths.push_back(e.path());std::sort(paths.begin(),paths.end());
 std::ofstream csv(out/"observations.csv"),audit(out/"audit.csv");csv<<std::setprecision(17)<<"path,cohort,Y,T,status";for(int j=0;j<22;j++)csv<<",f"<<j;csv<<'\n';audit<<"path,cohort,sha256,duplicate_of,status\n";std::map<std::string,std::string> hashes;int done=0,failed=0;
 for(auto& path:paths){auto rel=fs::relative(path,root).generic_u8string();std::string cohort=oldmixed.count(rel)?"mixed_development":rel.rfind("Theta/",0)==0?"theta_development":rel.rfind("Y/",0)==0?"y_development":rel.rfind("Y+",0)==0?"new_evaluation":"reference";
 double y=0,t=0;std::smatch match;std::string stem=path.stem().u8string();std::regex yr("Y([+-][0-9]+\\.[0-9]+)"),tr("T([+-][0-9]+\\.[0-9]+)");if(std::regex_search(stem,match,yr))y=std::stod(match[1]);if(std::regex_search(stem,match,tr))t=std::stod(match[1]);
 auto hash=qrwithbar::detail::sha256(path);std::string duplicate=hashes.count(hash)?hashes[hash]:"";hashes.emplace(hash,rel);t_only::Features features{};int status=0;
 try{auto image=qrwithbar::detail::read_gray(path),before=image.clone();auto w=qrwithbar::detail::align_qrs(image,module/"assets");auto init=qrwithbar::detail::analytic_observe(image,w);if(init.failures)throw std::runtime_error("init");auto o=qrwithbar::detail::interpolated_observe(image,reference,w,rw,init,ro,2);if(o.failures)throw std::runtime_error("registration");features=t_only::enriched_features(o,ro);if(cv::norm(before,image,cv::NORM_INF)!=0)throw std::runtime_error("mutated");}
 catch(const std::exception& e){status=1;failed++;std::cerr<<rel<<": "<<e.what()<<'\n';}
 csv<<rel<<','<<cohort<<','<<y<<','<<t<<','<<status;for(double x:features)csv<<','<<x;csv<<'\n';audit<<rel<<','<<cohort<<','<<hash<<','<<duplicate<<','<<status<<'\n';csv.flush();audit.flush();done++;if(done%10==0)std::cout<<"Extracted "<<done<<'/'<<paths.size()<<" failures="<<failed<<'\n'<<std::flush;
 }std::cout<<"Complete "<<done<<" failures="<<failed<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
