#include "alignment.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

// Input preparation and calls only. Motion and recapture belong to the caller.
int wmain(int argc,wchar_t** argv) {
 try {
  if(argc<3){std::cerr<<"Usage: main.exe joint|control IMAGE [JOINT_ASSETS] [CONTROL_MODEL]\n";return 2;}
  std::string mode=std::filesystem::path(argv[1]).u8string();
  if(mode!="joint"&&mode!="control")throw std::invalid_argument("Mode must be joint or control");
  std::string assets=argc>3?std::filesystem::path(argv[3]).u8string():"experiments/00000_final_Y_T_joint/assets";
  std::string model=argc>4?std::filesystem::path(argv[4]).u8string():"experiments/00000_final_Y_T control/assets/control_model.yml";
  initialize_alignment(assets,mode=="control"?model:"");
  std::ifstream file(std::filesystem::path(argv[2]),std::ios::binary);if(!file)throw std::runtime_error("Cannot read image");
  std::vector<unsigned char> raw{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
  cv::Mat image=cv::imdecode(raw,cv::IMREAD_GRAYSCALE);
  auto result=mode=="joint"?joint(image):control(image);auto d=alignment_diagnostics();
  std::cout<<std::setprecision(17)<<"{\"y\":"<<result.y<<",\"t\":"<<result.t<<",\"measurement_valid\":"<<(d.measurement_valid?"true":"false")<<",\"outside_training_range\":"<<(d.outside_training_range?"true":"false")<<"}\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
