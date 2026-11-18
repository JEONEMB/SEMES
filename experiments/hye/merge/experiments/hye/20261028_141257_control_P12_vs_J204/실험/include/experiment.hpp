#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
namespace study {
struct Sample {
 std::string file,date,axis,session;std::filesystem::path path;
 double y=0,t=0,value=0;int fold=-1;
 cv::Mat feature[2];bool valid[2]={false,false};std::string error[2];double measurement_ms[2]={0,0},decode_ms=0;
};
struct Prediction {double y=0,t=0,cost=0;std::string axis,branch;};
struct Branch {std::string axis,name;cv::Mat origin,slope,curve;};
class Model {
public:
 bool conditional=false;int dimension=0;double shrink=0;
 cv::Mat origin,projector,W;std::vector<Branch> branches;
 Prediction predict(const cv::Mat& feature)const;
 void save(const std::filesystem::path& path)const;
 static Model load(const std::filesystem::path& path);
};
Model train_y(const std::vector<Sample>& data,const std::vector<int>& ids,int feature);
Model train_yort(const std::vector<Sample>& data,const std::vector<int>& ids,int feature);
void require(bool condition,const std::string& message);
}
