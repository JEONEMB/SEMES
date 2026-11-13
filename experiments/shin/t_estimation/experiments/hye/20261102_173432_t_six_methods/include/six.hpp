#pragma once
#include "observation.hpp"
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
namespace six {
constexpr int dimensions=104;
constexpr double tick_deg=.01;
struct Row { std::string path,cohort; double y=0,t=0,time=0; int fold=0,session=0,status=0,index=0; cv::Mat x; };
struct Frame { qrwithbar::detail::Observation observation; qrwithbar::detail::Warps warps; };
cv::Mat describe(const Frame&,const Frame&);
void save_frame(const Frame&,const std::filesystem::path&);
Frame load_frame(const std::filesystem::path&);
std::vector<Row> read_rows(const std::filesystem::path&);
struct Config { int method=0,representation=0,order=1; double alpha=.01,shrink=.1,gamma=.2; };
struct Prediction { double t=0,cost=0; cv::Vec2d latent{0,0}; bool known_session=false; int low_quality_line_samples=0; };
struct Model {
 Config config; cv::Mat mean,scale,coef,precision,centers,pca,weights;
 double intercept=0,min_y=-2,max_y=2,min_t=-1,max_t=2;
 std::vector<std::shared_ptr<Model>> members;
 std::vector<double> session_start,session_end,session_offset;
};
std::vector<Config> candidates(int method);
Model fit(const std::vector<Row>&,const std::vector<int>&,Config);
Prediction predict(const Model&,const cv::Mat&,double time=0);
cv::Mat forward(const Model&,cv::Vec2d);
void save_model(const Model&,const std::filesystem::path&);
Model load_model(const std::filesystem::path&);
Prediction direct_refine(const Model&,const Row&,const Frame&,const Frame&,const cv::Mat&,const cv::Mat&);
}
