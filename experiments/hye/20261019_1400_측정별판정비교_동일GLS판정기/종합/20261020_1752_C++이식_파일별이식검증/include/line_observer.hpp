#pragma once
#include "numeric.hpp"
namespace semes {
struct LineMeasurement {double shift,rmse,correlation,gain,offset,contrast;bool valid;std::vector<std::string> reasons;};
class LineObserver {
    struct Block{std::string id,line;cv::Rect box;};
    cv::Mat center_;std::vector<Block> blocks_;std::vector<std::vector<double>> reference_;
    std::vector<std::vector<double>> profiles(const cv::Mat&)const;
public:
    LineObserver(const fs::path& center,const fs::path& protocol);
    static LineMeasurement match_profile(const std::vector<double>& reference,const std::vector<double>& current);
    Json extract(const std::vector<unsigned char>& current)const;
};
}
