#include "line_observer.hpp"
#include <chrono>
using namespace semes::numeric;
namespace semes {
LineObserver::LineObserver(const fs::path& center,const fs::path& protocol){
    auto definition=Json::parse(read(protocol));if(sha256(center)!=definition.at("center_sha256").str())throw std::runtime_error("Fixed center hash changed");center_=loadimage(center);
    for(auto& b:definition.at("blocks").array){auto& box=b.at("read_box");if(box.size()!=4)throw std::runtime_error("Line read box must have four coordinates");int x0=int(box.at(0).num()),y0=int(box.at(1).num()),x1=int(box.at(2).num()),y1=int(box.at(3).num());cv::Rect r(x0,y0,x1-x0,y1-y0);if((r&cv::Rect(0,0,center_.cols,center_.rows))!=r||r.height!=49||r.width<=0)throw std::runtime_error("Invalid fixed line profile region");auto line=b.at("line").str();if(line!="upper"&&line!="lower")throw std::runtime_error("Unknown permitted line");blocks_.push_back({b.at("id").str(),line,r});}
    reference_=profiles(center_);
}
std::vector<std::vector<double>> LineObserver::profiles(const cv::Mat& image)const{
    std::vector<std::vector<double>> output;for(auto& b:blocks_){std::vector<double> p(b.box.height);for(int y=0;y<b.box.height;++y){auto row=image.ptr<unsigned char>(b.box.y+y)+b.box.x;for(int x=0;x<b.box.width;++x)p[y]+=row[x];p[y]/=b.box.width;}output.push_back(gaussian_reflect(p));}return output;
}
LineMeasurement LineObserver::match_profile(const std::vector<double>& reference,const std::vector<double>& current){
    if(reference.size()!=49||current.size()!=49)throw std::runtime_error("Line profiles require 49 rows");CubicSplineUniform spline(current,-24);std::vector<double> target(reference.begin()+14,reference.begin()+35);double mean=std::accumulate(target.begin(),target.end(),0.)/21,ss=0;for(double x:target)ss+=(x-mean)*(x-mean);
    struct Evaluation{double rss,gain,offset,correlation,contrast;};
    auto evaluate=[&](double delta){std::vector<double> v(21);for(int i=0;i<21;++i)v[i]=spline(i-10+delta);double vm=std::accumulate(v.begin(),v.end(),0.)/21,vv=0,product=0;for(int i=0;i<21;++i){vv+=(v[i]-vm)*(v[i]-vm);product+=(target[i]-mean)*(v[i]-vm);}if(vv<1e-12){double nan=std::numeric_limits<double>::quiet_NaN();return Evaluation{1e9,nan,nan,nan,nan};}double gain=product/vv,offset=mean-gain*vm,rss=0;for(int i=0;i<21;++i){double r=target[i]-gain*v[i]-offset;rss+=r*r;}std::sort(v.begin(),v.end());return Evaluation{rss/21,gain,offset,product/std::sqrt(std::max(ss*vv,1e-24)),v[18]-v[2]};};
    double initial=-12,best=evaluate(-12).rss;for(int i=-11;i<=12;++i){double cost=evaluate(i).rss;if(cost<best){best=cost;initial=i;}}
    auto fitted=bounded_minimize([&](double d){return evaluate(d).rss;},std::max(-12.,initial-1),std::min(12.,initial+1),1e-7,60);auto e=evaluate(fitted.x);LineMeasurement result{fitted.x,std::sqrt(e.rss),e.correlation,e.gain,e.offset,e.contrast,true,{}};
    if(!fitted.success||!std::isfinite(result.shift)||!std::isfinite(e.rss)||!std::isfinite(e.gain)||!std::isfinite(e.correlation)||!std::isfinite(e.contrast))result.reasons.push_back("nonfinite_or_optimizer");
    if(e.correlation<.90)result.reasons.push_back("correlation_below0.90");if(e.contrast<5)result.reasons.push_back("normal_profile_contrast_below5");if(!(e.gain>=.25&&e.gain<=4))result.reasons.push_back("gain_outside0.25..4");if(std::abs(result.shift)>11.5)result.reasons.push_back("vertical_search_boundary");result.valid=result.reasons.empty();return result;
}
Json LineObserver::extract(const std::vector<unsigned char>& current)const{
    auto start=std::chrono::steady_clock::now();auto image=cv::imdecode(current,cv::IMREAD_GRAYSCALE);if(image.empty()||image.size()!=center_.size())throw std::runtime_error("Expected fixed center image dimensions");auto p=profiles(image);Json result=Json::dict();for(auto key:{"measured_shifts","valid_blocks","reasons","rmse","correlation","gain","offset","contrast","X_upper","X_lower"})result[key]=Json::list();bool upper=true,lower=true;
    for(size_t i=0;i<p.size();++i){auto m=match_profile(reference_[i],p[i]);result["measured_shifts"].push(m.shift);result["valid_blocks"].push(m.valid);Json reasons=Json::list();for(auto& s:m.reasons)reasons.push(s);result["reasons"].push(reasons);result["rmse"].push(m.rmse);result["correlation"].push(m.correlation);result["gain"].push(m.gain);result["offset"].push(m.offset);result["contrast"].push(m.contrast);auto key="X_"+blocks_[i].line;result[key].push(m.valid?Json(m.shift):Json());if(blocks_[i].line=="upper")upper&=m.valid;else lower&=m.valid;}
    result["success_upper"]=upper;result["success_lower"]=lower;result["image_sha256"]=sha256(current);result["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();result["scope"]="Only fixed protocol read boxes of the permitted upper/lower horizontal structures";return result;
}
}
