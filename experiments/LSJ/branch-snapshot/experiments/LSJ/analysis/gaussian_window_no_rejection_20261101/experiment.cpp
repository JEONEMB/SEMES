#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>
#include <chrono>
#include "original_nelder_mead.hpp"
using json=nlohmann::json;
namespace fs=std::filesystem;
json read_json(const fs::path& p) {std::ifstream f(p); if(!f) throw std::runtime_error("Cannot open JSON");json j; f>>j; return j;}
cv::Mat read_gray(const fs::path& p) {
    std::ifstream f(p,std::ios::binary); if(!f) throw std::runtime_error("Cannot open image");
    std::vector<uchar> b((std::istreambuf_iterator<char>(f)),{});return cv::imdecode(b,cv::IMREAD_GRAYSCALE);
}
bool occluded(int x,const json& ranges) {for(const auto& r:ranges) if(x>=r[0].get<int>() && x<=r[1].get<int>()) return true;return false;}
struct Profile {int x,cy;double peak;std::vector<double> samples;};
struct Fit {double init_A,A,mu,sigma,sse;int iterations;bool converged,finite;};
int main() {
 try {
    const auto start=std::chrono::steady_clock::now();
    const fs::path base=fs::path("..")/"reference_line_methods_108_20261030";
    auto manifest=read_json(base/"configs/inputs_manifest.json");
    auto targets=read_json(base/"configs/targets.json");
    auto run=read_json("run_config.json");
    const int max_n=run["maximum_sample_count"], half=(max_n-1)/2;
    const int max_iter=run["solver_max_iterations"];
    const double tol=run["solver_relative_objective_spread_tolerance"];
    std::vector<int> ns;for(int n=3;n<=max_n;n+=2)ns.push_back(n);
    std::ofstream out("fits.csv");out<<std::setprecision(17);
    out<<"ref,target,x,coarse_y,coarse_response,n,init_A,init_mu,init_sigma,A,mu,sigma,sse,iterations,converged,finite\n";
    std::ofstream profile_out("profiles.jsonl");
    size_t total_profiles=0,total_fits=0;
    json groups=json::array();
    for(const auto& ref:manifest["references"]) {
        std::string rid=ref["reference_id"];
        auto gray=read_gray(fs::u8path(ref["path"].get<std::string>()));
        if(gray.cols!=3840 || gray.rows!=2160)throw std::runtime_error("Unexpected dimensions");
        // Calculate the entire-image Sobel map once. Sampling never changes this map.
        cv::Mat f,gy;gray.convertTo(f,CV_64F);
        cv::Sobel(f,gy,CV_64F,0,1,3,1.,0.,cv::BORDER_REFLECT101);
        for(const auto& t:targets["targets"]) {
            if(t["kind"]!="external")continue;
            std::string target=t["id"];
            const int x0=t["x_range"][0],x1=t["x_range"][1],y0=t["band_y"][0],y1=t["band_y"][1],polarity=t["polarity"];
            if(y0-half<0 || y1+half>=gray.rows)throw std::runtime_error("Requested samples exceed image; no clipping applied");
            std::vector<Profile> profiles;
            for(int x=x0;x<=x1;++x) {
                if(occluded(x,t["exclude_x"]))continue;
                int cy=y0;double peak=-std::numeric_limits<double>::infinity();
                for(int y=y0;y<=y1;++y){double v=polarity*gy.at<double>(y,x);if(v>peak){peak=v;cy=y;}}
                Profile p{x,cy,peak,{}};
                for(int k=-half;k<=half;++k)p.samples.push_back(polarity*gy.at<double>(cy+k,x));
                // No response threshold, band-edge rejection, or fitted-position rejection.
                profile_out<<json({{"ref",rid},{"target",target},{"x",x},{"coarse_y",cy},{"coarse_response",peak},{"half_span",half},{"sobel_samples",p.samples}}).dump()<<'\n';
                profiles.push_back(std::move(p));
            }
            std::vector<Fit> results(profiles.size()*ns.size());
            #pragma omp parallel for schedule(dynamic)
            for(int pi=0;pi<(int)profiles.size();++pi) {
                const auto& p=profiles[pi];
                for(size_t ni=0;ni<ns.size();++ni) {
                    const int n=ns[ni],hw=(n-1)/2;
                    const double a0=*std::max_element(p.samples.begin()+half-hw,p.samples.begin()+half+hw+1);
                    auto sse=[&](const std::array<double,3>& q) {
                        double e=0.,ss=std::fabs(q[2])+1e-12;
                        for(int k=-hw;k<=hw;++k) {
                            double d=p.cy+k-q[1];
                            double r=q[0]*std::exp(-d*d/(2.*ss*ss))-p.samples[k+half];
                            e+=r*r;
                        }
                        return e;
                    };
                    // 2014 predecessor initialization: max response, coarse coordinate, N.
                    std::array<double,3> init={a0,(double)p.cy,(double)n};
                    // Existing implementation's simplex steps; not paper-prescribed values.
                    std::array<double,3> step={.2*a0,1.,.5*n};
                    int iterations=0;bool converged=false;
                    auto q=nelder_mead3(sse,init,step,max_iter,tol,&iterations,&converged);
                    double e=sse(q),sigma=std::fabs(q[2]);
                    bool finite=std::isfinite(q[0])&&std::isfinite(q[1])&&std::isfinite(sigma)&&std::isfinite(e);
                    results[pi*ns.size()+ni]={a0,q[0],q[1],sigma,e,iterations,converged,finite};
                }
            }
            int nonconverged=0,nonfinite=0;
            for(size_t pi=0;pi<profiles.size();++pi) {
                const auto& p=profiles[pi];
                for(size_t ni=0;ni<ns.size();++ni) {
                    const auto& r=results[pi*ns.size()+ni];
                    out<<rid<<','<<target<<','<<p.x<<','<<p.cy<<','<<p.peak<<','<<ns[ni]<<','<<r.init_A<<','<<p.cy<<','<<ns[ni]<<','<<r.A<<','<<r.mu<<','<<r.sigma<<','<<r.sse<<','<<r.iterations<<','<<r.converged<<','<<r.finite<<'\n';
                    nonconverged+=!r.converged;nonfinite+=!r.finite;
                }
            }
            out.flush();profile_out.flush();
            total_profiles+=profiles.size();total_fits+=results.size();
            groups.push_back({{"ref",rid},{"target",target},{"profiles",profiles.size()},{"fits",results.size()},{"nonconverged",nonconverged},{"nonfinite",nonfinite}});
            std::cout<<rid<<" "<<target<<": "<<profiles.size()<<" columns, "<<results.size()<<" fits; nonconverged="<<nonconverged<<", nonfinite="<<nonfinite<<std::endl;
        }
    }
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream meta("runtime.json");meta<<json({{"opencv_version",CV_VERSION},{"columns",total_profiles},{"fits",total_fits},{"sample_counts",ns},{"elapsed_seconds",elapsed},{"groups",groups}}).dump(2);
    std::cout<<"Completed "<<total_fits<<" fits in "<<elapsed<<" seconds."<<std::endl;
    return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
