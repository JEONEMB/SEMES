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
#include <string>
#include <vector>
#include "original_nelder_mead.hpp"
using json = nlohmann::json;
namespace fs = std::filesystem;

json read_json(const fs::path& p) {
    std::ifstream f(p); if (!f) throw std::runtime_error("Cannot read JSON");
    json j; f >> j; return j;
}
cv::Mat read_gray(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read image");
    std::vector<uchar> b((std::istreambuf_iterator<char>(f)), {});
    return cv::imdecode(b, cv::IMREAD_GRAYSCALE);
}
bool excluded_x(int x, const json& intervals) {
    for (const auto& v : intervals) if (x >= v[0].get<int>() && x <= v[1].get<int>()) return true;
    return false;
}

int main() {
    try {
        const fs::path base = fs::path("..") / "reference_line_methods_108_20261030";
        const json manifest = read_json(base / "configs/inputs_manifest.json");
        const json targets = read_json(base / "configs/targets.json");
        const json config = read_json(base / "configs/methods.json");
        const auto params = config["methods"]["E01/Q01"]["external"];
        const int column_step = config["common"]["column_step_px"]["external"];
        const int max_iter = params["nm_max_iter"];
        const double tol = params["nm_tol"], min_peak = params["min_peak"], max_shift = params["max_center_shift_px"];
        std::ofstream out("fits.csv");
        out << std::setprecision(17);
        out << "ref,target,x,coarse_y,peak,n,initial_sigma,A,mu,sigma,sse,iterations,converged,finite,legacy_selected,status\n";
        json profiles = json::array();
        size_t attempted = 0;
        for (const auto& ref : manifest["references"]) {
            const std::string refid = ref["reference_id"];
            cv::Mat gray = read_gray(fs::u8path(ref["path"].get<std::string>()));
            if (gray.cols != 3840 || gray.rows != 2160) throw std::runtime_error("Unexpected dimensions");
            for (const auto& t : targets["targets"]) {
                if (t["id"] != "lower_structure") continue;
                const std::string target = t["id"];
                const int x0=t["x_range"][0], x1=t["x_range"][1], y0=t["band_y"][0], y1=t["band_y"][1], polarity=t["polarity"];
                // A single maximal patch keeps Sobel samples identical for all window sizes.
                const int pad=34, ax=std::max(0,x0-pad), bx=std::min(gray.cols-1,x1+pad);
                const int ay=std::max(0,y0-pad), by=std::min(gray.rows-1,y1+pad);
                cv::Mat f,gy;
                gray(cv::Rect(ax,ay,bx-ax+1,by-ay+1)).convertTo(f,CV_64F);
                cv::Sobel(f,gy,CV_64F,0,1,3,1.0,0.0,cv::BORDER_REFLECT101);
                int eligible=0, total=0;
                for (int x=x0; x<=x1; x+=column_step) {
                    if (excluded_x(x,t["exclude_x"])) continue;
                    ++total;
                    int cy=-1; double peak=-1e300;
                    for (int y=y0; y<=y1; ++y) {
                        double v=polarity*gy.at<double>(y-ay,x-ax);
                        if (v>peak) {peak=v; cy=y;}
                    }
                    std::string coarse_status=peak<min_peak?"weak_peak":(cy==y0||cy==y1)?"peak_at_band_edge":"eligible";
                    std::vector<double> samples;
                    for (int k=-30;k<=30;++k) samples.push_back(polarity*gy.at<double>(cy+k-ay,x-ax));
                    profiles.push_back({{"ref",refid},{"target",target},{"x",x},{"coarse_y",cy},{"peak",peak},
                        {"coarse_status",coarse_status},{"sobel_samples_m30_p30",samples}});
                    if (coarse_status!="eligible") continue;
                    ++eligible;
                    for (int n=11;n<=61;n+=2) {
                        if (n>11 && n<21) continue;
                        const int hw=(n-1)/2;
                        auto sse = [&](const std::array<double,3>& q) {
                            double e=0,ss=std::fabs(q[2])+1e-12;
                            for (int k=-hw;k<=hw;++k) {
                                double d=cy+k-q[1];
                                double residual=q[0]*std::exp(-d*d/(2*ss*ss))-samples[k+30];
                                e+=residual*residual;
                            }
                            return e;
                        };
                        for (int start : {11,5,21}) {
                            std::array<double,3> init={peak,(double)cy,(double)start};
                            std::array<double,3> step={0.2*peak,1.0,0.5*start};
                            int it=0; bool conv=false;
                            auto q=nelder_mead3(sse,init,step,max_iter,tol,&it,&conv);
                            double e=sse(q), sigma=std::fabs(q[2]);
                            bool finite=std::isfinite(q[0])&&std::isfinite(q[1])&&std::isfinite(sigma)&&std::isfinite(e);
                            std::string status=!finite?"nonfinite":!conv?"not_converged":q[0]<=0?"non_positive_A":std::fabs(q[1]-cy)>max_shift?"shift_over_2px":"selected";
                            out<<refid<<','<<target<<','<<x<<','<<cy<<','<<peak<<','<<n<<','<<start<<','<<q[0]<<','<<q[1]<<','<<sigma<<','<<e<<','<<it<<','<<conv<<','<<finite<<','<<(status=="selected")<<','<<status<<'\n';
                            ++attempted;
                        }
                    }
                }
                std::cout<<refid<<" "<<target<<" eligible="<<eligible<<" / "<<total<<std::endl;
                out.flush();
            }
        }
        std::ofstream p("profiles.json"); p<<profiles.dump();
        std::ofstream m("runtime.json");
        m<<json({{"opencv_version",CV_VERSION},{"fit_attempts",attempted},{"columns",profiles.size()}}).dump(2);
        std::cout<<"Finished: "<<attempted<<" attempted fits."<<std::endl;
        return 0;
    } catch (const std::exception& e) {std::cerr<<e.what()<<std::endl; return 1;}
}
