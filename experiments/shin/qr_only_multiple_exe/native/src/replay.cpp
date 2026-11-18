#include "qr_pose.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
using Value=cv::Vec<double,5>;
using Grid=std::array<std::array<Value,41>,41>;

class MockStage : public qr::Stage {
public:
    explicit MockStage(qr::Position p):reported(p){}
    qr::Position position() override {return reported;}
    std::array<int,4> limits() override {return {{-2500,2500,-1200,1200}};}
    qr::Position move_absolute(qr::Position target) override {reported=target;return reported;}
private:
    qr::Position reported;
};

Value interpolate(const Grid& g,double y,double t) {
    const double iy=20.+y/.1,it=20.+t/.05;
    if (iy<0. || iy>40. || it<0. || it>40.) throw std::out_of_range("Pose left measured grid");
    const int j=std::min(39,int(iy)), k=std::min(39,int(it));
    const double v=iy-j,w=it-k;
    return (1-v)*((1-w)*g[j][k]+w*g[j][k+1])+v*((1-w)*g[j+1][k]+w*g[j+1][k+1]);
}
}

int main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::runtime_error("qr_pose_replay RIG_SURFACE.csv GEOMETRY.xml.gz");
        std::array<Grid,2> grids{};
        std::array<std::array<std::array<bool,41>,41>,2> seen{};
        std::ifstream stream(argv[1]);if(!stream) throw std::runtime_error("Cannot read replay surface");
        std::string row;
        while(std::getline(stream,row)) {
            std::replace(row.begin(),row.end(),',',' ');
            std::istringstream input(row);int c,j,k;Value v;
            if (!(input>>c>>j>>k>>v[0]>>v[1]>>v[2]>>v[3]>>v[4])) throw std::runtime_error("Invalid replay row");
            if(c<0 || c>1 || j<0 || j>40 || k<0 || k>40 || seen[c][j][k]) throw std::runtime_error("Invalid replay index");
            grids[c][j][k]=v;seen[c][j][k]=true;
        }
        for(const auto& g:seen) for(const auto& row:g) for(const auto present:row)
            if(!present) throw std::runtime_error("Missing replay grid point");
        const auto coeff=qr::matrix(qr::open_model(argv[2]),"coefficients");
        std::cout<<"correction_limit,initial_residual_mm,passed,center_candidate,max_moves,n\n";
        for(int mode:{0,1,3}) for(double bias:{-.1,-.06,-.045,-.025,0.,.025,.045,.06,.1}) {
            int n=0,passed=0,candidate=0,maximum=0;
            for(const auto& g:grids) for(int j=-20;j<=20;++j) for(int k=-20;k<=20;++k) {
                if (j==0&&k==0) continue;
                MockStage stage({j*100,k*50});qr::Settings settings;settings.cancellation_limit=mode;
                auto view=[&](){const auto p=stage.position();return interpolate(g,bias+p[0]/1000.,p[1]/1000.);};
                auto out=qr::run_correction(stage,settings,[&](){return view()[0];},[&](){
                    const auto o=view();const cv::Mat a=cv::Mat(qr::Features(o[1],o[2],o[3],o[4])).t()*coeff;
                    return qr::Pose(a.at<double>(0),a.at<double>(1));
                });
                const double y=bias+out.reported[0]/1000.,t=out.reported[1]/1000.;
                ++n;passed+=std::abs(y)<=.045+1e-12&&std::abs(t)<=.01+1e-12;
                candidate+=out.reason=="qr_center_candidate";maximum=std::max(maximum,out.moves);
            }
            std::cout<<mode<<','<<bias<<','<<passed<<','<<candidate<<','<<maximum<<','<<n<<'\n';
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
