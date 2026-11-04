#include "spline.hpp"
#include "io.hpp"
#include <iostream>
using namespace semes;
using namespace pixel_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try{
        if(argc!=2)throw std::runtime_error("historical_spline_checks new-output");
        fs::path out=fs::u8path(argv[1]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        cv::Mat original(39,52,CV_32F);
        for(int y=0; y<original.rows; ++y)for(int x=0; x<original.cols; ++x)original.at<float>(y,x)=float(13+.7*x+1.3*y+std::sin(.3*x));
        auto coefficients=spline_coefficients(original);
        double maximum=0;
        for(int y=1; y+1<original.rows; ++y)for(int x=1; x+1<original.cols; ++x)maximum=std::max(maximum,std::abs(spline_value(coefficients,x,y)-original.at<float>(y,x)));
        double tiny_change=spline_value(coefficients,13.1231,11.567)-spline_value(coefficients,13.123,11.567);
        if(maximum>1e-9||std::abs(tiny_change)<1e-8)throw std::runtime_error("Cubic cardinal reconstruction or subpixel continuity failed");
        Json report=Json::dict();
        report["created_kst"]=now();
        report["max_integer_reconstruction_error"]=maximum;
        report["intensity_change_for_0_0001px"]=tiny_change;
        report["passed"]=true;
        write(out/"checks.json",report.dump());
        std::cout<<report.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
