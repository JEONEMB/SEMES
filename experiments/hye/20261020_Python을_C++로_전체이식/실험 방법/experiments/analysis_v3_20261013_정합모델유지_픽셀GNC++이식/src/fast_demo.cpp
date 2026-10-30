#include "gn_pixel.hpp"
#include <iostream>
using namespace semes;
using namespace pixel_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try {
        if(argc<4)throw std::runtime_error("historical_pixel_fast original-workspace bmp-20261014 new-output [filename...]");
        cv::setNumThreads(4);
        fs::path root=fs::u8path(argv[1]),images=fs::u8path(argv[2]),out=fs::u8path(argv[3]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        FastPixel engine(root,images);
        auto model=engine.geometry.fit_model(0);
        double map_error=engine.check_linear();
        std::cout<<"linear_map_max_error="<<map_error<<std::endl;
        auto target=engine.mean_template(0,4);
        std::vector<std::string> names;
        if(argc==4)names={
            "Y+0.10.bmp","Y+0.50.bmp","T+0.09.bmp","Y-1.00.bmp","T-0.40.bmp"
        };
        else for(int i=4; i<argc; ++i)names.emplace_back(argv[i]);
        fs::create_directories(out);
        std::ofstream predictions(out/"demo.csv");
        predictions<<"file,true_T,true_Y,pred_T,pred_Y,initial_cost,final_cost,iteration,accepted_steps\n";
        for(auto& name:names){
            int index=int(std::find(engine.geometry.data.names.begin(),engine.geometry.data.names.end(),name)-engine.geometry.data.names.begin());
            if(index>=engine.geometry.data.x.rows)throw std::runtime_error("Unknown image");
            auto image=engine.read_image(name);
            cv::Mat initial=cv::Mat::zeros(1,5,CV_64F);
            initial.at<double>(0,0)=engine.geometry.mean_qr_shift[index]/80;
            std::ofstream trace(out/(name+"_steps.csv"));
            trace<<"iteration,lambda,current_cost,candidate_cost,accepted\n";
            auto result=engine.solve(image,model,initial,target,&trace);
            if(!cv::checkRange(result.p)||!std::isfinite(result.cost)||result.cost>result.initial_cost)throw std::runtime_error("Invalid pixel fit");
            predictions<<csv(name)<<','<<std::setprecision(17)<<engine.geometry.data.x.at<double>(index,0)<<','<<engine.geometry.data.x.at<double>(index,1)<<','<<result.p.at<double>(0,0)<<','<<result.p.at<double>(0,1)<<','<<result.initial_cost<<','<<result.cost<<','<<result.iteration<<','<<result.accepted<<'\n';
            std::cout<<name<<" T="<<result.p.at<double>(0,0)<<" Y="<<result.p.at<double>(0,1)<<" cost="<<result.cost<<std::endl;
        }
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["opencv_version"]=CV_VERSION;
        audit["examples"]=int(names.size());
        audit["linear_map_max_error_px"]=map_error;
        audit["mean_template_training_parity"]=0;
        audit["mean_template_stride"]=4;
        audit["retained"]="Original cubic spline mirror prefilter/nearest coefficient sampling, linearized corner map, five pose/nuisance parameters and affine brightness, 60-iteration damped GN. Demo uses every fourth training image and theta-only initialization as original __main__.";
        audit["changed"]="Native C++ spline and OpenCV linear algebra; independent pixels processed in parallel. No Python/SciPy runtime. Full run_g1 uses a separate evaluator with all training images and two starts.";
        write(out/"checks.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
