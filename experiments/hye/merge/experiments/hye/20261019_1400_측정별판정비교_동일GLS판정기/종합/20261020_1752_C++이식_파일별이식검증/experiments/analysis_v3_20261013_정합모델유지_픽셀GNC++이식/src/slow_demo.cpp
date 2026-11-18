#include "gn_pixel_slow.hpp"
#include <iostream>
using namespace semes;
using namespace pixel_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try {
        if(argc<4)throw std::runtime_error("historical_pixel_slow original-workspace bmp-20261014 new-output [filename...]");
        fs::path root=fs::u8path(argv[1]),data=fs::u8path(argv[2]),out=fs::u8path(argv[3]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        SlowPixel engine(root,data);
        auto model=engine.fit_model(0);
        std::vector<std::string> tests;
        if(argc==4)tests={
            "Y+0.50.bmp","T+0.09.bmp","Y+0.10.bmp","T-1.00.bmp"
        };
        else for(int i=4; i<argc; ++i)tests.emplace_back(argv[i]);
        fs::create_directories(out);
        std::ofstream predictions(out/"demo.csv");
        predictions<<"file,initial_kind,true_T,true_Y,pred_T,pred_Y,initial_cost,final_cost,iteration,accepted_steps\n";
        int solved=0;
        for(auto& name:tests){
            auto at=std::find(engine.data.names.begin(),engine.data.names.end(),name);
            if(at==engine.data.names.end())throw std::runtime_error("Unknown historical image name");
            int i=int(at-engine.data.names.begin());
            auto image=engine.read_image(name);
            for(int initial_kind=0; initial_kind<2; ++initial_kind){
                cv::Mat p=cv::Mat::zeros(1,2,CV_64F);
                p.at<double>(0,initial_kind)=engine.mean_qr_shift[i]/(initial_kind==0?80:14.25);
                std::ofstream trace(out/(name+"_init"+std::to_string(initial_kind)+"_steps.csv"));
                trace<<"iteration,lambda,current_cost,candidate_cost,accepted\n";
                auto fit=engine.solve(image,model,p,cv::Mat(),&trace);
                if(!cv::checkRange(fit.p)||!std::isfinite(fit.cost)||fit.cost>fit.initial_cost)throw std::runtime_error("Invalid or increasing pixel objective");
                predictions<<csv(name)<<','<<initial_kind<<','<<std::setprecision(17)<<engine.data.x.at<double>(i,0)<<','<<engine.data.x.at<double>(i,1)<<','<<fit.p.at<double>(0,0)<<','<<fit.p.at<double>(0,1)<<','<<fit.initial_cost<<','<<fit.cost<<','<<fit.iteration<<','<<fit.accepted<<'\n';
                ++solved;
                std::cout<<name<<" init"<<initial_kind<<" cost "<<fit.cost<<std::endl;
            }
        }
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["opencv_version"]=CV_VERSION;
        audit["solved_starts"]=solved;
        audit["training_parity"]=0;
        audit["template"]= "original center blurred crop";
        audit["max_iterations"]=40;
        audit["retained"]="Exact homography from fitted corner movement, finite-difference coordinate Jacobian at 1e-3, original linear remap, two axis initializations, affine brightness and damped normal equations";
        audit["changed"]="Native C++ OpenCV4.11 and SVD/LU replace Python. Explicit paths/new output/iteration traces. Historical cached QR mean shift remains an initialization observation; this evaluator is not an image-only product API.";
        audit["saved_original_demo_reference_available"]=false;
        write(out/"checks.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
