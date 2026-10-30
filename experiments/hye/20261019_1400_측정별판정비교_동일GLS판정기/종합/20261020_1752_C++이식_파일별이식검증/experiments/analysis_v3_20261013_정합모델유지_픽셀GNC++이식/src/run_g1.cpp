#include "gn_pixel.hpp"
#include "../../analysis_v1_20261013_QR관측유지_ArUcoC++이식/src/common.hpp"
#include <iostream>
using namespace semes;
using namespace pixel_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try {
        if(argc!=4&&argc!=5)throw std::runtime_error("historical_run_g1 original-workspace bmp-20261014 new-output [limit-per-fold; 0=all]");
        cv::setNumThreads(4);
        fs::path root=fs::u8path(argv[1]),images=fs::u8path(argv[2]),out=fs::u8path(argv[3]);
        int limit=argc==5?std::stoi(argv[4]):0;
        if(limit<0)throw std::runtime_error("Nonnegative limit required");
        if(fs::exists(out))throw std::runtime_error("New output required");
        FastPixel engine(root,images);
        int n=engine.geometry.data.x.rows;
        double nan=std::numeric_limits<double>::quiet_NaN();
        cv::Mat estimate(n,2,CV_64F,cv::Scalar(nan)),nuisance(n,3,CV_64F,cv::Scalar(nan)),cost(n,1,CV_64F,cv::Scalar(nan));
        fs::create_directories(out);
        std::ofstream table(out/"predictions.csv"),failures(out/"failures.csv");
        table<<"row,file,training_parity,true_T,true_Y,pred_T,pred_Y,nuisance_dy,nuisance_rotation,nuisance_scale,cost,iteration\n";
        failures<<"row,file,error\n";
        int attempted=0,failed=0,completed=0;
        for(int fold=0; fold<2; ++fold){
            auto model=engine.geometry.fit_model(fold),target=engine.mean_template(fold);
            std::cout<<"fold "<<fold<<" template ready"<<std::endl;
            int selected=0;
            for(int i=0; i<n; ++i)if(i%2!=fold){
                if(limit&&selected++>=limit)continue;
                ++attempted;
                try{
                    auto result=engine.solve_multi(engine.read_image(engine.geometry.data.names[i]),model,engine.geometry.mean_qr_shift[i],target);
                    if(!cv::checkRange(result.p)||!std::isfinite(result.cost)||result.cost>result.initial_cost)throw std::runtime_error("Invalid fit");
                    result.p.colRange(0,2).copyTo(estimate.row(i));
                    result.p.colRange(2,5).copyTo(nuisance.row(i));
                    cost.at<double>(i)=result.cost;
                    table<<i<<','<<csv(engine.geometry.data.names[i])<<','<<fold<<','<<std::setprecision(17)<<engine.geometry.data.x.at<double>(i,0)<<','<<engine.geometry.data.x.at<double>(i,1)<<','<<estimate.at<double>(i,0)<<','<<estimate.at<double>(i,1);
                    for(int j=0; j<3; ++j)table<<','<<nuisance.at<double>(i,j);
                    table<<','<<result.cost<<','<<result.iteration<<'\n';
                    ++completed;
                }
                catch(const std::exception& e){
                    ++failed;
                    failures<<i<<','<<csv(engine.geometry.data.names[i])<<','<<csv(e.what())<<'\n';
                }
                if(attempted%20==0)std::cout<<"processed "<<attempted<<" failures="<<failed<<std::endl;
            }
        }
        table.close();
        write(out/"g1_est.npy",qr_history::npy_blob(estimate,"<f8",{
            size_t(n),2
        }));
        write(out/"g1_nuisance.npy",qr_history::npy_blob(nuisance,"<f8",{
            size_t(n),3
        }));
        write(out/"g1_cost.npy",qr_history::npy_blob(cost,"<f8",{
            size_t(n)
        }));
        auto expected_estimate=historical_v1::array(npy(bytes(root/"analysis_v3/g1_est.npy"))),expected_nuisance=historical_v1::array(npy(bytes(root/"analysis_v3/g1_nuisance.npy")));
        auto expected_cost=npy(bytes(root/"analysis_v3/g1_cost.npy"));
        double max_t=0,max_y=0,max_nuisance=0,max_cost=0;
        for(int i=0; i<n; ++i)if(std::isfinite(cost.at<double>(i))){
            max_t=std::max(max_t,std::abs(estimate.at<double>(i,0)-expected_estimate.at<double>(i,0)));
            max_y=std::max(max_y,std::abs(estimate.at<double>(i,1)-expected_estimate.at<double>(i,1)));
            for(int j=0; j<3; ++j)max_nuisance=std::max(max_nuisance,std::abs(nuisance.at<double>(i,j)-expected_nuisance.at<double>(i,j)));
            max_cost=std::max(max_cost,std::abs(cost.at<double>(i)-expected_cost.number(i)));
        }
        Json report=Json::dict();
        report["created_kst"]=now();
        report["attempted"]=attempted;
        report["completed"]=completed;
        report["failures"]=failed;
        report["full_281_run"]=attempted==281;
        report["max_T_difference_from_original"]=max_t;
        report["max_Y_difference_from_original"]=max_y;
        report["max_nuisance_difference_from_original"]=max_nuisance;
        report["max_cost_difference_from_original"]=max_cost;
        report["linear_map_max_error_px"]=engine.check_linear();
        report["changes"]="Native C++ cubic spline/GN and OpenCV4.11, same original training parity, all training images for mean template, both initializations selected by lower cost. Per-image failures retained with NaNs instead of aborting. Optional limited runs are explicitly not full evaluation.";
        write(out/"checks.json",report.dump());
        std::cout<<report.dump()<<'\n';
        return failed?2:0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
