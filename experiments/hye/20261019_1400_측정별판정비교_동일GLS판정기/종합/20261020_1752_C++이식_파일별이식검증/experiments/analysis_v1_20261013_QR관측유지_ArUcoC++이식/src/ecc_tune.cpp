#include "common.hpp"
using namespace semes;
using namespace qr_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try {
        if(argc!=4&&argc!=5)throw std::runtime_error("historical_ecc_tune original-workspace bmp-20261014 new-output [variant-index 0..6]");
        const int selected_variant=argc==5?std::stoi(argv[4]):-1;
        if(argc==5&&(selected_variant<0||selected_variant>6))throw std::runtime_error("variant-index must be in 0..6");
        cv::setNumThreads(4);
        fs::path root=fs::u8path(argv[1]),data=fs::u8path(argv[2]),out=fs::u8path(argv[3]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Input input(root);
        auto center=gray(data/"center.bmp");
        auto quads=polygons(center);
        std::vector<int> theta_rows,selected_rows;
        for(int i=0; i<input.labels.rows; ++i)if(input.labels.at<double>(i,1)==0)theta_rows.push_back(i);
        for(size_t i=0; i<theta_rows.size(); i+=3)selected_rows.push_back(theta_rows[i]);
        std::vector<cv::Mat> images;
        for(int row:selected_rows)images.push_back(gray(input.path(data,row)));
        struct Variant{
            double blur;
            int expansion,filter;
        };
        std::vector<Variant> variants={
            {
                0,0,5
            },{
                0,0,1
            },{
                1.5,0,1
            },{
                3,0,1
            },{
                0,12,5
            },{
                0,25,5
            },{
                3,25,1
            }
        };
        fs::create_directories(out);
        write(out/"qr_polygons.json",polygon_json(quads).dump());
        std::ofstream metrics(out/"variants.csv"),measurements(out/"homographies.csv"),failures(out/"failures.csv");
        metrics<<"variant,blur,mask_expansion_px,ecc_filter,images,failures,T_noise,seconds\n";
        measurements<<"variant,row,file,true_T,h20,ecc_cc\n";
        failures<<"variant,row,file,error\n";
        int total_failures=0;
        cv::Rect crop(1450,940,920,260);
        for(size_t variant=0; variant<variants.size(); ++variant){
            if(selected_variant>=0&&int(variant)!=selected_variant)continue;
            auto start=std::chrono::steady_clock::now();
            auto v=variants[variant];
            cv::Mat reference;
            center(crop).convertTo(reference,CV_32F);
            if(v.blur)cv::GaussianBlur(reference,reference,cv::Size(),v.blur);
            cv::Mat mask=cv::Mat::zeros(reference.size(),CV_8U);
            for(auto& quad:quads){
                cv::Point2f mean;
                for(auto p:quad)mean+=p*.25f;
                std::vector<cv::Point> polygon;
                for(auto p:quad){
                    double dx=p.x-mean.x,dy=p.y-mean.y;
                    polygon.emplace_back(int(p.x-1450+(dx>0?1:dx<0?-1:0)*v.expansion),int(p.y-940+(dy>0?1:dy<0?-1:0)*v.expansion));
                }
                cv::fillConvexPoly(mask,polygon,255);
            }
            int failed=0;
            cv::Mat design,values;
            for(size_t i=0; i<images.size(); ++i){
                int row=selected_rows[i];
                try{
                    cv::Mat current;
                    images[i](crop).convertTo(current,CV_32F);
                    if(v.blur)cv::GaussianBlur(current,current,cv::Size(),v.blur);
                    cv::Mat correlation;
                    cv::matchTemplate(current,reference(cv::Rect(150,40,620,180)),correlation,cv::TM_CCOEFF_NORMED);
                    cv::Point location;
                    cv::minMaxLoc(correlation,nullptr,nullptr,nullptr,&location);
                    cv::Mat warp=(cv::Mat_<float>(3,3)<<1,0,location.x-150,0,1,location.y-40,0,0,1);
                    double cc=cv::findTransformECC(reference,current,warp,cv::MOTION_HOMOGRAPHY,cv::TermCriteria(3,500,1e-9),mask,v.filter);
                    cv::Mat sample=(cv::Mat_<double>(1,2)<<input.labels.at<double>(row,0),1);
                    design.push_back(sample);
                    cv::Mat response=(cv::Mat_<double>(1,1)<<double(warp.at<float>(2,0)));
                    values.push_back(response);
                    measurements<<variant<<','<<row<<','<<csv(input.files[row][2])<<','<<std::setprecision(17)<<input.labels.at<double>(row,0)<<','<<warp.at<float>(2,0)<<','<<cc<<'\n';
                }
                catch(const std::exception& e){
                    ++failed;
                    failures<<variant<<','<<row<<','<<csv(input.files[row][2])<<','<<csv(e.what())<<'\n';
                }
            }
            double noise=std::numeric_limits<double>::quiet_NaN();
            if(!failed){
                auto coefficients=least_squares(design,values);
                cv::Mat residual=values-design*coefficients;
                double mean=cv::mean(residual)[0],squared=0;
                for(int i=0; i<residual.rows; ++i)squared+=std::pow(residual.at<double>(i)-mean,2);
                noise=std::abs(std::sqrt(squared/residual.rows)/coefficients.at<double>(0));
            }
            double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            metrics<<variant<<','<<v.blur<<','<<v.expansion<<','<<v.filter<<','<<images.size()<<','<<failed<<','<<std::setprecision(17)<<noise<<','<<seconds<<'\n';
            total_failures+=failed;
            std::cout<<"variant "<<variant<<" T_noise="<<noise<<" failures="<<failed<<std::endl;
        }
        Json report=Json::dict();
        report["created_kst"]=now();
        report["opencv_version"]=CV_VERSION;
        report["variants"]=selected_variant<0?int(variants.size()):1;
        if(selected_variant>=0)report["selected_variant"]=selected_variant;
        report["selected_original_rows"]=int(selected_rows.size());
        report["failures"]=total_failures;
        report["exact_original_numeric_reference_available"]=false;
        report["changes"]="Original seven blur/mask/filter configurations, theta-row every-third selection, ECC500 EPS1e-9 and h20 slope-normalized noise retained. C++4.11 runtime replaces original Python runtime. Original abort-on-error becomes explicit per-image failure report; a variant with failures has no aggregate noise.";
        report["comparison_scope"]="Original script only printed noise values; no stored seven-variant baseline found. Execution alone is not labelled exact numerical parity.";
        write(out/"checks.json",report.dump());
        std::cout<<report.dump()<<'\n';
        return total_failures?2:0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
