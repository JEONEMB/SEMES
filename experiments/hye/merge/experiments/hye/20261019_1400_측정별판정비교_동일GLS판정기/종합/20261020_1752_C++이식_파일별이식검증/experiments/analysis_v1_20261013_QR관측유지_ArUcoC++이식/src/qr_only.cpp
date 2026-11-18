#include "common.hpp"
using namespace semes;
using namespace qr_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try {
        if(argc!=4)throw std::runtime_error("historical_qr_only original-workspace bmp-20261014 new-output");
        fs::path root=fs::u8path(argv[1]),data=fs::u8path(argv[2]),out=fs::u8path(argv[3]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Input input(root);
        auto center=gray(data/"center.bmp");
        auto quads=polygons(center);
        cv::Mat mask=cv::Mat::zeros(center.size(),CV_8U);
        for(auto& quad:quads){
            std::vector<cv::Point> integer;
            for(auto p:quad)integer.emplace_back(int(p.x),int(p.y));
            cv::fillConvexPoly(mask,integer,255);
        }
        std::vector<cv::Point2f> points;
        cv::goodFeaturesToTrack(center,points,300,.01,12,mask,5);
        if(points.empty())throw std::runtime_error("No QR points");
        constexpr int hp=10,sxr=110,syr=6;
        std::vector<cv::Mat> templates;
        for(auto p:points){
            int x=int(std::nearbyint(p.x)),y=int(std::nearbyint(p.y));
            cv::Mat patch;
            center(cv::Rect(x-hp,y-hp,2*hp,2*hp)).convertTo(patch,CV_32F);
            templates.push_back(patch);
        }
        cv::Mat displacements(input.labels.rows,int(points.size()),CV_64F),scores(displacements.size(),CV_64F);
        fs::create_directories(out);
        write(out/"qr_polygons.json",polygon_json(quads).dump());
        std::ofstream table(out/"observations.csv");
        table<<"row,file,point_id,dx,score\n";
        for(int row=0; row<input.labels.rows; ++row){
            auto image=gray(input.path(data,row));
            for(size_t k=0; k<points.size(); ++k){
                int x=int(std::nearbyint(points[k].x)),y=int(std::nearbyint(points[k].y));
                cv::Mat search,response;
                image(cv::Rect(x-hp-sxr,y-hp-syr,2*(hp+sxr),2*(hp+syr))).convertTo(search,CV_32F);
                cv::matchTemplate(search,templates[k],response,cv::TM_CCOEFF_NORMED);
                cv::Point location;
                double score;
                cv::minMaxLoc(response,nullptr,&score,nullptr,&location);
                double dx=subpixel(response,location.x,location.y)-sxr+(x-double(points[k].x));
                displacements.at<double>(row,int(k))=dx;
                scores.at<double>(row,int(k))=score;
                table<<row<<','<<csv(input.files[row][2])<<','<<k<<','<<std::setprecision(17)<<dx<<','<<score<<'\n';
            }
            if((row+1)%40==0)std::cout<<row+1<<'/'<<input.labels.rows<<std::endl;
        }
        cv::Mat coefficients=least_squares(input.labels,displacements),residuals=displacements-input.labels*coefficients;
        std::vector<int64_t> good;
        for(int k=0; k<displacements.cols; ++k){
            double maximum=0,minimum=1;
            for(int row=0; row<displacements.rows; ++row){
                maximum=std::max(maximum,std::abs(residuals.at<double>(row,k)));
                minimum=std::min(minimum,scores.at<double>(row,k));
            }
            if(maximum<2&&minimum>.8)good.push_back(k);
        }
        cv::Mat coordinates(int(points.size()),2,CV_32F);
        for(size_t k=0; k<points.size(); ++k){
            coordinates.at<float>(int(k),0)=points[k].x;
            coordinates.at<float>(int(k),1)=points[k].y;
        }
        cv::Mat good_storage(1,int(good.size()*sizeof(int64_t)),CV_8U);
        std::memcpy(good_storage.data,good.data(),good.size()*sizeof(int64_t));
        write_zip(out/"qr_tracks.npz",{
            {
                "pts.npy",npy_blob(coordinates,"<f4",{
                    points.size(),2
                })
            },{
                "D.npy",npy_blob(displacements,"<f8",{
                    size_t(displacements.rows),size_t(displacements.cols)
                })
            },{
                "good.npy",npy_blob(good_storage,"<i8",{
                    good.size()
                })
            }
        });
        auto expected=npz(root/"analysis_v3/v1_data/qr_tracks.npz");
        bool same_shape=expected.at("pts").shape==std::vector<size_t>{
            points.size(),2
        };
        double coordinate_difference=0,displacement_difference=0;
        if(same_shape)for(size_t i=0; i<points.size()*2; ++i)coordinate_difference=std::max(coordinate_difference,std::abs(double(coordinates.ptr<float>()[i])-expected.at("pts").number(i)));
        bool point_ids_match=same_shape&&coordinate_difference<1e-6;
        if(point_ids_match){
            if(expected.at("D").shape!=std::vector<size_t>{
                size_t(displacements.rows),points.size()
            })throw std::runtime_error("Reference row coverage differs");
            for(size_t i=0; i<displacements.total(); ++i)displacement_difference=std::max(displacement_difference,std::abs(displacements.ptr<double>()[i]-expected.at("D").number(i)));
        }
        bool good_match=expected.at("good").count()==good.size();
        if(good_match)for(size_t i=0; i<good.size(); ++i)good_match&=good[i]==int64_t(expected.at("good").number(i));
        cv::Mat diagnostic;
        cv::cvtColor(center,diagnostic,cv::COLOR_GRAY2BGR);
        for(auto p:points)cv::circle(diagnostic,p,4,cv::Scalar(0,200,0),1);
        png(out/"center_all_points_1x.png",diagnostic);
        png(out/"center_QR_mask_1x.png",mask);
        Json report=Json::dict();
        report["created_kst"]=now();
        report["opencv_version"]=CV_VERSION;
        report["rows"]=displacements.rows;
        report["points"]=displacements.cols;
        report["good_count"]=int(good.size());
        report["ordered_point_ids_match"]=point_ids_match;
        report["max_point_coordinate_difference"]=coordinate_difference;
        report["displacement_comparison_valid"]=point_ids_match;
        report["max_displacement_difference"]=displacement_difference;
        report["good_ids_match"]=good_match;
        report["center_sha256"]=sha256(data/"center.bmp");
        report["reference_sha256"]=sha256(root/"analysis_v3/v1_data/qr_tracks.npz");
        report["changes"]="Original ArUco QR mask, center-only goodFeaturesToTrack parameters, fixed IDs, NCC/subpixel formula and point filtering retained. Native OpenCV 4.11 C++; runtime/build optimizations differ from unknown original Python build. Original full-data quality selection is historical, not independent validation.";
        report["parity_passed"]=point_ids_match&&good_match&&displacement_difference<=.001;
        write(out/"checks.json",report.dump());
        std::cout<<report.dump()<<'\n';
        return report.at("parity_passed").num()?0:2;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
