#include "common.hpp"
#include <iostream>
using namespace semes;
using namespace historical_v1;
namespace {
    double subpixel(const cv::Mat& response,int x,int y,bool horizontal){
        int coordinate=horizontal?x:y,limit=horizontal?response.cols:response.rows;
        if(coordinate<=0||coordinate+1>=limit)return coordinate;
        float before=response.at<float>(y-(horizontal?0:1),x-(horizontal?1:0)),middle=response.at<float>(y,x),after=response.at<float>(y+(horizontal?0:1),x+(horizontal?1:0));
        // NumPy2 retains float32 through the correction and integer-coordinate
        // addition; promotion occurs only when adding the float64 point offset.
        float denominator=before-2.0f*middle+after,numerator=before-after;
        float correction=denominator!=0?(.5f*numerator)/denominator:0.f;
        return double(float(coordinate)+correction);
    }
}
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try{
        if(argc!=5)throw std::runtime_error("historical_v1_track original-workspace data-bmp-20261014 new-output frozen|detect");
        fs::path root=fs::u8path(argv[1]),images=fs::u8path(argv[2]),out=fs::u8path(argv[3]);
        std::string mode=argv[4];
        if(mode!="frozen"&&mode!="detect")throw std::runtime_error("Point mode must be frozen or detect");
        if(fs::exists(out))throw std::runtime_error("New output required");
        auto zip=read_zip(root/"analysis_v3/v1_data/tracks.npz");
        auto original_points=npy(zip.at("pts.npy")),original_m=npy(zip.at("M.npy")),object_files=npy(zip.at("files.npy"));
        auto files=PickleData(object_files.data).string_rows(object_files.shape.at(0),4);
        cv::Mat center=loadimage(images/"center.bmp"),points=array(original_points);
        if(mode=="detect"){
            std::vector<cv::Point2f> detected;
            cv::goodFeaturesToTrack(center,detected,400,.01,90,cv::noArray(),9);
            std::vector<cv::Point2d> positions={
                cv::Point2d(1846.7,1010.3)
            };
            for(auto p:detected)if(24+120+2<p.x&&p.x<center.cols-24-120-2&&24+12+2<p.y&&p.y<center.rows-24-12-2)positions.push_back(p);
            points=cv::Mat(int(positions.size()),2,CV_64F);
            for(int i=0; i<points.rows; ++i){
                points.at<double>(i,0)=positions[i].x;
                points.at<double>(i,1)=positions[i].y;
            }
        }
        constexpr int hp=24,sxr=120,syr=12;
        std::vector<cv::Mat> templates;
        for(int i=0; i<points.rows; ++i){
            int x=int(std::nearbyint(points.at<double>(i,0))),y=int(std::nearbyint(points.at<double>(i,1)));
            cv::Mat patch;
            center(cv::Rect(x-hp,y-hp,2*hp,2*hp)).convertTo(patch,CV_32F);
            templates.push_back(patch);
        }
        fs::create_directories(out);
        std::ofstream positions(out/"points.csv"),measurements(out/"measurements.csv"),differences(out/"parity.csv"),failures(out/"failures.csv");
        positions<<"point_id,x,y\n";
        for(int i=0; i<points.rows; ++i)positions<<i<<','<<std::setprecision(17)<<points.at<double>(i,0)<<','<<points.at<double>(i,1)<<'\n';
        measurements<<"row,file,point_id,dx,dy,score\n";
        differences<<"row,file,point_id,dx_difference,dy_difference,score_difference\n";
        failures<<"row,file,error\n";
        int failed=0;
        double max_dx=0,max_dy=0,max_score=0;
        for(size_t row=0; row<files.size(); ++row){
            const auto& file=files[row][2];
            try{
                auto path=file=="center.bmp"?images/"center.bmp":images/(file[0]=='T'?"Theta":"Y")/fs::u8path(file);
                auto image=loadimage(path);
                for(int i=0; i<points.rows; ++i){
                    double x=points.at<double>(i,0),y=points.at<double>(i,1);
                    int xi=int(std::nearbyint(x)),yi=int(std::nearbyint(y));
                    cv::Mat search,response;
                    image(cv::Rect(xi-hp-sxr,yi-hp-syr,2*(hp+sxr),2*(hp+syr))).convertTo(search,CV_32F);
                    cv::matchTemplate(search,templates[i],response,cv::TM_CCOEFF_NORMED);
                    cv::Point location;
                    double score;
                    cv::minMaxLoc(response,nullptr,&score,nullptr,&location);
                    float local_x=float(subpixel(response,location.x,location.y,true))-float(sxr),local_y=float(subpixel(response,location.x,location.y,false))-float(syr);
                    float dx=float(double(local_x)+(xi-x)),dy=float(double(local_y)+(yi-y)),stored_score=float(score);
                    measurements<<row<<','<<csv(file)<<','<<i<<','<<std::setprecision(17)<<dx<<','<<dy<<','<<stored_score<<'\n';
                    if(mode=="frozen"){
                        size_t offset=(row*size_t(points.rows)+i)*3;
                        double ex=dx-original_m.number(offset),ey=dy-original_m.number(offset+1),es=stored_score-original_m.number(offset+2);
                        max_dx=std::max(max_dx,std::abs(ex));
                        max_dy=std::max(max_dy,std::abs(ey));
                        max_score=std::max(max_score,std::abs(es));
                        differences<<row<<','<<csv(file)<<','<<i<<','<<ex<<','<<ey<<','<<es<<'\n';
                    }
                }
            }
            catch(const std::exception& e){
                ++failed;
                failures<<row<<','<<csv(file)<<','<<csv(e.what())<<'\n';
            }
            if((row+1)%40==0)std::cout<<row+1<<'/'<<files.size()<<std::endl;
        }
        Json report=Json::dict();
        report["created_kst"]=now();
        report["mode"]=mode;
        report["image_rows_including_repeated_center"]=int(files.size());
        report["points"]=points.rows;
        report["failures"]=failed;
        report["max_dx_difference"]=max_dx;
        report["max_dy_difference"]=max_dy;
        report["max_score_difference"]=max_score;
        report["parity_evaluated"]=mode=="frozen";
        report["opencv_version"]=CV_VERSION;
        report["changed"]="Explicit original snapshot point mode fixes IDs for measurement comparison; detect mode implements the original center-only detector parameters. NumPy pickle metadata is parsed symbolically without executing code. CSV replaces NPZ output. QR-outside points are historical comparison inputs, not current QR-only 630 points.";
        write(out/"checks.json",report.dump());
        std::cout<<report.dump()<<'\n';
        return failed?2:0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
