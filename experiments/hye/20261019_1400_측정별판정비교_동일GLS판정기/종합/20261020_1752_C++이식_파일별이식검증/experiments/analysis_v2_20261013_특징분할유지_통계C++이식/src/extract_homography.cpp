#include "alignment.hpp"
#include "table.hpp"
#include <iostream>
using namespace semes;
int wmain(int argc,wchar_t** wide_argv) {
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try {
        if(argc<4)throw std::runtime_error("historical_v2_homography original-QR-position.csv data-bmp-20261014 new-output [filenames...]");
        fs::path source=fs::u8path(argv[1]),root=fs::u8path(argv[2]),out=fs::u8path(argv[3]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        CsvTable table(source);
        size_t center_row=table.rows.size();
        std::map<std::string,size_t> index;
        for(size_t i=0; i<table.rows.size(); ++i){
            index.emplace(table.text(i,"file"),i);
            if(table.text(i,"file")=="center.bmp")center_row=i;
        }
        if(center_row==table.rows.size())throw std::runtime_error("Center metadata missing");
        auto value=[&](const char* key){
            return table.number(center_row,key);
        };
        int x0=int(value("Lx")-110),x1=int(value("Rx")+110),y0=int(std::min(value("Ly"),value("Ry"))-120),y1=int(std::max(value("Ly"),value("Ry"))+120);
        cv::Mat center;
        loadimage(root/"center.bmp").convertTo(center,CV_32F);
        cv::Mat reference=center(cv::Rect(x0,y0,x1-x0,y1-y0)).clone();
        std::vector<cv::Point2f> points={
            cv::Point2f(float(x0),float(y0)),cv::Point2f(float(x1),float(y0)),cv::Point2f(float(x1),float(y1)),cv::Point2f(float(x0),float(y1)),cv::Point2f(float(value("Lx")),float(value("Ly"))),cv::Point2f(float(value("Cx")),float(value("Cy"))),cv::Point2f(float(value("Rx")),float(value("Ry")))
        };
        const std::vector<std::string> names={
            u8"좌상",u8"우상",u8"우하",u8"좌하",u8"QR좌",u8"QR중",u8"QR우"
        };
        std::vector<std::string> files;
        if(argc==4){
            for(size_t i=0; i<table.rows.size(); ++i)files.push_back(table.text(i,"file"));
        }
        else for(int i=4; i<argc; ++i)files.emplace_back(argv[i]);
        fs::create_directories(out);
        std::ofstream features(out/"homography_features.csv"),failures(out/"failures.csv");
        features<<"file,cc,h20,h21";
        for(auto& name:names)features<<','<<name<<"_dx,"<<name<<"_dy";
        features<<'\n';
        failures<<"file,error\n";
        int failed=0;
        for(size_t i=0; i<files.size(); ++i){
            auto& file=files[i];
            try{
                size_t row=index.at(file);
                int sx=int(std::nearbyint((table.number(row,"dLx")+table.number(row,"dCx")+table.number(row,"dRx"))/3)),sy=int(std::nearbyint((table.number(row,"dLy")+table.number(row,"dCy")+table.number(row,"dRy"))/3));
                fs::path path=file=="center.bmp"?root/"center.bmp":root/(file[0]=='T'?"Theta":"Y")/fs::u8path(file);
                cv::Mat image;
                loadimage(path).convertTo(image,CV_32F);
                cv::Mat current=image(cv::Rect(x0+sx,y0+sy,x1-x0,y1-y0)),warp=cv::Mat::eye(3,3,CV_32F);
                double correlation=cv::findTransformECC(reference,current,warp,cv::MOTION_HOMOGRAPHY,cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,300,1e-7),cv::noArray(),5);
                std::vector<cv::Point2f> local,transformed;
                for(auto p:points)local.push_back(p-cv::Point2f(float(x0),float(y0)));
                cv::perspectiveTransform(local,transformed,warp);
                features<<csv(file)<<','<<std::setprecision(17)<<correlation<<','<<warp.at<float>(2,0)<<','<<warp.at<float>(2,1);
                for(size_t j=0; j<points.size(); ++j){
                    cv::Point2f displacement=transformed[j]+cv::Point2f(float(x0+sx),float(y0+sy))-points[j];
                    features<<','<<displacement.x<<','<<displacement.y;
                }
                features<<'\n';
            }
            catch(const std::exception& e){
                ++failed;
                failures<<csv(file)<<','<<csv(e.what())<<'\n';
            }
            if((i+1)%40==0)std::cout<<i+1<<'/'<<files.size()<<std::endl;
        }
        features.close();
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["attempted"]=int(files.size());
        audit["failures"]=failed;
        audit["opencv_version"]=CV_VERSION;
        audit["metadata_sha256"]=sha256(source);
        audit["center_sha256"]=sha256(root/"center.bmp");
        audit["changed"]=std::string("Original float32 plate ECC homography parameters retained. Native OpenCV ")+CV_VERSION+". Failures are logged per image instead of aborting the whole run. Current algorithm performance is not inferred from these historical features.";
        write(out/"audit.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return failed?2:0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
