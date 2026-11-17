#include "final04.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/version.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace fs=std::filesystem;
namespace {
const cv::Scalar blue(255,200,20),orange(20,145,255),green(80,255,90),grey(190,190,190),white(245,245,245),dark(22,22,22);
const cv::Rect qr_boxes[]={{1552,969,184,198},{1809,971,184,198},{2067,973,185,198}};
const cv::Point2d pins[]={{974,1352},{2864,1358},{1010,1677},{2817,1681}};
cv::Point px(cv::Point2d p){return {cvRound(p.x),cvRound(p.y)};}
std::string number(double v){std::ostringstream s;s<<std::fixed<<std::setprecision(3)<<v;return s.str();}
void label(cv::Mat& im,const std::string& s,cv::Point at,double scale=.8,cv::Scalar color=white) {
    int baseline=0;auto size=cv::getTextSize(s,cv::FONT_HERSHEY_SIMPLEX,scale,1,&baseline);
    cv::rectangle(im,cv::Rect(at.x-3,at.y-size.height-4,size.width+6,size.height+baseline+8),dark,-1);
    cv::putText(im,s,at,cv::FONT_HERSHEY_SIMPLEX,scale,color,1,cv::LINE_AA);
}
cv::Point2d project(const cv::Mat& h,cv::Point2d p){const double z=h.at<double>(2,0)*p.x+h.at<double>(2,1)*p.y+h.at<double>(2,2);return {(h.at<double>(0,0)*p.x+h.at<double>(0,1)*p.y+h.at<double>(0,2))/z,(h.at<double>(1,0)*p.x+h.at<double>(1,1)*p.y+h.at<double>(1,2))/z};}
void outline(cv::Mat& im,const cv::Mat& h,cv::Rect b,cv::Scalar color,int width=2){
    const cv::Point2d corners[]={{double(b.x),double(b.y)},{double(b.x+b.width),double(b.y)},{double(b.x+b.width),double(b.y+b.height)},{double(b.x),double(b.y+b.height)}};
    for(int i=0;i<4;++i)cv::line(im,px(project(h,corners[i])),px(project(h,corners[(i+1)%4])),color,width,cv::LINE_AA);
}
void dashed(cv::Mat& im,cv::Point a,cv::Point b,cv::Scalar color){const auto d=b-a;const double len=cv::norm(d);for(double k=0;k<len;k+=12)cv::line(im,a+d*(k/len),a+d*(std::min(k+6.,len)/len),color,1,cv::LINE_AA);}
void reference(cv::Mat& im,cv::Rect b){dashed(im,b.tl(),{b.x+b.width,b.y},grey);dashed(im,{b.x+b.width,b.y},b.br(),grey);dashed(im,b.br(),{b.x,b.y+b.height},grey);dashed(im,{b.x,b.y+b.height},b.tl(),grey);}
cv::Mat crop2(const cv::Mat& im,cv::Point2d center,int width,int height){
    const cv::Rect roi(cvRound(center.x)-width/2,cvRound(center.y)-height/2,width,height);
    if((roi&cv::Rect(0,0,im.cols,im.rows))!=roi)throw std::runtime_error("Detail crop outside image");
    cv::Mat result;cv::resize(im(roi),result,cv::Size(width*2,height*2),0,0,cv::INTER_NEAREST);return result;
}
void save(const fs::path& path,const cv::Mat& image){
    std::vector<uchar> bytes;cv::imencode(".png",image,bytes);
    std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot write overlay");f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
}
void render(const fs::path& assets,const fs::path& file,const fs::path& out){
    if(fs::exists(out))throw std::runtime_error("Choose a new output directory");
    cv::setNumThreads(1);const auto gray=alignment_detail::read_gray(file);
    final04::Estimator estimator(assets.u8string());const auto result=estimator.predict(gray);
    if(result.qr_transforms.size()!=3)throw std::runtime_error("Expected three measured QR transforms");
    cv::Mat plain;cv::cvtColor(gray,plain,cv::COLOR_GRAY2BGR);cv::Mat full=plain.clone();
    cv::FileStorage protocol((assets/"line_protocol.json").u8string(),cv::FileStorage::READ);
    std::vector<cv::Rect> blocks;for(auto n:protocol["blocks"]){auto b=n["read_box"];blocks.push_back({int(b[0]),int(b[1]),int(b[2])-int(b[0]),int(b[3])-int(b[1])});}
    std::vector<cv::Point2d> qpoints,pinpoints;
    fs::create_directories(out);
    std::ofstream points(out/"points.tsv");points<<"kind\tid\treference_x\treference_y\tcurrent_x\tcurrent_y\tdx\tdy\n"<<std::setprecision(17);
    for(int q=0;q<3;++q){const auto b=qr_boxes[q];reference(full,b);outline(full,result.qr_transforms[q],b,blue,3);
        int k=0;for(double fy:{.2,.5,.8})for(double fx:{.2,.5,.8}){
            cv::Point2d a(b.x+fx*b.width,b.y+fy*b.height),v(result.raw110.at<double>(q*18+k*2),result.raw110.at<double>(q*18+k*2+1)),c=a+v;
            if(cv::norm(c-project(result.qr_transforms[q],a))>1e-8)throw std::runtime_error("Overlay and model QR coordinates differ");
            cv::drawMarker(full,px(a),grey,cv::MARKER_CROSS,7,1);cv::arrowedLine(full,px(a),px(c),blue,1,cv::LINE_AA,0,.15);cv::circle(full,px(c),4,blue,1,cv::LINE_AA);
            points<<"qr_projected\t"<<q+1<<'.'<<k+1<<'\t'<<a.x<<'\t'<<a.y<<'\t'<<c.x<<'\t'<<c.y<<'\t'<<v.x<<'\t'<<v.y<<'\n';qpoints.push_back(c);++k;
        }
        const auto top=project(result.qr_transforms[q],cv::Point2d(b.x,b.y));label(full,"QR"+std::to_string(q+1),px(top)+cv::Point(0,-15),.85,blue);
    }
    for(int k=0;k<4;++k){cv::Point2d v(result.raw110.at<double>(102+k*2),result.raw110.at<double>(103+k*2)),c=pins[k]+v;
        reference(full,{int(pins[k].x)-45,int(pins[k].y)-45,91,91});cv::rectangle(full,{cvRound(c.x)-45,cvRound(c.y)-45,91,91},orange,2);
        cv::drawMarker(full,px(c),orange,cv::MARKER_CROSS,14,2);cv::arrowedLine(full,px(pins[k]),px(c),orange,1,cv::LINE_AA,0,.15);
        label(full,"S"+std::to_string(k+1),px(c)+cv::Point(-45,-58),.85,orange);pinpoints.push_back(c);
        points<<"screw_patch\t"<<k+1<<'\t'<<pins[k].x<<'\t'<<pins[k].y<<'\t'<<c.x<<'\t'<<c.y<<'\t'<<v.x<<'\t'<<v.y<<'\n';
    }
    for(int k=0;k<48;++k){const auto b=blocks[k];const double dy=result.raw110.at<double>(54+k);cv::Point2d a(b.x+b.width/2.,b.y+24.),c=a+cv::Point2d(0,dy);
        cv::rectangle(full,b,green,1);cv::line(full,{b.x,cvRound(c.y)},{b.x+b.width,cvRound(c.y)},green,2);
        cv::circle(full,px(c),2,green,-1);points<<"bar_profile_anchor\t"<<k+1<<'\t'<<a.x<<'\t'<<a.y<<'\t'<<c.x<<'\t'<<c.y<<"\t0\t"<<dy<<'\n';
        label(full,std::to_string(k+1),{b.x,b.y-6},.42,green);
    }
    label(full,"3 QR / 2 BARS (48 windows) / 4 SCREWS (8 x,y values)",{90,95},1.4);
    label(full,"BLUE: QR warp + 27 projected samples   GREEN: bar profile windows   ORANGE: 4 screw patches",{90,150},.95);
    label(full,"GRAY dashed/cross: fixed reference. Colored positions: measured current. Displacements are NOT magnified.",{90,202},.9);
    label(full,"Native 3840x2160. QR samples are virtual projections; screw crosses are tracked patch centers, not fitted circle centers.",{90,2070},.9);
    label(full,"OpenCV "+std::string(CV_VERSION)+" | local diagnostic run | Y="+number(result.pose.y)+" T="+number(result.pose.t),{90,2120},.9);
    save(out/"01_full_overlay.png",full);
    cv::Mat qrpanel(740,1680,CV_8UC3,dark);label(qrpanel,"QR: matching rectangle + 9 projected samples per code (detail 2x)",{20,38},.85,blue);
    label(qrpanel,"The points are generated by the fitted homography, not 27 independently detected corners.",{20,75},.7);
    for(int q=0;q<3;++q){auto c=project(result.qr_transforms[q],cv::Point2d(qr_boxes[q].x+qr_boxes[q].width/2.,qr_boxes[q].y+qr_boxes[q].height/2.));
        crop2(full,c,280,280).copyTo(qrpanel(cv::Rect(q*560,110,560,560)));label(qrpanel,"QR"+std::to_string(q+1)+" : 9 x (dx,dy) = 18 values",{q*560+15,710},.68,blue);}
    save(out/"02_qr_detail_2x.png",qrpanel);
    cv::Mat pinpanel(500,1280,CV_8UC3,dark);label(pinpanel,"4 screws: NCC patch matching -> translation GN (detail 2x)",{16,34},.8,orange);
    label(pinpanel,"Orange cross = tracked reference patch center; not automatic screw-circle center detection.",{16,68},.64);
    for(int k=0;k<4;++k){crop2(full,pinpoints[k],160,160).copyTo(pinpanel(cv::Rect(k*320,90,320,320)));
        label(pinpanel,"S"+std::to_string(k+1)+" dx="+number(result.raw110.at<double>(102+2*k)),{k*320+12,436},.62,orange);
        label(pinpanel,"dy="+number(result.raw110.at<double>(103+2*k))+" NCC="+number(result.pin_quality.at<double>(2*k)),{k*320+12,474},.56,orange);}
    save(out/"03_screw_detail_2x.png",pinpanel);
    cv::Mat barpanel(370,3840,CV_8UC3,dark);label(barpanel,"UPPER: 24 profile windows (native 1x crop)",{20,34},.85,green);
    full(cv::Rect(0,665,3840,100)).copyTo(barpanel(cv::Rect(0,50,3840,100)));
    label(barpanel,"LOWER: 24 profile windows (native 1x crop)",{20,190},.85,green);
    full(cv::Rect(0,1745,3840,100)).copyTo(barpanel(cv::Rect(0,205,3840,100)));
    label(barpanel,"Boxes = fixed 64x49 read windows. Colored segment = profile reference anchor + measured vertical shift; NOT an independently detected physical edge.",{20,350},.85);
    save(out/"04_bar_detail_1x.png",barpanel);
    cv::FileStorage metadata((out/"measurement.json").u8string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
    metadata<<"image_sha256"<<alignment_detail::sha256(file)<<"opencv"<<CV_VERSION<<"qr_objects"<<3<<"bar_objects"<<2<<"screw_objects"<<4<<"qr_projected_points"<<27<<"bar_windows"<<48<<"screw_values"<<8<<"Y"<<result.pose.y<<"T"<<result.pose.t<<"raw110"<<result.raw110<<"pin_quality"<<result.pin_quality<<"note"<<"Current-image diagnostic inference; not new machine or repeatability certification.";
    std::cout<<out.u8string()<<" : 3 QR,2 bars,4 screw patches; coordinate consistency PASS\n";
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv){
#else
int main(int argc,char** argv){
#endif
    try{if(argc!=4)throw std::runtime_error("final04_overlay ASSETS IMAGE NEW_OUTPUT_DIR");render(fs::path(argv[1]),fs::path(argv[2]),fs::path(argv[3]));return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
