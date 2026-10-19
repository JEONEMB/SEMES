#include "alignment.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
struct Line { cv::Point2f p, v; double angle, length; };
bool intersect(const Line& a, const Line& b, cv::Point2f& p) {
    double d = a.v.cross(b.v);
    if (std::abs(d) < 1e-6) return false;
    p = a.p + a.v * float((b.p-a.p).cross(b.v)/d);
    return true;
}
double support(const cv::Mat& distance, cv::Point2f a, cv::Point2f b) {
    int hit=0, total=0;
    // Rounded corners are intentionally omitted from the straight-edge score.
    for(int i=8;i<=92;++i) {
        cv::Point2f p=a+(b-a)*(i/100.f);
        int x=cvRound(p.x),y=cvRound(p.y);
        if(x<0||y<0||x>=distance.cols||y>=distance.rows) return 0;
        hit += distance.at<float>(y,x)<=3.f; ++total;
    }
    return double(hit)/total;
}
void showStep(const std::string& name,const cv::Mat& image) {
    cv::namedWindow(name,cv::WINDOW_NORMAL);
    cv::imshow(name,image);
    cv::resizeWindow(name,1200,650);
    cv::waitKey(0); // Press any key to advance.
    cv::destroyWindow(name);
}
} // namespace
Alignment_t alignment(cv::Mat& src) {
    return alignment(src, AlignmentOptions{}, nullptr);
}
Alignment_t alignment(cv::Mat& src, const AlignmentOptions& options,
                      PlateDetectionResult* detection) {
    if(detection) *detection = PlateDetectionResult{};
    if(src.empty() || src.depth()!=CV_8U)
        CV_Error(cv::Error::StsBadArg, "Expected a nonempty 8-bit image");
    cv::Mat image;
    if(src.channels()==1) cv::cvtColor(src,image,cv::COLOR_GRAY2BGR);
    else if(src.channels()==4) cv::cvtColor(src,image,cv::COLOR_BGRA2BGR);
    else if(src.channels()==3) image=src;
    else CV_Error(cv::Error::StsBadArg, "Expected 1, 3, or 4 channels");
    const std::string& prefix=options.outputPrefix;
    bool gui=options.visualize;
    auto save=[&](const std::string& suffix,const cv::Mat& mat) {
        if(!prefix.empty() && !cv::imwrite(prefix+suffix,mat))
            CV_Error(cv::Error::StsError, "Cannot save output image");
    };
    // One broad search ROI, not three QR-specific crops. Scale with input resolution.
    double sx=image.cols/3840.,sy=image.rows/2160.;
    cv::Rect roi(cvRound(1200*sx),cvRound(800*sy),cvRound(1400*sx),cvRound(550*sy));
    roi &= cv::Rect(0,0,image.cols,image.rows);
    if(roi.empty()) CV_Error(cv::Error::StsBadArg, "Search ROI is empty");
    cv::Mat crop=image(roi),gray,blurred,edges;
    cv::cvtColor(crop,gray,cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray,blurred,cv::Size(5,5),1.2);
    cv::Canny(blurred,edges,40,120);
    save("_01_canny.png",edges);
    if(gui) showStep("1 - Canny edges (search ROI)",edges);

    std::vector<cv::Vec4i> segments;
    cv::HoughLinesP(edges,segments,1,CV_PI/720,30,50*sx,10*sx);
    cv::Mat hough=crop.clone();
    std::vector<Line> horizontal,vertical;
    for(auto s:segments) {
        cv::line(hough,{s[0],s[1]},{s[2],s[3]},cv::Scalar(0,200,0),1,cv::LINE_AA);
        cv::Point2f p{float(s[0]),float(s[1])},v{float(s[2]-s[0]),float(s[3]-s[1])};
        double length=cv::norm(v); v*=float(1./length);
        double angle=std::atan2(v.y,v.x)*180/CV_PI;
        while(angle<0) angle+=180;
        Line l{p,v,angle,length};
        if(std::min(angle,180-angle)<8 && length>180*sx) horizontal.push_back(l);
        if(std::abs(angle-90)<8 && length>50*sy) vertical.push_back(l);
    }
    save("_02_hough.png",hough);
    if(gui) showStep("2 - Hough line segments (search ROI)",hough);
    auto trim=[](std::vector<Line>& lines) {
        std::sort(lines.begin(),lines.end(),[](const Line&a,const Line&b){return a.length>b.length;});
        if(lines.size()>40) lines.resize(40);
    };
    trim(horizontal);trim(vertical);
    std::cout<<"Horizontal/vertical candidates: "<<horizontal.size()<<"/"<<vertical.size()<<"\n";
    cv::Mat inverse,distance;
    cv::bitwise_not(edges,inverse);
    cv::distanceTransform(inverse,distance,cv::DIST_L2,3);
    double best=-1;std::array<cv::Point2f,4> corners{};
    for(size_t i=0;i<horizontal.size();++i) for(size_t j=i+1;j<horizontal.size();++j)
    for(size_t k=0;k<vertical.size();++k) for(size_t l=k+1;l<vertical.size();++l) {
        std::array<cv::Point2f,4> q;
        if(!intersect(horizontal[i],vertical[k],q[0]) || !intersect(horizontal[i],vertical[l],q[1]) ||
           !intersect(horizontal[j],vertical[l],q[2]) || !intersect(horizontal[j],vertical[k],q[3])) continue;
        double w1=cv::norm(q[1]-q[0])/sx,w2=cv::norm(q[2]-q[3])/sx;
        double h1=cv::norm(q[3]-q[0])/sy,h2=cv::norm(q[2]-q[1])/sy;
        // Size prior for the black plate surrounding all three QR labels.
        if(std::min(w1,w2)<850||std::max(w1,w2)>1100||std::min(h1,h2)<180||std::max(h1,h2)>300) continue;
        if(std::abs(w1-w2)>40||std::abs(h1-h2)>40) continue;
        double score=0,minSupport=1;
        for(int t=0;t<4;++t) {double s=support(distance,q[t],q[(t+1)%4]); score+=s;minSupport=std::min(minSupport,s);}
        if(minSupport<0.45) continue;
        score/=4;
        // Prefer the dark plate boundary over nearby metal housing edges.
        cv::Point2f center=(q[0]+q[1]+q[2]+q[3])*0.25f;
        double contrast=0; int samples=0;
        for(int t=0;t<4;++t) for(int z=20;z<=80;z+=5) {
            cv::Point2f p=q[t]+(q[(t+1)%4]-q[t])*(z/100.f);
            cv::Point2f n=center-p; n*=float(10*sx/cv::norm(n));
            cv::Point inside(cvRound(p.x+n.x),cvRound(p.y+n.y));
            cv::Point outside(cvRound(p.x-n.x),cvRound(p.y-n.y));
            if(cv::Rect(0,0,gray.cols,gray.rows).contains(inside)&&cv::Rect(0,0,gray.cols,gray.rows).contains(outside)) {
                contrast+=int(gray.at<uchar>(outside))-int(gray.at<uchar>(inside)); ++samples;
            }
        }
        if(samples) score+=0.4*contrast/(255*samples);
        if(score>best) {best=score;corners=q;}
    }
    std::cout<<"Best support: "<<best<<"\n";
    cv::Mat result=image.clone();
    if(best<0.65) {
        std::cerr<<"Plate not found: adjust ROI, Canny thresholds, or plate size limits.\n";
        cv::putText(result,"PLATE NOT FOUND",{30,60},cv::FONT_HERSHEY_SIMPLEX,1.5,{0,0,255},3);
    } else {
        std::vector<cv::Point> polygon;
        for(auto p:corners) {p+=cv::Point2f(float(roi.x),float(roi.y));polygon.emplace_back(cvRound(p.x),cvRound(p.y));std::cout<<p<<"\n";}
        cv::polylines(result,std::vector<std::vector<cv::Point>>{polygon},true,{0,0,255},4,cv::LINE_AA);
        std::cout<<"Boundary score: "<<best<<"\n";
    }
    save("_03_plate.png",result);
    if(gui) showStep("3 - QR plate boundary (red)",result);
    cv::Point2f plateCenter;
    std::array<QrRoiResult,3> qrRois{};
    cv::Mat roiVisualization=result.clone();
    if(best>=0.65) {
        // Detection returns cyclic vertices in arbitrary starting order.
        // At the supported +/-8 degree rotation, sorting rows is unambiguous.
        auto ordered=corners;
        for(auto& p:ordered) p+=cv::Point2f(float(roi.x),float(roi.y));
        std::sort(ordered.begin(),ordered.end(),[](const cv::Point2f& a,const cv::Point2f& b){return a.y<b.y;});
        if(ordered[0].x>ordered[1].x) std::swap(ordered[0],ordered[1]);
        if(ordered[2].x<ordered[3].x) std::swap(ordered[2],ordered[3]);
        const std::array<cv::Point2f,4> model{{{-.5f,-.5f},{.5f,-.5f},{.5f,.5f},{-.5f,.5f}}};
        cv::Mat modelToImage=cv::getPerspectiveTransform(model.data(),ordered.data());
        std::vector<cv::Point2f> centerModel{{0,0}},centerImage;
        cv::perspectiveTransform(centerModel,centerImage,modelToImage);
        plateCenter=centerImage[0];
        cv::drawMarker(roiVisualization,plateCenter,{0,255,255},cv::MARKER_CROSS,24,2);
        const double plateWidth=(cv::norm(ordered[1]-ordered[0])+cv::norm(ordered[2]-ordered[3]))*.5;
        const double plateHeight=(cv::norm(ordered[3]-ordered[0])+cv::norm(ordered[2]-ordered[1]))*.5;
        if(!std::isfinite(options.qrWidthRatio)||!std::isfinite(options.qrHeightRatio)||
           options.qrWidthRatio<=0||options.qrHeightRatio<=0||
           options.qrWidthRatio>2||options.qrHeightRatio>2)
            CV_Error(cv::Error::StsBadArg,"QR ROI ratios must be positive and at most 2");
        cv::Size size(std::max(2,cvRound(plateWidth*options.qrWidthRatio)),
                      std::max(2,cvRound(plateHeight*options.qrHeightRatio)));
        const std::array<cv::Point2f,4> target{{{0,0},{float(size.width-1),0},
                  {float(size.width-1),float(size.height-1)},{0,float(size.height-1)}}};
        const char* names[]={"L","C","R"};
        cv::Mat montage(size.height,size.width*3,CV_8UC3,cv::Scalar(255,255,255));
        for(int i=0;i<3;++i) {
            const float x=options.qrCenterOffsetsX[i],y=options.qrCenterOffsetY;
            if(!std::isfinite(x)||!std::isfinite(y)||std::abs(x)>2||std::abs(y)>2)
                CV_Error(cv::Error::StsBadArg,"Invalid QR ROI center offset");
            float dx=options.qrWidthRatio*.5f,dy=options.qrHeightRatio*.5f;
            std::vector<cv::Point2f> local{{x-dx,y-dy},{x+dx,y-dy},{x+dx,y+dy},{x-dx,y+dy},{x,y}}, projected;
            cv::perspectiveTransform(local,projected,modelToImage);
            auto& qr=qrRois[i]; qr.center=projected[4]; qr.valid=true;
            std::vector<cv::Point> polygon;
            for(int k=0;k<4;++k) {
                qr.corners[k]=projected[k]; polygon.emplace_back(cvRound(projected[k].x),cvRound(projected[k].y));
                if(!std::isfinite(projected[k].x)||!std::isfinite(projected[k].y)||
                   projected[k].x<0||projected[k].y<0||projected[k].x>image.cols-1||projected[k].y>image.rows-1) qr.valid=false;
            }
            if(!qr.valid) {std::cerr<<"QR ROI "<<names[i]<<" extends outside the image; crop skipped\n";continue;}
            cv::Mat transform=cv::getPerspectiveTransform(qr.corners.data(),target.data());
            cv::warpPerspective(image,qr.image,transform,size,cv::INTER_LINEAR);
            save(std::string("_05_qr_")+names[i]+".png",qr.image);
            qr.image.copyTo(montage(cv::Rect(i*size.width,0,size.width,size.height)));
            cv::polylines(roiVisualization,std::vector<std::vector<cv::Point>>{polygon},true,{255,100,0},3,cv::LINE_AA);
            cv::putText(roiVisualization,names[i],polygon[0],cv::FONT_HERSHEY_SIMPLEX,.8,{255,100,0},2);
        }
        save("_04_qr_rois.png",roiVisualization);
        save("_05_qr_crops.png",montage);
        if(gui) {
            showStep("4 - Plate center and QR ROIs",roiVisualization);
            showStep("5 - QR crops: L / C / R",montage);
        }
    }
    if(best>=0.65) for(int i=0;i<3;++i)
        if(qrRois[i].valid) analyzeQrFeatures(image,options,qrRois[i],i);
    if(detection) {
        detection->found = best>=0.65;
        detection->score = best;
        detection->searchRoi = roi;
        detection->edges = edges;
        detection->hough = hough;
        detection->visualization = result;
        detection->center = plateCenter;
        detection->qrRois = qrRois;
        detection->roiVisualization = roiVisualization;
        if(detection->found) for(int i=0;i<4;++i)
            detection->corners[i]=corners[i]+cv::Point2f(float(roi.x),float(roi.y));
    }
    return Alignment_t{};
}
