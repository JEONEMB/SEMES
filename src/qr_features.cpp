#include "alignment.hpp"
#include "third_party/qrcodegen.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
void display(const std::string& title,const cv::Mat& image) {
    cv::namedWindow(title,cv::WINDOW_NORMAL); cv::imshow(title,image);
    cv::resizeWindow(title,1000,700); cv::waitKey(0); cv::destroyWindow(title);
}
bool regenerate(const std::string& payload,const cv::Mat& observed,cv::Mat& best,double& mismatch) {
    using namespace qrcodegen;
    int n=observed.rows,version=(n-17)/4;
    if(observed.type()!=CV_8UC1||observed.cols!=n||version<1||version>40||n!=17+4*version) return false;
    std::vector<std::vector<QrSegment>> modes{QrSegment::makeSegments(payload.c_str()),
        {QrSegment::makeBytes(std::vector<std::uint8_t>(payload.begin(),payload.end()))}};
    int bestErrors=n*n+1,secondErrors=n*n+1;
    // Content alone is insufficient: enumerate ECC, mask and common encoding modes.
    // Mixed segmentation/ECI not covered here: unmatched symbols fail explicitly.
    for(const auto& segments:modes) for(int ec=0;ec<4;++ec) for(int mask=0;mask<8;++mask) {
        try {
            auto code=QrCode::encodeSegments(segments,static_cast<QrCode::Ecc>(ec),version,version,mask,false);
            cv::Mat candidate(n,n,CV_8U); int errors=0;
            for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
                candidate.at<uchar>(y,x)=code.getModule(x,y)?0:255;
                errors+=(candidate.at<uchar>(y,x)<128)!=(observed.at<uchar>(y,x)<128);
            }
            if(!best.empty() && cv::countNonZero(candidate!=best)==0) continue;
            if(errors<bestErrors) {secondErrors=bestErrors;bestErrors=errors;best=candidate;}
            else secondErrors=std::min(secondErrors,errors);
        } catch(const qrcodegen::data_too_long&) { /* Try another encoding/ECC. */ }
    }
    mismatch=double(bestErrors)/(n*n);
    return !best.empty() && mismatch<=0.03 && double(secondErrors-bestErrors)/(n*n)>0.02;
}
cv::Point2f project(const cv::Point2f& p,const cv::Mat& h) {
    double z=h.at<double>(2,0)*p.x+h.at<double>(2,1)*p.y+h.at<double>(2,2);
    return {float((h.at<double>(0,0)*p.x+h.at<double>(0,1)*p.y+h.at<double>(0,2))/z),
            float((h.at<double>(1,0)*p.x+h.at<double>(1,1)*p.y+h.at<double>(1,2))/z)};
}
bool inBounds(const cv::Point2f& p,const cv::Mat& image,int margin=0) {
    return std::isfinite(p.x)&&std::isfinite(p.y)&&p.x>=margin&&p.y>=margin&&
           p.x<image.cols-margin&&p.y<image.rows-margin;
}
}

void analyzeQrFeatures(const cv::Mat& source,const AlignmentOptions& options,QrRoiResult& qr,int index) {
    const std::string label=std::string(1,"LCR"[index]);
    auto save=[&](const std::string& suffix,const cv::Mat& mat) {
        if(!options.outputPrefix.empty()&&!cv::imwrite(options.outputPrefix+suffix+"_"+label+".png",mat))
            CV_Error(cv::Error::StsError,"Cannot save QR feature visualization");
    };
    auto stage=[&](const std::string& suffix,const std::string& title,const cv::Mat& mat) {
        save(suffix,mat); if(options.visualize) display(title+" / "+label,mat);
    };
    cv::Mat points,observed,decodeGray;
    cv::cvtColor(qr.image,decodeGray,cv::COLOR_BGR2GRAY);
    cv::QRCodeDetector detector;
    cv::Point2f decodeOffset;
    // Decode from original crop first. Upscale retry changes only initialization, never final measurement.
    double decodeScale=1;
    qr.payload=detector.detectAndDecode(decodeGray,points,observed);
    if(qr.payload.empty()) {
        cv::Mat up; cv::resize(decodeGray,up,{},2,2,cv::INTER_CUBIC);
        points.release(); observed.release();
        qr.payload=detector.detectAndDecode(up,points,observed);decodeScale=2;
    }
    if(qr.payload.empty()) {
        cv::Mat retry; cv::resize(decodeGray,retry,{},3,3,cv::INTER_LINEAR);
        cv::copyMakeBorder(retry,retry,40,40,40,40,cv::BORDER_CONSTANT,cv::Scalar(255));
        points.release();observed.release();
        qr.payload=detector.detectAndDecode(retry,points,observed);
        if(!qr.payload.empty()) {decodeScale=3;decodeOffset={-40.f/3,-40.f/3};}
    }
    if(qr.payload.empty()) {
        // Dark plate margins can hide the QR quiet zone. Locate the bright printed label,
        // retry with an added white border, and map all coordinates back to the crop.
        for(int threshold=110;threshold<=210 && qr.payload.empty();threshold+=20) {
            cv::Mat binary; cv::threshold(decodeGray,binary,threshold,255,cv::THRESH_BINARY);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
            for(const auto& contour:contours) {
                cv::Rect box=cv::boundingRect(contour);
                if(box.width<decodeGray.cols*.65 || box.height<decodeGray.rows*.65 ||
                   box.width>decodeGray.cols*.98 || box.height>decodeGray.rows*.98 ||
                   !box.contains(cv::Point(decodeGray.cols/2,decodeGray.rows/2))) continue;
                cv::Mat retry; cv::resize(decodeGray(box),retry,{},2,2,cv::INTER_CUBIC);
                cv::copyMakeBorder(retry,retry,32,32,32,32,cv::BORDER_CONSTANT,cv::Scalar(255));
                points.release();observed.release();
                qr.payload=detector.detectAndDecode(retry,points,observed);
                if(!qr.payload.empty()) {decodeScale=2;decodeOffset={float(box.x-16),float(box.y-16)};break;}
            }
        }
    }
    // Decode-only fallback. Never blur source or use this buffer for cornerSubPix.
    cv::Mat gaussianDecodeInput;
    if(qr.payload.empty()) for(double sigma:{0.6,0.9}) {
        cv::Mat smoothed;cv::GaussianBlur(decodeGray,smoothed,{3,3},sigma);
        for(int scale:{1,3}) {
            cv::Mat retry;
            if(scale==1) retry=smoothed;
            else {
                cv::resize(smoothed,retry,{},3,3,cv::INTER_LINEAR);
                cv::copyMakeBorder(retry,retry,40,40,40,40,cv::BORDER_CONSTANT,cv::Scalar(255));
            }
            points.release();observed.release();
            qr.payload=detector.detectAndDecode(retry,points,observed);
            if(!qr.payload.empty()) {
                decodeScale=scale;
                decodeOffset=scale==1?cv::Point2f():cv::Point2f(-40.f/3,-40.f/3);
                qr.decodeUsedGaussian=true;qr.decodeGaussianSigma=sigma;
                gaussianDecodeInput=retry.clone();break;
            }
        }
        if(!qr.payload.empty()) break;
    }
    qr.decoded=!qr.payload.empty();
    qr.decodedVisualization=qr.image.clone();
    std::vector<cv::Point2f> decodedCorners;
    if(points.total()==4) {
        cv::Mat flat=points.reshape(2,4);
        for(int i=0;i<4;++i) decodedCorners.push_back(flat.at<cv::Point2f>(i)/float(decodeScale)+decodeOffset);
        for(int i=0;i<4;++i) cv::line(qr.decodedVisualization,decodedCorners[i],decodedCorners[(i+1)%4],{0,255,0},1);
    }
    cv::putText(qr.decodedVisualization,qr.decoded?qr.payload:"DECODE FAILED",{4,18},cv::FONT_HERSHEY_SIMPLEX,.45,{0,0,255},1);
    if(qr.decodeUsedGaussian) {
        cv::putText(qr.decodedVisualization,"decode-only Gaussian",{4,qr.image.rows-6},cv::FONT_HERSHEY_SIMPLEX,.4,{0,0,255},1);
        stage("_06_gaussian_input","6a - Decode-only Gaussian input (not used for subpixel)",gaussianDecodeInput);
    }
    stage("_06_decode","6 - QR decode",qr.decodedVisualization);
    if(!qr.decoded||decodedCorners.size()!=4||observed.empty()) {qr.featureStatus="decode_failed";return;}
    qr.moduleCount=observed.rows;
    qr.modelMatched=regenerate(qr.payload,observed,qr.idealModules,qr.modelMismatch);
    if(!qr.modelMatched) {qr.featureStatus="model_mismatch_or_ambiguous";return;}

    // Crop pixels -> original image pixels. Keep subpixel refinement out of resampled crop.
    std::array<cv::Point2f,4> cropBounds{{{0,0},{float(qr.image.cols-1),0},
        {float(qr.image.cols-1),float(qr.image.rows-1)},{0,float(qr.image.rows-1)}}};
    cv::Mat cropToSource=cv::getPerspectiveTransform(cropBounds.data(),qr.corners.data());
    std::vector<cv::Point2f> sourceCorners;
    cv::perspectiveTransform(decodedCorners,sourceCorners,cropToSource);
    float n=float(qr.moduleCount);
    std::array<cv::Point2f,4> modelBounds{{{0,0},{n,0},{n,n},{0,n}}};
    cv::Mat h=cv::getPerspectiveTransform(modelBounds.data(),sourceCorners.data());
    cv::Mat sourceToCrop=cropToSource.inv(),gray,quality;
    cv::cvtColor(source,gray,cv::COLOR_BGR2GRAY);
    cv::cornerMinEigenVal(gray,quality,3,3);
    double pitch=0;
    for(int i=0;i<4;++i) pitch+=cv::norm(sourceCorners[i]-sourceCorners[(i+1)%4])/(4*n);
    int radius=std::max(1,int(std::floor(pitch*.30)));
    if(pitch<4) {qr.featureStatus="insufficient_pixels_per_module";return;}
    cv::Mat ideal; cv::resize(qr.idealModules,ideal,{},14,14,cv::INTER_NEAREST);
    cv::cvtColor(ideal,ideal,cv::COLOR_GRAY2BGR);
    qr.candidateVisualization=qr.image.clone();
    for(int row=1;row<qr.moduleCount;++row) for(int col=1;col<qr.moduleCount;++col) {
        auto black=[&](int y,int x){return qr.idealModules.at<uchar>(y,x)<128;};
        // Only isolated checkerboard-type intersections; do not treat all grid vertices as corners.
        bool a=black(row-1,col-1),b=black(row-1,col),c=black(row,col-1),d=black(row,col);
        if(!(a==d&&b==c&&a!=b)) continue;
        QrFeature f; f.id=row*(qr.moduleCount+1)+col; f.model={float(col),float(row)};
        f.initial=project(f.model,h); f.refined=f.initial;
        qr.features.push_back(f);
        auto p=project(f.initial,sourceToCrop);
        cv::circle(qr.candidateVisualization,p,2,{255,255,0},1,cv::LINE_AA);
        cv::circle(ideal,{col*14,row*14},3,{0,0,255},1);
    }
    stage("_07_model","7 - Verified ideal QR and X-corner candidates",ideal);
    stage("_08_candidates","8 - Projected candidates (cyan)",qr.candidateVisualization);
    for(auto& f:qr.features) {
        if(!inBounds(f.initial,gray,radius+3)) continue;
        std::vector<cv::Point2f> point{f.initial};
        cv::cornerSubPix(gray,point,{radius,radius},{-1,-1},
            {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,40,0.001});
        f.refined=point[0]; f.shift=cv::norm(f.refined-f.initial);
        if(!inBounds(f.refined,gray,radius+3)) continue;
        f.minEigenvalue=quality.at<float>(cvRound(f.refined.y),cvRound(f.refined.x));
        // Verify the expected four-quadrant polarity on the observed original pixels.
        double blackMean=0,whiteMean=0;int nb=0,nw=0;
        for(int dy=-1;dy<=1;dy+=2) for(int dx=-1;dx<=1;dx+=2) {
            cv::Point2f sample=project(f.model+cv::Point2f(dx*.28f,dy*.28f),h)+(f.refined-f.initial);
            if(!inBounds(sample,gray,1)) continue;
            cv::Mat patch;cv::getRectSubPix(gray,{1,1},sample,patch);
            int row=int(f.model.y)+(dy<0?-1:0),col=int(f.model.x)+(dx<0?-1:0);
            if(qr.idealModules.at<uchar>(row,col)<128) {blackMean+=patch.at<uchar>(0);++nb;}
            else {whiteMean+=patch.at<uchar>(0);++nw;}
        }
        f.contrast=nb==2&&nw==2?whiteMean/nw-blackMean/nb:0;
        f.localAccepted=f.shift<=pitch*.45&&f.minEigenvalue>0.0001&&f.contrast>25;
    }
    std::vector<cv::Point2f> modelPoints,imagePoints;
    for(const auto& f:qr.features) if(f.localAccepted) {modelPoints.push_back(f.model);imagePoints.push_back(f.refined);}
    cv::Mat inliers,fit;
    double limit=std::max(.75,pitch*.15);
    if(modelPoints.size()>=8) fit=cv::findHomography(modelPoints,imagePoints,cv::RANSAC,limit,inliers);
    // Leave each point out of its own geometric prediction. Do not score a point using a fit to itself.
    int accepted=0,localIndex=0;
    for(auto& f:qr.features) if(f.localAccepted) {
        int thisIndex=localIndex++;
        if(fit.empty()||!inliers.at<uchar>(thisIndex)) continue;
        std::vector<cv::Point2f> otherModel,otherImage;
        for(size_t j=0;j<modelPoints.size();++j) if(int(j)!=thisIndex&&inliers.at<uchar>(int(j))) {
            otherModel.push_back(modelPoints[j]);otherImage.push_back(imagePoints[j]);
        }
        if(otherModel.size()<7) continue;
        cv::Mat heldOut=cv::findHomography(otherModel,otherImage,0);
        if(heldOut.empty()) continue;
        f.reprojectionError=cv::norm(f.refined-project(f.model,heldOut));
        f.accepted=std::isfinite(f.reprojectionError)&&f.reprojectionError<=limit;
        accepted+=f.accepted;
    }
    std::vector<cv::Point2f> acceptedModel;
    for(const auto& f:qr.features) if(f.accepted) acceptedModel.push_back(f.model);
    cv::Rect coverage=acceptedModel.empty()?cv::Rect():cv::boundingRect(acceptedModel);
    qr.featuresValid=accepted>=8&&coverage.width>=n*.4&&coverage.height>=n*.4;
    qr.featureStatus=qr.featuresValid?"ok":"insufficient_consistent_features";
    qr.featureVisualization=qr.image.clone();
    cv::resize(qr.featureVisualization,qr.featureVisualization,{},3,3,cv::INTER_NEAREST);
    for(const auto& f:qr.features) {
        auto initial=project(f.initial,sourceToCrop)*3,refined=project(f.refined,sourceToCrop)*3;
        cv::drawMarker(qr.featureVisualization,initial,{255,255,0},cv::MARKER_CROSS,6,1);
        cv::line(qr.featureVisualization,initial,refined,{0,255,255},1,cv::LINE_AA);
        cv::circle(qr.featureVisualization,refined,3,f.accepted?cv::Scalar(0,255,0):cv::Scalar(0,0,255),1,cv::LINE_AA);
    }
    cv::putText(qr.featureVisualization,qr.payload+" accepted "+std::to_string(accepted)+"/"+std::to_string(qr.features.size()),
        {8,22},cv::FONT_HERSHEY_SIMPLEX,.55,{255,0,0},1);
    double meanResidual=0; for(const auto& f:qr.features) if(f.accepted) meanResidual+=f.reprojectionError;
    if(accepted) meanResidual/=accepted;
    const int footerY=qr.featureVisualization.rows;
    cv::copyMakeBorder(qr.featureVisualization,qr.featureVisualization,0,55,0,0,cv::BORDER_CONSTANT,cv::Scalar(255,255,255));
    cv::putText(qr.featureVisualization,"Held-out mean residual: "+std::to_string(meanResidual)+" px",
        {8,footerY+20},cv::FONT_HERSHEY_SIMPLEX,.48,{0,0,0},1);
    cv::putText(qr.featureVisualization,"Cyan: initial  Green: accepted  Red: rejected (not a probability)",
        {8,footerY+43},cv::FONT_HERSHEY_SIMPLEX,.40,{0,0,0},1);
    stage("_09_refined","9 - Subpixel: cyan initial, green accepted, red rejected",qr.featureVisualization);
    if(!options.outputPrefix.empty()) {
        std::ofstream csv(options.outputPrefix+"_10_features_"+label+".csv");
        if(!csv) CV_Error(cv::Error::StsError,"Cannot save feature CSV");
        csv<<"id,model_x,model_y,initial_x,initial_y,refined_x,refined_y,shift_px,min_eigenvalue,contrast,heldout_residual_px,local_accepted,accepted\n";
        csv.precision(10);
        for(const auto& f:qr.features) csv<<f.id<<','<<f.model.x<<','<<f.model.y<<','<<f.initial.x<<','<<f.initial.y<<','
            <<f.refined.x<<','<<f.refined.y<<','<<f.shift<<','<<f.minEigenvalue<<','<<f.contrast<<','
            <<f.reprojectionError<<','<<f.localAccepted<<','<<f.accepted<<'\n';
    }
    std::cout<<"QR "<<label<<": "<<qr.payload<<", model mismatch="<<qr.modelMismatch
             <<", accepted="<<accepted<<"/"<<qr.features.size()<<"\n";
}
