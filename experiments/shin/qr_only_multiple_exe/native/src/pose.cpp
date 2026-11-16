#include "qr_pose.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace qr {
cv::Mat read_frame(const std::string& path) {
    std::ifstream source(path,std::ios::binary);
    if (!source) throw std::runtime_error("Cannot read image file: "+path);
    const std::vector<unsigned char> data((std::istreambuf_iterator<char>(source)),{});
    if (data.empty()) throw std::runtime_error("Empty image file: "+path);
    const auto frame=cv::imdecode(data,cv::IMREAD_GRAYSCALE);check_frame(frame);return frame;
}

cv::FileStorage open_model(const std::string& path) {
    cv::FileStorage file(path,cv::FileStorage::READ);
    if (!file.isOpened()) throw std::runtime_error("Cannot open " + path);
    return file;
}

cv::Mat matrix(const cv::FileStorage& file,const std::string& key) {
    cv::Mat out; file[key] >> out;
    if (out.empty() || !cv::checkRange(out)) throw std::runtime_error("Bad model field: "+key);
    return out;
}

void check_frame(const cv::Mat& gray) {
    if (gray.size()!=cv::Size(3840,2160) || gray.type()!=CV_8UC1)
        throw std::runtime_error("Expected rectified 3840x2160 grayscale image");
}

std::vector<cv::Point> centers(const cv::Mat& coordinates) {
    if (coordinates.rows!=3 || coordinates.cols!=2) throw std::runtime_error("Expected three QR centers");
    cv::Mat xy; coordinates.convertTo(xy,CV_32S);
    std::vector<cv::Point> out;
    for (int i=0;i<3;++i) out.emplace_back(xy.at<int>(i,0),xy.at<int>(i,1));
    return out;
}

cv::Point2d match(const cv::Mat& frame,const cv::Mat& patch,cv::Point center,
                  int sx,int sy,double minimum,bool subpixel) {
    auto roi=cv::Rect(center.x-patch.cols/2-sx,center.y-patch.rows/2-sy,patch.cols+2*sx,patch.rows+2*sy);
    cv::Mat scores;
    cv::matchTemplate(frame(roi).clone(),patch,scores,cv::TM_CCOEFF_NORMED);
    cv::Point p; double confidence;
    cv::minMaxLoc(scores,nullptr,&confidence,nullptr,&p);
    if (!(confidence>=minimum) || p.x<=0 || p.x>=2*sx || p.y<=0 || p.y>=2*sy)
        throw std::runtime_error("QR correlation peak outside reliable range");
    if (!subpixel) return {double(p.x-sx),double(p.y-sy)};
    const double mid=scores.at<float>(p);
    const auto offset=[mid](double l,double r){
        const double curvature=l+r-2*mid;
        if (curvature>=-1e-9) throw std::runtime_error("Ambiguous QR peak");
        return .5*(l-r)/curvature;
    };
    return {p.x-sx+offset(scores.at<float>(p.y,p.x-1),scores.at<float>(p.y,p.x+1)),
            p.y-sy+offset(scores.at<float>(p.y-1,p.x),scores.at<float>(p.y+1,p.x))};
}

FeaturePose::FeaturePose(const std::string& path) {
    auto f=open_model(path);
    coarse_=matrix(f,"ref_sparse_1"); zero_=matrix(f,"ref_zero");
    anchors_=matrix(f,"anchors"); beta_=matrix(f,"beta"); rotate_=matrix(f,"rotate");
    horizontal_=matrix(f,"horizontal_fit"); centers_=centers(matrix(f,"ref_centers"));
    width_=double(f["width"]);wide_=double(f["wide"]);anchored_=int(f["anchored"])!=0;
    const std::string kind=f["kind"];
    if (kind=="track") weight_={{.5,.1,.5,.5}};
    else if (kind=="shape") weight_={{1.,.2,1.,0.}};
    else if (kind=="soft_shape") weight_={{1.,.03,.1,0.}};
    else if (kind=="motion") weight_={{1.,.08,0.,0.}};
    else throw std::runtime_error("Unknown QR feature embedding");
    if (!(width_>0.) || beta_.rows!=anchors_.rows || beta_.cols!=2 || rotate_.total()!=4 || horizontal_.total()!=3)
        throw std::runtime_error("Invalid feature regression matrices");
    int point_total=0;
    for (int i=0;i<3;++i) {
        const auto suffix=std::to_string(i);
        patches_[i]=matrix(f,"patch"+suffix); inverse_[i]=matrix(f,"inverse"+suffix);
        polynomial_[i]=matrix(f,"poly"+suffix);
        const cv::Mat p=matrix(f,"points"+suffix);
        if (p.type()!=CV_32F || p.cols!=2 || inverse_[i].size()!=cv::Size(2,2) || polynomial_[i].cols!=p.rows)
            throw std::runtime_error("Invalid QR track geometry");
        for (int j=0;j<p.rows;++j) points_[i].emplace_back(p.at<float>(j,0),p.at<float>(j,1));
        point_total+=p.rows;
    }
    if (zero_.rows!=point_total || zero_.cols!=2) throw std::runtime_error("Invalid QR reference residuals");
    for (int i=0;i<anchors_.rows;++i) {
        const double n=anchors_.row(i).dot(anchors_.row(i));
        anchor_sq_.push_back(n);anchor_zero_.push_back(kernel(n));
    }
}

double FeaturePose::kernel(double d) const {
    return (1.-wide_)*std::exp(-.5*d/(width_*width_))+wide_*std::exp(-.5*d/36.);
}

static cv::Mat track(const cv::Mat& patch,const cv::Mat& image,const std::vector<cv::Point2f>& points,cv::Point shift) {
    std::vector<cv::Point2f> found;
    std::vector<unsigned char> keep;
    cv::calcOpticalFlowPyrLK(patch,image,points,found,keep,cv::noArray(),{19,19},1,
                            {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,35,.001});
    const int n=static_cast<int>(points.size());
    cv::Mat design(n,3,CV_64F), delta(n,2,CV_64F);
    cv::Point2d mean(0,0);
    for (const auto p:points) mean+=cv::Point2d(p);
    mean*=1./n;
    for(int j=0;j<n;++j) {
        const auto move=found[j]-points[j];
        design.at<double>(j,0)=1.;design.at<double>(j,1)=points[j].x-mean.x;design.at<double>(j,2)=points[j].y-mean.y;
        delta.at<double>(j,0)=move.x;delta.at<double>(j,1)=move.y;
        keep[j]=keep[j] && std::isfinite(move.x) && std::isfinite(move.y) && cv::norm(move)<=8.;
    }
    cv::Mat affine;
    for (int pass=0;pass<2;++pass) {
        cv::Mat a,b,coeff;
        for(int j=0;j<n;++j) if (keep[j]) {a.push_back(design.row(j));b.push_back(delta.row(j));}
        if (a.rows<.75*n || !cv::solve(a,b,coeff,cv::DECOMP_SVD)) throw std::runtime_error("Unreliable QR tracking");
        affine=design*coeff;
        for(int j=0;j<n;++j) keep[j]=keep[j] && cv::norm(delta.row(j),affine.row(j))<=1.;
    }
    if (std::count(keep.begin(),keep.end(),1)<.75*n) throw std::runtime_error("Unreliable QR tracking after affine filter");
    for(int j=0;j<n;++j) {
        if (!keep[j]) affine.row(j).copyTo(delta.row(j));
        delta.at<double>(j,0)+=shift.x;delta.at<double>(j,1)+=shift.y;
    }
    return delta;
}

Pose FeaturePose::predict(const cv::Mat& frame) const {
    check_frame(frame);
    const cv::Point shift=match(frame,coarse_,centers_[1],240,45,.85);
    cv::Mat raw; std::array<int,4> boundaries{{0,0,0,0}};
    for(int i=0;i<3;++i) {
        const auto p=centers_[i]+shift;
        const cv::Mat image=frame(cv::Rect(p.x-101,p.y-101,203,203)).clone();
        raw.push_back(cv::Mat(track(patches_[i],image,points_[i],shift)*inverse_[i]));
        boundaries[i+1]=raw.rows;
    }
    raw-=zero_;
    const cv::Scalar avg=cv::mean(raw.reshape(2));
    std::vector<double> shape;
    for(int i=0;i<3;++i) {
        cv::Mat field=polynomial_[i]*raw.rowRange(boundaries[i],boundaries[i+1]);
        field.at<double>(0,0)-=avg[0];field.at<double>(0,1)-=avg[1];
        shape.insert(shape.end(),field.ptr<double>(),field.ptr<double>()+field.total());
    }
    const double norm=std::sqrt(double(shape.size()));
    std::vector<double> v{std::sqrt(weight_[0])*avg[0]/10.,std::sqrt(weight_[1])*avg[1]/.12};
    if (weight_[2]>0.) for (auto z:shape) v.push_back(std::sqrt(weight_[2])*z/(.05*norm));
    if (weight_[3]>0.) for (int i=0;i<raw.rows;++i) for (int j=0;j<2;++j)
        v.push_back(std::sqrt(weight_[3])*(raw.at<double>(i,j)-avg[j])/(.12*norm));
    if (v.size()!=static_cast<size_t>(anchors_.cols)) throw std::runtime_error("Feature model dimension mismatch");
    const cv::Mat vector(1,static_cast<int>(v.size()),CV_64F,v.data());
    const double sq=vector.dot(vector), base=kernel(sq);
    Pose z(0,0);
    for (int i=0;i<anchors_.rows;++i) {
        double k=kernel(std::max(0.,anchor_sq_[i]+sq-2*anchors_.row(i).dot(vector)));
        if (anchored_) k-=base*anchor_zero_[i];
        z[0]+=k*beta_.at<double>(i,0);z[1]+=k*beta_.at<double>(i,1);
    }
    const auto h=horizontal_.ptr<double>();
    z[0]+=h[0]+h[1]*avg[0]+h[2]*avg[0]*avg[0]/100.;
    return {(.045)*(z[0]*rotate_.at<double>(0,0)+z[1]*rotate_.at<double>(0,1)),
            (.01)*(z[0]*rotate_.at<double>(1,0)+z[1]*rotate_.at<double>(1,1))};
}

Geometry::Geometry(const cv::Mat& reference,const std::vector<cv::Point>& centers):centers_(centers) {
    check_frame(reference);
    for (int i=0;i<3;++i) templates_[i]=reference(cv::Rect(centers_[i].x-79,centers_[i].y-86,159,173)).clone();
    zero_=measure(reference);
    affine_zero_=measure_affine(reference);
}

Geometry::Geometry(const cv::FileStorage& file,const std::vector<cv::Point>& centers):centers_(centers) {
    for (int i=0;i<3;++i) templates_[i]=matrix(file,"template"+std::to_string(i));
}

cv::Point Geometry::seed(const cv::Mat& frame) const {
    return match(frame,templates_[1],centers_[1],240,45,.8,false);
}

AffineFeatures Geometry::measure_affine(const cv::Mat& frame) const {
    check_frame(frame);
    const auto shift=seed(frame);
    AffineFeatures terms;
    for(int i=0;i<3;++i) {
        const auto p=centers_[i]+shift;
        const cv::Mat current=frame(cv::Rect(p.x-79,p.y-86,159,173)).clone();
        cv::Mat a=cv::Mat::eye(2,3,CV_32F);
        const double score=cv::findTransformECC(templates_[i],current,a,cv::MOTION_AFFINE,
              {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,22,1e-7},cv::noArray(),5);
        if (!(score>=.9) || !cv::checkRange(a)) throw std::runtime_error("Unreliable QR affine fit");
        const double scale=a.at<float>(0,0)-1.;
        const double vscale=a.at<float>(1,1)-1.;
        terms[6*i]=scale*79+a.at<float>(0,1)*86+a.at<float>(0,2)+shift.x;
        terms[6*i+1]=double(a.at<float>(1,0))*79+vscale*86+a.at<float>(1,2)+shift.y;
        terms[6*i+2]=scale*160;
        terms[6*i+3]=double(a.at<float>(1,0))*160;
        terms[6*i+4]=double(a.at<float>(0,1))*174;
        terms[6*i+5]=vscale*174;
    }
    return terms;
}

Features Geometry::measure(const cv::Mat& frame) const {
    const auto t=measure_affine(frame);
    return {(t[0]+t[6]+t[12])/3.,t[0]-2*t[6]+t[12],t[14]-t[2],t[17]-t[5]};
}

Features Geometry::predict(const cv::Mat& frame) const {return measure(frame)-zero_;}
AffineFeatures Geometry::affine(const cv::Mat& frame) const {return measure_affine(frame)-affine_zero_;}

Cancellation::Cancellation(const std::string& path,const cv::Mat& reference):zero_dx_(0.) {
    const auto f=open_model(path);
    centers_=centers(matrix(f,"centers"));coefficients_=matrix(f,"coefficients");
    if (coefficients_.size()!=cv::Size(2,4)) throw std::runtime_error("Invalid Rig geometry model");
    geometry_=Geometry(reference,centers_);
    const std::vector<cv::Point> horizontal_centers{{1650,1130},{1907,1130},{2166,1132}};
    centers_=horizontal_centers;
    for (int i=0;i<3;++i) patches_[i]=reference(cv::Rect(centers_[i].x-50,centers_[i].y-50,101,101)).clone();
    zero_dx_=horizontal(reference);
}

Pose Cancellation::predict(const cv::Mat& frame) const {
    const auto x=geometry_.predict(frame);
    const cv::Mat y=cv::Mat(x).t()*coefficients_;
    return {y.at<double>(0),y.at<double>(1)};
}

double Cancellation::horizontal(const cv::Mat& frame) const {
    check_frame(frame);std::array<double,3> x;
    for(int i=0;i<3;++i) x[i]=match(frame,patches_[i],centers_[i],140,22,.9).x;
    std::sort(x.begin(),x.end());return x[1]-zero_dx_;
}
}
