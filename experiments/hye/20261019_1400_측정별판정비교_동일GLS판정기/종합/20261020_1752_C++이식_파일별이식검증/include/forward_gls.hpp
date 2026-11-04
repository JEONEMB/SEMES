#pragma once
#include "native_lapack.hpp"
namespace semes {
struct ForwardDesign{cv::Mat value,dy,dt;};
inline ForwardDesign forward_design(const cv::Mat& q,int degree){
    if(q.cols!=2||degree<1||degree>8)throw std::runtime_error("Forward polynomial requires two commands and degree1..8");int count=(degree+1)*(degree+2)/2-1;ForwardDesign r{cv::Mat(q.rows,count,CV_64F),cv::Mat::zeros(q.rows,count,CV_64F),cv::Mat::zeros(q.rows,count,CV_64F)};
    for(int i=0;i<q.rows;++i){double y=q.at<double>(i,0),t=q.at<double>(i,1);int k=0;for(int power=1;power<=degree;++power)for(int b=0;b<=power;++b){int a=power-b;r.value.at<double>(i,k)=std::pow(y,a)*std::pow(t,b);if(a)r.dy.at<double>(i,k)=a*std::pow(y,a-1)*std::pow(t,b);if(b)r.dt.at<double>(i,k)=b*std::pow(y,a)*std::pow(t,b-1);++k;}}
    return r;
}
class ForwardGLS {
public:
    int degree=2;double shrink=.1;cv::Mat origin,coef,stddev,projector,gram,initial;cv::Vec2d scale,lower,upper;
    ForwardGLS(int d=2,double s=.1):degree(d),shrink(s){if(d<1||d>8||s<=0||s>1)throw std::runtime_error("Invalid forward GLS settings");}
    ForwardGLS& fit(const cv::Mat& X,const cv::Mat& commands,const cv::Mat& supplied_weights={}){
        if(X.rows!=commands.rows||commands.cols!=2||!cv::checkRange(X)||!cv::checkRange(commands))throw std::runtime_error("Finite aligned training rows required");int n=X.rows;cv::Mat weights=supplied_weights.empty()?cv::Mat::ones(n,1,CV_64F):supplied_weights.clone();if(weights.rows!=n||weights.cols!=1||!cv::checkRange(weights))throw std::runtime_error("Training weight shape mismatch");double mean=cv::sum(weights)[0]/n;weights/=mean;
        cv::Mat q=commands.clone();for(int k=0;k<2;++k){double minimum,maximum;cv::minMaxLoc(cv::abs(commands.col(k)),nullptr,&maximum);scale[k]=std::max(maximum,.05);q.col(k)/=scale[k];cv::minMaxLoc(q.col(k),&minimum,&maximum);lower[k]=minimum-.25;upper[k]=maximum+.25;}
        auto p=forward_design(q,degree).value;cv::Mat A;cv::hconcat(cv::Mat::ones(n,1,CV_64F),p,A);auto weighted_A=A.clone(),weighted_X=X.clone();for(int i=0;i<n;++i){double w=std::sqrt(weights.at<double>(i));weighted_A.row(i)*=w;weighted_X.row(i)*=w;}auto coefficients=lapack::least_squares(weighted_A,weighted_X,1e-10);origin=coefficients.row(0).clone();coef=coefficients.rowRange(1,coefficients.rows).clone();cv::Mat residual=X-A*coefficients;for(int i=0;i<n;++i)residual.row(i)*=std::sqrt(weights.at<double>(i));stddev=cv::Mat(1,X.cols,CV_64F);
        for(int k=0;k<X.cols;++k){std::vector<double> v(n);for(int i=0;i<n;++i)v[i]=residual.at<double>(i,k)*residual.at<double>(i,k);stddev.at<double>(k)=std::max(std::sqrt(lapack::pairwise_sum(v.data(),v.size())/n),1e-6);}
        cv::Mat R=residual.clone(),V=coef.t();for(int k=0;k<X.cols;++k){R.col(k)/=stddev.at<double>(k);V.row(k)/=stddev.at<double>(k);}double a=shrink,b=(1-a)/n;cv::Mat system=a*cv::Mat::eye(n,n,CV_64F)+b*R*R.t();cv::Mat Q=(V-b*R.t()*lapack::solve(system,R*V))/a;projector=Q.clone();for(int k=0;k<X.cols;++k)projector.row(k)/=stddev.at<double>(k);gram=coef*projector;gram=(gram+gram.t())*.5;initial=lapack::pinv(gram(cv::Rect(0,0,2,2)));return *this;
    }
    cv::Mat predict(const cv::Mat& X)const{
        cv::Mat centered=X.clone();for(int i=0;i<X.rows;++i)centered.row(i)-=origin;cv::Mat h=centered*projector,q=h.colRange(0,2)*initial;
        if(degree>1){for(int i=0;i<q.rows;++i)for(int k=0;k<2;++k)q.at<double>(i,k)=std::clamp(q.at<double>(i,k),lower[k],upper[k]);
            for(int iteration=0;iteration<35;++iteration){auto design=forward_design(q,degree);cv::Mat residual=h-design.value*gram,yg=design.dy*gram,tg=design.dt*gram;double maximum_step=0;
                for(int i=0;i<q.rows;++i){double g0=design.dy.row(i).dot(residual.row(i)),g1=design.dt.row(i).dot(residual.row(i)),a=yg.row(i).dot(design.dy.row(i)),b=yg.row(i).dot(design.dt.row(i)),c=tg.row(i).dot(design.dt.row(i));double damp=1e-8*(a+c+1);a+=damp;c+=damp;double determinant=std::max(a*c-b*b,1e-20);cv::Vec2d step(std::clamp((c*g0-b*g1)/determinant,-.25,.25),std::clamp((a*g1-b*g0)/determinant,-.25,.25));maximum_step=std::max({maximum_step,std::abs(step[0]),std::abs(step[1])});cv::Mat pg=design.value.row(i)*gram-2*h.row(i);double old=design.value.row(i).dot(pg);
                    for(double factor:{1.,.5,.25,.125,.0625}){cv::Mat trial=(cv::Mat_<double>(1,2)<<std::clamp(q.at<double>(i,0)+factor*step[0],lower[0],upper[0]),std::clamp(q.at<double>(i,1)+factor*step[1],lower[1],upper[1]));auto tp=forward_design(trial,degree).value;cv::Mat cost=tp*gram-2*h.row(i);if(tp.dot(cost)<old){trial.copyTo(q.row(i));break;}}
                }if(maximum_step<1e-8)break;
            }
        }
        for(int k=0;k<2;++k)q.col(k)*=scale[k];return q;
    }
    Json json()const{Json j=Json::dict();j["degree"]=degree;j["shrink"]=shrink;for(auto pair:{std::pair<const char*,cv::Mat>{"origin",origin},{"coef",coef},{"std",stddev},{"projector",projector},{"gram",gram},{"initial",initial}})j[pair.first]=jsonmatrix(pair.second);for(auto pair:{std::pair<const char*,cv::Vec2d>{"scale",scale},{"lower",lower},{"upper",upper}}){Json a=Json::list();a.push(pair.second[0]);a.push(pair.second[1]);j[pair.first]=a;}return j;}
    static ForwardGLS from_json(const Json& j){ForwardGLS f(int(j.at("degree").num()),j.at("shrink").num());f.origin=matrix(j.at("origin"));f.coef=matrix(j.at("coef"));f.stddev=matrix(j.at("std"));f.projector=matrix(j.at("projector"));f.gram=matrix(j.at("gram"));f.initial=matrix(j.at("initial"));for(int k=0;k<2;++k){f.scale[k]=j.at("scale").at(k).num();f.lower[k]=j.at("lower").at(k).num();f.upper[k]=j.at("upper").at(k).num();}return f;}
};
}
