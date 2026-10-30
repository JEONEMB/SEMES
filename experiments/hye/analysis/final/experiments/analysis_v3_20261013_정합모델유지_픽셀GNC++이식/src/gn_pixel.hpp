#pragma once
#include "gn_pixel_slow.hpp"
#include "spline.hpp"
namespace pixel_history {
    struct SplineImage {
        cv::Mat intensity,gx,gy;
    };
    class FastPixel {
        public: SlowPixel geometry;
        cv::Mat phi_x,phi_y;
        FastPixel(const fs::path& root,const fs::path& images):geometry(root,images){
            int n=geometry.width*geometry.height;
            phi_x=cv::Mat(n,8,CV_64F);
            phi_y=cv::Mat(n,8,CV_64F);
            for(int k=0; k<8; ++k){
                cv::Mat d=cv::Mat::zeros(1,8,CV_64F);
                d.at<double>(0,k)=.5;
                auto a=exact_map(d);
                d.at<double>(0,k)=-.5;
                auto b=exact_map(d);
                for(int i=0; i<n; ++i){
                    auto difference=a.at<cv::Vec2d>(i)-b.at<cv::Vec2d>(i);
                    phi_x.at<double>(i,k)=difference[0];
                    phi_y.at<double>(i,k)=difference[1];
                }
            }
        }
        cv::Mat exact_map(const cv::Mat& d)const{
            std::vector<cv::Point2f> destination;
            for(int i=0; i<4; ++i)destination.emplace_back(float(double(geometry.corners[i].x)+d.at<double>(0,2*i)),float(double(geometry.corners[i].y)+d.at<double>(0,2*i+1)));
            auto h=cv::getPerspectiveTransform(geometry.corners,destination);
            cv::Mat result;
            cv::perspectiveTransform(geometry.pixels,result,h);
            return result;
        }
        std::pair<cv::Mat,cv::Mat> linear_map(const cv::Mat& d)const{
            cv::Mat dx=phi_x*d.t(),dy=phi_y*d.t(),mx(geometry.height,geometry.width,CV_32F),my(mx.size(),CV_32F);
            for(int i=0; i<geometry.width*geometry.height; ++i){
                auto base=geometry.pixels.at<cv::Vec2d>(i);
                mx.ptr<float>()[i]=float(base[0]+dx.at<double>(i)-geometry.bx0);
                my.ptr<float>()[i]=float(base[1]+dy.at<double>(i)-geometry.by0);
            }
            return {
                mx,my
            };
        }
        SplineImage read_image(const std::string& name)const{
            auto raw=geometry.read_image(name);
            return {
                spline_coefficients(raw.intensity),spline_coefficients(raw.gx),spline_coefficients(raw.gy)
            };
        }
        cv::Mat mean_template(int training_parity,int stride=1)const{
            cv::Mat sum=cv::Mat::zeros(geometry.height,geometry.width,CV_64F);
            int eligible=0,count=0;
            for(int i=0; i<geometry.data.x.rows; ++i)if(i%2==training_parity){
                if(eligible++%stride)continue;
                auto maps=linear_map(geometry.corner_displacements.row(i));
                auto raw=geometry.read_image(geometry.data.names[i]);
                auto sampled=spline_warp(spline_coefficients(raw.intensity),maps.first,maps.second);
                cv::Mat values;
                sampled.convertTo(values,CV_64F);
                sum+=values;
                ++count;
            }
            sum/=count;
            cv::Mat output;
            sum.convertTo(output,CV_32F);
            return output;
        }
        double check_linear()const{
            double worst=0;
            for(auto name:{
                "T-1.00.bmp","T+1.00.bmp","Y-2.00.bmp","Y+2.00.bmp","T+0.50.bmp"
            }){
                int i=int(std::find(geometry.data.names.begin(),geometry.data.names.end(),name)-geometry.data.names.begin());
                auto d=geometry.corner_displacements.row(i);
                auto exact=exact_map(d);
                cv::Mat dx=phi_x*d.t(),dy=phi_y*d.t();
                for(int k=0; k<exact.rows; ++k){
                    auto point=exact.at<cv::Vec2d>(k),base=geometry.pixels.at<cv::Vec2d>(k);
                    worst=std::max({
                        worst,std::abs(point[0]-base[0]-dx.at<double>(k)),std::abs(point[1]-base[1]-dy.at<double>(k))
                    });
                }
            }
            return worst;
        }
        Fit solve(const SplineImage& image,const cv::Mat& model,const cv::Mat& initial,const cv::Mat& target,std::ostream* trace=nullptr)const{
            int count=initial.cols,n=geometry.width*geometry.height;
            if(count!=2&&count!=5)throw std::runtime_error("Pixel parameter count must be 2 or 5");
            cv::Mat directions(8,count,CV_64F);
            for(int axis=0; axis<2; ++axis){
                cv::Mat column=model.row(axis).t();
                column.copyTo(directions.col(axis));
            }
            for(int k=2; k<count; ++k)geometry.nuisance.col(k-2).copyTo(directions.col(k));
            cv::Mat dx64=phi_x*directions,dy64=phi_y*directions,dx,dy;
            dx64.convertTo(dx,CV_32F);
            dy64.convertTo(dy,CV_32F);
            cv::Mat p=initial.clone(),target64;
            target.convertTo(target64,CV_64F);
            double gain=1,offset=0;
            auto displacement=[&](const cv::Mat& candidate){
                cv::Mat active=(cv::Mat_<double>(1,3)<<candidate.at<double>(0,0),candidate.at<double>(0,1),1);
                cv::Mat d=active*model;
                if(count>2)d+=candidate.colRange(2,count)*geometry.nuisance.colRange(0,count-2).t();
                return d;
            };
            auto evaluate=[&](const cv::Mat& candidate,double a,double b,cv::Mat& residual,cv::Mat& mx,cv::Mat& my){
                auto maps=linear_map(displacement(candidate));
                mx=maps.first;
                my=maps.second;
                auto sampled=spline_warp(image.intensity,mx,my);
                cv::Mat observed;
                sampled.convertTo(observed,CV_64F);
                residual=observed-(a*target64+b);
                return cv::norm(residual,cv::NORM_L2SQR)/residual.total();
            };
            cv::Mat residual,mx,my;
            double cost=evaluate(p,gain,offset,residual,mx,my),initial_cost=cost,lambda=1e-4;
            int iteration=0,accepted=0;
            for(iteration=0; iteration<60; ++iteration){
                auto gx=spline_warp(image.gx,mx,my),gy=spline_warp(image.gy,mx,my);
                cv::Mat jac(n,count+2,CV_64F);
                cv::parallel_for_(cv::Range(0,n),[&](const cv::Range& range){
                    for(int i=range.start; i<range.end; ++i){
                        auto* row=jac.ptr<double>(i); for(int k=0; k<count; ++k){
                            float x=gx.ptr<float>()[i]*dx.at<float>(i,k),y=gy.ptr<float>()[i]*dy.at<float>(i,k); row[k]=float(x+y);
                        }
                        row[count]=-target.ptr<float>()[i]; row[count+1]=-1;
                    }
                });
                cv::Mat h=jac.t()*jac,g=-jac.t()*residual.reshape(1,n),step;
                while(true){
                    cv::Mat damped=h.clone();
                    for(int k=0; k<h.cols; ++k)damped.at<double>(k,k)+=lambda*h.at<double>(k,k);
                    if(!cv::solve(damped,g,step,cv::DECOMP_LU))throw std::runtime_error("Pixel normal equations singular");
                    cv::Mat candidate=p+step.rowRange(0,count).t(),rn,mxn,myn;
                    double an=gain+step.at<double>(count),bn=offset+step.at<double>(count+1),next=evaluate(candidate,an,bn,rn,mxn,myn);
                    bool accept=next<cost;
                    if(trace)*trace<<iteration<<','<<std::setprecision(17)<<lambda<<','<<cost<<','<<next<<','<<accept<<'\n';
                    if(accept){
                        p=candidate;
                        gain=an;
                        offset=bn;
                        residual=rn;
                        mx=mxn;
                        my=myn;
                        cost=next;
                        lambda=std::max(lambda/10,1e-9);
                        ++accepted;
                        break;
                    }
                    lambda*=10;
                    if(lambda>1e6)return {
                        p,cost,initial_cost,iteration,accepted
                    };
                }
                if(cv::norm(step.rowRange(0,count),cv::NORM_INF)<1e-7)break;
            }
            return {
                p,cost,initial_cost,iteration,accepted
            };
        }
        Fit solve_multi(const SplineImage& image,const cv::Mat& model,double mean,const cv::Mat& target)const{
            Fit best;
            bool found=false;
            for(int axis=0; axis<2; ++axis){
                cv::Mat initial=cv::Mat::zeros(1,5,CV_64F);
                initial.at<double>(0,axis)=mean/(axis==0?80:14.25);
                auto result=solve(image,model,initial,target);
                if(!found||result.cost<best.cost){
                    found=true;
                    best=result;
                }
            }
            return best;
        }
    };
}
