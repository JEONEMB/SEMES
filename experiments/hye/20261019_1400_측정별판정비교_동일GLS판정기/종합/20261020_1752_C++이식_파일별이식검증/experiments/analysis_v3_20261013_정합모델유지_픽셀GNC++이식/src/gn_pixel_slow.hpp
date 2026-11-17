#pragma once
#include "../../analysis_v1_20261013_고정점분할유지_선형계산C++이식/src/common.hpp"
#include <chrono>
namespace pixel_history {
    using namespace semes;
    using namespace historical_v1;
    struct Image {
        cv::Mat intensity,gx,gy;
    };
    struct Fit {
        cv::Mat p;
        double cost=0,initial_cost=0;
        int iteration=0,accepted=0;
    };
    class SlowPixel {
        fs::path workspace,images;
        public: Data data;
        int x0,x1,y0,y1,bx0,by0,width,height;
        cv::Mat corner_displacements,nuisance,pixels,center_template;
        std::vector<cv::Point2f> corners;
        std::vector<double> mean_qr_shift;
        SlowPixel(const fs::path& root,const fs::path& image_root):workspace(root),images(image_root),data(root){
            CsvTable q(root/fs::u8path(u8"14_데이터_QR마커위치.csv")),hom(root/"analysis_v2/homography_features.csv");
            std::map<std::string,size_t> qi,hi;
            for(size_t i=0; i<q.rows.size(); ++i)qi.emplace(q.text(i,"file"),i);
            for(size_t i=0; i<hom.rows.size(); ++i)hi.emplace(hom.text(i,"file"),i);
            size_t c=qi.at("center.bmp");
            x0=int(q.number(c,"Lx")-110);
            x1=int(q.number(c,"Rx")+110);
            y0=int(std::min(q.number(c,"Ly"),q.number(c,"Ry"))-120);
            y1=int(std::max(q.number(c,"Ly"),q.number(c,"Ry"))+120);
            width=x1-x0;
            height=y1-y0;
            bx0=x0-130;
            by0=y0-40;
            corners={
                cv::Point2f(float(x0),float(y0)),cv::Point2f(float(x1),float(y0)),cv::Point2f(float(x1),float(y1)),cv::Point2f(float(x0),float(y1))
            };
            nuisance=cv::Mat(8,3,CV_64F);
            for(int i=0; i<4; ++i){
                float cx=corners[i].x-float(x0+x1)/2,cy=corners[i].y-float(y0+y1)/2;
                nuisance.at<double>(2*i,0)=0;
                nuisance.at<double>(2*i+1,0)=1;
                nuisance.at<double>(2*i,1)=float(-cy/1000);
                nuisance.at<double>(2*i+1,1)=float(cx/1000);
                nuisance.at<double>(2*i,2)=float(cx/1000);
                nuisance.at<double>(2*i+1,2)=float(cy/1000);
            }
            corner_displacements=cv::Mat(data.x.rows,8,CV_64F);
            const std::vector<std::string> names={
                u8"좌상",u8"우상",u8"우하",u8"좌하"
            };
            for(int i=0; i<data.x.rows; ++i){
                for(int k=0; k<4; ++k)for(int axis=0; axis<2; ++axis)corner_displacements.at<double>(i,2*k+axis)=hom.number(hi.at(data.names[i]),names[k]+(axis==0?"_dx":"_dy"));
                size_t row=qi.at(data.names[i]);
                mean_qr_shift.push_back((q.number(row,"dLx")+q.number(row,"dCx")+q.number(row,"dRx"))/3);
            }
            pixels=cv::Mat(width*height,1,CV_64FC2);
            for(int y=0; y<height; ++y)for(int x=0; x<width; ++x)pixels.at<cv::Vec2d>(y*width+x)=cv::Vec2d(x+x0,y+y0);
            center_template=read_image("center.bmp").intensity(cv::Rect(130,40,width,height)).clone();
        }
        Image read_image(const std::string& name)const{
            fs::path path=name=="center.bmp"?images/"center.bmp":images/(name[0]=='T'?"Theta":"Y")/fs::u8path(name);
            auto gray=cv::imdecode(bytes(path),cv::IMREAD_GRAYSCALE);
            if(gray.size()!=cv::Size(3840,2160))throw std::runtime_error("Expected original image dimensions");
            cv::Mat raw,big,gx,gy;
            gray(cv::Rect(bx0,by0,width+260,height+80)).convertTo(raw,CV_32F);
            cv::GaussianBlur(raw,big,cv::Size(),1);
            cv::Sobel(big,gx,CV_32F,1,0,3);
            cv::Sobel(big,gy,CV_32F,0,1,3);
            gx/=8;
            gy/=8;
            return {
                big,gx,gy
            };
        }
        cv::Mat fit_model(int training_parity)const{
            std::vector<int> train;
            for(int i=0; i<data.x.rows; ++i)if(i%2==training_parity)train.push_back(i);
            return least_squares(append_one(selected(data.x,train)),selected(corner_displacements,train));
        }
        std::pair<cv::Mat,cv::Mat> coords(const cv::Mat& model,const cv::Mat& p)const{
            cv::Mat active(1,3,CV_64F);
            active.at<double>(0,0)=p.at<double>(0,0);
            active.at<double>(0,1)=p.at<double>(0,1);
            active.at<double>(0,2)=1;
            cv::Mat d=active*model;
            if(p.cols>2)d+=p.colRange(2,5)*nuisance.t();
            std::vector<cv::Point2f> destination;
            for(int i=0; i<4; ++i)destination.emplace_back(corners[i].x+float(d.at<double>(0,2*i)),corners[i].y+float(d.at<double>(0,2*i+1)));
            auto homography=cv::getPerspectiveTransform(corners,destination);
            cv::Mat mapped;
            cv::perspectiveTransform(pixels,mapped,homography);
            cv::Mat mx(height,width,CV_32F),my(height,width,CV_32F);
            for(int i=0; i<width*height; ++i){
                auto p=mapped.at<cv::Vec2d>(i);
                mx.ptr<float>()[i]=float(p[0]-bx0);
                my.ptr<float>()[i]=float(p[1]-by0);
            }
            return {
                mx,my
            };
        }
        cv::Mat mean_template(int training_parity)const{
            cv::Mat accumulator=cv::Mat::zeros(height,width,CV_32F);
            int count=0;
            for(int i=0; i<data.x.rows; ++i)if(i%2==training_parity){
                cv::Mat model=cv::Mat::zeros(3,8,CV_64F);
                corner_displacements.row(i).copyTo(model.row(2));
                auto maps=coords(model,cv::Mat::zeros(1,2,CV_64F));
                auto image=read_image(data.names[i]);
                cv::Mat aligned;
                cv::remap(image.intensity,aligned,maps.first,maps.second,cv::INTER_LINEAR);
                accumulator+=aligned;
                ++count;
            }
            accumulator/=count;
            return accumulator;
        }
        Fit solve(const Image& image,const cv::Mat& model,const cv::Mat& initial,const cv::Mat& custom_template=cv::Mat(),std::ostream* trace=nullptr)const{
            if(initial.rows!=1||(initial.cols!=2&&initial.cols!=5))throw std::runtime_error("Historical pixel parameter count must be 2 or 5");
            const cv::Mat& target=custom_template.empty()?center_template:custom_template;
            cv::Mat target64;
            target.convertTo(target64,CV_64F);
            cv::Mat p=initial.clone();
            double gain=1,offset=0;
            auto evaluate=[&](const cv::Mat& candidate,double a,double b,cv::Mat& r,cv::Mat& mx,cv::Mat& my){
                auto maps=coords(model,candidate);
                mx=maps.first;
                my=maps.second;
                cv::Mat sampled,observed;
                cv::remap(image.intensity,sampled,mx,my,cv::INTER_LINEAR);
                sampled.convertTo(observed,CV_64F);
                r=observed-(a*target64+b);
                return cv::norm(r,cv::NORM_L2SQR)/r.total();
            };
            cv::Mat residual,mx,my;
            double cost=evaluate(p,gain,offset,residual,mx,my),initial_cost=cost,lambda=1e-4;
            int accepted=0,iteration=0;
            for(iteration=0; iteration<40; ++iteration){
                int k=p.cols;
                cv::Mat gx,gy;
                cv::remap(image.gx,gx,mx,my,cv::INTER_LINEAR);
                cv::remap(image.gy,gy,mx,my,cv::INTER_LINEAR);
                cv::Mat jac(width*height,k+2,CV_64F);
                for(int j=0; j<k; ++j){
                    cv::Mat plus=p.clone(),minus=p.clone();
                    plus.at<double>(0,j)+=1e-3;
                    minus.at<double>(0,j)-=1e-3;
                    auto a=coords(model,plus),b=coords(model,minus);
                    cv::Mat dx=a.first-b.first,dy=a.second-b.second,tx=gx.mul(dx),ty=gy.mul(dy);
                    for(int i=0; i<width*height; ++i)jac.at<double>(i,j)=double(tx.ptr<float>()[i])/.002+double(ty.ptr<float>()[i])/.002;
                }
                for(int i=0; i<width*height; ++i){
                    jac.at<double>(i,k)=-target.ptr<float>()[i];
                    jac.at<double>(i,k+1)=-1;
                }
                cv::Mat h=jac.t()*jac,g=-jac.t()*residual.reshape(1,width*height),step;
                while(true){
                    cv::Mat damped=h.clone();
                    for(int j=0; j<h.cols; ++j)damped.at<double>(j,j)+=lambda*h.at<double>(j,j);
                    if(!cv::solve(damped,g,step,cv::DECOMP_LU))throw std::runtime_error("Historical pixel normal equations singular");
                    cv::Mat candidate=p+step.rowRange(0,k).t(),rn,mxn,myn;
                    double an=gain+step.at<double>(k),bn=offset+step.at<double>(k+1),next=evaluate(candidate,an,bn,rn,mxn,myn);
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
                if(cv::norm(step.rowRange(0,k),cv::NORM_INF)<1e-7)break;
            }
            return {
                p,cost,initial_cost,iteration,accepted
            };
        }
    };
}
