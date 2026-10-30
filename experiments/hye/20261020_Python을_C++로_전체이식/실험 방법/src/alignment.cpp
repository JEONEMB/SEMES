#include "alignment.hpp"
#include <chrono>
#include <algorithm>
#include <numeric>
#include <limits>
#include <set>
namespace semes {
    using Clock=std::chrono::steady_clock;
    static double elapsed(Clock::time_point t){
        return std::chrono::duration<double>(Clock::now()-t).count();
    }
    cv::Mat matrix(const Json& j){
        if(j.type!=Json::Array||j.size()==0)throw std::runtime_error("Nonempty numeric array required");
        bool nested=j.at(0).type==Json::Array;
        int rows=nested?int(j.size()):1,cols=nested?int(j.at(0).size()):int(j.size());
        cv::Mat m(rows,cols,CV_64F);
        for(int i=0; i<rows; i++)for(int k=0; k<cols; k++)m.at<double>(i,k)=nested?j.at(i).at(k).num():j.at(k).num();
        return m;
    }
    Json jsonmatrix(const cv::Mat& m){
        Json j=Json::list();
        for(int i=0; i<m.rows; i++){
            Json r=Json::list();
            for(int k=0; k<m.cols; k++)r.push(m.at<double>(i,k));
            j.push(r);
        }
        return j;
    }
    cv::Mat loadimage(const fs::path& p){
        auto raw=bytes(p);
        auto image=cv::imdecode(raw,cv::IMREAD_GRAYSCALE);
        if(image.empty())throw std::runtime_error("Cannot decode "+p.u8string());
        return image;
    }
    Json Prediction::json()const{
        Json j=Json::dict();
        j["axis"]=axis;
        j["branch"]=branch;
        j["value"]=value;
        j["Y"]=y;
        j["T"]=t;
        j["cost"]=cost;
        return j;
    }
    struct Window {
        std::vector<cv::Point2d> xy;
        std::vector<double> pixels;
    };
    struct Fit {
        cv::Vec<double,6> p;
        double rmse=0;
        bool ok=false;
    };
    struct Objective {
        double cost=0;
        cv::Mat r,J;
    };
    static Objective objective(const cv::Vec<double,6>& p,const Window& w,const Corner& c){
        Objective o;
        o.r=cv::Mat(int(w.xy.size()),1,CV_64F);
        o.J=cv::Mat(int(w.xy.size()),6,CV_64F);
        double du=c.patch[1]-c.patch[0],dv=c.patch[2]-c.patch[0],cross=c.patch[3]-c.patch[1]-c.patch[2]+c.patch[0];
        for(int i=0; i<o.r.rows; i++){
            auto d=w.xy[i]-cv::Point2d(p[0],p[1]);
            double u=d.x*c.inverse(0,0)+d.y*c.inverse(0,1),v=d.x*c.inverse(1,0)+d.y*c.inverse(1,1),zu=u/p[2],zv=v/p[3];
            double U=.5*(1+std::erf(zu/std::sqrt(2.))),V=.5*(1+std::erf(zv/std::sqrt(2.)));
            double F=c.patch[0]*(1-U)*(1-V)+c.patch[1]*U*(1-V)+c.patch[2]*(1-U)*V+c.patch[3]*U*V;
            double r=p[4]+p[5]*(1-F)-w.pixels[i];
            o.r.at<double>(i)=r;
            o.cost+=36*(std::sqrt(1+(r/6)*(r/6))-1);
            double gu=(du+cross*V)*std::exp(-.5*zu*zu)/std::sqrt(2*CV_PI),gv=(dv+cross*U)*std::exp(-.5*zv*zv)/std::sqrt(2*CV_PI);
            auto j=o.J.ptr<double>(i);
            j[0]=p[5]*(gu/p[2]*c.inverse(0,0)+gv/p[3]*c.inverse(1,0));
            j[1]=p[5]*(gu/p[2]*c.inverse(0,1)+gv/p[3]*c.inverse(1,1));
            j[2]=p[5]*gu*u/(p[2]*p[2]);
            j[3]=p[5]*gv*v/(p[3]*p[3]);
            j[4]=1;
            j[5]=1-F;
        }
        return o;
    }
    static double quantile(std::vector<double> v,double q){
        std::sort(v.begin(),v.end());
        double f=q*(v.size()-1);
        size_t k=size_t(f);
        return v[k]*(1-(f-k))+v[std::min(k+1,v.size()-1)]*(f-k);
    }
    static Fit fit(const Window& w,const cv::Point2d& start,const Corner& c){
        if(w.pixels.size()<6)throw std::runtime_error("Insufficient boundary pixels");
        double low=quantile(w.pixels,.1),high=quantile(w.pixels,.9);
        Fit f;
        f.p={
            start.x,start.y,1,1,low,std::max(high-low,20.)
        };
        cv::Vec<double,6> lower={
            start.x-3.8,start.y-3.8,.35,.35,0,10
        },upper={
            start.x+3.8,start.y+3.8,2.7,2.7,240,255
        },scales={
            1,1,1,1,60,120
        };
        // This is the existing compiled.py bounded LM variant, not SciPy TRF.
        double mu=1e-3;
        auto o=objective(f.p,w,c);
        int evaluations=1;
        for(int iteration=0; iteration<85; iteration++){
            cv::Mat H=cv::Mat::zeros(6,6,CV_64F),g=cv::Mat::zeros(6,1,CV_64F);
            for(int i=0; i<o.r.rows; i++){
                double r=o.r.at<double>(i),root=std::sqrt(1+(r/6)*(r/6)),weight=1/(root*root*root);
                const auto j=o.J.ptr<double>(i);
                for(int k=0; k<6; k++){
                    double jj=j[k]*scales[k];
                    g.at<double>(k)+=jj*r/root;
                    for(int l=0; l<6; l++)H.at<double>(k,l)+=jj*j[l]*scales[l]*weight;
                }
            }
            double projected=0;
            for(int j=0; j<6; j++){
                double gg=g.at<double>(j);
                if((f.p[j]-lower[j]<1e-9&&gg>0)||(upper[j]-f.p[j]<1e-9&&gg<0))gg=0;
                projected=std::max(projected,std::abs(gg));
            }
            if(projected<1e-7){
                f.ok=true;
                break;
            }
            for(int j=0; j<6; j++)if((f.p[j]-lower[j]<1e-8&&g.at<double>(j)>0)||(upper[j]-f.p[j]<1e-8&&g.at<double>(j)<0)){
                H.row(j).setTo(0);
                H.col(j).setTo(0);
                H.at<double>(j,j)=1;
                g.at<double>(j)=0;
            }
            bool accepted=false;
            for(int retry=0; retry<14; retry++){
                cv::Mat A=H.clone(),delta;
                for(int j=0; j<6; j++)A.at<double>(j,j)+=mu*std::max(H.at<double>(j,j),1e-6);
                if(!cv::solve(A,-g,delta,cv::DECOMP_LU))throw std::runtime_error("LM singular system");
                auto trial=f.p;
                for(int j=0; j<6; j++)trial[j]=std::min(std::max(f.p[j]+delta.at<double>(j)*scales[j],lower[j]+1e-12),upper[j]-1e-12);
                auto n=objective(trial,w,c);
                ++evaluations;
                double step=cv::norm(trial-f.p),pnorm=cv::norm(f.p);
                if(step<1e-7*(1e-7+pnorm)){
                    f.ok=true;
                    if(n.cost<o.cost){
                        f.p=trial;
                        o=std::move(n);
                    }
                    accepted=true;
                    break;
                }
                if(n.cost<o.cost){
                    double improvement=o.cost-n.cost;
                    f.p=trial;
                    o=std::move(n);
                    mu=std::max(mu*.3,1e-12);
                    accepted=true;
                    if(improvement<1e-7*std::max(o.cost,1e-12)||step<1e-7*(1e-7+pnorm))f.ok=true;
                    break;
                }
                mu=std::min(mu*10,1e14);
                if(evaluations>=85)break;
            }
            if(f.ok||evaluations>=85||!accepted)break;
        }
        f.rmse=cv::norm(o.r)/std::sqrt(double(w.pixels.size()));
        return f;
    }
    static cv::Rect box(int k){
        static const int b[3][4]={
            {
                1552,969,1736,1167
            },{
                1809,971,1993,1169
            },{
                2067,973,2252,1171
            }
        };
        return {
            b[k][0],b[k][1],b[k][2]-b[k][0],b[k][3]-b[k][1]
        };
    }
    static Window window(const cv::Mat& img,cv::Point2d est,int radius,cv::Rect clip){
        Window w;
        int x0=int(std::nearbyint(est.x)),y0=int(std::nearbyint(est.y));
        for(int y=y0-radius; y<=y0+radius; y++)for(int x=x0-radius; x<=x0+radius; x++)if(clip.contains({
            x,y
        })){
            if(x<0||y<0||x>=img.cols||y>=img.rows)throw std::runtime_error("Boundary window outside image");
            w.xy.emplace_back(x,y);
            w.pixels.push_back(img.at<unsigned char>(y,x));
        }
        return w;
    }
    Engine::Engine(const fs::path& assets,const fs::path& center){
        cv::setNumThreads(1);
        auto provenance=Json::parse(read(assets/"provenance.json"));
        if(sha256(center)!=provenance.at("center_sha256").str())throw std::runtime_error("Fixed center SHA256 mismatch");
        center_=loadimage(center);
        auto j=Json::parse(read(assets/"corners.json"));
        auto reference=npz(assets/"reference.npz");
        const auto& pts=reference.at("points");
        const auto& ids=reference.at("ids");
        const auto& regions=reference.at("regions");
        if(j.size()!=630||pts.shape!=std::vector<size_t>{
            630,2
        }
        ||ids.count()!=630||regions.count()!=630)throw std::runtime_error("Expected fixed 630 IDs");
        std::set<int> unique;
        for(size_t i=0; i<630; i++){
            auto& x=j.at(i);
            Corner c;
            c.id=int(x.at("id").num());
            c.region=int(x.at("qr").num())-1;
            c.reference={
                pts.number(2*i),pts.number(2*i+1)
            };
            if(c.id!=ids.number(i)||c.region!=regions.number(i)||c.region<0||c.region>2||!unique.insert(c.id).second)throw std::runtime_error("Reference ID/region mismatch");
            c.initial={
                x.at("cx").num(),x.at("cy").num()
            };
            auto a=matrix(x.at("axes"));
            cv::Mat af,inv;
            a.convertTo(af,CV_32F);
            cv::invert(af,inv,cv::DECOMP_LU);
            for(int y=0; y<2; y++)for(int k=0; k<2; k++){
                c.inverse(y,k)=inv.at<float>(y,k);
                c.patch[2*y+k]=x.at("patch").at(y).at(k).num();
            }
            corners_.push_back(c);
        }
        basis_=cv::Mat(630,6,CV_64F);
        for(int i=0; i<630; i++){
            double u=(corners_[i].reference.x-1906)/486,v=(corners_[i].reference.y-1070)/122;
            auto r=basis_.ptr<double>(i);
            r[0]=1;
            r[1]=u;
            r[2]=v;
            r[3]=u*u;
            r[4]=u*v;
            r[5]=v*v;
        }
        cv::invert(basis_,pinv_,cv::DECOMP_SVD);
        model_=Json::parse(read(assets/"model.json"));
        if(matrix(model_.at("W")).cols!=12||model_.at("zero_feature").size()!=12)throw std::runtime_error("Invalid P12 model");
    }
    Measurement Engine::measure(const cv::Mat& image,int search)const{
        if(image.empty()||image.type()!=CV_8UC1||image.size()!=center_.size())throw std::runtime_error("Original fixed-center image dimensions required");
        if(search<1||search>512)throw std::runtime_error("Search radius must be 1..512");
        auto begin=Clock::now();
        Measurement m;
        m.current=cv::Mat(630,2,CV_64F);
        m.displacement=cv::Mat(630,2,CV_64F);
        m.quality=cv::Mat(630,5,CV_64F);
        m.valid.resize(630);
        m.flags.resize(630);
        std::array<cv::Point,3> shifts;
        for(int k=0; k<3; k++){
            auto b=box(k);
            cv::Rect s(b.x-search,b.y-search,b.width+2*search,b.height+2*search);
            if((s&cv::Rect(0,0,image.cols,image.rows))!=s)throw std::runtime_error("NCC search outside image");
            cv::Mat corr;
            cv::matchTemplate(image(s),center_(b),corr,cv::TM_CCOEFF_NORMED);
            cv::Point loc;
            cv::minMaxLoc(corr,nullptr,&m.ncc[k],nullptr,&loc);
            if(!std::isfinite(m.ncc[k])||m.ncc[k]<.6)throw std::runtime_error("QR NCC below 0.6");
            shifts[k]=loc-cv::Point(search,search);
        }
        cv::parallel_for_(cv::Range(0,630),[&](const cv::Range& r){
            for(int i=r.start; i<r.end; i++){
                auto& c=corners_[i]; cv::Point2d start=c.initial+cv::Point2d(shifts[c.region]),est=start; auto b=box(c.region)+shifts[c.region]; Fit f; for(int iteration=0; iteration<3; iteration++){
                    f=fit(window(image,est,4,b),start,c); cv::Point2d next(f.p[0],f.p[1]); bool moved=std::nearbyint(next.x)!=std::nearbyint(est.x)||std::nearbyint(next.y)!=std::nearbyint(est.y); est=next; if(!moved)break;
                }
                auto f2=fit(window(image,est,5,b),start,c); double stability=cv::norm(cv::Point2d(f.p[0]-f2.p[0],f.p[1]-f2.p[1])); int flags=0; if(!(f.ok&&f2.ok))flags|=1; if(f.rmse>15)flags|=2; if(f.p[5]<45)flags|=4; if(stability>.65)flags|=8; if(std::max(std::abs(f.p[0]-start.x),std::abs(f.p[1]-start.y))>3.6)flags|=16; if(std::max(f.p[2],f.p[3])>2.6)flags|=32; m.flags[i]=flags; m.valid[i]=flags==0; m.current.at<double>(i,0)=f.p[0]; m.current.at<double>(i,1)=f.p[1]; m.displacement.at<double>(i,0)=f.p[0]-c.reference.x; m.displacement.at<double>(i,1)=f.p[1]-c.reference.y; auto q=m.quality.ptr<double>(i); q[0]=f.rmse; q[1]=stability; q[2]=f.p[5]; q[3]=f.p[2]; q[4]=f.p[3];
            }
        });
        if(!cv::checkRange(m.displacement))throw std::runtime_error("Nonfinite displacement");
        m.seconds=elapsed(begin);
        return m;
    }
    cv::Mat Engine::features(const cv::Mat& d,const std::string& name)const{
        if(d.rows!=630||d.cols!=2||d.type()!=CV_64F||!cv::checkRange(d))throw std::runtime_error("Finite 630x2 displacements required");
        if(name=="raw1260")return d.reshape(1,1).clone();
        if(name=="P12"){
            cv::Mat c=pinv_*d;
            return c.reshape(1,1).clone();
        }
        if(name!="QR18"&&name!="QR36")throw std::runtime_error("Unknown feature map");
        cv::Mat output(1,name=="QR18"?18:36,CV_64F);
        int offset=0;
        for(int k=0; k<3; k++){
            std::vector<int> ix;
            cv::Point2d mean;
            for(int i=0; i<630; i++)if(corners_[i].region==k){
                ix.push_back(i);
                mean+=corners_[i].reference;
            }
            mean*=1./ix.size();
            int cols=name=="QR18"?3:6;
            cv::Mat B(int(ix.size()),cols,CV_64F),D(int(ix.size()),2,CV_64F);
            for(int i=0; i<int(ix.size()); i++){
                auto p=(corners_[ix[i]].reference-mean)*.01;
                double r[]={
                    1,p.x,p.y,p.x*p.x,p.x*p.y,p.y*p.y
                };
                for(int j=0; j<cols; j++)B.at<double>(i,j)=r[j];
                d.row(ix[i]).copyTo(D.row(i));
            }
            cv::Mat C;
            cv::solve(B,D,C,cv::DECOMP_SVD);
            C=C.reshape(1,1);
            C.copyTo(output.colRange(offset,offset+2*cols));
            offset+=2*cols;
        }
        return output;
    }
    Prediction predict_model(const cv::Mat& feature,const Json& model,bool sign_gate){
        if(feature.rows!=1||!cv::checkRange(feature))throw std::runtime_error("Finite feature required");
        auto W=matrix(model.at("W"));
        cv::Mat x=model.has("zero_feature")?feature-matrix(model.at("zero_feature")):feature.clone();
        Prediction best;
        best.cost=std::numeric_limits<double>::infinity();
        for(auto& key:model.at("branches").object_order){
            auto& b=model.at("branches").at(key);
            cv::Mat z=W*(x-matrix(b.at(1))).t(),s=W*matrix(b.at(2)).t(),c=W*matrix(b.at(3)).t();
            double value=z.dot(s)/s.dot(s);
            for(int k=0; k<8; k++){
                cv::Mat jac=s+2*value*c,res=z-value*s-value*value*c;
                value+=jac.dot(res)/std::max(jac.dot(jac),1e-20);
            }
            cv::Mat residual=z-value*s-value*value*c;
            double cost=residual.dot(residual);
            if(sign_gate&&value*(key.back()=='+'?1:-1)<0)cost=std::numeric_limits<double>::infinity();
            if(cost<best.cost){
                best.axis=b.at(0).str();
                best.branch=key;
                best.value=value;
                best.cost=cost;
            }
        }
        if(!std::isfinite(best.cost)||!std::isfinite(best.value))throw std::runtime_error("Model produced nonfinite prediction");
        best.y=best.axis=="Y"?best.value:0;
        best.t=best.axis=="T"?best.value:0;
        return best;
    }
    Prediction Engine::predict(const cv::Mat& feature)const{
        return predict_model(feature,model_);
    }
    Json Engine::infer(const std::vector<unsigned char>& raw,int search,Measurement* measured)const{
        auto begin=Clock::now();
        cv::Mat image=cv::imdecode(raw,cv::IMREAD_GRAYSCALE);
        auto m=measure(image,search);
        auto f=features(m.displacement);
        auto p=predict(f);
        if(cv::countNonZero(image!=center_)==0){
            p.axis="CENTER";
            p.value=p.y=p.t=p.cost=0;
        }
        Json j=p.json();
        j["points"]=630;
        j["valid"]=std::accumulate(m.valid.begin(),m.valid.end(),0);
        j["seconds"]=elapsed(begin);
        j["measurement_seconds"]=m.seconds;
        j["solver"]="compiled_bounded_lm_cpp";
        j["model"]="v1_edge630_boundary_fixed";
        j["search_radius"]=search;
        j["features"]=jsonmatrix(f);
        if(measured)*measured=std::move(m);
        return j;
    }
    void selftest(){
        Corner c;
        c.inverse=cv::Matx22d(1,0,.006,1);
        c.patch={
            0,0,0,1
        };
        Window w;
        for(int y=-4; y<=4; y++)for(int x=-4; x<=4; x++){
            w.xy.emplace_back(x,y);
            w.pixels.push_back(0);
        }
        cv::Vec<double,6> p{
            .13,-.27,1.1,.8,35,140
        };
        auto initial=objective(p,w,c);
        for(int i=0; i<initial.r.rows; i++)w.pixels[i]=initial.r.at<double>(i);
        auto exact=objective(p,w,c);
        if(cv::norm(exact.r)>1e-10)throw std::runtime_error("Synthetic boundary construction");
        for(int k=0; k<6; k++){
            auto a=p,b=p;
            a[k]+=1e-5;
            b[k]-=1e-5;
            auto ap=objective(a,w,c),bp=objective(b,w,c);
            cv::Mat numeric=(ap.r-bp.r)/2e-5;
            if(cv::norm(numeric-exact.J.col(k),cv::NORM_INF)>1e-6)throw std::runtime_error("Boundary Jacobian finite difference");
        }
        auto f=fit(w,{
            0,0
        },c);
        if(!f.ok||cv::norm(f.p-p)>1e-4)throw std::runtime_error("Synthetic bounded fit");
        auto j=Json::parse("{\"q\":[1,true,\"\\uD55C\"],\"x\":-1.3e-5}");
        if(Json::parse(j.dump()).at("q").at(2).str()!=u8"한")throw std::runtime_error("JSON Unicode");
        bool rejected=false;
        try{
            Json::parse("[1] trailing");
        }
        catch(...){
            rejected=true;
        }
        if(!rejected)throw std::runtime_error("JSON trailing data accepted");
    }
}
