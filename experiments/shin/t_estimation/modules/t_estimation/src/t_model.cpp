#include "t_model.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
namespace t_only {
std::vector<double> representation(const Features& x,int k) {
    if(k==0)return {x.begin(),x.begin()+10};
    if(k==6)return {x.begin(),x.end()};
    if(k==7)return {x.begin()+10,x.end()};
    if(k==1)return {x[0],x[1],x[2],x[3],x[4],x[5]};
    if(k==2)return {x[0],x[2],x[4]};
    if(k==3)return {(x[0]+x[2]+x[4])/3,(x[1]+x[3]+x[5])/3};
    if(k==4)return {(x[0]+x[2]+x[4])/3,x[4]-x[0],x[2]-(x[0]+x[4])/2,(x[1]+x[3]+x[5])/3,x[5]-x[1],x[6],x[8]};
    if(k==5)return {x[4]-x[0],x[5]-x[1],x[7],x[9],x[8]-x[6]};
    throw std::invalid_argument("Unknown T representation");
}
static std::vector<double> design(const Model& m,const Features& f) {
    auto z=representation(f,m.config.representation);
    for(int j=0;j<(int)z.size();j++)z[j]=(z[j]-m.mean.at<double>(0,j))/m.scale.at<double>(0,j);
    auto v=z;
    if(m.config.degree>=2)for(int j=0;j<(int)z.size();j++)for(int k=j;k<(int)z.size();k++)v.push_back(z[j]*z[k]);
    if(m.config.degree==3)for(int j=0;j<(int)z.size();j++)for(int k=j;k<(int)z.size();k++)for(int l=k;l<(int)z.size();l++)v.push_back(z[j]*z[k]*z[l]);
    return v;
}
Model fit(const std::vector<Sample>& rows,const std::vector<int>& ids,Config cfg) {
    if(ids.empty())throw std::invalid_argument("Empty T training set");
    Model m;m.config=cfg;m.minimum_t=rows[ids[0]].target;m.maximum_t=m.minimum_t;for(int i:ids){m.minimum_t=std::min(m.minimum_t,rows[i].target);m.maximum_t=std::max(m.maximum_t,rows[i].target);}int n=(int)ids.size(),p=(int)representation(rows[ids[0]].features,cfg.representation).size();
    m.mean=cv::Mat::zeros(1,p,CV_64F);m.scale=cv::Mat::zeros(1,p,CV_64F);
    for(int i:ids){auto v=representation(rows[i].features,cfg.representation);for(int j=0;j<p;j++)m.mean.at<double>(0,j)+=v[j]/n;m.intercept+=rows[i].target/tick/n;}
    for(int i:ids){auto v=representation(rows[i].features,cfg.representation);for(int j=0;j<p;j++)m.scale.at<double>(0,j)+=pow(v[j]-m.mean.at<double>(0,j),2)/n;}
    for(int j=0;j<p;j++)m.scale.at<double>(0,j)=std::max(1e-8,sqrt(m.scale.at<double>(0,j)));
    int d=(int)design(m,rows[ids[0]].features).size();cv::Mat x(n,d,CV_64F),y(n,1,CV_64F),center;
    for(int r=0;r<n;r++){auto v=design(m,rows[ids[r]].features);for(int j=0;j<d;j++)x.at<double>(r,j)=v[j];y.at<double>(r)=rows[ids[r]].target/tick-m.intercept;}
    cv::reduce(x,center,0,cv::REDUCE_AVG);
    for(int r=0;r<n;r++)x.row(r)-=center;
    cv::SVD svd(x);cv::Mat projection=svd.u.t()*y;
    for(int r=0;r<svd.w.rows;r++){double s=svd.w.at<double>(r);projection.row(r)*=(cfg.rank&&r>=cfg.rank)?0:cfg.alpha?s/(s*s+cfg.alpha):(s>svd.w.at<double>(0)*1e-10?1/s:0);}
    m.coefficient=svd.vt.t()*projection;if(cfg.loss){
        cv::Mat a(n,d+1,CV_64F),b(n,1,CV_64F),beta=cv::Mat::zeros(d+1,1,CV_64F);
        m.coefficient.copyTo(beta.rowRange(1,d+1));
        for(int iteration=0;iteration<12;iteration++){
            for(int r=0;r<n;r++){
                double residual=y.at<double>(r)-beta.at<double>(0);
                for(int j=0;j<d;j++)residual-=x.at<double>(r,j)*beta.at<double>(j+1);
                double w=cfg.loss==1?1/std::max(.25,std::abs(residual)):std::max(.25,std::min(20.,std::abs(residual)));
                double root=sqrt(w);a.at<double>(r,0)=root;
                for(int j=0;j<d;j++)a.at<double>(r,j+1)=x.at<double>(r,j)*root;
                b.at<double>(r)=y.at<double>(r)*root;
            }
            cv::Mat gram=a.t()*a,rhs=a.t()*b,next;
            for(int j=1;j<=d;j++)gram.at<double>(j,j)+=cfg.alpha;
            cv::solve(gram,rhs,next,cv::DECOMP_SVD);beta=.5*beta+.5*next;
        }
        m.intercept+=beta.at<double>(0);m.coefficient=beta.rowRange(1,d+1).clone();
    }
    cv::Mat offset=center*m.coefficient;m.intercept-=offset.at<double>(0);return m;
}
double predict(const Model& m,const Features& f){auto v=design(m,f);double t=m.intercept;for(int j=0;j<(int)v.size();j++)t+=v[j]*m.coefficient.at<double>(j);if(!std::isfinite(t))throw std::runtime_error("Nonfinite T prediction");return t*tick;}
Config select(const std::vector<Sample>& rows,const std::vector<int>& ids,int mode){
    std::set<int> groups;for(int i:ids)groups.insert(rows[i].group);
    Config best;double score=1e300;
    for(int degree=1;degree<=(mode?2:1);degree++)for(int kind=0;kind<6;kind++)for(double alpha:{0.,.001,.01,.1,1.,10.,100.}){
        for(int loss=0;loss<=((mode&&degree==1)?2:0);loss++){ if(loss && alpha!=0. && alpha!=.1 && alpha!=1.)continue; for(int rank : {0,2,3,4,6,8}) { if(rank && (mode<2||degree!=1||loss||kind!=0||alpha!=0.))continue; Config cfg{kind,degree,alpha,loss,rank};double worst=0,sse=0;int count=0;
        for(int g:groups){std::vector<int> train,test;for(int i:ids)(rows[i].group==g?test:train).push_back(i);auto m=fit(rows,train,cfg);for(int i:test){double e=std::abs(predict(m,rows[i].features)-rows[i].target)/tick;worst=std::max(worst,e);sse+=e*e;count++;}}
        double cost=worst+.1*sqrt(sse/count);if(cost<score){score=cost;best=cfg;}}}
    }return best;
}
void save(const Model& m,const std::string& path){cv::FileStorage f(path,cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);if(!f.isOpened())throw std::runtime_error("Cannot write T model");f<<"requires_y_fixed_at_reference"<<(int)m.requires_y_fixed_at_reference<<"schema"<<"t_only.v1"<<"tick"<<tick<<"representation"<<m.config.representation<<"degree"<<m.config.degree<<"rank"<<m.config.rank<<"loss"<<m.config.loss<<"alpha"<<m.config.alpha<<"mean"<<m.mean<<"scale"<<m.scale<<"coefficient"<<m.coefficient<<"intercept_ticks"<<m.intercept<<"minimum_t"<<m.minimum_t<<"maximum_t"<<m.maximum_t;}
Model load(const std::string& path){cv::FileStorage f(path,cv::FileStorage::READ);if(!f.isOpened()||(std::string)f["schema"]!="t_only.v1"||std::abs((double)f["tick"]-tick)>1e-12)throw std::runtime_error("Incompatible T model");Model m;m.requires_y_fixed_at_reference=(int)f["requires_y_fixed_at_reference"]!=0;m.config={(int)f["representation"],(int)f["degree"],(double)f["alpha"],(int)f["loss"],(int)f["rank"]};f["mean"]>>m.mean;f["scale"]>>m.scale;f["coefficient"]>>m.coefficient;m.intercept=(double)f["intercept_ticks"];if(!f["minimum_t"].empty()){m.minimum_t=(double)f["minimum_t"];m.maximum_t=(double)f["maximum_t"];}if(!std::isfinite(m.minimum_t)||!std::isfinite(m.maximum_t)||m.minimum_t>m.maximum_t)throw std::runtime_error("Invalid T calibration range");Features zero{};int p=(int)representation(zero,m.config.representation).size();int d=p+(m.config.degree>=2?p*(p+1)/2:0)+(m.config.degree==3?p*(p+1)*(p+2)/6:0);if((m.config.degree<1||m.config.degree>3)||m.mean.type()!=CV_64F||m.mean.rows!=1||m.mean.cols!=p||m.scale.type()!=CV_64F||m.scale.rows!=1||m.scale.cols!=p||m.coefficient.type()!=CV_64F||m.coefficient.rows!=d||m.coefficient.cols!=1||!cv::checkRange(m.mean)||!cv::checkRange(m.scale)||!cv::checkRange(m.coefficient)||!std::isfinite(m.intercept))throw std::runtime_error("Invalid T model matrices");for(int j=0;j<p;j++)if(m.scale.at<double>(j)<=0)throw std::runtime_error("Invalid T scale");return m;}
}
