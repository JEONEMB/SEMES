#include "joint.hpp"
#include <algorithm>
namespace semes {
    static cv::Mat numeric(const Npy& a){
        if(a.shape.size()!=2)throw std::runtime_error("Joint matrix must have rank 2");
        cv::Mat m(int(a.shape[0]),int(a.shape[1]),CV_64F);
        for(size_t i=0; i<a.count(); i++)m.ptr<double>()[i]=a.number(i);
        return m;
    }
    JointOrigins::JointOrigins(const fs::path& model){
        auto a=npz(model);
        A_=numeric(a.at("A"));
        centers_=numeric(a.at("centers"));
        auto& w=a.at("W");
        if(w.shape.size()==1){
            W_=cv::Mat::zeros(int(w.count()),int(w.count()),CV_64F);
            for(int i=0; i<W_.rows; i++)W_.at<double>(i,i)=w.number(i);
        }
        else W_=numeric(w);
        degree_=int(a.at("degree").number(0));
        feature_=a.count("feature")?a.at("feature").text(0):"P12";
        for(int i=0; i<2; i++){
            scale_[i]=a.at("scale").number(i);
            lower_[i]=a.at("lower").number(i);
            upper_[i]=a.at("upper").number(i);
            if(scale_[i]<=0||lower_[i]>upper_[i])throw std::runtime_error("Invalid joint domain");
        }
        for(size_t i=0; i<a.at("origin_ids").count(); i++)ids_.push_back(a.at("origin_ids").text(i));
        if((degree_!=1&&degree_!=2)||A_.rows!=(degree_==1?2:5)||W_.rows!=A_.cols||W_.cols!=A_.cols||centers_.cols!=A_.cols||ids_.size()!=centers_.rows)throw std::runtime_error("Joint model dimension mismatch");
        Aw_=A_*W_.t();
        cw_=centers_*W_.t();
        cv::invert(Aw_.rowRange(0,2),linear_pinv_,cv::DECOMP_SVD);
        if(a.count("As")){
            auto& local=a.at("As");
            if(local.shape!=std::vector<size_t>{
                size_t(centers_.rows),size_t(A_.rows),size_t(A_.cols)
            })throw std::runtime_error("Local surface dimensions mismatch");
            for(int k=0; k<centers_.rows; k++){
                cv::Mat ak(A_.size(),CV_64F);
                for(size_t i=0; i<ak.total(); i++)ak.ptr<double>()[i]=local.number(k*ak.total()+i);
                cv::Mat aw=ak*W_.t(),pinv;
                cv::invert(aw.rowRange(0,2),pinv,cv::DECOMP_SVD);
                local_aw_.push_back(aw);
                local_pinv_.push_back(pinv);
            }
        }
    }
    Json JointOrigins::predict(const cv::Mat& feature)const{
        if(feature.rows!=1||feature.cols!=A_.cols||!cv::checkRange(feature))throw std::runtime_error("Finite matching joint feature required");
        cv::Mat whitened=feature*W_.t();
        struct Candidate{
            double cost;
            cv::Vec2d output;
            int index;
        };
        std::vector<Candidate> candidates;
        for(int k=0; k<centers_.rows; k++){
            const cv::Mat& aw=local_aw_.empty()?Aw_:local_aw_[k];
            const cv::Mat& pinv=local_pinv_.empty()?linear_pinv_:local_pinv_[k];
            cv::Mat z=whitened-cw_.row(k),initial=z*pinv;
            cv::Vec2d q;
            for(int i=0; i<2; i++)q[i]=std::clamp(initial.at<double>(i),lower_[i],upper_[i]);
            auto surface=[&](const cv::Vec2d& v){
                cv::Mat p=v[0]*aw.row(0)+v[1]*aw.row(1);
                if(degree_==2)p+=v[0]*v[0]*aw.row(2)+v[0]*v[1]*aw.row(3)+v[1]*v[1]*aw.row(4);
                return p;
            };
            for(int iteration=0; iteration<16; iteration++){
                cv::Mat j0=aw.row(0).clone(),j1=aw.row(1).clone();
                if(degree_==2){
                    j0+=2*q[0]*aw.row(2)+q[1]*aw.row(3);
                    j1+=q[0]*aw.row(3)+2*q[1]*aw.row(4);
                }
                cv::Mat residual=z-surface(q);
                double a=j0.dot(j0),b=j0.dot(j1),c=j1.dot(j1),g0=j0.dot(residual),g1=j1.dot(residual),damp=1e-8*(a+c+1);
                a+=damp;
                c+=damp;
                double det=std::max(a*c-b*b,1e-20);
                cv::Vec2d step{
                    (c*g0-b*g1)/det,(a*g1-b*g0)/det
                };
                for(int i=0; i<2; i++)q[i]=std::clamp(q[i]+std::clamp(step[i],-.2,.2),lower_[i],upper_[i]);
            }
            cv::Mat residual=z-surface(q);
            candidates.push_back({
                residual.dot(residual),{
                    q[0]*scale_[0],q[1]*scale_[1]
                },k
            });
        }
        std::stable_sort(candidates.begin(),candidates.end(),[](auto& a,auto& b){
            return a.cost<b.cost;
        });
        auto& best=candidates[0];
        auto& second=candidates[candidates.size()>1?1:0];
        Json j=Json::dict();
        j["Y"]=best.output[0];
        j["T"]=best.output[1];
        j["center_index"]=best.index;
        j["center_id"]=ids_[best.index];
        j["cost"]=best.cost;
        j["second_cost"]=second.cost;
        j["margin"]=(second.cost-best.cost)/(second.cost+best.cost+1e-12);
        j["alternative_Y"]=second.output[0];
        j["alternative_T"]=second.output[1];
        j["feature"]=feature_;
        j["model"]="joint_origins_cpp_frozen";
        return j;
    }
}
