#include "alignment.hpp"
#include <set>
namespace semes {
    static std::string session(const std::string& n){
        double v=std::stod(n.substr(1));
        return (n[0]=='T'?std::string("T"):std::abs(std::lround(v/.05))%2?std::string("Y2"):std::string("Y1"))+(v>0?"+":"-");
    }
    static cv::Mat whitening(const cv::Mat& R,int dof,double shrink){
        cv::Mat S=R.t()*R/std::max(R.rows-dof,1),C(S.size(),CV_64F);
        std::vector<double> sd(S.rows);
        for(int i=0; i<S.rows; i++)sd[i]=std::sqrt(std::max(S.at<double>(i,i),1e-14));
        for(int i=0; i<S.rows; i++)for(int j=0; j<S.cols; j++)C.at<double>(i,j)=(1-shrink)*S.at<double>(i,j)/sd[i]/sd[j]+(i==j?shrink:0);
        cv::Mat inv;
        cv::invert(C,inv,cv::DECOMP_LU);
        cv::Mat L=cv::Mat::zeros(inv.size(),CV_64F);
        for(int i=0; i<L.rows; i++)for(int j=0; j<=i; j++){
            double s=inv.at<double>(i,j);
            for(int k=0; k<j; k++)s-=L.at<double>(i,k)*L.at<double>(j,k);
            if(i==j){
                if(s<=0)throw std::runtime_error("Whitening covariance is not positive definite");
                L.at<double>(i,j)=std::sqrt(s);
            }
            else L.at<double>(i,j)=s/L.at<double>(j,j);
        }
        cv::Mat W=L.t();
        for(int i=0; i<W.rows; i++)for(int j=0; j<W.cols; j++)W.at<double>(i,j)/=sd[j];
        return W;
    }
    Json train_axis_model(const cv::Mat& X,const std::vector<std::string>& names,const std::vector<int>& training,const std::string& mode,bool quadratic,double shrink,int covariance_dof){
        if(X.rows!=int(names.size())||training.size()<10||shrink<=0||shrink>1)throw std::runtime_error("Invalid calibration input");
        if(mode!="center"&&mode!="free"&&mode!="inferred_session"&&mode!="separate_session")throw std::runtime_error("Unsupported origin mode");
        std::set<int> unique(training.begin(),training.end());
        if(unique.size()!=training.size())throw std::runtime_error("Duplicate training index");
        Json model=Json::dict(),branches=Json::dict();
        cv::Mat residual;
        int dof=0;
        std::vector<std::string> groups;
        if(mode=="separate_session"){
            std::set<std::string> ss;
            for(int i:training)ss.insert(session(names.at(i)));
            groups.assign(ss.begin(),ss.end());
        }
        else groups={
            "Y","T"
        };
        for(auto& group:groups){
            std::vector<int> ix;
            for(int i:training){
                if(i<0||i>=X.rows||names[i]=="center.bmp")throw std::runtime_error("Invalid training membership");
                if(mode=="separate_session"?session(names[i])==group:names[i][0]==group[0])ix.push_back(i);
            }
            if(ix.size()<3)throw std::runtime_error("Insufficient branch training");
            std::set<std::string> ss;
            for(int i:ix)ss.insert(session(names[i]));
            std::vector<std::string> sessions(ss.begin(),ss.end());
            int nb=quadratic?2:1,extra=mode=="inferred_session"?int(sessions.size()):(mode=="free"||mode=="separate_session"?1:0);
            cv::Mat A=cv::Mat::zeros(int(ix.size()),nb+extra,CV_64F),Z(int(ix.size()),X.cols,CV_64F);
            for(int k=0; k<int(ix.size()); k++){
                double value=std::stod(names[ix[k]].substr(1));
                A.at<double>(k,0)=value;
                if(quadratic)A.at<double>(k,1)=value*value;
                if(mode=="inferred_session"){
                    for(int j=0; j<extra; j++)A.at<double>(k,nb+j)=session(names[ix[k]])==sessions[j]?1:0;
                }
                else if(extra)A.at<double>(k,nb)=1;
                X.row(ix[k]).copyTo(Z.row(k));
            }
            cv::Mat co;
            cv::solve(A,Z,co,cv::DECOMP_SVD);
            cv::Mat r=Z-A*co;
            residual.push_back(r);
            dof+=A.cols;
            cv::Mat slope=co.row(0),curve=quadratic?co.row(1).clone():cv::Mat::zeros(1,X.cols,CV_64F);
            auto branch=[&](const std::string& key,const cv::Mat& origin){
                Json b=Json::list();
                b.push(std::string(1,group[0]));
                b.push(jsonmatrix(origin).at(0));
                b.push(jsonmatrix(slope).at(0));
                b.push(jsonmatrix(curve).at(0));
                branches[key]=b;
            };
            if(mode=="inferred_session")for(int j=0; j<extra; j++)branch(sessions[j],co.row(nb+j));
            else branch(group,extra?co.row(nb):cv::Mat::zeros(1,X.cols,CV_64F));
        }
        model["branches"]=branches;
        model["W"]=jsonmatrix(whitening(residual,covariance_dof?covariance_dof:dof,shrink));
        model["zero_feature"]=jsonmatrix(cv::Mat::zeros(1,X.cols,CV_64F)).at(0);
        model["mode"]=mode;
        model["shrink"]=shrink;
        model["quadratic"]=quadratic;
        Json tr=Json::list();
        for(int i:training)tr.push(names[i]);
        model["training_files"]=tr;
        return model;
    }
}
