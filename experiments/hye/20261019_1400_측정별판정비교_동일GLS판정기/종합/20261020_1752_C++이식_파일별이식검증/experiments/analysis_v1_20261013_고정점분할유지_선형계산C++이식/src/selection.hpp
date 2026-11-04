#pragma once
#include "common.hpp"
namespace historical_v1 {
    struct Tracks {
        cv::Mat x,dx,score,points;
        explicit Tracks(const fs::path& root){
            auto raw=read_zip(root/"analysis_v3/v1_data/tracks.npz");
            x=array(npy(raw.at("X.npy")));
            points=array(npy(raw.at("pts.npy")));
            auto m=npy(raw.at("M.npy"));
            if(m.shape!=std::vector<size_t>{
                size_t(x.rows),size_t(points.rows),3
            })throw std::runtime_error("Track shape");
            dx=cv::Mat(x.rows,points.rows,CV_64F);
            score=dx.clone();
            for(int i=0; i<dx.rows; ++i)for(int j=0; j<dx.cols; ++j){
                dx.at<double>(i,j)=m.number((size_t(i)*dx.cols+j)*3);
                score.at<double>(i,j)=m.number((size_t(i)*dx.cols+j)*3+2);
            }
        }
    };
    struct Score{
        double t,y,condition;
    };
    inline Score evaluate(const Tracks& data,const std::vector<int>& cols,const std::vector<int>& train,const std::vector<int>& test,const std::vector<double>& weights={
    }){
        auto observed=columns(data.dx,cols);
        cv::Mat jac=least_squares(selected(data.x,train),selected(observed,train)).t(),target=selected(observed,test).t();
        for(int i=0; i<jac.rows; ++i){
            double weight=weights.empty()?1:weights.at(i);
            jac.row(i)*=weight;
            target.row(i)*=weight;
        }
        cv::Mat prediction=least_squares(jac,target).t(),error=prediction-selected(data.x,test),singular;
        cv::SVD::compute(jac,singular);
        Score result{
            0,0,singular.at<double>(0)/singular.at<double>(singular.rows-1)
        };
        for(int i=0; i<error.rows; ++i){
            result.t+=std::pow(error.at<double>(i,0),2);
            result.y+=std::pow(error.at<double>(i,1),2);
        }
        result.t=std::sqrt(result.t/error.rows);
        result.y=std::sqrt(result.y/error.rows);
        return result;
    }
    inline Json ids_json(const std::vector<int>& values){
        Json out=Json::list();
        for(int value:values)out.push(value);
        return out;
    }
}
