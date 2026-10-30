#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>
namespace pixel_history {
    // Cubic cardinal B-spline coefficients with whole-sample mirror boundaries.
    // The recurrence and boundary equations are the order-3 spline equations;
    // SciPy ndimage's public/source definitions are cited in the experiment README.
    inline void spline_line(std::vector<double>& samples){
        if(samples.size()<2)return;
        constexpr double pole=-0.267949192431122706472553658494127633;
        double gain=(1-pole)*(1-1/pole);
        for(double& value:samples)value*=gain;
        const int n=int(samples.size());
        double last_power=std::pow(pole,n-1),first=samples[0]+last_power*samples[n-1],power=pole;
        for(int i=1; i<n-1; ++i){
            first+=power*(samples[i]+last_power*samples[n-1-i]);
            power*=pole;
        }
        samples[0]=first/(1-last_power*last_power);
        for(int i=1; i<n; ++i)samples[i]+=pole*samples[i-1];
        samples[n-1]=(pole*samples[n-2]+samples[n-1])*pole/(pole*pole-1);
        for(int i=n-2; i>=0; --i)samples[i]=pole*(samples[i+1]-samples[i]);
    }
    inline cv::Mat spline_coefficients(const cv::Mat& image){
        cv::Mat output;
        image.convertTo(output,CV_64F);
        std::vector<double> line(output.rows);
        for(int x=0; x<output.cols; ++x){
            for(int y=0; y<output.rows; ++y)line[y]=output.at<double>(y,x);
            spline_line(line);
            for(int y=0; y<output.rows; ++y)output.at<double>(y,x)=line[y];
        }
        line.resize(output.cols);
        for(int y=0; y<output.rows; ++y){
            for(int x=0; x<output.cols; ++x)line[x]=output.at<double>(y,x);
            spline_line(line);
            for(int x=0; x<output.cols; ++x)output.at<double>(y,x)=line[x];
        }
        return output;
    }
    inline void spline_weights(double coordinate,double* w){
        double t=coordinate-std::floor(coordinate),r=1-t;
        w[1]=(t*t*(t-2)*3+4)/6;
        w[2]=(r*r*(r-2)*3+4)/6;
        w[0]=r*r*r/6;
        w[3]=1-w[0]-w[1]-w[2];
    }
    inline double spline_value(const cv::Mat& coefficients,double x,double y){
        if(!std::isfinite(x)||!std::isfinite(y)||std::abs(x)>1e8||std::abs(y)>1e8)throw std::runtime_error("Invalid spline coordinate");
        double wx[4],wy[4];
        spline_weights(x,wx);
        spline_weights(y,wy);
        int start_x=int(std::floor(x))-1,start_y=int(std::floor(y))-1;
        double result=0;
        for(int row=0; row<4; ++row){
            int iy=std::clamp(start_y+row,0,coefficients.rows-1);
            const double* values=coefficients.ptr<double>(iy);
            for(int col=0; col<4; ++col){
                int ix=std::clamp(start_x+col,0,coefficients.cols-1);
                result+=values[ix]*wy[row]*wx[col];
            }
        }
        return result;
    }
    inline cv::Mat spline_warp(const cv::Mat& coefficients,const cv::Mat& mx,const cv::Mat& my){
        cv::Mat result(mx.size(),CV_32F);
        cv::parallel_for_(cv::Range(0,result.rows),[&](const cv::Range& range){
            for(int y=range.start; y<range.end; ++y){
                auto* output=result.ptr<float>(y); const auto* xs=mx.ptr<float>(y); const auto* ys=my.ptr<float>(y); for(int x=0; x<result.cols; ++x)output[x]=float(spline_value(coefficients,xs[x],ys[x]));
            }
        });
        return result;
    }
}
