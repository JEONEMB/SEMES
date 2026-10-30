#pragma once
#include "io.hpp"
#include "pickle_data.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>
namespace qr_history {
    using namespace semes;
    inline cv::Mat matrix(const Npy& array){
        if(array.shape.size()!=2)throw std::runtime_error("Expected numeric matrix");
        cv::Mat out(int(array.shape[0]),int(array.shape[1]),CV_64F);
        for(size_t i=0; i<array.count(); ++i)out.ptr<double>()[i]=array.number(i);
        return out;
    }
    inline cv::Mat least_squares(const cv::Mat& a,const cv::Mat& b){
        cv::Mat w,u,vt;
        cv::SVD::compute(a,w,u,vt);
        cv::Mat projected=u.t()*b;
        double cutoff=std::numeric_limits<double>::epsilon()*std::max(a.rows,a.cols)*w.at<double>(0);
        for(int i=0; i<w.rows; ++i){
            double value=w.at<double>(i);
            if(value>cutoff)projected.row(i)/=value;
            else projected.row(i).setTo(0);
        }
        return vt.t()*projected;
    }
    inline cv::Mat gray(const fs::path& file){
        auto image=cv::imdecode(bytes(file),cv::IMREAD_GRAYSCALE);
        if(image.size()!=cv::Size(3840,2160))throw std::runtime_error("Expected full 3840x2160 image "+file.u8string());
        return image;
    }
    struct Input {
        cv::Mat labels;
        std::vector<std::vector<std::string>> files;
        Input(const fs::path& workspace){
            auto zip=read_zip(workspace/"analysis_v3/v1_data/tracks.npz");
            labels=matrix(npy(zip.at("X.npy")));
            auto metadata=npy(zip.at("files.npy"));
            files=PickleData(metadata.data).string_rows(size_t(labels.rows),4);
            for(int i=0; i<labels.rows; ++i){
                double command=std::stod(files[i][1]);
                bool theta=files[i][0]==u8"Θ";
                if(labels.at<double>(i,theta?0:1)!=command||labels.at<double>(i,theta?1:0)!=0)throw std::runtime_error("Original row labels disagree with metadata");
            }
        }
        fs::path path(const fs::path& data,int row)const{
            auto file=files.at(row)[2];
            return file=="center.bmp"?data/"center.bmp":data/(file[0]=='T'?"Theta":"Y")/fs::u8path(file);
        }
    };
    inline std::vector<std::vector<cv::Point2f>> polygons(const cv::Mat& center){
        cv::QRCodeDetectorAruco detector;
        cv::Mat detected;
        if(!detector.detectMulti(center,detected))throw std::runtime_error("ArUco QR detectMulti failed");
        if(detected.total()!=12||detected.channels()!=2)throw std::runtime_error("Expected three QR quadrilaterals");
        detected=detected.reshape(2,3);
        std::vector<std::vector<cv::Point2f>> output;
        for(int i=0; i<3; ++i){
            std::vector<cv::Point2f> quad;
            for(int j=0; j<4; ++j)quad.push_back(detected.at<cv::Point2f>(i,j));
            output.push_back(quad);
        }
        std::sort(output.begin(),output.end(),[](auto& a,auto& b){
            double ax=0,bx=0; for(auto p:a)ax+=p.x; for(auto p:b)bx+=p.x; return ax<bx;
        });
        return output;
    }
    inline Json polygon_json(const std::vector<std::vector<cv::Point2f>>& quads){
        Json result=Json::list();
        for(auto& quad:quads){
            Json q=Json::list();
            for(auto p:quad){
                Json point=Json::list();
                point.push(double(p.x));
                point.push(double(p.y));
                q.push(point);
            }
            result.push(q);
        }
        return result;
    }
    inline std::string npy_blob(const cv::Mat& array,const std::string& dtype,const std::vector<size_t>& shape){
        std::ostringstream header;
        header<<"{'descr': '"<<dtype<<"', 'fortran_order': False, 'shape': (";
        for(auto value:shape)header<<value<<", ";
        header<<"), }";
        auto text=header.str();
        text.append((64-(10+text.size()+1)%64)%64,' ');
        text+='\n';
        if(text.size()>65535)throw std::runtime_error("NPY header too long");
        std::string out("\x93NUMPY\x01\x00",8);
        out+=char(text.size()&255);
        out+=char((text.size()>>8)&255);
        out+=text;
        auto data=array.isContinuous()?array:array.clone();
        out.append(reinterpret_cast<const char*>(data.data),data.total()*data.elemSize());
        return out;
    }
    inline void png(const fs::path& path,const cv::Mat& image){
        std::vector<uchar> encoded;
        if(!cv::imencode(".png",image,encoded))throw std::runtime_error("PNG failed");
        std::ofstream file(path,std::ios::binary);
        file.write(reinterpret_cast<const char*>(encoded.data()),encoded.size());
    }
    inline double subpixel(const cv::Mat& response,int x,int y){
        if(x==0||x+1==response.cols)return x;
        float a=response.at<float>(y,x-1),b=response.at<float>(y,x),c=response.at<float>(y,x+1);
        float d=a-2.0f*b+c;
        return x+(d!=0?.5*double(a-c)/d:0);
    }
}
