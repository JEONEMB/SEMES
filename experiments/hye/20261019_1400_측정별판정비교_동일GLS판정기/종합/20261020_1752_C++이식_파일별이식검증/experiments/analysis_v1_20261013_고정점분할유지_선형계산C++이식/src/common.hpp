#pragma once
#include "alignment.hpp"
#include "table.hpp"
#include "pickle_data.hpp"
namespace historical_v1 {
    using namespace semes;
    inline cv::Mat array(const Npy& values){
        if(values.shape.size()!=2)throw std::runtime_error("Expected 2D array");
        cv::Mat result(int(values.shape[0]),int(values.shape[1]),CV_64F);
        for(size_t i=0; i<values.count(); ++i)result.ptr<double>()[i]=values.number(i);
        return result;
    }
    inline cv::Mat selected(const cv::Mat& values,const std::vector<int>& rows){
        cv::Mat result;
        for(int row:rows)result.push_back(values.row(row));
        return result;
    }
    inline cv::Mat columns(const cv::Mat& values,const std::vector<int>& indices){
        cv::Mat out(values.rows,int(indices.size()),CV_64F);
        for(int j=0; j<out.cols; ++j)values.col(indices[j]).copyTo(out.col(j));
        return out;
    }
    inline std::vector<int> indices(const Json& list){
        std::vector<int> out;
        for(auto& value:list.array)out.push_back(int(value.num()));
        return out;
    }
    inline cv::Mat append_one(const cv::Mat& values){
        cv::Mat result;
        cv::hconcat(values,cv::Mat::ones(values.rows,1,CV_64F),result);
        return result;
    }
    inline cv::Mat least_squares(const cv::Mat& a,const cv::Mat& b){
        cv::Mat singular,u,vt;
        cv::SVD::compute(a,singular,u,vt);
        cv::Mat projected=u.t()*b;
        double cutoff=std::numeric_limits<double>::epsilon()*std::max(a.rows,a.cols)*singular.at<double>(0);
        for(int i=0; i<singular.rows; ++i){
            double value=singular.at<double>(i);
            if(value>cutoff)projected.row(i)/=value;
            else projected.row(i).setTo(0);
        }
        return vt.t()*projected;
    }
    inline cv::Mat population_sd(const cv::Mat& values){
        cv::Mat mean,squared,sd;
        cv::reduce(values,mean,0,cv::REDUCE_AVG);
        auto residual=values-cv::repeat(mean,values.rows,1);
        cv::multiply(residual,residual,squared);
        cv::reduce(squared,sd,0,cv::REDUCE_AVG);
        cv::sqrt(sd,sd);
        return sd;
    }
    struct Configuration{
        std::string name,group;
        cv::Mat values;
    };
    struct Data {
        cv::Mat x,displacements,measurements,points,ecc,qr;
        std::vector<std::vector<std::string>> metadata;
        std::vector<int> keep;
        std::vector<std::string> names;
        std::vector<Configuration> configurations;
        Json combinations,stable;
        Data(const fs::path& root){
            fs::path dir=root/"analysis_v3/v1_data";
            auto raw=read_zip(dir/"tracks.npz");
            auto labels=npy(raw.at("X.npy")),measurement=npy(raw.at("M.npy")),point_data=npy(raw.at("pts.npy")),files=npy(raw.at("files.npy"));
            points=array(point_data);
            cv::Mat all_x=array(labels);
            if(files.shape.size()!=2||files.shape[0]!=size_t(all_x.rows)||files.shape[1]!=4||files.dtype.size()<2||files.dtype[1]!='O')throw std::runtime_error("Expected original files object metadata");
            auto all_metadata=PickleData(files.data).string_rows(files.shape[0],files.shape[1]);
            bool center_seen=false;
            for(int i=0; i<all_x.rows; ++i){
                auto& row=all_metadata[i];
                double command=std::stod(row[1]);
                bool theta=row[0]==u8"Θ";
                if(std::abs(all_x.at<double>(i,theta?0:1)-command)>1e-12||std::abs(all_x.at<double>(i,theta?1:0))>1e-12)throw std::runtime_error("Metadata/label order mismatch");
                if(row[2]=="center.bmp"){
                    if(center_seen)continue;
                    center_seen=true;
                }
                keep.push_back(i);
                metadata.push_back(row);
                names.push_back(row[2]);
            }
            if(!center_seen||keep.size()!=281)throw std::runtime_error("Expected 281 unique images after center deduplication");
            x=selected(all_x,keep);
            if(measurement.shape!=std::vector<size_t>{
                size_t(all_x.rows),size_t(points.rows),3
            })throw std::runtime_error("Track array shape mismatch");
            measurements=cv::Mat(all_x.rows,points.rows*3,CV_64F);
            for(size_t i=0; i<measurement.count(); ++i)measurements.ptr<double>()[i]=measurement.number(i);
            measurements=selected(measurements,keep);
            displacements=cv::Mat(x.rows,points.rows,CV_64F);
            for(int i=0; i<x.rows; ++i)for(int j=0; j<points.rows; ++j)displacements.at<double>(i,j)=measurements.at<double>(i,3*j);
            ecc=selected(array(npy(bytes(dir/"ecc_feats.npy"))),keep);
            auto qr_data=npz(dir/"qr_tracks.npz");
            qr=selected(array(qr_data.at("D")),keep);
            std::vector<int> good_qr;
            for(size_t i=0; i<qr_data.at("good").count(); ++i)good_qr.push_back(int(qr_data.at("good").number(i)));
            combinations=Json::parse(read(dir/"combos.json"));
            stable=Json::parse(read(dir/"stable.json"));
            CsvTable geometry(root/fs::u8path(u8"14_데이터_QR마커위치.csv")),finder(root/fs::u8path(u8"15_데이터_파인더_볼트이동량.csv"));
            auto from_table=[&](const CsvTable& table,const std::vector<std::string>& fields){
                cv::Mat out=cv::Mat::zeros(x.rows,int(fields.size()),CV_64F);
                std::map<std::string,size_t> index;
                for(size_t i=0; i<table.rows.size(); ++i)if(!index.emplace(table.text(i,"file"),i).second)throw std::runtime_error("Duplicate table filename");
                for(int i=0; i<x.rows; ++i)if(index.count(names[i]))for(int j=0; j<out.cols; ++j)out.at<double>(i,j)=table.number(index.at(names[i]),fields[j]);
                return out;
            };
            cv::Mat g14=from_table(geometry,{
                "dLx","dCx","dRx","dLy","dCy","dRy","Lw","Cw","Rw","Lh","Ch","Rh","LC_gap","CR_gap"
            }),f15=from_table(finder,{
                "finder_dx","finder_dy","bolt_dx"
            });
            int center=int(std::find(names.begin(),names.end(),"center.bmp")-names.begin());
            for(int j=6; j<g14.cols; ++j){
                double origin=g14.at<double>(center,j);
                for(int i=0; i<g14.rows; ++i)g14.at<double>(i,j)-=origin;
            }
            cv::Mat mean;
            cv::reduce(g14.colRange(0,3),mean,1,cv::REDUCE_AVG);
            cv::Mat combined;
            cv::hconcat(g14,columns(displacements,indices(stable.at("best"))),combined);
            configurations={
                {
                    u8"Q1 QR 3개 평균 이동량 (1개)",u8"QR만",mean
                },{
                    u8"Q2 파인더 이동량 (1개)",u8"QR만",f15.col(0)
                }, {
                    u8"Q3 QR 3개 가로/세로 이동량 (6개)",u8"QR만",g14.colRange(0,6)
                },{
                    u8"Q4 QR 3개 이동량+크기+간격 (14개)",u8"QR만",g14
                }, {
                    u8"Q5 QR 전체 호모그래피 (8개)",u8"QR만",ecc.colRange(0,8)
                },{
                    u8"Q6 QR 호모그래피+QR별 affine (26개)",u8"QR만",ecc
                }, {
                    u8"Q7 QR 내부 특징점 (263개)",u8"QR만",columns(qr,good_qr)
                },{
                    u8"O1 파인더+볼트 (2개)",u8"QR 밖 포함",columns(f15,{
                        0,2
                    })
                }, {
                    u8"O2 조합A 파인더+벽볼트 (2개)",u8"QR 밖 포함",columns(displacements,{
                        0,47
                    })
                },{
                    u8"O3 조합B2 브래킷 윗판+아래 (2개)",u8"QR 밖 포함",columns(displacements,indices(stable.at("best")))
                }, {
                    u8"O4 안 바뀌는 특징점 전부 (51개)",u8"QR 밖 포함",columns(displacements,indices(stable.at("ok")))
                },{
                    u8"O5 특징점 전부, 라벨 포함 (74개)",u8"QR 밖 포함",columns(displacements,indices(combinations.at("good")))
                }, {
                    u8"O6 QR 기하 14개 + 조합B2 2개 (16개)",u8"QR 밖 포함",combined
                }
            };
        }
    };
    inline cv::Mat predict(const cv::Mat& features,const cv::Mat& labels,const std::vector<int>& train,const std::vector<int>& test,int algorithm,double* selected_alpha=nullptr){
        if(algorithm==1&&features.cols!=1)return cv::Mat();
        if((algorithm==4||algorithm==5)&&features.cols<2)return cv::Mat();
        auto ftr=selected(features,train),fte=selected(features,test),target=selected(labels,train);
        if(algorithm==1){
            cv::Mat a,b;
            for(int i=0; i<target.rows; ++i)if(target.at<double>(i,1)==0){
                cv::Mat row=(cv::Mat_<double>(1,2)<<target.at<double>(i,0),1);
                a.push_back(row);
                b.push_back(ftr.row(i));
            }
            auto co=least_squares(a,b);
            cv::Mat out=cv::Mat::zeros(fte.rows,2,CV_64F);
            for(int i=0; i<fte.rows; ++i)out.at<double>(i,0)=(fte.at<double>(i,0)-co.at<double>(1,0))/co.at<double>(0,0);
            return out;
        }
        if(algorithm==2||algorithm==3){
            cv::Mat mean;
            cv::reduce(ftr,mean,0,cv::REDUCE_AVG);
            auto sd=population_sd(ftr)+1e-12;
            cv::Mat a,b;
            cv::divide(ftr-cv::repeat(mean,ftr.rows,1),cv::repeat(sd,ftr.rows,1),a);
            cv::divide(fte-cv::repeat(mean,fte.rows,1),cv::repeat(sd,fte.rows,1),b);
            a=append_one(a);
            b=append_one(b);
            if(algorithm==2){
                if(a.cols>a.rows*.8)return cv::Mat();
                return b*least_squares(a,target);
            }
            auto fit=[](const cv::Mat& design,const cv::Mat& y,double alpha){
                cv::Mat gram=design.t()*design,coef;
                for(int j=0; j+1<gram.cols; ++j)gram.at<double>(j,j)+=alpha;
                if(!cv::solve(gram,design.t()*y,coef,cv::DECOMP_LU))throw std::runtime_error("Historical ridge solve failed");
                return coef;
            };
            double best_score=std::numeric_limits<double>::infinity(),best_alpha=0;
            for(double alpha:{
                1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0
            }){
                double squared=0;
                int count=0;
                for(int fold=0; fold<2; ++fold){
                    std::vector<int> inner_train,inner_test;
                    for(int i=0; i<a.rows; ++i)(i%2==fold?inner_test:inner_train).push_back(i);
                    cv::Mat error=selected(a,inner_test)*fit(selected(a,inner_train),selected(target,inner_train),alpha)-selected(target,inner_test);
                    for(int i=0; i<error.rows; ++i){
                        squared+=std::pow(error.at<double>(i,0)*80,2)+std::pow(error.at<double>(i,1)*14.3,2);
                        count+=2;
                    }
                }
                double score=std::sqrt(squared/count);
                if(score<best_score){
                    best_score=score;
                    best_alpha=alpha;
                }
            }
            if(selected_alpha)*selected_alpha=best_alpha;
            return b*fit(a,target,best_alpha);
        }
        auto a=append_one(target),co=least_squares(a,ftr);
        cv::Mat jacobian=co.rowRange(0,2).t(),intercept=co.row(2),residual=ftr-a*co,observed=fte-cv::repeat(intercept,fte.rows,1);
        if(algorithm==4){
            auto sd=population_sd(residual)+1e-12;
            cv::Mat scaled_j,scaled_y;
            cv::divide(jacobian,cv::repeat(sd.t(),1,2),scaled_j);
            cv::divide(observed,cv::repeat(sd,observed.rows,1),scaled_y);
            return (least_squares(scaled_j,scaled_y.t())).t();
        }
        cv::Mat mean;
        cv::reduce(residual,mean,0,cv::REDUCE_AVG);
        cv::Mat centered=residual-cv::repeat(mean,residual.rows,1),covariance=centered.t()*centered/(residual.rows-1);
        double jitter=.01*cv::trace(covariance)[0]/covariance.cols;
        for(int j=0; j<covariance.cols; ++j)covariance.at<double>(j,j)+=jitter;
        cv::Mat inverse;
        if(!cv::invert(covariance,inverse,cv::DECOMP_LU))throw std::runtime_error("Historical GLS covariance singular");
        cv::Mat mapping;
        if(!cv::solve(jacobian.t()*inverse*jacobian,jacobian.t()*inverse,mapping,cv::DECOMP_LU))throw std::runtime_error("Historical GLS solve failed");
        return observed*mapping.t();
    }
}
