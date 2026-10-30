#include "common.hpp"
#include <iostream>
using namespace semes;
namespace {
    void scatter(cv::Mat& image,cv::Rect panel,const std::vector<double>& x,const std::vector<double>& y,const std::vector<std::string>& axis,const std::string& xlabel,const std::string& ylabel) {
        double xmin=*std::min_element(x.begin(),x.end()),xmax=*std::max_element(x.begin(),x.end()),ymin=*std::min_element(y.begin(),y.end()),ymax=*std::max_element(y.begin(),y.end());
        double px=std::max((xmax-xmin)*.06,1e-9),py=std::max((ymax-ymin)*.06,1e-9);
        xmin-=px;
        xmax+=px;
        ymin-=py;
        ymax+=py;
        cv::Rect plot(panel.x+85,panel.y+35,panel.width-120,panel.height-100);
        cv::rectangle(image,plot,cv::Scalar(170,170,170),1);
        for(int k=0; k<=4; ++k){
            int xx=plot.x+k*plot.width/4,yy=plot.y+k*plot.height/4;
            cv::line(image,{
                xx,plot.y
            },{
                xx,plot.y+plot.height
            },{
                235,235,235
            });
            cv::line(image,{
                plot.x,yy
            },{
                plot.x+plot.width,yy
            },{
                235,235,235
            });
            cv::putText(image,agent_history::num(xmin+(xmax-xmin)*k/4).substr(0,8),{
                xx-(k==4?55:20),plot.y+plot.height+22
            },cv::FONT_HERSHEY_SIMPLEX,.42,{
                60,60,60
            });
            cv::putText(image,agent_history::num(ymax-(ymax-ymin)*k/4).substr(0,8),{
                panel.x+4,yy+4
            },cv::FONT_HERSHEY_SIMPLEX,.42,{
                60,60,60
            });
        }
        for(size_t i=0; i<x.size(); ++i){
            int xx=plot.x+int((x[i]-xmin)/(xmax-xmin)*plot.width),yy=plot.y+plot.height-int((y[i]-ymin)/(ymax-ymin)*plot.height);
            cv::circle(image,{
                xx,yy
            },2,axis[i]=="T"?cv::Scalar(2,95,217):cv::Scalar(119,158,27),-1);
        }
        cv::putText(image,xlabel,{
            plot.x,plot.y+plot.height+48
        },cv::FONT_HERSHEY_SIMPLEX,.46,{
            25,25,25
        });
        cv::putText(image,ylabel,{
            plot.x,plot.y-12
        },cv::FONT_HERSHEY_SIMPLEX,.46,{
            25,25,25
        });
        cv::circle(image,{
            panel.x+panel.width-110,panel.y+18
        },4,{
            2,95,217
        },-1);
        cv::putText(image,"T",{
            panel.x+panel.width-100,panel.y+23
        },cv::FONT_HERSHEY_SIMPLEX,.45,{
            25,25,25
        });
        cv::circle(image,{
            panel.x+panel.width-65,panel.y+18
        },4,{
            119,158,27
        },-1);
        cv::putText(image,"Y",{
            panel.x+panel.width-55,panel.y+23
        },cv::FONT_HERSHEY_SIMPLEX,.45,{
            25,25,25
        });
    }
}
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try {
        if(argc!=3)throw std::runtime_error("historical_agent_evaluate qr_measurements.csv new-output");
        CsvTable data(fs::u8path(argv[1]));
        fs::path out=fs::u8path(argv[2]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        fs::create_directories(out);
        std::vector<size_t> moving,train,test;
        std::vector<double> shift,bend,vertical,gradient;
        std::vector<std::string> axis;
        for(size_t i=0; i<data.rows.size(); ++i)if(data.text(i,"axis")!="center") {
            auto p=agent_history::positions(data,i);
            moving.push_back(i);
            axis.push_back(data.text(i,"axis"));
            shift.push_back((p[0]+p[2]+p[4])/3);
            bend.push_back(p[2]-(p[0]+p[4])/2);
            vertical.push_back(p[5]-p[1]);
            gradient.push_back(data.number(i,"q2_a00")-data.number(i,"q0_a00"));
            double value=data.number(i,"value"),step=std::nearbyint(std::abs(value)*(axis.back()=="T"?100:20));
            (int(step)%2?train:test).push_back(moving.size()-1);
        }
        cv::Mat X(int(train.size()),3,CV_64F),target(int(train.size()),2,CV_64F);
        cv::Vec2d mean{
        },scale{
        };
        for(auto i:train){
            mean[0]+=shift[i];
            mean[1]+=bend[i];
        }
        mean*=1./train.size();
        for(auto i:train){
            scale[0]+=std::pow(shift[i]-mean[0],2);
            scale[1]+=std::pow(bend[i]-mean[1],2);
        }
        for(int k=0; k<2; ++k)scale[k]=std::sqrt(scale[k]/train.size());
        for(int k=0; k<int(train.size()); ++k){
            auto i=train[k];
            double v=data.number(moving[i],"value");
            X.at<double>(k,0)=(shift[i]-mean[0])/scale[0];
            X.at<double>(k,1)=(bend[i]-mean[1])/scale[1];
            X.at<double>(k,2)=1;
            target.at<double>(k,0)=axis[i]=="T"?v:0;
            target.at<double>(k,1)=axis[i]=="Y"?v:0;
        }
        cv::Mat coefficient;
        cv::solve(X,target,coefficient,cv::DECOMP_SVD);
        std::ofstream output(out/"heldout_predictions.csv");
        auto header=data.header;
        for(auto name:{
            "x0","y0","x1","y1","x2","y2","shift_x","outer_spacing_change","middle_bend_x","outer_vertical_change","scale_gradient","shear_gradient","step","theta_true","y_true","theta_pred","y_pred"
        })header.push_back(name);
        agent_history::write_row(output,header);
        std::vector<double> truthT,predT;
        std::vector<std::string> testaxis;
        double sumT=0,sumY=0;
        std::array<int,6> subset_count{
        };
        std::array<double,6> subset_T{
        },subset_Y{
        };
        const double shift_limits[]={
            2,5,10,15,30,80
        };
        for(auto i:test){
            cv::Mat input=(cv::Mat_<double>(1,3)<<(shift[i]-mean[0])/scale[0],(bend[i]-mean[1])/scale[1],1);
            cv::Mat pred=input*coefficient;
            double v=data.number(moving[i],"value"),t=axis[i]=="T"?v:0,y=axis[i]=="Y"?v:0;
            auto row=data.rows[moving[i]];
            auto positions=agent_history::positions(data,moving[i]);
            for(double value:positions)row.push_back(agent_history::num(value));
            double shear=data.number(moving[i],"q2_a01")-data.number(moving[i],"q0_a01");
            for(double value:{
                shift[i],positions[4]-positions[0],bend[i],vertical[i],gradient[i],shear,std::nearbyint(std::abs(v)*(axis[i]=="T"?100:20)),t,y,pred.at<double>(0),pred.at<double>(1)
            })row.push_back(agent_history::num(value));
            agent_history::write_row(output,row);
            double errorT=std::abs(pred.at<double>(0)-t),errorY=std::abs(pred.at<double>(1)-y);
            sumT+=errorT;
            sumY+=errorY;
            for(int k=0; k<6; ++k)if(std::abs(shift[i])<=shift_limits[k]){
                ++subset_count[k];
                subset_T[k]+=errorT;
                subset_Y[k]+=errorY;
            }
            truthT.push_back(t);
            predT.push_back(pred.at<double>(0));
            testaxis.push_back(axis[i]);
        }
        cv::Mat figure(1100,1600,CV_8UC3,cv::Scalar(255,255,255));
        scatter(figure,{
            0,0,800,550
        },shift,bend,axis,"Mean QR horizontal shift (px)","Middle QR bend (px)");
        scatter(figure,{
            800,0,800,550
        },shift,vertical,axis,"Mean QR horizontal shift (px)","Outer QR vertical difference (px)");
        scatter(figure,{
            0,550,800,550
        },shift,gradient,axis,"Mean QR horizontal shift (px)","Outer QR scale gradient");
        scatter(figure,{
            800,550,800,550
        },truthT,predT,testaxis,"Commanded T","Predicted T");
        std::vector<unsigned char> png;
        cv::imencode(".png",figure,png);
        write(out/"qr_geometry.png",std::string(png.begin(),png.end()));
        Json summary=Json::dict(),subsets=Json::list(),line_statistics=Json::list();
        for(int k=0; k<6; ++k)if(subset_count[k]){
            Json s=Json::dict();
            s["max_shift_px"]=shift_limits[k];
            s["n"]=subset_count[k];
            s["T_MAE"]=subset_T[k]/subset_count[k];
            s["Y_MAE"]=subset_Y[k]/subset_count[k];
            subsets.push(s);
        }
        for(auto ax:{
            "T","Y"
        })for(int feature=0; feature<2; ++feature){
            std::vector<size_t> selected;
            for(size_t i=0; i<moving.size(); ++i)if(axis[i]==ax)selected.push_back(i);
            cv::Mat a(int(selected.size()),2,CV_64F),b(int(selected.size()),1,CV_64F),co;
            for(int k=0; k<int(selected.size()); ++k){
                size_t i=selected[k];
                a.at<double>(k,0)=data.number(moving[i],"value");
                a.at<double>(k,1)=1;
                b.at<double>(k)=feature?bend[i]:shift[i];
            }
            cv::solve(a,b,co,cv::DECOMP_SVD);
            cv::Mat residual=b-a*co;
            double sd=cv::norm(residual)/std::sqrt(double(selected.size()-1));
            Json s=Json::dict();
            s["axis"]=ax;
            s["feature"]=feature?"middle_bend_x":"shift_x";
            s["slope"]=co.at<double>(0);
            s["intercept"]=co.at<double>(1);
            s["residual_sample_sd"]=sd;
            line_statistics.push(s);
        }
        summary["n"]=int(test.size());
        summary["train_n"]=int(train.size());
        summary["T_MAE"]=sumT/test.size();
        summary["Y_MAE"]=sumY/test.size();
        summary["shift_subsets"]=subsets;
        summary["line_statistics"]=line_statistics;
        summary["protocol"]="historical_odd_magnitudes_train_even_magnitudes_test";
        summary["render_change"]="Same scatter quantities and all points; OpenCV raster renderer replaces matplotlib layout.";
        summary["coefficients"]=jsonmatrix(coefficient);
        summary["input_sha256"]=sha256(fs::u8path(argv[1]));
        summary["created_kst"]=now();
        write(out/"summary.json",summary.dump());
        std::cout<<summary.dump()<<'\n';
        return 0;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
