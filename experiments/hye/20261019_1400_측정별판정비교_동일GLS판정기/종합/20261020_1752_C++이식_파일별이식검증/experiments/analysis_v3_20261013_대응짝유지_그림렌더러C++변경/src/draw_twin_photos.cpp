#include "common.hpp"
using namespace twin_history;
struct PhotoRenderer {
    Dataset& dataset;
    fs::path data;
    int cx,cy;
    std::map<std::string,cv::Mat> cache;
    PhotoRenderer(Dataset& d,const fs::path& root,const fs::path& images):dataset(d),data(images){
        CsvTable q(root/fs::u8path(u8"14_데이터_QR마커위치.csv"));
        bool found=false;
        for(size_t i=0; i<q.rows.size(); ++i)if(q.text(i,"file")=="center.bmp"){
            cx=int(q.number(i,"Cx"));
            cy=int(q.number(i,"Cy"));
            found=true;
        }
        if(!found)throw std::runtime_error("No center metadata");
    }
    cv::Mat crop(char axis,int command,int size){
        std::string name=command==0?"center.bmp":std::string(1,axis)+signed_value(command/100.)+".bmp";
        auto at=cache.find(name);
        if(at==cache.end()){
            auto image=read_image(data,name);
            cv::Rect roi(cx-450,cy-450,900,900);
            if((roi&cv::Rect(0,0,image.cols,image.rows))!=roi)throw std::runtime_error("Crop outside image");
            at=cache.emplace(name,image(roi).clone()).first;
        }
        cv::Mat result;
        cv::resize(at->second,result,cv::Size(size,size),0,0,cv::INTER_AREA);
        cv::cvtColor(result,result,cv::COLOR_GRAY2BGR);
        return result;
    }
    Json draw(int limit,int tsz,const fs::path& out,const std::string& filename){
        int count=limit*2+1,gap=std::max(2,tsz/12),width=count*(tsz+gap)+120,ysz=std::min((width-120)/81-gap,3*tsz);
        int head=60+std::max(30,tsz/2)+std::max(20,tsz/3)+60,ph=ysz+tsz+520+std::max(28,tsz/2)+std::max(18,ysz/4),height=head+2*ph+60;
        cv::Mat image(height,width,CV_8UC3,cv::Scalar::all(255));
        auto xt=[&](double v){
            return 60+(v+limit/100.)/(2*limit/100.)*(count-1)*(tsz+gap)+tsz/2.;
        };
        auto xy=[&](double v){
            return 60+(v+2)/4*80*(ysz+gap)+ysz/2.+((width-120)-81*(ysz+gap))/2.;
        };
        text(image,u8"QR 위치가 같은 Y/T 대응 짝 · Y 80장 + center / T "+std::to_string(count-1)+u8"장 + center",{
            60,30
        },std::max(30,tsz/2),ink,true);
        text(image,u8"굵은 테두리: 구분 불가 · 흐린 사진: 구분 가능 · 파란 실선 Y→T / 빨간 점선 T→Y. 원본의 동일한 900×900px 영역 사용.",{
            60,44+std::max(30,tsz/2)
        },std::max(20,tsz/3),sub);
        int pasted=0,clipped_connections=0;
        for(int p=0; p<2; ++p){
            auto& panel=dataset.panels[p];
            auto by=bad(panel.yt),bt=bad(panel.ty);
            int top=head+p*ph,yt=top+40+std::max(28,tsz/2)+std::max(18,ysz/4)+30,yb=yt+ysz+330;
            std::string title=panel.label+u8"  Y 쪽 "+std::to_string(panel.yt.size())+u8"세트 / T 쪽 "+std::to_string(panel.ty.size())+u8"세트";
            text(image,title,{
                60,top+20
            },std::max(28,tsz/2),ink,true);
            for(auto& pair:panel.yt){
                if(std::abs(pair.target)<=limit/100.)connection(image,{
                    xy(pair.source),double(yt+ysz+6)
                },{
                    xt(pair.target),double(yb-6)
                },blue,std::max(2,tsz/25));
                else ++clipped_connections;
            }
            for(auto& pair:panel.ty){
                if(std::abs(pair.source)<=limit/100.)connection(image,{
                    xt(pair.source),double(yb-6)
                },{
                    xy(pair.target),double(yt+ysz+6)
                },red,std::max(1,tsz/35),true);
                else ++clipped_connections;
            }
            for(int row=0; row<2; ++row){
                int lo=row?-limit:-200,hi=row?limit:200,step=row?1:5,size=row?tsz:ysz,y=row?yb:yt;
                auto& flags=row?bt:by;
                auto color=row?green:orange;
                int index=0;
                for(int k=lo; k<=hi; k+=step,++index){
                    double xcenter=row?xt(k/100.):xy(k/100.);
                    int x=int(xcenter-size/2.);
                    auto photo=crop(row?'T':'Y',k,size);
                    bool is_bad=flags.count(k);
                    if(!is_bad&&k)photo=photo*.4+cv::Scalar::all(255)*.6;
                    cv::Rect rect(x,y,size,size);
                    if((rect&cv::Rect(0,0,width,height))!=rect)throw std::runtime_error("Thumbnail outside canvas");
                    photo.copyTo(image(rect));
                    int border=(is_bad||!k)?std::max(3,size/14):std::max(1,size/40);
                    cv::rectangle(image,cv::Rect(x-border,y-border,size+2*border,size+2*border),!k?ink:color,border);
                    ++pasted;
                    int tick=row?(limit>=100?10:5):4;
                    if(index%tick==0||!k){
                        int font=std::max(18,size/4);
                        text(image,k?(std::string(row?"T ":"Y ")+signed_value(k/100.)):"center",{
                            int(xcenter),row?y+size+10:y-font-14
                        },font,sub,false,true);
                    }
                }
            }
            if(!panel.yt.empty()){
                auto bounds=std::minmax_element(panel.yt.begin(),panel.yt.end(),[](const Pair& a,const Pair& b){
                    return a.source<b.source;
                });
                double target_lo=1e100,target_hi=-1e100;
                for(auto& pair:panel.yt){
                    target_lo=std::min(target_lo,pair.target);
                    target_hi=std::max(target_hi,pair.target);
                }
                text(image,u8"구분 불가 Y: "+signed_value(bounds.first->source)+" ~ "+signed_value(bounds.second->source)+u8"mm / 대응 T: "+signed_value(target_lo,3)+" ~ "+signed_value(target_hi,3),{
                    100+text_width(title,std::max(28,tsz/2),true),top+20
                },std::max(28,tsz/2),blue,true);
            }
        }
        std::ostringstream scale;
        scale<<u8"축소 배율: Y "<<std::setprecision(5)<<ysz/900.<<u8"배 / T "<<tsz/900.<<u8"배. 명령값 보간 짝이며 실제 촬영 시간순 궤적이 아님. T 물리 단위 미확정.";
        text(image,scale.str(),{
            60,height-46
        },24,sub);
        save(out/fs::u8path(filename),image);
        cv::Mat preview;
        cv::resize(image,preview,cv::Size(2400,int(std::round(image.rows*2400./image.cols))),0,0,cv::INTER_AREA);
        save(out/fs::u8path(filename+".preview.png"),preview);
        Json item=Json::dict();
        item["file"]=filename;
        item["width"]=width;
        item["height"]=height;
        item["Y_thumbnail_px"]=ysz;
        item["T_thumbnail_px"]=tsz;
        item["photos_pasted"]=pasted;
        item["connections_outside_display_range"]=clipped_connections;
        item["fixed_crop_px"]=900;
        return item;
    }
};
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try{
        if(argc!=5)throw std::runtime_error("historical_twin_photos original-workspace bmp-20261014 cpp-v2-decisions.csv new-output");
        fs::path root=fs::u8path(argv[1]),data=fs::u8path(argv[2]),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Dataset dataset(root,fs::u8path(argv[3]));
        fs::create_directories(out);
        PhotoRenderer renderer(dataset,root,data);
        Json figures=Json::list();
        figures.push(renderer.draw(40,150,out,u8"28_구분안되는짝_사진연결도_확대.png"));
        figures.push(renderer.draw(100,64,out,u8"28_구분안되는짝_사진연결도_전체.png"));
        auto audit=dataset.audit(root,out);
        audit["figures"]=figures;
        audit["unique_source_photos_read"]=int(renderer.cache.size());
        audit["passed"]=true;
        write(out/"checks.json",audit.dump());
        std::cout<<audit.dump()<<'\n';
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
