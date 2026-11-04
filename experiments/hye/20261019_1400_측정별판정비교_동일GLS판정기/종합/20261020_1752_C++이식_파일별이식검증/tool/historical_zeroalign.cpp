#include "numeric.hpp"
#include "table.hpp"
#include "wintext.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
using namespace semes;
using namespace semes::numeric;
struct Shot{std::string file;double value,x,bolt,time,direction=0;int segment=0;};
struct Fit{std::vector<double> offset;double slope=0;int parameters=0,changes=0;};
static cv::Mat values(const std::vector<Shot>& a){cv::Mat b(int(a.size()),1,CV_64F);for(int i=0;i<b.rows;++i)b.at<double>(i)=a[i].x;return b;}
static std::vector<int> changepoints(const std::vector<double>& t,const std::vector<double>& r,double penalty){
    int n=int(t.size());std::vector<double> cost(n+1,std::numeric_limits<double>::infinity());std::vector<int> last(n+1);cost[0]=-penalty;
    for(int j=6;j<=n;++j)for(int i=0;i<=j-6;++i){
        if(!std::isfinite(cost[i])||(i>0&&i<6))continue;
        double tm=0;for(int k=i;k<j;++k)tm+=t[k];tm/=j-i;
        cv::Mat a(j-i,2,CV_64F),b(j-i,1,CV_64F);
        for(int k=i;k<j;++k){a.at<double>(k-i,0)=1;a.at<double>(k-i,1)=t[k]-tm;b.at<double>(k-i)=r[k];}
        auto residual=b-a*least_squares(a,b);double c=cost[i]+residual.dot(residual)+penalty;
        if(c<cost[j]){cost[j]=c;last[j]=i;}
    }
    if(!std::isfinite(cost[n]))throw std::runtime_error("No feasible change-point segmentation");
    std::vector<int> cps;for(int j=n;j>0;j=last[j])cps.push_back(j);std::sort(cps.begin(),cps.end());cps.pop_back();return cps;
}
static Fit fit(const std::vector<Shot>& g,const std::vector<int>& tr,int method,const std::pair<double,double>& witness){
    const int n=int(g.size());Fit f;f.offset.assign(n,0);auto x=values(g);
    if(method==2){
        for(int i=0;i<n;++i)f.offset[i]=witness.first*g[i].time+witness.second;
        cv::Mat a(int(tr.size()),1,CV_64F),b(int(tr.size()),1,CV_64F);
        for(int i=0;i<int(tr.size());++i){a.at<double>(i)=g[tr[i]].value;b.at<double>(i)=g[tr[i]].x-f.offset[tr[i]];}
        f.slope=least_squares(a,b).at<double>(0);f.parameters=1;return f;
    }
    if(method==5){
        cv::Mat v(n,1,CV_64F);for(int i=0;i<n;++i)v.at<double>(i)=g[i].value;
        f.slope=least_squares(rows(v,tr),rows(x,tr)).at<double>(0);
        for(int iteration=0;iteration<3;++iteration){
            std::vector<double> t,r,all(n),differences;
            for(int i=0;i<n;++i)all[i]=g[i].x-f.slope*g[i].value;
            for(int i:tr){t.push_back(g[i].time);r.push_back(all[i]);}
            for(size_t i=1;i<r.size();++i)differences.push_back(std::pow(r[i]-r[i-1],2));
            auto cps=changepoints(t,r,4*(median(differences)/(2*.455)+1e-9)*std::log(double(tr.size())));
            f.changes=int(cps.size());std::vector<int> bounds={0};bounds.insert(bounds.end(),cps.begin(),cps.end());bounds.push_back(int(tr.size()));
            std::vector<int> piece(n);
            for(int i=0;i<n;++i){int nearest=0;for(int j=1;j<int(tr.size());++j)if(std::abs(g[i].time-t[j])<std::abs(g[i].time-t[nearest]))nearest=j;piece[i]=int(std::upper_bound(cps.begin(),cps.end(),nearest)-cps.begin());}
            for(int s=0;s+1<int(bounds.size());++s){
                std::vector<double> tx(t.begin()+bounds[s],t.begin()+bounds[s+1]),rx(r.begin()+bounds[s],r.begin()+bounds[s+1]);auto p=line(tx,rx);
                for(int i=0;i<n;++i)if(piece[i]==s)f.offset[i]=p.first*g[i].time+p.second;
            }
            cv::Mat corrected(n,1,CV_64F);for(int i=0;i<n;++i)corrected.at<double>(i)=g[i].x-f.offset[i];
            f.slope=least_squares(rows(v,tr),rows(corrected,tr)).at<double>(0);
        }
        f.parameters=1+2*(f.changes+1);return f;
    }
    int segments=g.back().segment+1,columns=method==0?1:method==1?5:method==3?2:1+segments;
    cv::Mat a=cv::Mat::zeros(n,columns,CV_64F);
    for(int i=0;i<n;++i){
        a.at<double>(i,0)=g[i].value;
        if(method==1){a.at<double>(i,1)=1;a.at<double>(i,2)=g[i].direction;a.at<double>(i,3)=g[i].time;a.at<double>(i,4)=1-std::exp(-g[i].time/3);}
        if(method==3)a.at<double>(i,1)=g[i].value*g[i].value;
        if(method==4)a.at<double>(i,1+g[i].segment)=1;
    }
    auto p=least_squares(rows(a,tr),rows(x,tr));f.slope=p.at<double>(0);f.parameters=columns;
    cv::Mat off=a*p-a.col(0)*f.slope;for(int i=0;i<n;++i)f.offset[i]=off.at<double>(i);return f;
}
static std::string escape(const std::string& s){std::string r;for(char c:s){if(c=='&')r+="&amp;";else if(c=='<')r+="&lt;";else if(c=='"')r+="&quot;";else r+=c;}return r;}
static void raster(const std::vector<Shot>& shots,const std::vector<Fit>& fits,double scale,const std::vector<double>& witness_t,const std::vector<double>& witness_x,const fs::path& path){
    cv::Mat image(1650,3300,CV_8UC3,cv::Scalar(255,255,255));
    auto label=[&](const std::string& s,int x,int y,int size=17){cv::Rect box(x,y,std::min(510,image.cols-x),std::min(size*2+4,image.rows-y));if(box.empty())return;auto roi=image(box);wintext(roi,s,{0,0},size,RGB(35,35,35));};
    const std::vector<std::string> names={u8"보정 전",u8"백래시 + 드리프트",u8"직교축 증인",u8"회전 중심 기하 (2차)",u8"구간 절편",u8"변화점 + 시간 직선"};
    const cv::Scalar colors[]={cv::Scalar(208,114,74),cv::Scalar(63,112,210),cv::Scalar(118,167,90),cv::Scalar(160,122,207)};
    double xmin=0,xmax=0,low=0,high=0;
    for(auto& s:shots){xmin=std::min(xmin,s.value);xmax=std::max(xmax,s.value);double r=(s.x-fits[0].slope*s.value)*scale;low=std::min(low,r);high=std::max(high,r);}
    double pad=(high-low)*.12;low-=pad;high+=pad;
    for(int method=0;method<6;++method){
        int left=method*550+45;label(names[method],left,18,22);double maximum=1e-6;for(int i=0;i<int(shots.size());++i)maximum=std::max(maximum,std::abs((shots[i].x-fits[method].offset[i]-fits[method].slope*shots[i].value)*scale));
        for(int panel=0;panel<3;++panel){
            cv::Rect box(left,90+panel*510,475,385);cv::rectangle(image,box,cv::Scalar(210,210,210));
            double ymin=panel==1?-maximum*1.15:low,ymax=panel==1?maximum*1.15:high;
            auto point=[&](double x,double y){double nx=panel==2?x/shots.back().time:(x-xmin)/(xmax-xmin);return cv::Point(int(std::lround(box.x+nx*box.width)),int(std::lround(box.y+box.height-(y-ymin)/(ymax-ymin)*box.height)));};
            auto horizontal=[&](double y){auto a=point(panel==2?0:xmin,y),b=point(panel==2?shots.back().time:xmax,y);cv::line(image,a,b,cv::Scalar(225,225,225),1,cv::LINE_AA);};horizontal(0);
            cv::Point previous;bool had=false;int clipped=0;
            for(int i=0;i<int(shots.size());++i){auto& s=shots[i];double y=panel==2?fits[method].offset[i]*scale:(s.x-fits[method].offset[i]-fits[method].slope*s.value)*scale;auto p=point(panel==2?s.time:s.value,y);
                if(panel==2){auto raw=point(s.time,(s.x-fits[0].slope*s.value)*scale);if(box.contains(raw))cv::circle(image,raw,2,cv::Scalar(210,210,210),-1,cv::LINE_AA);}
                if(had){auto a=previous,b=p;if(cv::clipLine(box,a,b))cv::line(image,a,b,cv::Scalar(190,190,190),1,cv::LINE_AA);}previous=p;had=true;
                if(box.contains(p))cv::circle(image,p,3,colors[s.segment%4],-1,cv::LINE_AA);else ++clipped;
            }
            auto zero=point(0,0);if(box.contains(zero))cv::drawMarker(image,zero,cv::Scalar(25,25,25),cv::MARKER_STAR,16,2,cv::LINE_AA);
            if(panel==2&&method==2)for(int i=0;i<int(witness_t.size());++i){auto p=point(witness_t[i],witness_x[i]*scale);if(box.contains(p))cv::circle(image,p,2,cv::Scalar(170,90,135),-1,cv::LINE_AA);}
            label(panel==2?u8"촬영 시각 / 추정 0점":panel==1?u8"명령값 / 잔차 (개별 확대)":u8"명령값 / 잔차 (공통 눈금)",left,box.y+box.height+10);
            std::ostringstream range;range<<std::fixed<<std::setprecision(3)<<"range "<<ymin<<" .. "<<ymax<<"; outside="<<clipped;label(range.str(),left,box.y+box.height+38,14);
        }
    }
    label(u8"과거 실험 재현: 시간순 연결. 독립 검증 또는 실제 로봇 궤적이 아님.",45,1620,17);
    std::vector<unsigned char> encoded;if(!cv::imencode(".png",image,encoded))throw std::runtime_error("PNG encoding failed");write(path,std::string(encoded.begin(),encoded.end()));
}
static void plot(const std::vector<Shot>& shots,const std::vector<Fit>& fits,double scale,const fs::path& path){
    std::ostringstream out;out<<"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1800\" height=\"900\" viewBox=\"0 0 1800 900\"><rect width=\"1800\" height=\"900\" fill=\"white\"/><g font-family=\"Malgun Gothic,sans-serif\" font-size=\"14\">";
    const std::vector<std::string> labels={"baseline","backlash + drift","orthogonal witness","quadratic geometry","segment intercept","change points"};
    double xmin=shots.front().value,xmax=xmin;for(auto& s:shots){xmin=std::min(xmin,s.value);xmax=std::max(xmax,s.value);}
    for(int method=0;method<6;++method){double left=method*300+40,width=240;std::vector<double> residual;double range=1e-6;for(int i=0;i<int(shots.size());++i){double r=(shots[i].x-fits[method].offset[i]-fits[method].slope*shots[i].value)*scale;residual.push_back(r);range=std::max(range,std::abs(r));}
        out<<"<text x=\""<<left<<"\" y=\"35\">"<<escape(labels[method])<<"</text>";
        for(int panel=0;panel<2;++panel){double top=60+panel*420,height=340,limit=range*1.1;out<<"<rect x=\""<<left<<"\" y=\""<<top<<"\" width=\"240\" height=\"340\" fill=\"none\" stroke=\"#ccc\"/>";
            for(int i=0;i<int(shots.size());++i){double x=panel?shots[i].time/shots.back().time:(shots[i].value-xmin)/(xmax-xmin),y=panel?fits[method].offset[i]*scale:residual[i];if(panel){limit=1;for(double o:fits[method].offset)limit=std::max(limit,std::abs(o*scale)*1.1);}out<<"<circle cx=\""<<left+x*width<<"\" cy=\""<<top+height*.5-y/limit*height*.45<<"\" r=\"2\" fill=\"#4369ac\"/>";}
            out<<"<text x=\""<<left<<"\" y=\""<<top+height+25<<"\">"<<(panel?"capture time / estimated offset":"command / corrected residual")<<"</text>";
        }
    }out<<"</g></svg>";write(path,out.str());
}
int wmain(int argc,wchar_t** wide){
    Utf8Arguments args(argc,wide);char** argv=args.data();
    try{
        if(argc!=4)throw std::runtime_error("historical_zeroalign reference-root bmp-20261014 new-output");
        fs::path root=fs::u8path(argv[1]),data=fs::u8path(argv[2]),out=fs::u8path(argv[3]);if(fs::exists(out))throw std::runtime_error("New output required");
        CsvTable q(root/fs::u8path(u8"14_데이터_QR마커위치.csv")),b(root/fs::u8path(u8"15_데이터_파인더_볼트이동량.csv"));
        std::map<std::string,double> bolts;for(size_t i=0;i<b.rows.size();++i)bolts[b.text(i,"file")]=b.number(i,"bolt_dx");
        std::vector<Shot> all;double earliest=std::numeric_limits<double>::infinity();
        for(size_t i=0;i<q.rows.size();++i){Shot s;s.file=q.text(i,"file");s.value=q.number(i,"value");s.x=(q.number(i,"dLx")+q.number(i,"dCx")+q.number(i,"dRx"))/3;s.bolt=s.file=="center.bmp"?0:bolts.at(s.file);
            auto path=s.file=="center.bmp"?data/"center.bmp":data/(s.file[0]=='T'?"Theta":"Y")/fs::u8path(s.file);
            s.time=std::chrono::duration<double>(fs::last_write_time(path).time_since_epoch()).count();earliest=std::min(earliest,s.time);all.push_back(s);
        }
        std::vector<Shot> axes[2];for(auto s:all){s.time=(s.time-earliest)/60;if(s.file[0]=='Y')axes[0].push_back(s);if(s.file[0]=='T')axes[1].push_back(s);}
        double ratios[2];for(auto& g:axes){std::stable_sort(g.begin(),g.end(),[](auto& a,auto& b){return a.time<b.time;});for(int i=1;i<int(g.size());++i)g[i].segment=g[i-1].segment+(g[i].time-g[i-1].time>3||std::signbit(g[i].value)!=std::signbit(g[i-1].value));
            for(int s=0;s<=g.back().segment;++s){std::vector<double> t,v;for(auto& x:g)if(x.segment==s){t.push_back(x.time);v.push_back(x.value);}double direction=line(t,v).first>0?1:-1;for(auto& x:g)if(x.segment==s)x.direction=direction;}
        }
        for(int k=0;k<2;++k){std::vector<double>x,b;for(auto& s:axes[k]){x.push_back(s.x);b.push_back(s.bolt);}ratios[k]=dot(x,b)/dot(x,x);}
        fs::create_directories(out);std::ofstream metrics(out/"metrics.csv"),predictions(out/"predictions.csv");metrics<<"axis,method,parameters,fit_rms,fit_max,cv_rms,left_intercept,right_intercept,intercept_gap,scale\n";predictions<<"axis,method,file,time_minutes,segment,direction,command,observed_px,offset_px,slope,residual_px\n";metrics<<std::setprecision(17);predictions<<std::setprecision(17);
        std::ofstream compatible(out/fs::u8path(u8"33_결과표_0점정렬_5방법.csv"));compatible<<"\xef\xbb\xbf"<<u8"축,방법,관점,파라미터수,RMS_적합,최대_적합,RMS_CV,좌절편,우절편,좌우차,단위,메모\n"<<std::fixed<<std::setprecision(3);
        CsvTable gold(root/fs::u8path(u8"33_결과표_0점정렬_5방법.csv"));double max_delta=0;int compared=0,outside=0;
        Json summary=Json::dict();summary["created_kst"]=now();summary["source"]=u8"analysis_v4/zero_align.py";summary["scope"]="Historical fit and alternating chronological cross-validation; includes historical bolt witness, not an adopted QR-only model";summary["metrics"]=Json::list();
        for(int axis=0;axis<2;++axis){auto& g=axes[axis];std::vector<double> v,x,tw={0},ew={0};for(auto& s:g){v.push_back(s.value);x.push_back(s.x);}double unit=1000/line(v,x).first;
            for(auto& s:axes[1-axis]){tw.push_back(s.time);ew.push_back((s.bolt-ratios[1-axis]*s.x)/(ratios[axis]-ratios[1-axis]));}auto witness=line(tw,ew);std::vector<Fit> fits;
            for(int method=0;method<6;++method){auto f=fit(g,indices(int(g.size())),method,witness);fits.push_back(f);double squared=0,maxabs=0,cvsquared=0;std::vector<double> negv,negr,posv,posr;
                for(int i=0;i<int(g.size());++i){auto& s=g[i];double residual=s.x-f.offset[i]-f.slope*s.value;squared+=residual*residual;maxabs=std::max(maxabs,std::abs(residual));(s.value<0?negv:posv).push_back(s.value);(s.value<0?negr:posr).push_back(residual);predictions<<(axis?"T":"Y")<<','<<method<<','<<csv(s.file)<<','<<s.time<<','<<s.segment<<','<<s.direction<<','<<s.value<<','<<s.x<<','<<f.offset[i]<<','<<f.slope<<','<<residual<<'\n';}
                for(int parity=0;parity<2;++parity){auto c=fit(g,indices(int(g.size()),parity),method,witness);for(int i:indices(int(g.size()),1-parity)){double r=g[i].x-c.offset[i]-c.slope*g[i].value;cvsquared+=r*r;}}
                double left=line(negv,negr).second*unit,right=line(posv,posr).second*unit,frms=std::sqrt(squared/g.size())*unit,crms=std::sqrt(cvsquared/g.size())*unit;
                size_t gi=size_t(axis*6+method);if(gi>=gold.rows.size())throw std::runtime_error("Missing original metric row");
                const std::vector<std::pair<std::string,double>> comparison={{u8"파라미터수",double(f.parameters)},{u8"RMS_적합",frms},{u8"최대_적합",maxabs*unit},{u8"RMS_CV",crms},{u8"좌절편",left},{u8"우절편",right},{u8"좌우차",std::abs(left-right)}};
                for(auto& c:comparison){double d=std::abs(c.second-gold.number(gi,c.first));max_delta=std::max(max_delta,d);++compared;outside+=d>.0005000001;}
                compatible<<csv(axis?u8"Θ":"Y")<<','<<csv(gold.text(gi,u8"방법"))<<','<<csv(gold.text(gi,u8"관점"))<<','<<f.parameters<<','<<frms<<','<<maxabs*unit<<','<<crms<<','<<left<<','<<right<<','<<std::abs(left-right)<<','<<csv(gold.text(gi,u8"단위"))<<','<<csv("Native C++ fit; full per-image offsets and slopes in predictions.csv")<<'\n';
                metrics<<(axis?"T":"Y")<<','<<method<<','<<f.parameters<<','<<frms<<','<<maxabs*unit<<','<<crms<<','<<left<<','<<right<<','<<std::abs(left-right)<<','<<unit<<'\n';
                Json j=Json::dict();j["axis"]=axis?"T":"Y";j["method"]=method;j["fit_rms"]=frms;j["cv_rms"]=crms;j["fit_max"]=maxabs*unit;j["left_intercept"]=left;j["right_intercept"]=right;j["parameters"]=f.parameters;summary["metrics"].push(j);
            }
            plot(g,fits,unit,out/(axis?"theta_zero_alignment.svg":"y_zero_alignment.svg"));
            raster(g,fits,unit,tw,ew,out/fs::u8path(axis?u8"35_0점정렬_Θ_5방법_경로.png":u8"34_0점정렬_Y_5방법_경로.png"));
        }
        summary["native_compute"]=true;summary["models"]=12;summary["python_invocations"]=0;summary["original_rounded_values_compared"]=compared;summary["max_difference_from_rounded_reference"]=max_delta;summary["outside_original_rounding_precision"]=outside;summary["parity_passed"]=outside==0&&compared==84;summary["render_change"]="Native OpenCV/GDI raster renderer; shared residual range, zoomed residuals, chronological segments, offset time series and orthogonal witness retained. PNG layout/style differs from Matplotlib.";write(out/"checks.json",summary.dump());std::cout<<summary.dump()<<'\n';return outside?2:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
