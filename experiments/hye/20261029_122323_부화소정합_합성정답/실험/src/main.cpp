// 1단계 합성 정답 실험: center.bmp에 알려진 변형을 넣어 630개 고정 점의 변위 정답을 만들고,
// 부화소 정합 방법별 점 오차·실패·시간과 정·역방향 오차의 실패 검출력을 잰다.
#include "bench.hpp"
#include "control_internal.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
using namespace bench;
namespace {
const double kNan=std::numeric_limits<double>::quiet_NaN();
const std::string kCenterSha="01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685";
struct Pool {std::vector<double> mag,mag536;double sx=0,sy=0;long ok=0,fail=0,bad05=0,gt01=0,total=0;std::vector<double> ms;};
double quantile(std::vector<double> v,double q){if(v.empty())return kNan;std::sort(v.begin(),v.end());return v[std::min(v.size()-1,size_t(q*(v.size()-1)+.5))];}
double rms(const std::vector<double>& v){if(v.empty())return kNan;double s=0;for(auto x:v)s+=x*x;return std::sqrt(s/v.size());}
std::string num(double v,int p=4){if(!std::isfinite(v))return "";std::ostringstream o;o<<std::fixed<<std::setprecision(p)<<v;return o.str();}
std::string pool_row(const Pool& p){
    std::ostringstream o;o<<p.total<<','<<p.fail<<','<<num(rms(p.mag))<<','<<num(rms(p.mag536))<<','<<num(p.ok?p.sx/p.ok:kNan)<<','<<num(p.ok?p.sy/p.ok:kNan)<<','
     <<num(quantile(p.mag,.5))<<','<<num(quantile(p.mag,.95))<<','<<num(p.mag.empty()?kNan:*std::max_element(p.mag.begin(),p.mag.end()))<<','
     <<num(double(p.gt01)/p.total,4)<<','<<num(double(p.bad05)/p.total,4)<<','<<num(quantile(p.ms,.5),2)<<','<<num(quantile(p.ms,.95),2);return o.str();}
const char* kPoolHeader="points,fail,rmse_px,rmse536_px,bias_x_px,bias_y_px,median_px,p95_px,max_px,frac_gt0.1,frac_fail_or_gt0.5,time_median_ms,time_p95_ms";
std::vector<Condition> conditions(double sigma){
    std::vector<Condition> out;
    for(int i=0;i<=20;++i){double f=i*.05;Condition c;c.group="평행이동_x";c.tx=3+f;c.ty=-2+.35;c.name="tx+"+num(f,2);out.push_back(c);}
    for(int i=0;i<=20;++i){double f=i*.05;Condition c;c.group="평행이동_y";c.tx=3.35;c.ty=-2+f;c.name="ty+"+num(f,2);out.push_back(c);}
    for(int i=0;i<=10;++i){double f=i*.1;Condition c;c.group="평행이동+잡음";c.tx=3+f;c.ty=-1.65;c.noise=sigma;c.seed=100+i;c.name="tx+"+num(f,1)+"_noise";out.push_back(c);}
    for(double a:{-.1,-.05,-.02,.02,.05,.1}){Condition c;c.group="회전·배율";c.tx=3.4;c.ty=-1.7;c.angle_deg=a;c.noise=sigma;c.seed=200+int(out.size());c.name="rot"+num(a,2)+"deg";out.push_back(c);}
    for(double s:{.999,1.001}){Condition c;c.group="회전·배율";c.tx=3.4;c.ty=-1.7;c.scale=s;c.noise=sigma;c.seed=200+int(out.size());c.name="scale"+num(s,3);out.push_back(c);}
    struct P{const char* n;double g,o,b,k;};
    for(auto p:{P{"gain0.7_off+20",.7,20,0,1},P{"gain1.3_off-20",1.3,-20,0,1},P{"off+30",1,30,0,1},P{"blur0.7",1,0,.7,1},P{"blur1.2",1,0,1.2,1},P{"noise2x",1,0,0,2},P{"noise4x",1,0,0,4}}){
        Condition c;c.group="밝기·흐림·잡음";c.tx=3.4;c.ty=-1.7;c.gain=p.g;c.offset=p.o;c.blur=p.b;c.noise=sigma*p.k;c.seed=300+int(out.size());c.name=p.n;out.push_back(c);}
    for(int s=1;s<=4;++s){Condition c;c.group="가림";c.tx=3.4;c.ty=-1.7;c.noise=sigma;c.occlusions=3;c.seed=400+s;c.name="occlusion_seed"+std::to_string(s);out.push_back(c);}
    return out;
}
}
int main(int argc,char** argv){
    try{
        // 인자: [결과 폴더] [quick]. quick은 조건을 10개 중 1개만 돌리는 빌드 확인용이다.
        fs::path repo=fs::u8path(REPO_DIR),experiment=fs::u8path(EXPERIMENT_DIR),results=argc>1?fs::u8path(argv[1]):experiment/"results";
        bool quick=argc>2&&std::string(argv[2])=="quick";
        if(fs::exists(results))throw std::runtime_error("results already exists; refusing to overwrite: "+results.u8string());
        cv::setNumThreads(1);
        auto data14=repo/"data/clean/clean_20261014",data20=repo/fs::u8path(u8"data/clean/clean_20261020/기준 위치");
        auto control=repo/fs::u8path(u8"experiments/hye/00000_final_Y_T control/assets");
        if(legacy_control::sha256(data14/"center.bmp")!=kCenterSha)throw std::runtime_error("center.bmp hash mismatch");
        cv::Mat center=legacy_control::read_gray(data14/"center.bmp");
        auto points=load_points(control/"points.json");

        // 20261020 반복 센터 3장의 평탄부로 촬영 잡음 σ 추정
        std::vector<cv::Mat> repeats;for(int i=1;i<=3;++i)repeats.push_back(legacy_control::read_gray(data20/("center"+std::to_string(i)+".bmp")));
        std::vector<double> sigmas{estimate_noise(repeats[0],repeats[1]),estimate_noise(repeats[0],repeats[2]),estimate_noise(repeats[1],repeats[2])};
        double sigma=quantile(sigmas,.5);std::cout<<"noise sigma pairs "<<sigmas[0]<<' '<<sigmas[1]<<' '<<sigmas[2]<<" -> "<<sigma<<"\n";

        Methods methods(center,points,control);int M=Methods::count();auto conds=conditions(sigma);
        if(quick){std::vector<Condition> few;for(size_t i=0;i<conds.size();i+=10)few.push_back(conds[i]);conds=few;}
        fs::create_directories(results);
        std::ofstream per((results/"per_image.csv")),fail((results/"failures.csv")),lock((results/"locking.csv"));
        per<<"condition,group,method,"<<kPoolHeader<<",own_valid\n";fail<<"condition,method,point_id,selected536,ok,err_x_px,err_y_px,err_px,fb_px\n";lock<<"group,method,fraction,bias_px,rmse_axis_px\n";
        std::map<std::pair<std::string,int>,Pool> groups;std::vector<Pool> overall(M);
        struct Fb{double fb;bool bad;};std::map<int,std::vector<Fb>> fbs;std::map<int,std::array<long,4>> own;// own: [valid&good, valid&bad, invalid&good, invalid&bad]
        int index=0;
        for(const auto& c:conds){
            auto image=synthesize(center,c);std::vector<cv::Point2d> truth;for(auto p:points.p)truth.push_back(warp_point(c,p)-p);
            for(int m=0;m<M;++m){
                auto t0=std::chrono::steady_clock::now();auto e=methods.run(m,image);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
                Pool p;p.ms.push_back(ms);long valid=0;double lock_sum=0,lock_sq=0;long lock_n=0;bool xs=c.group=="평행이동_x",ys=c.group=="평행이동_y";
                for(size_t i=0;i<truth.size();++i){++p.total;bool sel=points.selected[i];cv::Point2d err=e.d[i]-truth[i];double mag=cv::norm(err);
                    if(!e.ok[i]){++p.fail;++p.bad05;fail<<c.name<<','<<Methods::name(m)<<','<<i<<','<<int(sel)<<",0,,,,"<<num(e.fb[i])<<"\n";continue;}
                    ++p.ok;p.mag.push_back(mag);if(sel)p.mag536.push_back(mag);p.sx+=err.x;p.sy+=err.y;if(mag>.1)++p.gt01;bool bad=mag>.5;if(bad){++p.bad05;fail<<c.name<<','<<Methods::name(m)<<','<<i<<','<<int(sel)<<",1,"<<num(err.x)<<','<<num(err.y)<<','<<num(mag)<<','<<num(e.fb[i])<<"\n";}
                    if(e.own_valid[i])++valid;own[m][(e.own_valid[i]?0:2)+(bad?1:0)]++;
                    if(std::isfinite(e.fb[i]))fbs[m].push_back({e.fb[i],bad});
                    if(xs||ys){double a=xs?err.x:err.y;lock_sum+=a;lock_sq+=a*a;++lock_n;}}
                per<<c.name<<','<<c.group<<','<<Methods::name(m)<<','<<pool_row(p)<<','<<valid<<"\n";
                if(lock_n)lock<<c.group<<','<<Methods::name(m)<<','<<num(xs?c.tx-3:c.ty+2,2)<<','<<num(lock_sum/lock_n)<<','<<num(std::sqrt(lock_sq/lock_n))<<"\n";
                for(auto* g:{&groups[{c.group,m}],&overall[m]}){g->mag.insert(g->mag.end(),p.mag.begin(),p.mag.end());g->mag536.insert(g->mag536.end(),p.mag536.begin(),p.mag536.end());
                    g->sx+=p.sx;g->sy+=p.sy;g->ok+=p.ok;g->fail+=p.fail;g->bad05+=p.bad05;g->gt01+=p.gt01;g->total+=p.total;g->ms.push_back(ms);}
            }
            std::cout<<"["<<++index<<"/"<<conds.size()<<"] "<<c.name<<"\n";
        }
        std::ofstream sum((results/"summary.csv"));sum<<"group,method,"<<kPoolHeader<<"\n";
        std::vector<std::string> order;for(const auto& c:conds)if(std::find(order.begin(),order.end(),c.group)==order.end())order.push_back(c.group);
        for(const auto& g:order)for(int m=0;m<M;++m)sum<<g<<','<<Methods::name(m)<<','<<pool_row(groups[{g,m}])<<"\n";
        for(int m=0;m<M;++m)sum<<"전체,"<<Methods::name(m)<<','<<pool_row(overall[m])<<"\n";

        // 정·역방향 오차로 |오차|>0.5px 점을 걸러낼 때의 정밀도·재현율과 AUC
        std::ofstream roc((results/"fb_detection.csv"));roc<<"method,threshold_px,flagged,bad_total,true_positive,precision,recall,auc\n";
        for(auto& kv:fbs){auto v=kv.second;long bad=0;for(auto& f:v)bad+=f.bad;std::sort(v.begin(),v.end(),[](const Fb& a,const Fb& b){return a.fb<b.fb;});
            double auc=kNan;if(bad&&bad<long(v.size())){double rank=0;for(size_t i=0;i<v.size();++i)if(v[i].bad)rank+=double(i+1);auc=(rank-double(bad)*(bad+1)/2)/(double(bad)*(v.size()-bad));}
            for(double th:{.1,.2,.3,.5,.75,1.,1.5,2.}){long flagged=0,tp=0;for(auto& f:v)if(f.fb>th){++flagged;tp+=f.bad;}
                roc<<Methods::name(kv.first)<<','<<th<<','<<flagged<<','<<bad<<','<<tp<<','<<num(flagged?double(tp)/flagged:kNan)<<','<<num(bad?double(tp)/bad:kNan)<<','<<num(auc)<<"\n";}}
        std::ofstream ov((results/"own_validity.csv"));ov<<"method,valid_good,valid_bad,invalid_good,invalid_bad\n";
        for(auto& kv:own)ov<<Methods::name(kv.first)<<','<<kv.second[0]<<','<<kv.second[1]<<','<<kv.second[2]<<','<<kv.second[3]<<"\n";

        std::ofstream man((results/"manifest.json"));
        man<<"{\n  \"center\": \"data/clean/clean_20261014/center.bmp\",\n  \"center_sha256\": \""<<kCenterSha<<"\",\n  \"points\": \"experiments/hye/00000_final_Y_T control/assets/points.json\",\n  \"points_sha256\": \""<<legacy_control::sha256(control/"points.json")
           <<"\",\n  \"noise_source\": \"data/clean/clean_20261020/기준 위치/center1~3.bmp 평탄부 차이, 쌍별 σ 중앙값\",\n  \"noise_sigma_pairs\": ["<<sigmas[0]<<", "<<sigmas[1]<<", "<<sigmas[2]<<"],\n  \"noise_sigma\": "<<sigma
           <<",\n  \"opencv\": \""<<CV_VERSION<<"\",\n  \"threads\": 1,\n  \"conditions\": [\n";
        for(size_t i=0;i<conds.size();++i){auto& c=conds[i];man<<"    {\"name\": \""<<c.name<<"\", \"group\": \""<<c.group<<"\", \"tx\": "<<c.tx<<", \"ty\": "<<c.ty<<", \"angle_deg\": "<<c.angle_deg<<", \"scale\": "<<c.scale
            <<", \"gain\": "<<c.gain<<", \"offset\": "<<c.offset<<", \"blur\": "<<c.blur<<", \"noise\": "<<c.noise<<", \"occlusions\": "<<c.occlusions<<", \"seed\": "<<c.seed<<"}"<<(i+1<conds.size()?",":"")<<"\n";}
        man<<"  ]\n}\n";
        std::cout<<"done: "<<results.u8string()<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
