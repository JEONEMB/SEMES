#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <omp.h>
using json=nlohmann::json;
using Q=std::array<double,3>;
// A, mu, abs(sigma), SSE, iterations, converged, relative objective spread, relative parameter spread.
using Record=std::array<double,8>;
struct Stop {std::string id;int cap;double ftol,xtol;};
struct Profile {int x,cy;double peak;std::vector<double> g;};
struct Group {std::string ref,target;std::vector<Profile> profiles;};
const std::vector<Stop> stops={
    {"legacy_800",800,1e-10,-1},{"legacy_1600",1600,1e-10,-1},
    {"legacy_3200",3200,1e-10,-1},{"legacy_6400",6400,1e-10,-1},
    {"dual_1e-6",6400,1e-6,1e-6},{"dual_1e-8",6400,1e-8,1e-8},
    {"dual_1e-10",6400,1e-10,1e-10}};
const std::array<double,3> fixed_sigma={5,11,21};
#include "original_nelder_mead.hpp"

template<class F>
std::array<Record,7> solve_checkpoints(F f,Q x0,const Q& step,double cy) {
    std::array<Q,4> s;std::array<double,4> v;
    s[0]=x0;
    for(int i=0;i<3;++i){s[i+1]=x0;s[i+1][i]+=step[i];}
    for(int i=0;i<4;++i)v[i]=f(s[i]);
    std::array<Record,7> out;std::array<bool,7> done={};
    auto snapshot=[&](size_t k,int it,bool conv){
        const int best=(int)(std::min_element(v.begin(),v.end())-v.begin());
        double fspan=0,xspan=0;
        for(int j=0;j<4;++j){
            fspan=std::max(fspan,std::fabs(v[j]-v[best]));
            for(int d=0;d<3;++d){
                // Center mu's relative scale on the measured coarse edge, not the image origin.
                double origin=d==1?cy:0.;
                double span=std::fabs(s[j][d]-s[best][d])/(1.+std::fabs(s[best][d]-origin));
                xspan=std::max(xspan,span);
            }
        }
        out[k]={s[best][0],s[best][1],std::fabs(s[best][2]),v[best],(double)it,conv?1.:0.,
                fspan/(std::fabs(v[best])+1e-30),xspan};
        done[k]=true;
    };
    for(int it=0;it<=6400;++it){
        // Preserve original max_iter semantics: stopping tests occur only before the cap.
        for(size_t k=0;k<stops.size();++k)if(!done[k]&&it==stops[k].cap)snapshot(k,it,false);
        if(std::all_of(done.begin(),done.end(),[](bool b){return b;}))break;
        std::array<int,4> ix={0,1,2,3};
        std::sort(ix.begin(),ix.end(),[&](int a,int b){return v[a]<v[b];});
        auto ss=s;auto vv=v;
        for(int i=0;i<4;++i){s[i]=ss[ix[i]];v[i]=vv[ix[i]];}
        double xspan=0.;
        for(int j=1;j<4;++j)for(int d=0;d<3;++d){
            double origin=d==1?cy:0.;
            xspan=std::max(xspan,std::fabs(s[j][d]-s[0][d])/(1.+std::fabs(s[0][d]-origin)));
        }
        for(size_t k=0;k<stops.size();++k)if(!done[k]){
            const auto& stop=stops[k];
            bool fsmall=std::fabs(v[3]-v[0])<=stop.ftol*(std::fabs(v[0])+1e-30);
            if(fsmall&&(stop.xtol<0||xspan<=stop.xtol))snapshot(k,it,true);
        }
        if(std::all_of(done.begin(),done.end(),[](bool b){return b;}))break;
        Q c={0,0,0};
        for(int i=0;i<3;++i)for(int k=0;k<3;++k)c[k]+=s[i][k]/3.0;
        auto mix=[&](double a){Q r;for(int k=0;k<3;++k)r[k]=c[k]+a*(s[3][k]-c[k]);return r;};
        auto xr=mix(-1.);double fr=f(xr);
        if(fr<v[0]){
            auto xe=mix(-2.);double fe=f(xe);
            if(fe<fr){s[3]=xe;v[3]=fe;}else{s[3]=xr;v[3]=fr;}
        }else if(fr<v[2]){s[3]=xr;v[3]=fr;}
        else{
            auto xc=fr<v[3]?mix(-.5):mix(.5);double fc=f(xc);
            if(fc<std::min(fr,v[3])){s[3]=xc;v[3]=fc;}
            else for(int i=1;i<4;++i){for(int k=0;k<3;++k)s[i][k]=s[0][k]+.5*(s[i][k]-s[0][k]);v[i]=f(s[i]);}
        }
    }
    return out;
}

int main(int argc,char** argv){
 try{
    if(argc>1 && std::string(argv[1])=="--self-test"){
        int checks=0;
        for(int n:{3,11,61})for(double sigma:{5.,11.,21.,(double)n})for(int sign:{1,-1}){
            auto objective=[&](const Q& q){double e=0,ss=std::fabs(q[2])+1e-12;
                for(int k=-(n-1)/2;k<=(n-1)/2;++k){
                    double y=700.+k;
                    double g=sign*30.*std::exp(-(y-700.4)*(y-700.4)/(2.*2.5*2.5))+std::sin(k*.7);
                    double d=y-q[1],r=q[0]*std::exp(-d*d/(2.*ss*ss))-g;e+=r*r;
                }return e;};
            Q initial={(double)sign*30.,700.,sigma},step={6.,1.,5.5};
            auto batched=solve_checkpoints(objective,initial,step,700.);
            for(size_t i=0;i<4;++i){int it=0;bool conv=false;
                auto original=nelder_mead3(objective,initial,step,stops[i].cap,1e-10,&it,&conv);
                const auto& r=batched[i];
                if(r[0]!=original[0]||r[1]!=original[1]||r[2]!=std::fabs(original[2])||r[3]!=objective(original)||r[4]!=it||r[5]!=(conv?1.:0.))
                    throw std::runtime_error("Checkpoint/original solver mismatch");
                ++checks;
            }
        }
        std::cout<<"PASS "<<checks<<" exact comparisons against independent original solver calls"<<std::endl;
        return 0;
    }
    auto start=std::chrono::steady_clock::now();
    std::ifstream input("../gaussian_window_no_rejection_20261101/profiles.jsonl");
    if(!input)throw std::runtime_error("Missing audited profiles");
    std::vector<Group> groups;std::map<std::string,size_t> indexes;
    std::string line;
    while(std::getline(input,line)){
        auto j=json::parse(line);std::string ref=j["ref"],target=j["target"],key=ref+"_"+target;
        if(!indexes.count(key)){indexes[key]=groups.size();groups.push_back({ref,target,{}});}
        if(j["half_span"]!=30)throw std::runtime_error("Unexpected profile length");
        groups[indexes[key]].profiles.push_back({j["x"],j["coarse_y"],j["coarse_response"],j["sobel_samples"].get<std::vector<double>>()});
    }
    std::filesystem::create_directories("raw");
    json meta={{"shape_order",{"profile","n","initialization","stopping","field"}},
      {"fields",{"A","mu","sigma_abs","sse","iterations","converged","relative_f_spread","relative_parameter_spread"}},
      {"sample_counts",json::array()},{"initializations",{"fixed_5","fixed_11","fixed_21","paper_N"}},
      {"stopping",json::array()},{"groups",json::array()},{"dtype","little-endian IEEE754 float64"}};
    for(int n=3;n<=61;n+=2)meta["sample_counts"].push_back(n);
    for(const auto& st:stops)meta["stopping"].push_back({{"id",st.id},{"max_iterations",st.cap},{"f_tolerance",st.ftol},{"parameter_tolerance",st.xtol<0?json(nullptr):json(st.xtol)}});
    size_t total=0;
    for(const auto& gr:groups){
        const int P=(int)gr.profiles.size();
        std::vector<Record> records((size_t)P*30*4*7);
        std::cout<<"Starting "<<gr.ref<<" "<<gr.target<<" "<<P<<" profiles"<<std::endl;
        #pragma omp parallel for schedule(dynamic)
        for(int pi=0;pi<P;++pi){
            const auto& p=gr.profiles[pi];
            for(int ni=0;ni<30;++ni){
                const int n=3+2*ni,hw=(n-1)/2;
                auto sse=[&](const Q& q){
                    double e=0,ss=std::fabs(q[2])+1e-12;
                    for(int k=-hw;k<=hw;++k){
                        double d=p.cy+k-q[1];
                        double r=q[0]*std::exp(-d*d/(2.*ss*ss))-p.g[k+30];e+=r*r;
                    }
                    return e;
                };
                for(int si=0;si<4;++si){
                    // Fixed arms isolate window and initial-width effects: same A0, mu0, simplex steps.
                    // paper_N is a separate reproduction arm, not included in that isolation claim.
                    const double a0=si==3?*std::max_element(p.g.begin()+30-hw,p.g.begin()+30+hw+1):p.peak;
                    const double sigma=si==3?(double)n:fixed_sigma[si];
                    Q initial={a0,(double)p.cy,sigma};
                    Q step={.2*a0,1.,si==3?.5*n:5.5};
                    auto fits=solve_checkpoints(sse,initial,step,(double)p.cy);
                    size_t index=(((size_t)pi*30+ni)*4+si)*7;
                    for(size_t k=0;k<fits.size();++k)records[index+k]=fits[k];
                }
            }
        }
        std::string file="raw/"+gr.ref+"_"+gr.target+".bin";
        std::ofstream out(file,std::ios::binary);
        out.write(reinterpret_cast<const char*>(records.data()),records.size()*sizeof(Record));
        if(!out)throw std::runtime_error("Could not write raw returns");
        json xs=json::array(),cy=json::array(),peaks=json::array();
        for(const auto& p:gr.profiles){xs.push_back(p.x);cy.push_back(p.cy);peaks.push_back(p.peak);}
        meta["groups"].push_back({{"ref",gr.ref},{"target",gr.target},{"profiles",P},{"file",file},{"shape",{P,30,4,7,8}},
                                 {"x",xs},{"coarse_y",cy},{"coarse_response",peaks}});
        total+=records.size();
        std::cout<<"Saved "<<file<<" ("<<records.size()<<" returns)"<<std::endl;
        std::ofstream("metadata.json")<<meta.dump(2);
    }
    meta["optimization_trajectories"]=23056*30*4;
    meta["stored_returns"]=total;
    meta["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    meta["openmp_max_threads"]=omp_get_max_threads();
    std::ofstream("metadata.json")<<meta.dump(2);
    std::cout<<"Completed: "<<total<<" returns, "<<meta["elapsed_seconds"]<<" seconds"<<std::endl;
    return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
