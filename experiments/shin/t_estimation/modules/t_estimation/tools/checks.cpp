#include "t_estimator.hpp"
#include "t_features.hpp"
#include <iostream>
#include <stdexcept>
int main(){try{
 std::vector<t_only::Sample> rows;
 for(int g=0;g<5;g++)for(int r=0;r<3;r++){t_only::Features x{};x[0]=g+.01*r;x[2]=g*.9+.02*r;x[4]=g*1.1-.01*r;rows.push_back({"unused",g,.01*g,x});}
 std::vector<int> ids;for(int i=0;i<12;i++)ids.push_back(i);
 t_only::Config cfg{2,1,.01,0};auto before=t_only::fit(rows,ids,cfg);double expected=t_only::predict(before,rows[12].features);
 rows[12].target=1e12;rows[13].target=-1e12;rows[14].target=0;
 auto after=t_only::fit(rows,ids,cfg);if(std::abs(t_only::predict(after,rows[12].features)-expected)>1e-12)throw std::runtime_error("Held label affected fit");
 for(int loss=0;loss<3;loss++){cfg.loss=loss;auto model=t_only::fit(rows,ids,cfg);if(!std::isfinite(t_only::predict(model,rows[12].features)))throw std::runtime_error("Invalid loss fit");}
 bool threw=false;try{t_only::fit(rows,{},cfg);}catch(const std::invalid_argument&){threw=true;}if(!threw)throw std::runtime_error("Empty train accepted");
 std::vector<t_only::Sample> cubic;std::vector<int> all;
 for(int i=0;i<8;i++){t_only::Features x{};x[0]=x[2]=x[4]=i;cubic.push_back({"",i,.01*(i*i*i-.5*i*i),x});all.push_back(i);}
 auto cubicModel=t_only::fit(cubic,all,{3,3,0,0,0});
 for(auto& r:cubic)if(std::abs(t_only::predict(cubicModel,r.features)-r.target)>1e-9)throw std::runtime_error("Cubic scalar fit failed");
 qrwithbar::detail::Observation reference{},current{};
 for(int i=0;i<630;i++){
     int q=i/210;double x=(i%14)*10.+q*250.,y=((i/14)%15)*12.;
     reference.points[i]={x,y,q};current.points[i]={x+.004*x+.007*y+3,y-.002*x+.003*y-1,q};
 }
 auto shape=t_only::enriched_features(current,reference);
 for(int q=0;q<3;q++)for(int j=0;j<4;j++){
     const double expected[4]={.4,.7,-.2,.3};
     if(std::abs(shape[10+q*4+j]-expected[j])>1e-9)throw std::runtime_error("Wrong spatial slope extraction");
 }
 t_only::Estimator estimator(T_CALIBRATION,T_MODEL_FILE);
 for(const cv::Mat image : {cv::Mat{},cv::Mat(20,20,CV_8UC1),cv::Mat(2160,3840,CV_8UC2)}){
     bool rejected=false;try{estimator.estimate(image);}catch(const std::invalid_argument&){rejected=true;}
     if(!rejected)throw std::runtime_error("Invalid image accepted by T estimator");
 }
 std::cout<<"Held-label isolation, scalar losses and empty-training checks passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
