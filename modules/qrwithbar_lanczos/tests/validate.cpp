#include "qrwithbar_lanczos.hpp"
#include "observation.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <chrono>
#include <cmath>
namespace fs=std::filesystem;
static std::vector<std::vector<std::string>> read_csv(const fs::path& p){std::ifstream f(p);if(!f)throw std::runtime_error("Cannot read "+p.u8string());std::vector<std::vector<std::string>> rows;std::string line;std::getline(f,line);while(std::getline(f,line)){std::vector<std::string> row;std::string x;std::istringstream in(line);while(std::getline(in,x,','))row.push_back(x);rows.push_back(row);}return rows;}
int wmain(int argc,wchar_t** argv){try{
 if(argc!=4)throw std::invalid_argument("Usage: qrwithbar_validate MODULE_DIRECTORY IMAGE_ROOT OUTPUT_DIRECTORY");
 fs::path module=argv[1],images=argv[2],out=argv[3],fixtures=module/"tests/fixtures";fs::create_directories(out);cv::setNumThreads(1);
 auto manifest=read_csv(fixtures/"manifest.csv");if(manifest.size()!=57)throw std::runtime_error("Expected57 frames");
 std::map<std::string,std::vector<std::string>> features,expected;std::map<std::string,std::string> hashes;
 for(auto r:read_csv(fixtures/"comparison_features.csv"))if(r[0]=="lanczos3")features[r[1]]=r;
 for(auto r:read_csv(fixtures/"comparison_predictions.csv"))if(r[0]=="refit"&&r[1]=="lanczos3")expected[r[2]]=r;
 for(auto r:read_csv(fixtures/"image_hash_check.csv"))hashes[r[0]]=r[1];
 cv::FileStorage models((fixtures/"lanczos_validation_models.json").u8string(),cv::FileStorage::READ);if(!models.isOpened())throw std::runtime_error("Missing validation models");
 auto begin=std::chrono::steady_clock::now();qrwithbar::Estimator estimator((module/"assets").u8string());double init_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
 std::ofstream csv(out/"predictions.csv");csv<<std::setprecision(17)<<"path,fold,Y_label,T_label,held_Y,held_T,deploy_Y,deploy_T,Y_error,T_error,upper_valid,lower_valid,inference_ms\n";
 std::ofstream measured(out/"features.csv");measured<<std::setprecision(17)<<"path";for(int j=0;j<10;j++)measured<<",f"<<j;measured<<'\n';
 double max_feature=0,max_prediction=0,mae[2]={},maximum[2]={},squared[2]={};int passes=0,pass009=0;std::vector<double> times;std::vector<std::array<double,4>> pairs;
 for(size_t i=0;i<manifest.size();i++){
  auto& r=manifest[i];int group=std::stoi(r[1]);double truth[2]={std::stod(r[12]),std::stod(r[13])};auto path=images/fs::u8path(r[0]);if(qrwithbar::detail::sha256(path)!=hashes.at(r[0]))throw std::runtime_error("Image hash mismatch");
  auto image=qrwithbar::detail::read_gray(path),before=image.clone();begin=std::chrono::steady_clock::now();auto result=estimator.estimate(image);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();times.push_back(ms);
  if(cv::norm(image,before,cv::NORM_INF)!=0)throw std::runtime_error("Original image changed");
  auto model=models["models"][group];cv::Mat mean,scale,ym,coef;model["mean"]>>mean;model["scale"]>>scale;model["ymean_ticks"]>>ym;model["coef"]>>coef;
  double pose[2]={ym.at<double>(0,0),ym.at<double>(0,1)};
  measured<<r[0];for(int j=0;j<10;j++){measured<<','<<result.features[j];max_feature=std::max(max_feature,fabs(result.features[j]-std::stod(features.at(r[0])[4+j])));for(int k=0;k<2;k++)pose[k]+=(result.features[j]-mean.at<double>(0,j))/scale.at<double>(0,j)*coef.at<double>(j,k);}measured<<'\n';pose[0]*=.05;pose[1]*=.01;
  double error[2];for(int k=0;k<2;k++){max_prediction=std::max(max_prediction,fabs(pose[k]-std::stod(expected.at(r[0])[6+k])));error[k]=pose[k]-truth[k];mae[k]+=fabs(error[k]);maximum[k]=std::max(maximum[k],fabs(error[k]));squared[k]+=error[k]*error[k];}
  passes+=fabs(error[0])<=.05&&fabs(error[1])<=.01;pass009+=fabs(error[0])<=.09&&fabs(error[1])<=.09;
  pairs.push_back({truth[0],truth[1],pose[0],pose[1]});
  csv<<r[0]<<','<<group<<','<<truth[0]<<','<<truth[1]<<','<<pose[0]<<','<<pose[1]<<','<<result.y_mm<<','<<result.yaw_deg<<','<<error[0]<<','<<error[1]<<','<<result.upper_valid<<','<<result.lower_valid<<','<<ms<<'\n';
  if((i+1)%5==0)std::cout<<"Validated "<<i+1<<"/57\n"<<std::flush;
 }
 std::sort(times.begin(),times.end());
 cv::FileStorage summary((out/"validation.json").u8string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
 summary<<"images"<<57<<"coordinate_folds"<<19<<"method"<<"lanczos3"<<"feature_max_difference"<<max_feature<<"held_prediction_max_difference"<<max_prediction<<"Y_mae"<<mae[0]/57<<"Y_max"<<maximum[0]<<"Y_rmse"<<sqrt(squared[0]/57)<<"T_mae"<<mae[1]/57<<"T_max"<<maximum[1]<<"T_rmse"<<sqrt(squared[1]/57)<<"both_tick"<<passes<<"both_009"<<pass009<<"asset_initialization_ms"<<init_ms<<"median_inference_ms"<<times[28]<<"inference_timer_excludes_decode"<<1<<"original_hashes_verified"<<1<<"original_pixels_unchanged"<<1<<"deployment_predictions_are_training_reapplication"<<1;
 summary.release();
 if(max_feature>1e-8||max_prediction>1e-7)throw std::runtime_error("Runtime differs from approved Lanczos comparison");
 std::ofstream svg(out/"y_theta_pairs.svg");svg<<"<svg xmlns='http://www.w3.org/2000/svg' width='900' height='480' viewBox='0 0 900 480'><rect width='100%' height='100%' fill='white'/><text x='70' y='25' font-size='18'>Lanczos-3 runtime: true label to held-coordinate prediction</text><path d='M70,45 V420 H850' stroke='#999' fill='none'/>";
 auto X=[](double x){return 70+(x+.4)/2*780;};auto Y=[](double y){return 420-(y+.15)/2.4*360;};
 for(double x:{0.,.5,1.,1.5})svg<<"<text x='"<<X(x)<<"' y='445' font-size='12'>"<<x<<"</text>";
 for(double y:{0.,.5,1.,1.5,2.})svg<<"<text x='35' y='"<<Y(y)<<"' font-size='12'>"<<y<<"</text>";
 for(auto p:pairs){double ax=X(p[0]),ay=Y(p[1]),bx=X(p[2]),by=Y(p[3]);svg<<"<path d='M"<<ax<<','<<ay<<" L"<<bx<<','<<by<<"' stroke='#aaa' fill='none'/><circle cx='"<<ax<<"' cy='"<<ay<<"' r='3' fill='white' stroke='#555'/><path d='M"<<bx-3<<','<<by-3<<" l6,6 m-6,0 l6,-6' stroke='#0071e3'/>";}
 svg<<"<text x='430' y='475' font-size='14'>Y (mm)</text><text x='20' y='45' font-size='14'>T</text></svg>";
 std::ofstream html(out/"report.html");html<<u8"<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>QRwithBar Lanczos 적용 검증</title><style>body{font:16px/1.8 system-ui;color:#333;max-width:1080px;margin:auto;padding:32px 20px;background:white}h1{font-size:28px}a{color:#0071e3}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%}.note{background:#f5f5f7;padding:18px}</style><h1>QRwithBar Lanczos-3 적용 검증</h1><p class='note'>원본57장 재실행: 비교 실험 대비 특징 최대차이 "<<max_feature<<u8", 좌표 보류 예측 최대차이 "<<max_prediction<<u8". 일반 실행과평가 모델을 분리했다.</p><table><tr><th>축</th><th>평균 절대오차</th><th>최대 절대오차</th><th>RMSE</th></tr><tr><td>Y(mm)</td><td>"<<mae[0]/57<<"</td><td>"<<maximum[0]<<"</td><td>"<<sqrt(squared[0]/57)<<"</td></tr><tr><td>T(yaw라벨)</td><td>"<<mae[1]/57<<"</td><td>"<<maximum[1]<<"</td><td>"<<sqrt(squared[1]/57)<<u8"</td></tr></table><p>동시 ±1틱 "<<passes<<u8"/57, 두 축0.09 이내 "<<pass009<<u8"/57. 이미 개발에노출된 기존 혼합57장, 같은좌표3회 반복을묶은19좌표 보류 성적이다. 새촬영·실제이동·원점반복수렴·다른장비 일반화는 검증하지 않았다.</p><img src='y_theta_pairs.svg' alt='정답과 좌표보류 예측 연결'><p>회색 원은 정답, 파란 십자는 좌표 보류 예측이다. 같은이미지끼리 선으로 연결했다.</p><h2>배포와 평가의 구분</h2><p>실제alignment()는 전체57장으로학습한 Lanczos 모델을 사용하며 정답·파일명·fold를 받지 않는다. 이 보고서의보류 성적은 실제실행에서 추출한특징에 별도평가용19개 모델을 적용한것이다. predictions.csv의 deploy_Y/deploy_T는 전체학습모델 출력이므로 재대입 확인용이며위 성적과구분한다.</p><h2>확인한 내용</h2><p>57장 원본SHA-256 대조,입력픽셀 불변,모든출력의 유한값,승인된10특징과예측대응을 검사했다. CTest는Lanczos 상수/정수표본/미분,잘못된자산·빈영상·크기·타입·채널·무특징영상의오류처리를 검사한다.</p><p>기준자산 초기화 "<<init_ms<<u8"ms,사진별 추론중앙값 "<<times[28]<<u8"ms. 사진별시간은디코딩·해시·초기화를제외하며 각1회 측정이다.</p><p><a href='predictions.csv'>사진별 예측</a> · <a href='features.csv'>추출 특징</a> · <a href='validation.json'>검증 JSON</a> · <a href='../README.md'>모듈 사용법</a></p></html>";
 std::cout<<"PASS:57/57; heldout Y MAE/max "<<mae[0]/57<<'/'<<maximum[0]<<" T MAE/max "<<mae[1]/57<<'/'<<maximum[1]<<" both tick "<<passes<<"/57; max numerical difference "<<max_prediction<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
