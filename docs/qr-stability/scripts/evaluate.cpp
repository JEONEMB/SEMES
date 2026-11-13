#include "alignment.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){
 if(argc!=3)return 1;
 cv::setNumThreads(1); // Parallelism is managed across independent worker processes.
 std::ifstream list(argv[1]);std::ofstream csv(std::string(argv[2])+"/points.csv"),meta(std::string(argv[2])+"/frames.csv"),log(std::string(argv[2])+"/processing.log");
 std::ofstream qrmeta(std::string(argv[2])+"/qr_status.csv");
 if(!list||!csv||!meta||!log||!qrmeta)return 1;
 qrmeta<<"frame,slot,roi_valid,payload,decoded,model_matched,features_valid,status,candidates,accepted,model_mismatch,score,gaussian,sigma\n";
 auto original=std::cout.rdbuf(log.rdbuf());
 csv<<"frame,payload,id,x,y,accepted,localAccepted,shift,eigenvalue,contrast,heldout_residual\n";csv<<std::setprecision(10);
 meta<<"frame,path,plate,decoded,valid_features,error\n";
 std::string path;int frame=0;
 while(std::getline(list,path)){
  int decoded=0,valid=0;bool found=false;std::string error;
  try{cv::Mat image=cv::imread(path);AlignmentOptions o;o.visualize=false;PlateDetectionResult r;alignment(image,o,&r);found=r.found;
   int slot=0; for(const auto&q:r.qrRois){decoded+=q.decoded;valid+=q.featuresValid; int n=0;for(const auto&f:q.features)n+=f.accepted;
 qrmeta<<frame<<','<<slot++<<','<<q.valid<<','<<q.payload<<','<<q.decoded<<','<<q.modelMatched<<','<<q.featuresValid<<','<<q.featureStatus<<','<<q.features.size()<<','<<n<<','<<q.modelMismatch<<','<<r.score<<','<<q.decodeUsedGaussian<<','<<q.decodeGaussianSigma<<'\n';
    for(const auto&f:q.features)csv<<frame<<','<<q.payload<<','<<f.id<<','<<f.refined.x<<','<<f.refined.y<<','<<f.accepted<<','<<f.localAccepted<<','<<f.shift<<','<<f.minEigenvalue<<','<<f.contrast<<','<<f.reprojectionError<<'\n';}
  }catch(const std::exception&e){error="processing_exception";log<<e.what()<<'\n';}
  meta<<frame<<','<<path<<','<<found<<','<<decoded<<','<<valid<<','<<error<<'\n';
  if(frame%20==0)std::cerr<<"Processed "<<frame+1<<" images\n";
  ++frame;
 }
 std::cout.rdbuf(original);std::cout<<"Finished "<<frame<<" images\n";
}
