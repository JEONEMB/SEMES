from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
EXP=ROOT/'qrwithbar-lanczos-src/experiments/hye/20261102_173432_t_six_methods'
s=(EXP/'src/models.cpp').read_text(encoding='utf-8')
s=s.replace('namespace six {','''namespace six {
static std::map<std::string,double> source_weight;
static std::map<std::string,std::string> pose_key;
void register_training_metadata(const std::vector<Row>& base,const std::vector<double>& w){
 source_weight.clear();pose_key.clear();
 for(size_t i=0;i<base.size();i++){source_weight[base[i].path]=w.at(i);pose_key[base[i].path]=std::to_string(std::lround(base[i].y*100))+":"+std::to_string(std::lround(base[i].t*100));}
}
''',1)
s=s.replace('cv::Mat raw(n,p,CV_64F),target(n,1,CV_64F);','cv::Mat raw(n,p,CV_64F),target(n,1,CV_64F),sampleWeight(n,1,CV_64F);for(int i=0;i<n;i++)sampleWeight.at<double>(i)=source_weight.at(rows[ids[i]].path);')
start=s.index(' cv::reduce(raw,m.mean,0,cv::REDUCE_AVG);')
end=s.index('\n if(config.method==1||',start)
s=s[:start]+''' double totalWeight=cv::sum(sampleWeight)[0];m.mean=cv::Mat::zeros(1,p,CV_64F);
 for(int i=0;i<n;i++)m.mean+=raw.row(i)*(sampleWeight.at<double>(i)/totalWeight);
 m.scale=cv::Mat(1,p,CV_64F);for(int j=0;j<p;j++){double ss=0;for(int i=0;i<n;i++)ss+=sampleWeight.at<double>(i)*pow(raw.at<double>(i,j)-m.mean.at<double>(j),2);m.scale.at<double>(j)=std::max(1e-8,sqrt(ss/totalWeight));for(int i=0;i<n;i++)raw.at<double>(i,j)=(raw.at<double>(i,j)-m.mean.at<double>(j))/m.scale.at<double>(j);}
'''+s[end:]
s=s.replace('regress(phi,raw,config.alpha,{})','regress(phi,raw,config.alpha,sampleWeight)')
s=s.replace('regress(with_intercept(whitened),target,config.alpha,{})','regress(with_intercept(whitened),target,config.alpha,sampleWeight)')
s=s.replace('regress(x,target,config.alpha,{})','regress(x,target,config.alpha,sampleWeight)')
s=s.replace('regress(design,target,config.alpha,{})','regress(design,target,config.alpha,sampleWeight)')
s=s.replace('n*q[group[i]]/counts[group[i]]','sampleWeight.at<double>(i)*n*q[group[i]]/counts[group[i]]')
s=s.replace('std::map<std::pair<int,int>,std::vector<int>> groups;for(int i=0;i<n;i++)groups[{(int)std::lround(rows[ids[i]].y*100),(int)std::lround(rows[ids[i]].t*100)}].push_back(i);',
'''std::map<std::string,std::vector<int>> groups;for(int i=0;i<n;i++)groups[pose_key.at(rows[ids[i]].path)].push_back(i);''')
# Covariance uses actual nominal repeat groups. Soft weights do not drop any observation.
s=s.replace('int degrees=0;', 'double degrees=0;')
s=s.replace('for(int i:g)mean+=raw.row(i)/(double)g.size();for(int i:g){cv::Mat e=raw.row(i)-mean;cov+=e.t()*e;}degrees+=(int)g.size()-1;',
'''double sw=0,sw2=0;for(int i:g){double w=sampleWeight.at<double>(i);sw+=w;sw2+=w*w;mean+=raw.row(i)*w;}mean/=sw;for(int i:g){cv::Mat e=raw.row(i)-mean;cov+=sampleWeight.at<double>(i)*e.t()*e;}degrees+=sw-sw2/sw;''')
(P/'weighted_models.cpp').write_text(s,encoding='utf-8')

# Standalone extractor/CLI uses precisely the previous 104-feature image pipeline.
s=(EXP/'tools/main.cpp').read_text(encoding='utf-8')
s=s.replace('auto model=six::load_model(argv[3]);','cv::FileStorage metadata(std::filesystem::path(argv[3]).u8string(),cv::FileStorage::READ);if((std::string)metadata["coordinate_frame"]!="standard_origin_v3_assumed")throw std::runtime_error("Expected tagged v3 standard-coordinate model");metadata.release();auto model=six::load_model(argv[3]);')
s=s.replace('<<"{\\\"t_deg\\\":"','<<"{\\\"coordinate_frame\\\":\\\"standard_origin_v3_assumed\\\",\\\"physical_accuracy_certified\\\":false,\\\"t_deg\\\":"')
(P/'estimate.cpp').write_text(s,encoding='utf-8')
print('Generated weighted fitting implementation and tagged v3 image CLI.')
