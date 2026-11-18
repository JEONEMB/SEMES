#include "qr_common.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#ifdef _WIN32
#include <locale>
#include <codecvt>
int wmain(int argc,wchar_t **argv){
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> cvt;
    if(argc<3||argc>4){std::cerr<<"Usage: qr_common_predict IMAGE CALIBRATION_DIR [evaluation_fold]\n";return 1;}
    std::string image=cvt.to_bytes(argv[1]),cal=cvt.to_bytes(argv[2]);int fold=argc==4?std::stoi(argv[3]):-1;
#else
int main(int argc,char **argv){
    if(argc<3||argc>4){std::cerr<<"Usage: qr_common_predict IMAGE CALIBRATION_DIR [evaluation_fold]\n";return 1;}
    std::string image=argv[1],cal=argv[2];int fold=argc==4?std::atoi(argv[3]):-1;
#endif
    qc_result result;int status=qc_predict_image(image.c_str(),cal.c_str(),fold,&result);
    if(status){std::cerr<<qc_status_string(status)<<"\n";return status;}
    std::cout<<std::setprecision(17)<<"{\"Y_mm\":"<<result.pose.y_mm<<",\"yaw_deg\":"<<result.pose.yaw_deg<<",\"features\":[";
    for(int i=0;i<10;i++)std::cout<<(i?",":"")<<result.features[i];
    std::cout<<"],\"qr_points\":"<<result.qr_points<<",\"upper_valid\":"<<result.upper_valid<<",\"lower_valid\":"<<result.lower_valid<<"}\n";
    return 0;
}
