#include "slide_data.hpp"
#include "source_provenance.hpp"
using namespace slide_history;
int wmain(int argc,wchar_t** wide_argv){
    Utf8Arguments args(argc,wide_argv);
    char** argv=args.data();
    try{
        if(argc!=5)throw std::runtime_error("historical_slide_data original-workspace cpp-v2-decisions.csv cpp-v3-predictions.csv new-output");
        fs::path root=fs::u8path(argv[1]),out=fs::u8path(argv[4]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        SlideData data(root,fs::u8path(argv[2]),fs::u8path(argv[3]));
        fs::create_directories(out);
        Json methods=Json::list(),control=Json::list();
        int compared=0,failures=0;
        for(auto& m:data.methods){
            methods.push(serialize(m));
            fs::path original=root/"analysis_v3/slide_img"/fs::u8path(m.id+".png");
            auto png=cv::imdecode(bytes(original),cv::IMREAD_COLOR);
            if(png.size()!=cv::Size(3948,332))throw std::runtime_error("Unexpected original slide panel dimensions");
            int failed=0;
            for(int axis=0; axis<2; ++axis)for(int i=0; i<=80; ++i){
                int command=axis?-40+i:-200+5*i;
                if(!command)continue;
                std::string file=std::string(axis?"T":"Y")+signed_value(command/100.)+".bmp";
                auto& flags=axis?m.tflags:m.yflags;
                bool expected=flags.at(file)==u8"불가";
                int x=30+i*48+22,y=(axis?248:34)-3;
                auto pixel=png.at<cv::Vec3b>(y,x);
                auto color=axis?green:orange;
                bool observed=pixel==cv::Vec3b(uchar(color[0]),uchar(color[1]),uchar(color[2]));
                failed+=expected!=observed;
                ++compared;
            }
            failures+=failed;
            Json item=Json::dict();
            item["method"]=m.id;
            item["original_panel_sha256"]=sha256(original);
            item["border_flag_mismatches"]=failed;
            control.push(item);
            std::cout<<m.name<<" Y "<<summary(m.yflags).dump()<<" T "<<summary(m.tflags).dump()<<std::endl;
        }
        write(out/"methods.json",methods.dump());
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["source_sha256"]=semes::original_source_hash(root,"analysis_v3/slide_data.py");
        audit["methods"]=int(data.methods.size());
        audit["basis_methods"]=8;
        audit["other_methods"]=6;
        audit["flags_generated"]=int(data.methods.size()*280);
        audit["original_png_border_flags_compared"]=compared;
        audit["original_png_border_mismatches"]=failures;
        audit["controls"]=control;
        audit["best_pair_grid_points"]=20001;
        audit["retained"]="Center-inclusive observations; fixed original three thresholds, seven mean bases, dense20001-point minimax search, original v2/v3 flags, same-sign/magnitude range summary. Inputs from already checked native C++ CSV computations.";
        audit["changed"]="C++ structured methods.json replaces Python module dictionaries/pickles. Archived original14 PNG borders independently check visible flags, not precise interpolated endpoint equality. No Python runtime or temporary pickle access.";
        audit["passed"]=failures==0;
        write(out/"checks.json",audit.dump());
        return failures?2:0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
