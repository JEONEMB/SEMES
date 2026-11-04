#include "features.hpp"
#include <iostream>
using namespace semes;
using namespace v2_history;
namespace {
    // Regularized incomplete gamma, followed by bisection for chi-square quantiles.
    // Computation is double precision and has no SciPy runtime dependency.
    double gamma_cdf(double a,double x) {
        if(x<=0)return 0;
        const double factor=std::exp(a*std::log(x)-x-std::lgamma(a));
        if(x<a+1){
            double term=1/a,sum=term;
            for(int i=1; i<10000; ++i){
                term*=x/(a+i);
                sum+=term;
                if(std::abs(term)<std::abs(sum)*1e-15)return sum*factor;
            }
            throw std::runtime_error("Gamma series did not converge");
        }
        double b=x+1-a,c=1e300,d=1/b,h=d;
        for(int i=1; i<10000; ++i){
            double an=-i*(i-a);
            b+=2;
            d=an*d+b;
            if(std::abs(d)<1e-300)d=1e-300;
            c=b+an/c;
            if(std::abs(c)<1e-300)c=1e-300;
            d=1/d;
            double delta=d*c;
            h*=delta;
            if(std::abs(delta-1)<1e-15)return 1-factor*h;
        }
        throw std::runtime_error("Gamma continued fraction did not converge");
    }
    double chi_square_quantile(double probability,int dimensions) {
        double low=0,high=dimensions;
        while(gamma_cdf(dimensions/2.0,high/2)<probability)high*=2;
        for(int i=0; i<160; ++i){
            double mid=(low+high)/2;
            if(gamma_cdf(dimensions/2.0,mid/2)<probability)low=mid;
            else high=mid;
        }
        return (low+high)/2;
    }
    cv::Mat residual(const Observation& row,const std::vector<Observation>& reference,const std::vector<std::string>& fields) {
        auto upper=std::upper_bound(reference.begin(),reference.end(),row.mean_shift,[](double m,const Observation& r){
            return m<r.mean_shift;
        });
        const auto& b=upper==reference.end()?reference.back():*upper;
        const auto& a=upper==reference.begin()?reference.front():*(upper-1);
        double fraction=b.mean_shift==a.mean_shift?0:(row.mean_shift-a.mean_shift)/(b.mean_shift-a.mean_shift);
        cv::Mat r(1,int(fields.size()),CV_64F);
        for(int j=0; j<r.cols; ++j)r.at<double>(0,j)=row.features.at(fields[j])-(a.features.at(fields[j])+fraction*(b.features.at(fields[j])-a.features.at(fields[j])));
        return r;
    }
    cv::Mat loo(const std::vector<Observation>& rows,const std::vector<std::string>& fields) {
        cv::Mat result;
        for(size_t i=1; i+1<rows.size(); ++i){
            auto reference=rows;
            reference.erase(reference.begin()+i);
            result.push_back(residual(rows[i],reference,fields));
        }
        return result;
    }
    cv::Mat population_sd(const cv::Mat& values) {
        cv::Mat mean,sd;
        cv::reduce(values,mean,0,cv::REDUCE_AVG);
        cv::Mat centered=values-cv::repeat(mean,values.rows,1);
        cv::multiply(centered,centered,centered);
        cv::reduce(centered,sd,0,cv::REDUCE_AVG);
        cv::sqrt(sd,sd);
        return sd;
    }
    bool rule1(const cv::Mat& r,const cv::Mat& threshold){
        for(int i=0; i<r.cols; ++i)if(std::abs(r.at<double>(0,i))>threshold.at<double>(0,i))return true;
        return false;
    }
    double distance(const cv::Mat& r,const cv::Mat& inverse){
        cv::Mat d=r*inverse*r.t();
        return d.at<double>(0,0);
    }
}
int wmain(int argc,wchar_t** wide_argv) {
    Utf8Arguments arguments(argc,wide_argv);
    char** argv=arguments.data();
    try {
        if(argc!=3)throw std::runtime_error("historical_v2_judge original-workspace new-output");
        fs::path root=fs::u8path(argv[1]),out=fs::u8path(argv[2]);
        if(fs::exists(out))throw std::runtime_error("New output required");
        Features data(root);
        fs::create_directories(out);
        std::ofstream summary(out/"judge_summary.csv"),decisions(out/"decisions.csv"),loo_output(out/"loo_residuals.csv"),heatmap(out/"heatmap.csv");
        summary<<"configuration,dimensions,residual_count,rule1_Y_unavailable,rule1_T_unavailable,rule1_false_positives,rule1_fp_rate,rule2_Y_unavailable,rule2_T_unavailable,rule2_false_positives,rule2_fp_rate,distance_threshold,Y_in_range,T_in_range\n";
        decisions<<"configuration,file,axis,filename_command,mean_qr_dx,in_range,rule1,rule2,distance\n";
        loo_output<<"configuration,axis,row,file,feature,residual\n";
        heatmap<<"configuration,Y_abs_lower,Y_abs_upper,n,unavailable,fraction\n";
        Json models=Json::list();
        for(auto& config:data.configurations){
            const auto& fields=config.second;
            auto ly=loo(data.ys,fields),lt=loo(data.ts,fields);
            cv::Mat l;
            cv::vconcat(ly,lt,l);
            cv::Mat threshold;
            cv::max(population_sd(ly),population_sd(lt),threshold);
            threshold*=3;
            cv::Mat mean;
            cv::reduce(l,mean,0,cv::REDUCE_AVG);
            cv::Mat centered=l-cv::repeat(mean,l.rows,1),covariance=(centered.t()*centered)/(l.rows-1);
            double jitter=1e-9*cv::trace(covariance)[0]/covariance.cols;
            for(int j=0; j<covariance.cols; ++j)covariance.at<double>(j,j)+=jitter;
            cv::Mat inverse;
            if(cv::invert(covariance,inverse,cv::DECOMP_LU)==0)throw std::runtime_error("Historical covariance singular");
            double dthr=chi_square_quantile(.9973,int(fields.size()));
            int fp1=0,fp2=0;
            for(int i=0; i<l.rows; ++i){
                fp1+=rule1(l.row(i),threshold);
                fp2+=distance(l.row(i),inverse)>dthr;
            }
            int unavailable[2][2]={
                {
                    0,0
                },{
                    0,0
                }
            },in_range[2]={
                0,0
            };
            std::array<int,6> bin_count{
            },bin_unavailable{
            };
            const std::array<double,7> bins={
                0,.25,.5,.75,1,1.5,2.01
            };
            for(int axis=0; axis<2; ++axis){
                const auto& rows=axis==0?data.ys:data.ts;
                const auto& reference=axis==0?data.ts:data.ys;
                const cv::Mat& residuals=axis==0?ly:lt;
                for(int i=0; i<residuals.rows; ++i)for(int j=0; j<residuals.cols; ++j)loo_output<<csv(config.first)<<','<<(axis==0?'Y':'T')<<','<<i+1<<','<<csv(rows[i+1].file)<<','<<csv(fields[j])<<','<<std::setprecision(17)<<residuals.at<double>(i,j)<<'\n';
                for(auto& row:rows){
                    bool supported=row.mean_shift>=reference.front().mean_shift&&row.mean_shift<=reference.back().mean_shift;
                    bool center=row.file=="center.bmp";
                    std::string first=u8"범위 밖",second=first;
                    double d=0;
                    if(center){
                        first=second=u8"기준";
                    }
                    else if(supported){
                        auto r=residual(row,reference,fields);
                        d=distance(r,inverse);
                        bool separated1=rule1(r,threshold),separated2=d>dthr;
                        first=separated1?u8"가능":u8"불가";
                        second=separated2?u8"가능":u8"불가";
                        ++in_range[axis];
                        unavailable[axis][0]+=!separated1;
                        unavailable[axis][1]+=!separated2;
                        if(axis==0)for(int bin=0; bin<6; ++bin)if(std::abs(row.value)>bins[bin]&&std::abs(row.value)<=bins[bin+1]){
                            ++bin_count[bin];
                            bin_unavailable[bin]+=!separated2;
                        }
                    }
                    decisions<<csv(config.first)<<','<<csv(row.file)<<','<<(axis==0?'Y':'T')<<','<<std::setprecision(17)<<row.value<<','<<row.mean_shift<<','<<(supported&&!center)<<','<<csv(first)<<','<<csv(second)<<',';
                    if(supported&&!center)decisions<<d;
                    decisions<<'\n';
                }
            }
            summary<<csv(config.first)<<','<<fields.size()<<','<<l.rows<<','<<unavailable[0][0]<<','<<unavailable[1][0]<<','<<fp1<<','<<std::setprecision(17)<<fp1/double(l.rows)<<','<<unavailable[0][1]<<','<<unavailable[1][1]<<','<<fp2<<','<<fp2/double(l.rows)<<','<<dthr<<','<<in_range[0]<<','<<in_range[1]<<'\n';
            for(int bin=0; bin<6; ++bin){
                heatmap<<csv(config.first)<<','<<bins[bin]<<','<<bins[bin+1]<<','<<bin_count[bin]<<','<<bin_unavailable[bin]<<',';
                if(bin_count[bin])heatmap<<bin_unavailable[bin]/double(bin_count[bin]);
                heatmap<<'\n';
            }
            Json model=Json::dict();
            model["configuration"]=config.first;
            Json columns=Json::list();
            for(auto& field:fields)columns.push(field);
            model["fields"]=columns;
            model["threshold3sigma"]=jsonmatrix(threshold);
            model["inverse_covariance"]=jsonmatrix(inverse);
            model["chi2_threshold"]=dthr;
            models.push(model);
            std::cout<<config.first<<": rule2 unavailable Y "<<unavailable[0][1]<<"/"<<in_range[0]<<", T "<<unavailable[1][1]<<"/"<<in_range[1]<<", false positives "<<fp2<<"/"<<l.rows<<std::endl;
        }
        write(out/"historical_calibration.json",models.dump());
        Json audit=Json::dict();
        audit["created_kst"]=now();
        audit["joined_center_present"]=data.joined_center_present;
        audit["limitations"]="Reproduces full-data historical feature distinguishability calibration. This is neither held-out classification accuracy nor the current four-fold OOF inference. Command values are used only for historical analysis and output grouping.";
        audit["changed"]="Native C++ incomplete-gamma chi-square inverse and OpenCV linear algebra replace SciPy/NumPy. JSON and CSV replace Python pickle. Workbook and plot packaging remain separate.";
        write(out/"audit.json",audit.dump());
        return 0;
    }
    catch(const std::exception& e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
