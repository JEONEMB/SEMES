#include "table.hpp"
#include <iostream>
#include <set>
#include <limits>
using namespace semes;
int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try {
        if(argc<7)throw std::runtime_error("compare_csv expected.csv actual.csv key-column new-output tolerance numeric-columns...");
        CsvTable expected(fs::u8path(argv[1])),actual(fs::u8path(argv[2]));
        std::string key=argv[3];
        fs::path out=fs::u8path(argv[4]);
        double tolerance=std::stod(argv[5]);
        auto key_columns=csvrow(key);
        auto row_key=[&](const CsvTable& table,size_t row){
            if(key_columns.size()==1)return table.text(row,key_columns[0]);
            Json values=Json::list();
            for(auto& column:key_columns)values.push(table.text(row,column));
            return values.dump();
        };
        if(fs::exists(out))throw std::runtime_error("New output required");
        std::map<std::string,size_t> lookup;
        for(size_t i=0; i<actual.rows.size(); ++i)if(!lookup.emplace(row_key(actual,i),i).second)throw std::runtime_error("Duplicate actual comparison key");
        fs::create_directories(out);
        Json columns=Json::list();
        int missing=0,failed=0;
        std::ofstream differences(out/"differences.csv");
        differences<<"key,column,expected,actual,abs_difference\n";
        for(int c=6; c<argc; ++c) {
            std::string field=argv[c];
            const bool float32=field.rfind("f32:",0)==0;
            if(float32)field=field.substr(4);
            double maximum=0,squared=0;
            int count=0,nan_mismatch=0,different=0,outside=0;
            for(size_t i=0; i<expected.rows.size(); ++i) {
                auto name=row_key(expected,i);
                if(!lookup.count(name)){
                    if(c==6)++missing;
                    continue;
                }
                double a=expected.number(i,field),b=actual.number(lookup.at(name),field);
                if(float32){a=double(float(a));b=double(float(b));}
                if(!std::isfinite(a)||!std::isfinite(b)){
                    nan_mismatch+=!(std::isnan(a)&&std::isnan(b))&&a!=b;
                    continue;
                }
                double error=std::abs(a-b);
                maximum=std::max(maximum,error);
                squared+=error*error;
                ++count;
                different+=error!=0;
                double representation_slack=tolerance>0?16*std::numeric_limits<double>::epsilon()*std::max({
                    1.0,std::abs(a),std::abs(b)
                }):0;
                if(error>tolerance+representation_slack){
                    ++outside;
                    differences<<std::setprecision(17)<<csv(name)<<','<<csv(field)<<','<<a<<','<<b<<','<<error<<'\n';
                }
            }
            Json j=Json::dict();
            j["column"]=field;
            if(float32)j["comparison_dtype"]="float32 restored from CSV; caller must establish original field dtype";
            j["n"]=count;
            j["nonzero_differences"]=different;
            j["outside_tolerance"]=outside;
            j["max_abs_difference"]=maximum;
            j["rmse"]=count?std::sqrt(squared/count):0;
            j["nonfinite_mismatches"]=nan_mismatch;
            j["passed"]=!outside&&!nan_mismatch;
            failed+=!j.at("passed").num();
            columns.push(j);
        }
        Json summary=Json::dict();
        summary["expected_rows"]=int(expected.rows.size());
        summary["actual_rows"]=int(actual.rows.size());
        summary["missing_expected_keys"]=missing;
        summary["failed_numeric_columns"]=failed;
        summary["tolerance"]=tolerance;
        summary["columns"]=columns;
        summary["expected_sha256"]=sha256(fs::u8path(argv[1]));
        summary["actual_sha256"]=sha256(fs::u8path(argv[2]));
        summary["created_kst"]=now();
        write(out/"checks.json",summary.dump());
        std::cout<<summary.dump()<<'\n';
        return missing||failed||expected.rows.size()!=actual.rows.size()?2:0;
    }
    catch(const std::exception&e){
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
