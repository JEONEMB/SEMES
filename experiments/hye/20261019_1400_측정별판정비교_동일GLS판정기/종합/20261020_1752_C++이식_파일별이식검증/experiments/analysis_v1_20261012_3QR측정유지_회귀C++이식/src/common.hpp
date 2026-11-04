#pragma once
#include "alignment.hpp"
#include "table.hpp"
namespace agent_history {
    inline std::array<double,6> positions(const semes::CsvTable& table,size_t row) {
        std::array<double,6> v{
        };
        for(int i=0; i<3; ++i) {
            std::string prefix="q"+std::to_string(i)+"_";
            v[2*i]=table.number(row,prefix+"dx")+(table.number(row,prefix+"a00")-1)*90+table.number(row,prefix+"a01")*96;
            v[2*i+1]=table.number(row,prefix+"dy")+table.number(row,prefix+"a10")*90+(table.number(row,prefix+"a11")-1)*96;
        }
        return v;
    }
    inline void write_row(std::ostream& out,const std::vector<std::string>& row) {
        for(size_t i=0; i<row.size(); ++i){
            if(i)out<<',';
            out<<semes::csv(row[i]);
        }
        out<<'\n';
    }
    inline std::string num(double value){
        std::ostringstream s;
        s<<std::setprecision(17)<<value;
        return s.str();
    }
}
