#include "core.hpp"
#include "measurements.hpp"
#include <iostream>
int wmain(int argc,wchar_t**argv) {
    SetConsoleOutputCP(CP_UTF8);
    if(argc!=3) return 2;
    const auto ds=hw::loadSet(hw::u8(hw::fs::path(argv[1])));
    const auto m=hw::measureSet(ds);
    hw::Csv out(hw::fs::path(argv[2]),{"file","Y","T","rep","center","id","qr","diagonal","all_shots_usable","ref_x","ref_y","dx","dy","finite"});
    for(size_t i=0;i<ds.shots.size();++i) for(size_t k=0;k<m.corners.size();++k){
        const auto&s=ds.shots[i];const auto&c=m.corners[k];const auto&p=m.shots[i].pos[k];
        out.row({s.rel,hw::num(s.y,4),hw::num(s.t,4),std::to_string(s.rep),s.center?"1":"0",std::to_string(k),std::to_string(c.qr),c.diagonal?"1":"0",m.cornerUsable[k]?"1":"0",hw::num(c.p.x,6),hw::num(c.p.y,6),hw::num(p.x-c.p.x,6),hw::num(p.y-c.p.y,6),std::isfinite(p.x)&&std::isfinite(p.y)?"1":"0"});
    }
    std::cout<<"exported "<<ds.shots.size()<<" x "<<m.corners.size()<<std::endl;
}
