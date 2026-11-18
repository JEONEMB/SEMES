"""Apply scoped C++ edits with checked anchors."""
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'reference_line_methods_108_20261030/cpp'
def edit(name,old,new,count=1):
    p=BASE/name;s=p.read_text(encoding='utf-8-sig')
    assert s.count(old)==count,(name,old[:60],s.count(old))
    p.write_text(s.replace(old,new),encoding='utf-8')

edit('src/methods_points.cpp','#include "methods.hpp"','#include "methods.hpp"\n#include "zernike_location.hpp"')
edit('src/methods_points.cpp','MethodResult method_zernike(MethodContext& c) {\n    MethodResult r;',
     'MethodResult method_zernike(MethodContext& c) {\n    validate_method_parameters(c.target.kind=="qr"?"Q05":"E05",c.params);\n    MethodResult r;')
edit('src/methods_points.cpp','    const ZernikeMasks& Z = zernike_masks(Np, ss);\n    const int h = Np / 2;\n    const double wbar = 2.0 * k_edge * blur_px / Np;',
     '    const int h = Np / 2;\n    const double wbar = 2.0 * k_edge * blur_px / Np;\n    if (!std::isfinite(wbar) || wbar<=0) { r.fail("invalid_parameters", "Zernike normalized width must be positive and finite"); return r; }\n    const ZernikeMasks& Z = zernike_masks(Np, ss);')
p=BASE/'src/methods_points.cpp';s=p.read_text(encoding='utf-8')
start=s.index('            psi = std::atan2(aim, are);',s.index('MethodResult method_zernike'))
end=s.index('            if (ex_every > 0',start)
s=s[:start]+'''            auto location=zernike_location(are,aim,a20,wbar,Np,x,best_y,t.polarity,max_dev,max_shift);
            psi=location.psi;Q=location.ratio;l=location.distance;shift=location.shift;u=location.u;v=location.v;
            if (!location.ok) state=excluded(location.reason);
            else { state="selected"; pts.emplace_back(u,v); }
'''+s[end:]
s=s.replace('double psi = 0, Q = 0, l = 0, shift = 0, u = 0, v = 0, are = 0, aim = 0, a20 = 0;',
'''const double missing=std::numeric_limits<double>::quiet_NaN();
        double psi=missing,Q=missing,l=missing,shift=missing,u=missing,v=missing,are=0,aim=0,a20=0;''')
s=s.replace('{"A20", a20}, {"psi_deg", psi * 180 / kPi}, {"Q_E", Q}, {"l_ramp", l},',
'''{"A20", a20}, {"A11_projected", location.denominator}, {"ramp_discriminant", location.discriminant},
                                    {"failure_reason", location.ok ? json(nullptr) : json(location.reason)},
                                    {"psi_deg", psi * 180 / kPi}, {"Q_E", Q}, {"l_ramp", l},''')
s=s.replace('    r.diagnostics["w_bar_normalised"] = wbar;',
'''    r.diagnostics["w_bar_normalised"] = wbar;
    r.diagnostics["numeric_guard_policy"] = "finite moments/denominator/ratio/discriminant/location; exact zero denominator and negative discriminant rejected; no new near-zero tolerance";
    json counts=json::object();for(const auto& row:cand){std::string key=row[2];counts[key]=counts.value(key,0)+1;}
    r.diagnostics["candidate_state_counts"]=counts;''')
p.write_text(s,encoding='utf-8')

# Authoritative raw coordinates are written centrally, before any common fit or failure.
for name,var,digits in [('methods_points.cpp','pts',5),('methods_hough.cpp','near',1),('methods_qr.cpp','pts',1)]:
    block=f'    json fitpts = json::array();\n    for (auto& q : {var}) fitpts.push_back(round_pt(q.x, q.y, {digits}));\n    r.intermediates["fit_points"] = fitpts;\n'
    edit('src/'+name,block,'',4 if name=='methods_points.cpp' else 1)
edit('src/methods_segments.cpp','    json fitpts = json::array();\n    for (auto& q : pts) fitpts.push_back({q.x,q.y}); // Preserve clipped/open-boundary membership in stored coordinates.\n    r.intermediates["fit_points"] = fitpts;\n','')
edit('src/line_fit.cpp','    FitResult f = robust_fit(pts, fp);',
'''    json raw=json::array();for(const auto& q:pts)raw.push_back({q.x,q.y});
    r.intermediates["fit_points"]=raw;
    r.intermediates["fit_points_precision"]={{"storage","binary64_json_roundtrip"},
        {"role","all input points passed to common fit, before trimming"},
        {"display_rounding_is_separate",true},
        {"nonfinite_policy","invalid coordinates cause fit failure; JSON nonfinite values are null"}};
    FitResult f = robust_fit(pts, fp);''')
edit('src/methods_qr.cpp','    r.n_selected = 2;',
'''    r.n_selected = 2;
    r.intermediates["line_input_points"]={{a.x,a.y},{b.x,b.y}};
    r.intermediates["line_input_points_precision"]={{"storage","binary64_json_roundtrip"},
        {"role","two endpoints passed to normalize_line; upstream corner precision unchanged"}};''')
edit('src/methods_qr.cpp','contour points are centres of the outermost bright pixels; the physical boundary lies about 0.5 px above (constant offset, no slope effect)',
     'contour points are centres of the outermost bright pixels; offset from the physical boundary and its effect on slope require separate validation; no fixed 0.5 px correction is applied')
edit('src/main.cpp','        meta["settings_schema_version"] = 2;',
'''        meta["settings_schema_version"] = 2;
        meta["point_record_schema_version"] = 2;
        meta["point_record_precision"] = "fit_points/line_input_points preserve finite calculation coordinates without decimal rounding; viewer coordinates are display-only; upstream precision is unchanged";''')
edit('CMakeLists.txt','    src/segment_support.cpp src/effective_settings.cpp',
     '    src/segment_support.cpp src/effective_settings.cpp src/zernike_location.cpp')
p=BASE/'CMakeLists.txt';s=p.read_text(encoding='utf-8')
s+='''
# Arithmetic domains and calculation-coordinate JSON round-trip; no captured images.
add_executable(numeric_record_test tests/numeric_record_test.cpp src/zernike_location.cpp src/line_fit.cpp src/io_utils.cpp)
target_include_directories(numeric_record_test PRIVATE include ${TP}/nlohmann_json/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(numeric_record_test PRIVATE ${OpenCV_LIBS})
if(MSVC)
    target_compile_definitions(numeric_record_test PRIVATE _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(numeric_record_test PRIVATE /utf-8 /W4)
endif()
if(WIN32)
    add_custom_command(TARGET numeric_record_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
            "$<TARGET_FILE_DIR:numeric_record_test>")
endif()
'''
p.write_text(s,encoding='utf-8')
print('C++ edits applied')
