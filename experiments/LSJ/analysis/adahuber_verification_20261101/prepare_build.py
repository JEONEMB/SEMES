from pathlib import Path
import json,hashlib,difflib
HERE=Path(__file__).resolve().parent
source=(HERE/'sources/adaHuber/src/adaHuber.cpp').read_text(encoding='utf-8')
out=source
out=out.replace('  return 0.5 * (low + up);','  audit::events.push_back({{"kind","root"},{"low",low},{"up",up},{"iteration",ite},{"rhs",rhs},{"equation_residual",f1(0.5*(low+up),resSq,n,rhs)}});\n  return 0.5 * (low + up);')
out=out.replace('  return mu + mx;', '  audit::events.push_back({{"kind","mean"},{"gradient",derNew},{"iteration",ite},{"tau",tau},{"mu",mu+mx}});\n  return mu + mx;')
out=out.replace('  mu += mx;','  audit::events.push_back({{"kind","mean"},{"gradient",derNew},{"iteration",ite},{"tau",tau},{"mu",mu+mx}});\n  mu += mx;')
out=out.replace('  beta.rows(1, p) %= sx1;', '  audit::events.push_back({{"kind","regression_pre_intercept"},{"gradient_inf",arma::norm(gradNew,"inf")},{"iteration",ite},{"tau",tau},{"beta_standardized",Rcpp::encode(beta)}});\n  beta.rows(1, p) %= sx1;')
(HERE/'generated').mkdir(exist_ok=True)
(HERE/'generated/observed.cpp').write_text(out,encoding='utf-8')
# Isolated cause-confirmation probes, never used for real-data result table.
patched=source.replace('cross / derDiff * derDiff','cross / (derDiff * derDiff)').replace('arma::vec resSq = arma::square(Y);','arma::vec resSq = arma::square(res);')
(HERE/'generated/diagnostic_patch.cpp').write_text(patched,encoding='utf-8')
for name,text in [('observation',out),('diagnostic_patch',patched)]:
    (HERE/'generated'/f'{name}.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),fromfile='upstream',tofile=name)),encoding='utf-8')
print('Prepared unchanged upstream and separately labelled observation/probe variants.')
