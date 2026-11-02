#pragma once
// Test adapter only: original Armadillo math, JSON instead of R List transport.
#include <armadillo>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
namespace Rcpp {
using json = nlohmann::json;
template<class T> json encode(const T& x) { return json(x); }
template<class T> json encode(const arma::Col<T>& v) { json a=json::array(); for(auto z:v) a.push_back(z); return a; }
template<class T> json encode(const arma::Mat<T>& m) { json a=json::array(); for(arma::uword i=0;i<m.n_rows;++i){json row=json::array();for(arma::uword j=0;j<m.n_cols;++j)row.push_back(m(i,j));a.push_back(row);}return a; }
struct Named {
  std::string key;
  explicit Named(const char* name):key(name){}
  template<class T> std::pair<std::string,json> operator=(const T& v) const { return {key,encode(v)}; }
};
struct List {
  json value=json::object();
  template<class... P> static List create(P... pairs) { List out; ((out.value[pairs.first]=pairs.second),...); return out; }
};
}
