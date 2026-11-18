// C++17. Input TSV: stratum<TAB>group_id (no header).
// Output TSV: stratum<TAB>group_id<TAB>test_fold.
// Metadata only: does not copy images or train a model.
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: fivefold_group_split groups.tsv > assignments.tsv");
        const std::array<std::string,4> strata{"1014_Y","1014_T","1020","1029"};
        std::ifstream in(argv[1]);
        if (!in) throw std::runtime_error("Cannot open input");
        std::map<std::string,std::vector<std::string>> groups;
        std::set<std::string> seen;
        std::string line;
        while (std::getline(in,line)) {
            if (!line.empty() && line.back()=='\r') line.pop_back();
            if (line.empty()) continue;
            const auto tab=line.find('\t');
            if (tab==std::string::npos || tab==0 || tab+1==line.size() || line.find('\t',tab+1)!=std::string::npos)
                throw std::runtime_error("Expected two nonempty TSV fields");
            auto s=line.substr(0,tab), id=line.substr(tab+1);
            if (std::find(strata.begin(),strata.end(),s)==strata.end()) throw std::runtime_error("Unknown stratum");
            if (!seen.insert(id).second) throw std::runtime_error("Duplicate group ID");
            groups[s].push_back(id);
        }
        if (groups.size()!=4) throw std::runtime_error("All four strata required");
        std::cout << "stratum\tgroup_id\ttest_fold\n";
        for (unsigned k=0;k<strata.size();++k) {
            auto v=groups.at(strata[k]);
            std::sort(v.begin(),v.end()); // group IDs must be canonical ASCII.
            std::mt19937 rng(20261104u+k);
            // Fisher-Yates with rejection sampling: no implementation-dependent shuffle.
            for (std::size_t n=v.size();n>1;--n) {
                if (n>0xffffffffULL) throw std::runtime_error("Too many groups");
                const auto bound=static_cast<unsigned>(n);
                const auto threshold=static_cast<unsigned>(-bound)%bound;
                unsigned r;
                do { r=static_cast<unsigned>(rng()); } while (r<threshold);
                std::swap(v[n-1],v[r%bound]);
            }
            std::size_t pos=0;
            for (unsigned fold=1;fold<=5;++fold) {
                auto count=v.size()/5+(fold<=v.size()%5 ? 1:0);
                for (std::size_t j=0;j<count;++j)
                    std::cout << strata[k] << '\t' << v[pos++] << '\t' << fold << '\n';
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
