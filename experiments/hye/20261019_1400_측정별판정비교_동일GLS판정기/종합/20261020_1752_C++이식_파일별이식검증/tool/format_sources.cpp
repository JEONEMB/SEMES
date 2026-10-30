// Minimal whitespace-only formatting for this migration's own generated C++.
// This tool preserves literal/comment contents and never operates on legacy source files.
#include "io.hpp"
#include <iostream>
#include <cctype>
using namespace semes;

static std::string format(const std::string& s) {
    std::string out;
    int indent = 0;
    int parentheses = 0;
    bool beginning = true;
    bool pending_space = false;
    auto emit = [&](char c) {
        if (beginning) {
            out.append(size_t(std::max(indent, 0)) * 4, ' ');
            beginning = false;
        }
        if (pending_space && !out.empty() && out.back() != '\n' && out.back() != ' ')
            out += ' ';
        pending_space = false;
        out += c;
    };
    auto newline = [&] {
        while (!out.empty() && out.back() == ' ') out.pop_back();
        if (!out.empty() && out.back() != '\n') out += '\n';
        beginning = true;
        pending_space = false;
    };
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!beginning) pending_space = true;
            continue;
        }
        if (c == '#') {
            newline();
            while (i < s.size() && s[i] != '\n') emit(s[i++]);
            newline();
            continue;
        }
        if(c=='R'&&i+1<s.size()&&s[i+1]=='"'){
            size_t open=s.find('(',i+2);
            if(open==std::string::npos||open-(i+2)>16)throw std::runtime_error("Invalid raw string delimiter");
            auto delimiter=s.substr(i+2,open-i-2),ending=")"+delimiter+"\"";
            size_t close=s.find(ending,open+1);
            if(close==std::string::npos)throw std::runtime_error("Unterminated raw string");
            emit(c);
            out.append(s,i+1,close+ending.size()-i-1);
            i=close+ending.size()-1;
            continue;
        }
        if (c == '"' || c == '\'') {
            const char quote = c;
            emit(c);
            while (++i < s.size()) {
                emit(s[i]);
                if (s[i] == '\\' && i + 1 < s.size()) emit(s[++i]);
                else if (s[i] == quote) break;
            }
            continue;
        }
        if (c == '/' && i + 1 < s.size() && s[i + 1] == '/') {
            emit(c);
            while (++i < s.size() && s[i] != '\n') emit(s[i]);
            newline();
            continue;
        }
        if (c == '/' && i + 1 < s.size() && s[i + 1] == '*') {
            emit(c);
            emit(s[++i]);
            while (++i < s.size()) {
                emit(s[i]);
                if (s[i] == '*' && i + 1 < s.size() && s[i + 1] == '/') {
                    emit(s[++i]);
                    break;
                }
            }
            continue;
        }
        if (c == '(') ++parentheses;
        if (c == ')') --parentheses;
        if (c == '{') {
            emit(c);
            newline();
            ++indent;
        } else if (c == '}') {
            newline();
            --indent;
            emit(c);
            size_t next = s.find_first_not_of(" \r\n\t", i + 1);
            if (next == std::string::npos || (s[next] != ';' && s[next] != ',' && s[next] != ')' && s[next] != ']')) newline();
        } else if (c == ';') {
            emit(c);
            if (parentheses == 0) newline();
            else pending_space = true;
        } else emit(c);
    }
    newline();
    return out;
}

int wmain(int argc,wchar_t** wide_argv){
    semes::Utf8Arguments unicode_arguments(argc,wide_argv);
    char** argv=unicode_arguments.data();
    try {
        if (argc != 2) throw std::runtime_error("format_sources final-root");
        if(std::string(argv[1])=="--selftest"){
            std::vector<std::string> raw={"R\"rx([\"']([^\"']{1,240}\\.csv)[\"'])rx\"","R\"({\n\"k\": \"v\",   \"n\": 4;\n})\"","u8R\"tag( braces { } quotes \" ' and ; )tag\""};
            for(auto& literal:raw){auto result=format("int main(){auto value="+literal+";return 0;}");if(result.find(literal)==std::string::npos)throw std::runtime_error("Formatter changed raw string bytes");if(format(result)!=result)throw std::runtime_error("Formatter not idempotent on raw literal sample");}
            std::cout<<"Raw literal preservation and idempotence:3 samples passed\n";
            return 0;
        }
        fs::path root = fs::weakly_canonical(fs::u8path(argv[1]));
        int count = 0;
        for (auto directory : {"include", "src", "tests", "examples", "tool", "experiments"}) {
            for (fs::recursive_directory_iterator it(root / directory), end; it != end; ++it) {
                if (it->is_directory() && it->path().filename() == "hw") {
                    it.disable_recursion_pending();
                    continue;
                }
                if (!it->is_regular_file()) continue;
                auto extension = it->path().extension();
                if (extension != ".cpp" && extension != ".hpp") continue;
                if (it->path().filename() == "format_sources.cpp") continue;
                auto text = read(it->path());
                auto formatted=format(text);
                if(formatted!=text){write(it->path(),formatted);++count;}
            }
        }
        std::cout << "Updated formatting in " << count << " migration source files\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
