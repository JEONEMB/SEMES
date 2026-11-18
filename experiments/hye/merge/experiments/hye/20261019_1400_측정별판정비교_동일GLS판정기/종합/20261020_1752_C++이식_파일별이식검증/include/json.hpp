#pragma once
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
namespace semes {
    struct Json {
        enum Type {
            Null, Bool, Number, String, Array, Object
        }
        type = Null;
        double number=0;
        std::string string;
        std::vector<Json> array;
        std::map<std::string,Json> object;
        std::vector<std::string> object_order;
        Json()=default;
        Json(double v):type(Number),number(v){
        }
        Json(int v):Json(double(v)){
        }
        Json(bool v):type(Bool),number(v){
        }
        Json(const std::string& v):type(String),string(v){
        }
        Json(const char* v):Json(std::string(v)){
        }
        static Json list(){
            Json j;
            j.type=Array;
            return j;
        }
        static Json dict(){
            Json j;
            j.type=Object;
            return j;
        }
        Json& operator[](const std::string& k){
            if(type==Null)type=Object;
            if(type!=Object)throw std::runtime_error("JSON object required");
            if(!object.count(k))object_order.push_back(k);
            return object[k];
        }
        const Json& at(const std::string& k)const{
            auto i=object.find(k);
            if(i==object.end())throw std::runtime_error("Missing JSON key: "+k);
            return i->second;
        }
        const Json& at(size_t i)const{
            return array.at(i);
        }
        size_t size()const{
            return type==Array?array.size():object.size();
        }
        bool has(const std::string& k)const{
            return object.count(k)!=0;
        }
        void push(const Json& j){
            if(type==Null)type=Array;
            array.push_back(j);
        }
        std::string str()const{
            if(type!=String)throw std::runtime_error("JSON string required");
            return string;
        }
        double num()const{
            if(type!=Number&&type!=Bool)throw std::runtime_error("JSON number required");
            return number;
        }
        static std::string quote(const std::string& s){
            std::ostringstream o;
            o<<'"';
            for(unsigned char c:s){
                switch(c){
                    case '"':o<<"\\\"";
                    break;
                    case '\\':o<<"\\\\";
                    break;
                    case '\n':o<<"\\n";
                    break;
                    case '\r':o<<"\\r";
                    break;
                    case '\t':o<<"\\t";
                    break;
                    default:if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
                    else o<<c;
                }
            }
            o<<'"';
            return o.str();
        }
        std::string dump()const{
            std::ostringstream o;
            o<<std::setprecision(17);
            switch(type){
                case Null:return "null";
                case Bool:return number?"true":"false";
                case Number:if(!std::isfinite(number))return "null";
                o<<number;
                break;
                case String:return quote(string);
                case Array:{
                    o<<'[';
                    bool first=true;
                    for(auto& v:array){
                        if(!first)o<<',';
                        first=false;
                        o<<v.dump();
                    }
                    o<<']';
                    break;
                }
                case Object:{
                    o<<'{';
                    bool first=true;
                    for(auto& key:object_order){
                        if(!first)o<<',';
                        first=false;
                        o<<quote(key)<<':'<<object.at(key).dump();
                    }
                    o<<'}';
                    break;
                }
            }
            return o.str();
        }
        struct Parser;
        static Json parse(const std::string& text);
    };
    struct Json::Parser {
        const std::string& s;
        size_t p=0;
        void ws(){
            while(p<s.size()&&(s[p]==' '||s[p]=='\n'||s[p]=='\r'||s[p]=='\t'))++p;
        }
        char get(){
            if(p>=s.size())throw std::runtime_error("Unexpected end of JSON");
            return s[p++];
        }
        void utf8(std::string& o,unsigned c){
            if(c<128)o+=char(c);
            else if(c<2048){
                o+=char(192|(c>>6));
                o+=char(128|(c&63));
            }
            else if(c<65536){
                o+=char(224|(c>>12));
                o+=char(128|((c>>6)&63));
                o+=char(128|(c&63));
            }
            else{
                o+=char(240|(c>>18));
                o+=char(128|((c>>12)&63));
                o+=char(128|((c>>6)&63));
                o+=char(128|(c&63));
            }
        }
        unsigned hex4(){
            unsigned u=0;
            for(int i=0; i<4; i++){
                char h=get();
                u=u*16;
                if(h>='0'&&h<='9')u+=h-'0';
                else if(h>='a'&&h<='f')u+=h-'a'+10;
                else if(h>='A'&&h<='F')u+=h-'A'+10;
                else throw std::runtime_error("Invalid Unicode escape");
            }
            return u;
        }
        std::string text(){
            if(get()!='"')throw std::runtime_error("Expected JSON string");
            std::string o;
            while(true){
                char c=get();
                if(c=='"')return o;
                if(c!='\\'){
                    o+=c;
                    continue;
                }
                c=get();
                switch(c){
                    case 'n':o+='\n';
                    break;
                    case 'r':o+='\r';
                    break;
                    case 't':o+='\t';
                    break;
                    case 'b':o+='\b';
                    break;
                    case 'f':o+='\f';
                    break;
                    case '"':case '\\':case '/':o+=c;
                    break;
                    case 'u':{
                        unsigned u=hex4();
                        if(u>=0xd800&&u<=0xdbff){
                            if(get()!='\\'||get()!='u')throw std::runtime_error("Missing Unicode surrogate");
                            unsigned v=hex4();
                            if(v<0xdc00||v>0xdfff)throw std::runtime_error("Invalid Unicode surrogate");
                            u=0x10000+((u-0xd800)<<10)+(v-0xdc00);
                        }
                        utf8(o,u);
                        break;
                    }
                    default:throw std::runtime_error("Invalid JSON escape");
                }
            }
        }
        Json value(){
            ws();
            if(p>=s.size())throw std::runtime_error("Empty JSON");
            char c=s[p];
            if(c=='"')return Json(text());
            if(c=='['||c=='{'){
                ++p;
                bool arr=c=='[';
                Json j=arr?Json::list():Json::dict();
                ws();
                if(p<s.size()&&s[p]==(arr?']':'}')){
                    ++p;
                    return j;
                }
                while(true){
                    if(arr)j.push(value());
                    else{
                        ws();
                        std::string k=text();
                        ws();
                        if(get()!=':')throw std::runtime_error("Expected colon");
                        j[k]=value();
                    }
                    ws();
                    char d=get();
                    if(d==(arr?']':'}'))break;
                    if(d!=',')throw std::runtime_error("Expected comma");
                }
                return j;
            }
            for(auto kv:{
                std::pair<const char*,Json>{
                    "null",Json()
                }, {
                    "true",Json(true)
                }, {
                    "false",Json(false)
                }
            }){
                std::string t=kv.first;
                if(s.compare(p,t.size(),t)==0){
                    p+=t.size();
                    return kv.second;
                }
            }
            char* end=nullptr;
            double n=std::strtod(s.c_str()+p,&end);
            if(end==s.c_str()+p||!std::isfinite(n))throw std::runtime_error("Invalid JSON number");
            p=end-s.c_str();
            return Json(n);
        }
    };
    inline Json Json::parse(const std::string& s){
        Parser p{
            s
        };
        if(s.size()>=3&&(unsigned char)s[0]==239)p.p=3;
        Json j=p.value();
        p.ws();
        if(p.p!=s.size())throw std::runtime_error("Trailing JSON input");
        return j;
    }
}
