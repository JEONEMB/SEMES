#pragma once
#include "io.hpp"
#include <memory>
#include <cstring>
namespace semes {
    // A symbolic parser for a NumPy object-array metadata payload. GLOBAL/REDUCE/BUILD
    // become inert nodes: no Python runtime, imports, constructors or code execution.
    // Only flat metadata lists are exposed; unsupported opcodes are rejected.
    class PickleData {
        struct Node;
        using Ptr=std::shared_ptr<Node>;
        struct Node{
            enum Kind{
                Null,Number,Text,List,Tuple,Symbol,Object,Mark
            }
            kind=Null;
            double number=0;
            std::string text;
            std::vector<Ptr> items;
        };
        const std::vector<unsigned char>& data;
        size_t cursor=0;
        std::vector<Ptr> stack,lists;
        std::map<uint32_t,Ptr> memo;
        static Ptr node(Node::Kind kind){
            auto p=std::make_shared<Node>();
            p->kind=kind;
            return p;
        }
        uint64_t integer(int count){
            if(count<0||count>8||cursor+count>data.size())throw std::runtime_error("Truncated pickle integer");
            uint64_t v=0;
            for(int i=0; i<count; ++i)v|=uint64_t(data[cursor++])<<(8*i);
            return v;
        }
        std::string string(size_t count){
            if(count>16*1024*1024||cursor+count>data.size())throw std::runtime_error("Pickle string length");
            std::string s(reinterpret_cast<const char*>(data.data()+cursor),count);
            cursor+=count;
            return s;
        }
        std::string line(){
            size_t begin=cursor;
            while(cursor<data.size()&&data[cursor]!='\n')++cursor;
            if(cursor==data.size())throw std::runtime_error("Truncated pickle line");
            return string_at(begin,cursor++-begin);
        }
        std::string string_at(size_t at,size_t count){
            return std::string(reinterpret_cast<const char*>(data.data()+at),count);
        }
        Ptr pop(){
            if(stack.empty())throw std::runtime_error("Pickle stack underflow");
            auto n=stack.back();
            stack.pop_back();
            return n;
        }
        void scalar(double value){
            auto p=node(Node::Number);
            p->number=value;
            stack.push_back(p);
        }
        void text(const std::string& value,Node::Kind kind=Node::Text){
            auto p=node(kind);
            p->text=value;
            stack.push_back(p);
        }
        std::vector<Ptr> marked(){
            std::vector<Ptr> items;
            while(!stack.empty()&&stack.back()->kind!=Node::Mark)items.push_back(pop());
            if(stack.empty())throw std::runtime_error("Pickle MARK missing");
            pop();
            std::reverse(items.begin(),items.end());
            return items;
        }
        void group(Node::Kind kind,int count){
            auto p=node(kind);
            if(count<0)p->items=marked();
            else{
                for(int i=0; i<count; ++i)p->items.push_back(pop());
                std::reverse(p->items.begin(),p->items.end());
            }
            stack.push_back(p);
            if(kind==Node::List)lists.push_back(p);
        }
        public: explicit PickleData(const std::vector<unsigned char>& bytes):data(bytes){
            if(data.size()>32*1024*1024)throw std::runtime_error("Metadata pickle exceeds limit");
            bool stopped=false;
            while(cursor<data.size()&&!stopped){
                unsigned op=unsigned(integer(1));
                switch(op){
                    case 0x80:if(integer(1)>5)throw std::runtime_error("Unsupported pickle protocol");
                    break;
                    case 0x95:{
                        uint64_t frame=integer(8);
                        if(frame>data.size()-cursor)throw std::runtime_error("Invalid pickle frame");
                        break;
                    }
                    case '(':stack.push_back(node(Node::Mark));
                    break;
                    case ')':group(Node::Tuple,0);
                    break;
                    case ']':group(Node::List,0);
                    break;
                    case 'l':group(Node::List,-1);
                    break;
                    case 't':group(Node::Tuple,-1);
                    break;
                    case 0x85:group(Node::Tuple,1);
                    break;
                    case 0x86:group(Node::Tuple,2);
                    break;
                    case 0x87:group(Node::Tuple,3);
                    break;
                    case 'N':stack.push_back(node(Node::Null));
                    break;
                    case 0x88:scalar(1);
                    break;
                    case 0x89:scalar(0);
                    break;
                    case 'K':scalar(double(integer(1)));
                    break;
                    case 'M':scalar(double(integer(2)));
                    break;
                    case 'J':scalar(double(int32_t(integer(4))));
                    break;
                    case 'G':{
                        uint64_t bits=0;
                        for(int i=0; i<8; ++i)bits=(bits<<8)|integer(1);
                        double value;
                        std::memcpy(&value,&bits,8);
                        scalar(value);
                        break;
                    }
                    case 'X':{
                        auto count=integer(4);
                        text(string(size_t(count)));
                        break;
                    }
                    case 0x8c:{
                        auto count=integer(1);
                        text(string(size_t(count)));
                        break;
                    }
                    case 0x8d:{
                        auto count=integer(8);
                        text(string(size_t(count)));
                        break;
                    }
                    case 'C':{
                        auto count=integer(1);
                        text(string(size_t(count)));
                        break;
                    }
                    case 'B':{
                        auto count=integer(4);
                        text(string(size_t(count)));
                        break;
                    }
                    case 0x94:if(stack.empty())throw std::runtime_error("Empty memo value");
                    memo[uint32_t(memo.size())]=stack.back();
                    break;
                    case 'q':{
                        auto key=uint32_t(integer(1));
                        if(stack.empty())throw std::runtime_error("Empty memo value");
                        memo[key]=stack.back();
                        break;
                    }
                    case 'r':{
                        auto key=uint32_t(integer(4));
                        if(stack.empty())throw std::runtime_error("Empty memo value");
                        memo[key]=stack.back();
                        break;
                    }
                    case 'h':{
                        auto key=uint32_t(integer(1));
                        stack.push_back(memo.at(key));
                        break;
                    }
                    case 'j':{
                        auto key=uint32_t(integer(4));
                        stack.push_back(memo.at(key));
                        break;
                    }
                    case 'c':{
                        auto module=line(),name=line();
                        text(module+"."+name,Node::Symbol);
                        break;
                    }
                    case 0x93:{
                        auto name=pop(),module=pop();
                        text(module->text+"."+name->text,Node::Symbol);
                        break;
                    }
                    case 'R':case 0x81:{
                        auto args=pop(),symbol=pop();
                        auto p=node(Node::Object);
                        p->items={
                            symbol,args
                        };
                        stack.push_back(p);
                        break;
                    }
                    case 'b':{
                        auto state=pop();
                        if(stack.empty())throw std::runtime_error("BUILD target missing");
                        stack.back()->items.push_back(state);
                        break;
                    }
                    case 'e':{
                        auto items=marked();
                        if(stack.empty()||stack.back()->kind!=Node::List)throw std::runtime_error("APPENDS target not a list");
                        stack.back()->items.insert(stack.back()->items.end(),items.begin(),items.end());
                        break;
                    }
                    case 'a':{
                        auto item=pop();
                        if(stack.empty()||stack.back()->kind!=Node::List)throw std::runtime_error("APPEND target not a list");
                        stack.back()->items.push_back(item);
                        break;
                    }
                    case '.':stopped=true;
                    break;
                    default:throw std::runtime_error("Unsupported symbolic pickle opcode "+std::to_string(op));
                }
                if(stack.size()>1000000||memo.size()>1000000)throw std::runtime_error("Pickle metadata limit");
            }
            if(!stopped||cursor!=data.size()||stack.size()!=1)throw std::runtime_error("Invalid pickle termination");
        }
        std::vector<std::vector<std::string>> string_rows(size_t count,size_t width)const{
            Ptr selected;
            for(auto& list:lists)if(list->items.size()==count*width){
                bool flat=true;
                for(auto& p:list->items)flat&=p->kind==Node::Text||p->kind==Node::Number;
                if(flat){
                    if(selected)throw std::runtime_error("Ambiguous metadata list");
                    selected=list;
                }
            }
            if(!selected)throw std::runtime_error("Object-array metadata rows missing");
            std::vector<std::vector<std::string>> rows(count,std::vector<std::string>(width));
            for(size_t i=0; i<count*width; ++i){
                auto& p=selected->items[i];
                if(p->kind==Node::Text)rows[i/width][i%width]=p->text;
                else{
                    std::ostringstream s;
                    s<<std::setprecision(17)<<p->number;
                    rows[i/width][i%width]=s.str();
                }
            }
            return rows;
        }
    };
}
