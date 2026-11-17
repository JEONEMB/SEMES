#include "io.hpp"
#include <windows.h>
#include <bcrypt.h>
#include <zlib.h>
#include <fstream>
#include <cstring>
#include <regex>
#include <chrono>
namespace semes {
    std::vector<unsigned char> bytes(const fs::path& p){
        std::ifstream f(p,std::ios::binary);
        if(!f)throw std::runtime_error("Cannot read "+p.u8string());
        return {
            std::istreambuf_iterator<char>(f),{
            }
        };
    }
    std::string read(const fs::path& p){
        auto b=bytes(p);
        return {
            b.begin(),b.end()
        };
    }
    void write(const fs::path& p,const std::string& s){
        fs::create_directories(p.parent_path());
        std::ofstream f(p,std::ios::binary);
        if(!f)throw std::runtime_error("Cannot write "+p.u8string());
        f<<s;
        if(!f)throw std::runtime_error("Write failed "+p.u8string());
    }
    std::string sha256(const fs::path& p){
        BCRYPT_ALG_HANDLE a=nullptr;
        BCRYPT_HASH_HANDLE h=nullptr;
        auto check=[](NTSTATUS s){
            if(s<0)throw std::runtime_error("SHA256 failed");
        };
        check(BCryptOpenAlgorithmProvider(&a,BCRYPT_SHA256_ALGORITHM,nullptr,0));
        check(BCryptCreateHash(a,&h,nullptr,0,nullptr,0,0));
        std::ifstream f(p,std::ios::binary);
        if(!f)throw std::runtime_error("Hash read failed "+p.u8string());
        std::vector<unsigned char> buf(1<<20);
        while(f){
            f.read((char*)buf.data(),buf.size());
            auto n=f.gcount();
            if(n)check(BCryptHashData(h,buf.data(),ULONG(n),0));
        }
        unsigned char digest[32];
        check(BCryptFinishHash(h,digest,32,0));
        BCryptDestroyHash(h);
        BCryptCloseAlgorithmProvider(a,0);
        std::ostringstream o;
        for(auto c:digest)o<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c);
        return o.str();
    }
    std::string sha256(const std::vector<unsigned char>& data){
        BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
        auto check=[](NTSTATUS status){if(status<0)throw std::runtime_error("SHA256 byte hashing failed");};
        try{
            check(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0));check(BCryptCreateHash(algorithm,&hash,nullptr,0,nullptr,0,0));
            for(size_t offset=0;offset<data.size();){auto n=ULONG(std::min<size_t>(data.size()-offset,1<<20));check(BCryptHashData(hash,const_cast<PUCHAR>(data.data()+offset),n,0));offset+=n;}
            unsigned char digest[32];check(BCryptFinishHash(hash,digest,32,0));BCryptDestroyHash(hash);hash=nullptr;BCryptCloseAlgorithmProvider(algorithm,0);algorithm=nullptr;
            std::ostringstream output;for(auto c:digest)output<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c);return output.str();
        }catch(...){if(hash)BCryptDestroyHash(hash);if(algorithm)BCryptCloseAlgorithmProvider(algorithm,0);throw;}
    }
    std::string csv(const std::string& s){
        std::string o="\"";
        for(char c:s){
            if(c=='"')o+='"';
            o+=c;
        }
        return o+'"';
    }
    std::vector<std::string> csvrow(const std::string& s){
        std::vector<std::string> o;
        std::string v;
        bool quoted=false;
        for(size_t i=0; i<s.size(); i++){
            char c=s[i];
            if(c=='"'){
                if(quoted&&i+1<s.size()&&s[i+1]=='"'){
                    v+='"';
                    i++;
                }
                else quoted=!quoted;
            }
            else if(c==','&&!quoted){
                o.push_back(v);
                v.clear();
            }
            else if(c!='\r')v+=c;
        }
        o.push_back(v);
        return o;
    }
    static std::string timefmt(FILETIME t,bool korea){
        ULARGE_INTEGER n;
        n.LowPart=t.dwLowDateTime;
        n.HighPart=t.dwHighDateTime;
        if(korea)n.QuadPart+=9ULL*3600*10000000;
        t.dwLowDateTime=n.LowPart;
        t.dwHighDateTime=n.HighPart;
        SYSTEMTIME s;
        FileTimeToSystemTime(&t,&s);
        char b[64];
        sprintf_s(b,"%04d-%02d-%02dT%02d:%02d:%02d.%03d%s",s.wYear,s.wMonth,s.wDay,s.wHour,s.wMinute,s.wSecond,s.wMilliseconds,korea?"+09:00":"Z");
        return b;
    }
    std::string timestamp(const fs::path& p,bool creation,bool korea){
        WIN32_FILE_ATTRIBUTE_DATA a;
        if(!GetFileAttributesExW(p.c_str(),GetFileExInfoStandard,&a))throw std::runtime_error("Cannot stat "+p.u8string());
        return timefmt(creation?a.ftCreationTime:a.ftLastWriteTime,korea);
    }
    std::string now(){
        FILETIME f;
        GetSystemTimeAsFileTime(&f);
        return timefmt(f,true);
    }
    template<class T>static T get(const unsigned char* p){
        T v;
        memcpy(&v,p,sizeof(v));
        return v;
    }
    size_t Npy::count()const{
        size_t n=1;
        for(auto s:shape)n*=s;
        return n;
    }
    double Npy::number(size_t i)const{
        if(i>=count())throw std::runtime_error("NPY index out of bounds");
        std::string d=dtype.substr(1);
        size_t n=std::stoul(d.substr(1));
        const auto* p=data.data()+i*n;
        if(dtype[0]=='>')throw std::runtime_error("Big endian NPY unsupported");
        if(d=="f8")return get<double>(p);
        if(d=="f4")return get<float>(p);
        if(d=="i8")return double(get<int64_t>(p));
        if(d=="i4")return get<int32_t>(p);
        if(d=="u8")return double(get<uint64_t>(p));
        if(d=="u4")return get<uint32_t>(p);
        if(d=="b1"||d=="u1"||d=="?")return *p;
        throw std::runtime_error("Unsupported NPY dtype "+dtype);
    }
    std::string Npy::text(size_t i)const{
        if(i>=count())throw std::runtime_error("NPY text index");
        char type=dtype.at(1);
        size_t n=std::stoul(dtype.substr(2));
        if(type=='S'){
            auto p=(const char*)data.data()+i*n;
            return std::string(p,strnlen(p,n));
        }
        if(type!='U')throw std::runtime_error("Expected unicode NPY");
        std::string s;
        Json::Parser parser{
            s
        };
        for(size_t j=0; j<n; j++){
            auto u=get<uint32_t>(data.data()+4*(i*n+j));
            if(!u)break;
            parser.utf8(s,u);
        }
        return s;
    }
    Npy npy(const std::vector<unsigned char>& b){
        if(b.size()<10||b[0]!=147||memcmp(b.data()+1,"NUMPY",5))throw std::runtime_error("Invalid NPY magic");
        size_t offset=b[6]==1?10:12;
        size_t len=b[6]==1?get<uint16_t>(b.data()+8):get<uint32_t>(b.data()+8);
        if(offset+len>b.size())throw std::runtime_error("Truncated NPY header");
        std::string h((char*)b.data()+offset,len);
        std::smatch m;
        Npy out;
        if(!std::regex_search(h,m,std::regex("'descr':\\s*'([^']+)'")))throw std::runtime_error("NPY dtype missing");
        out.dtype=m[1];
        bool fortran=h.find("True")!=std::string::npos;
        if(!std::regex_search(h,m,std::regex("'shape':\\s*\\(([^)]*)\\)")))throw std::runtime_error("NPY shape missing");
        std::string sh=m[1];
        std::regex re("[0-9]+");
        for(std::sregex_iterator i(sh.begin(),sh.end(),re),e; i!=e; ++i)out.shape.push_back(std::stoull(i->str()));
        out.data.assign(b.begin()+offset+len,b.end());
        // Object arrays contain an opaque pickle payload rather than fixed-size elements.
        // Callers must explicitly opt into a non-executing data parser.
        if(out.dtype.size()>1&&out.dtype[1]=='O')return out;
        size_t width=std::stoul(out.dtype.substr(2));
        if(out.dtype[1]=='U')width*=4;
        if(out.count()*width!=out.data.size())throw std::runtime_error("NPY byte count mismatch");
        if(fortran&&out.shape.size()>1){
            auto original=out.data;
            for(size_t index=0; index<out.count(); index++){
                size_t rest=index,src=0,stride=1;
                std::vector<size_t> coordinate(out.shape.size());
                for(size_t d=out.shape.size(); d-->0; ){
                    coordinate[d]=rest%out.shape[d];
                    rest/=out.shape[d];
                }
                for(size_t d=0; d<out.shape.size(); d++){
                    src+=coordinate[d]*stride;
                    stride*=out.shape[d];
                }
                memcpy(out.data.data()+index*width,original.data()+src*width,width);
            }
        }
        return out;
    }
    std::map<std::string,std::vector<unsigned char>> read_zip(const fs::path& p){
        auto b=bytes(p);
        if(b.size()<22)throw std::runtime_error("Invalid ZIP");
        size_t e=b.size()-22;
        while(e>0&&get<uint32_t>(&b[e])!=0x06054b50)--e;
        if(!e)throw std::runtime_error("ZIP directory missing");
        size_t off=get<uint32_t>(&b[e+16]);
        unsigned entries=get<uint16_t>(&b[e+10]);
        std::map<std::string,std::vector<unsigned char>> out;
        for(unsigned i=0; i<entries; i++){
            if(off+46>b.size()||get<uint32_t>(&b[off])!=0x02014b50)throw std::runtime_error("Invalid ZIP entry");
            unsigned method=get<uint16_t>(&b[off+10]);
            size_t compressed=get<uint32_t>(&b[off+20]),size=get<uint32_t>(&b[off+24]),nl=get<uint16_t>(&b[off+28]),el=get<uint16_t>(&b[off+30]),cl=get<uint16_t>(&b[off+32]),local=get<uint32_t>(&b[off+42]);
            if(off+46+nl+el+cl>b.size())throw std::runtime_error("Truncated ZIP central entry");
            std::string name((char*)&b[off+46],nl);
            off+=46+nl+el+cl;
            if(local+30>b.size())throw std::runtime_error("Truncated ZIP local");
            size_t data=local+30+get<uint16_t>(&b[local+26])+get<uint16_t>(&b[local+28]);
            if(data+compressed>b.size()||size>size_t(1)<<31)throw std::runtime_error("Invalid ZIP length");
            std::vector<unsigned char> raw(size);
            if(method==0){
                if(size!=compressed)throw std::runtime_error("ZIP stored length mismatch");
                memcpy(raw.data(),b.data()+data,size);
            }
            else if(method==8){
                z_stream z{
                };
                z.next_in=b.data()+data;
                z.avail_in=uInt(compressed);
                z.next_out=raw.data();
                z.avail_out=uInt(size);
                if(inflateInit2(&z,-MAX_WBITS)!=Z_OK)throw std::runtime_error("ZIP inflate init");
                int status=inflate(&z,Z_FINISH);
                inflateEnd(&z);
                if(status!=Z_STREAM_END||z.total_out!=size)throw std::runtime_error("ZIP inflate failed");
            }
            else throw std::runtime_error("Unsupported ZIP method");
            if(crc32(0,raw.data(),uInt(raw.size()))!=get<uint32_t>(&b[off-46-nl-el-cl+16]))throw std::runtime_error("ZIP CRC mismatch");
            if(!out.emplace(name,std::move(raw)).second)throw std::runtime_error("Duplicate ZIP entry");
        }
        return out;
    }
    std::map<std::string,Npy> npz(const fs::path& p){
        std::map<std::string,Npy> out;
        for(auto& item:read_zip(p))if(item.first.size()>4&&item.first.substr(item.first.size()-4)==".npy")out[item.first.substr(0,item.first.size()-4)]=npy(item.second);
        return out;
    }
    void write_zip(const fs::path& path,const std::map<std::string,std::string>& entries){
        if(fs::exists(path))throw std::runtime_error("New ZIP destination required");
        if(entries.size()>65535)throw std::runtime_error("ZIP64 is not supported");
        std::ofstream out(path,std::ios::binary);
        if(!out)throw std::runtime_error("Cannot create ZIP");
        auto word=[&](uint64_t value,int bytes){
            for(int i=0; i<bytes; ++i)out.put(char((value>>(8*i))&255));
        };
        struct Entry{
            std::string name;
            uint32_t size,crc,offset;
        };
        std::vector<Entry> directory;
        for(auto& item:entries){
            if(item.first.size()>65535||item.second.size()>UINT32_MAX||uint64_t(out.tellp())>UINT32_MAX)throw std::runtime_error("ZIP32 limit");
            Entry entry{
                item.first,uint32_t(item.second.size()),uint32_t(crc32(0,reinterpret_cast<const Bytef*>(item.second.data()),uInt(item.second.size()))),uint32_t(out.tellp())
            };
            word(0x04034b50,4);
            word(20,2);
            word(0x800,2);
            word(0,2);
            word(0,2);
            word(0x21,2);
            word(entry.crc,4);
            word(entry.size,4);
            word(entry.size,4);
            word(entry.name.size(),2);
            word(0,2);
            out<<entry.name;
            out.write(item.second.data(),item.second.size());
            directory.push_back(entry);
        }
        uint32_t start=uint32_t(out.tellp());
        for(auto& entry:directory){
            word(0x02014b50,4);
            word(20,2);
            word(20,2);
            word(0x800,2);
            word(0,2);
            word(0,2);
            word(0x21,2);
            word(entry.crc,4);
            word(entry.size,4);
            word(entry.size,4);
            word(entry.name.size(),2);
            word(0,2);
            word(0,2);
            word(0,2);
            word(0,2);
            word(0,4);
            word(entry.offset,4);
            out<<entry.name;
        }
        uint32_t end=uint32_t(out.tellp());
        word(0x06054b50,4);
        word(0,2);
        word(0,2);
        word(directory.size(),2);
        word(directory.size(),2);
        word(end-start,4);
        word(start,4);
        word(0,2);
        out.close();
        if(!out)throw std::runtime_error("ZIP write failed");
    }
}
