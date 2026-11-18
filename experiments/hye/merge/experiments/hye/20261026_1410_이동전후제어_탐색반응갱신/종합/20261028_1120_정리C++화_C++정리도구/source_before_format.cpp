// 원본 ZIP -> clean. C++17 / OpenCV3.4 / Windows BCrypt / Windows tar.exe.
// Python, 외부 실험 폴더, 평가 캐시를 참조하거나 실행하지 않습니다.
// 영상/명령값은 가공하지 않고 모든 원본 BMP/TXT를 보존합니다.
#include <windows.h>
#include <bcrypt.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs=std::filesystem;
using Bytes=std::vector<unsigned char>;
using Row=std::map<std::string,std::string>;
static void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
static std::string utf8(const std::wstring& s){
    if(s.empty())return {};
    int n=WideCharToMultiByte(CP_UTF8,0,s.data(),static_cast<int>(s.size()),nullptr,0,nullptr,nullptr);
    require(n>0,"UTF-8 encoding failed");std::string out(n,'\0');
    WideCharToMultiByte(CP_UTF8,0,s.data(),static_cast<int>(s.size()),&out[0],n,nullptr,nullptr);return out;
}
static std::wstring decode(const std::string& s,UINT cp=CP_UTF8){
    if(s.empty())return {};
    DWORD flags=cp==CP_UTF8?MB_ERR_INVALID_CHARS:0;
    int n=MultiByteToWideChar(cp,flags,s.data(),static_cast<int>(s.size()),nullptr,0);
    require(n>0,"Invalid text encoding");std::wstring out(n,L'\0');
    MultiByteToWideChar(cp,flags,s.data(),static_cast<int>(s.size()),&out[0],n);return out;
}
static std::string display(const fs::path& p){return utf8(p.wstring());}
static std::string relativeName(const fs::path& p){return p.generic_u8string();}
static std::wstring lower(std::wstring s){for(auto& c:s)c=static_cast<wchar_t>(towlower(c));return s;}
static fs::path safeRelative(std::wstring s){
    std::replace(s.begin(),s.end(),L'\\',L'/');
    require(!s.empty()&&s[0]!=L'/',"Unsafe relative path");
    std::wstringstream in(s);std::wstring part;fs::path out;
    while(std::getline(in,part,L'/')){
        require(!part.empty()&&part!=L"."&&part!=L"..","Unsafe path component");
        require(part.find_first_of(L"<>:\"|?*")==std::wstring::npos,"Invalid Windows path character");
        require(part.back()!=L'.'&&part.back()!=L' ',"Trailing dot/space in archive path");
        for(wchar_t c:part)require(c>=32,"Control character in archive path");
        auto base=lower(fs::path(part).stem().wstring());
        require(!std::regex_match(utf8(base),std::regex("con|prn|aux|nul|com[1-9]|lpt[1-9]")),"Reserved Windows filename");
        out/=part;
    }
    require(!out.empty()&&!out.has_root_path(),"Absolute archive path rejected");return out;
}
static bool within(const fs::path& p,const fs::path& root){
    auto a=fs::weakly_canonical(p),b=fs::weakly_canonical(root);auto i=a.begin();
    for(auto j=b.begin();j!=b.end();++j,++i)if(i==a.end()||lower(i->wstring())!=lower(j->wstring()))return false;
    return true;
}
static Bytes bytes(const fs::path& p){
    require(fs::is_regular_file(p),"Missing file: "+display(p));
    const auto n=fs::file_size(p);require(n<=512ull*1024*1024,"Unexpectedly large data member: "+display(p));
    std::ifstream in(p,std::ios::binary);Bytes b(static_cast<size_t>(n));
    if(n)in.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(n));
    require(static_cast<bool>(in),"Read failed: "+display(p));return b;
}
static std::string text(const fs::path& p){auto b=bytes(p);return std::string(b.begin(),b.end());}
static void writeText(const fs::path& p,const std::string& s){
    require(!fs::exists(p),"Refusing overwrite: "+display(p));
    std::ofstream out(p,std::ios::binary);out.write(s.data(),static_cast<std::streamsize>(s.size()));require(static_cast<bool>(out),"Write failed: "+display(p));
}
static std::string hex(const unsigned char* p,size_t n){std::ostringstream o;o<<std::hex<<std::setfill('0');for(size_t i=0;i<n;i++)o<<std::setw(2)<<static_cast<unsigned>(p[i]);return o.str();}
class Sha256 {
    BCRYPT_ALG_HANDLE alg_=nullptr;BCRYPT_HASH_HANDLE hash_=nullptr;Bytes object_;
public:
    Sha256(){
        require(BCryptOpenAlgorithmProvider(&alg_,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA256 provider failed");
        DWORD size=0,got=0;require(BCryptGetProperty(alg_,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&size),sizeof(size),&got,0)>=0,"SHA256 size failed");
        object_.resize(size);require(BCryptCreateHash(alg_,&hash_,object_.data(),size,nullptr,0,0)>=0,"SHA256 initialization failed");
    }
    ~Sha256(){if(hash_)BCryptDestroyHash(hash_);if(alg_)BCryptCloseAlgorithmProvider(alg_,0);}
    Sha256(const Sha256&)=delete;Sha256& operator=(const Sha256&)=delete;
    void update(const unsigned char* p,size_t n){require(n<=(std::numeric_limits<ULONG>::max)(),"SHA256 chunk too large");require(BCryptHashData(hash_,const_cast<PUCHAR>(p),static_cast<ULONG>(n),0)>=0,"SHA256 update failed");}
    std::string finish(){std::array<unsigned char,32> out{};require(BCryptFinishHash(hash_,out.data(),static_cast<ULONG>(out.size()),0)>=0,"SHA256 finish failed");return hex(out.data(),out.size());}
};
static std::string sha(const Bytes& b){Sha256 h;if(!b.empty())h.update(b.data(),b.size());return h.finish();}
static std::string shaFile(const fs::path& p){
    std::ifstream in(p,std::ios::binary);require(static_cast<bool>(in),"Cannot hash: "+display(p));
    Bytes chunk(8*1024*1024);Sha256 h;
    while(in){in.read(reinterpret_cast<char*>(chunk.data()),static_cast<std::streamsize>(chunk.size()));auto n=in.gcount();if(n>0)h.update(chunk.data(),static_cast<size_t>(n));}
    require(in.eof(),"Hash read failed: "+display(p));return h.finish();
}
static uint32_t crc32(const Bytes& b){
    static const auto table=[](){std::array<uint32_t,256> t{};for(uint32_t i=0;i<256;i++){uint32_t c=i;for(int k=0;k<8;k++)c=(c&1)?0xedb88320u^(c>>1):(c>>1);t[i]=c;}return t;}();
    uint32_t c=0xffffffffu;for(auto x:b)c=table[(c^x)&255]^(c>>8);return c^0xffffffffu;
}
static std::string crcHex(uint32_t c){std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(8)<<c;return s.str();}
static std::string quoted(const std::string& s){
    std::string out="\"";for(unsigned char c:s){if(c=='"'||c=='\\'){out+='\\';out+=static_cast<char>(c);}else if(c<32){std::ostringstream n;n<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<static_cast<unsigned>(c);out+=n.str();}else out+=static_cast<char>(c);}return out+'"';
}
static std::string jsonString(const std::string& s,const std::string& key){std::smatch m;require(std::regex_search(s,m,std::regex("\""+key+"\"\\s*:\\s*\"([^\"]*)\"")),"Missing JSON field: "+key);return m[1];}
static std::string csvValue(const std::string& s){if(s.find_first_of(",\"\r\n")==std::string::npos)return s;std::string r="\"";for(char c:s){if(c=='"')r+='"';r+=c;}return r+'"';}
static void writeCsv(const fs::path& p,const std::vector<std::string>& columns,const std::vector<Row>& rows){
    std::ostringstream out;out<<"\xef\xbb\xbf";
    auto line=[&](const std::vector<std::string>& v){for(size_t i=0;i<v.size();i++){if(i)out<<',';out<<csvValue(v[i]);}out<<"\r\n";};line(columns);
    for(const auto& r:rows){std::vector<std::string> v;for(const auto& c:columns){auto i=r.find(c);v.push_back(i==r.end()?"":i->second);}line(v);}writeText(p,out.str());
}
static std::vector<Row> readCsv(const fs::path& p){
    std::string s=text(p);if(s.compare(0,3,"\xef\xbb\xbf")==0)s.erase(0,3);
    std::vector<std::vector<std::string>> lines;std::vector<std::string> fields;std::string field;bool inQuote=false;
    for(size_t i=0;i<s.size();i++){
        char c=s[i];if(c=='"'){if(inQuote&&i+1<s.size()&&s[i+1]=='"'){field+='"';i++;}else inQuote=!inQuote;}
        else if(!inQuote&&c==','){fields.push_back(field);field.clear();}
        else if(!inQuote&&(c=='\r'||c=='\n')){if(c=='\r'&&i+1<s.size()&&s[i+1]=='\n')i++;fields.push_back(field);field.clear();lines.push_back(fields);fields.clear();}
        else field+=c;
    }
    require(!inQuote,"Unterminated CSV quotation");if(!field.empty()||!fields.empty()){fields.push_back(field);lines.push_back(fields);}
    require(!lines.empty(),"Empty CSV");std::set<std::string> names(lines[0].begin(),lines[0].end());require(names.size()==lines[0].size(),"Duplicate CSV column");
    std::vector<Row> rows;for(size_t i=1;i<lines.size();i++){require(lines[i].size()==lines[0].size(),"CSV column count mismatch");Row r;for(size_t c=0;c<lines[0].size();c++)r[lines[0][c]]=lines[i][c];rows.push_back(r);}return rows;
}
static uint16_t u16(const Bytes& b,size_t p){require(p+2<=b.size(),"Truncated ZIP value");return static_cast<uint16_t>(b[p]|(b[p+1]<<8));}
static uint32_t u32(const Bytes& b,size_t p){require(p+4<=b.size(),"Truncated ZIP value");return uint32_t(b[p])|(uint32_t(b[p+1])<<8)|(uint32_t(b[p+2])<<16)|(uint32_t(b[p+3])<<24);}
static uint64_t u64(const Bytes& b,size_t p){return u32(b,p)|(uint64_t(u32(b,p+4))<<32);}
static void put16(Bytes& b,size_t p,uint16_t n){b[p]=static_cast<unsigned char>(n);b[p+1]=static_cast<unsigned char>(n>>8);}
static void put32(Bytes& b,size_t p,uint32_t n){for(int i=0;i<4;i++)b[p+i]=static_cast<unsigned char>(n>>(8*i));}
static void put64(Bytes& b,size_t p,uint64_t n){for(int i=0;i<8;i++)b[p+i]=static_cast<unsigned char>(n>>(8*i));}
static Bytes readAt(std::istream& f,uint64_t p,size_t n){f.clear();f.seekg(static_cast<std::streamoff>(p));Bytes b(n);if(n)f.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(n));require(static_cast<bool>(f),"Truncated ZIP record");return b;}
static void writeAt(std::fstream& f,uint64_t p,const Bytes& b){f.clear();f.seekp(static_cast<std::streamoff>(p));f.write(reinterpret_cast<const char*>(b.data()),static_cast<std::streamsize>(b.size()));require(static_cast<bool>(f),"ZIP normalization write failed");}
struct EndRecord{uint64_t at;Bytes b;};
static EndRecord zipEnd(std::istream& f,uint64_t size){
    require(size>=22,"Too small for ZIP");const auto n=static_cast<size_t>(std::min<uint64_t>(size,70000));auto tail=readAt(f,size-n,n);
    for(size_t k=n-22;;k--){if(u32(tail,k)==0x06054b50&&k+22+u16(tail,k+20)==n)return {size-n+k,Bytes(tail.begin()+k,tail.begin()+k+22)};if(k==0)break;}
    throw std::runtime_error("ZIP end record not found");
}
struct Member{fs::path relative;std::string name;uint64_t size;uint32_t crc;};
static std::vector<Member> zipMembers(const fs::path& zip){
    std::ifstream in(zip,std::ios::binary);auto end=zipEnd(in,fs::file_size(zip));
    require(u16(end.b,4)==0&&u16(end.b,6)==0,"Normalize split ZIP before reading members");
    auto total=u16(end.b,10);uint64_t at=u32(end.b,16);require(total!=0xffff&&at!=0xffffffffu,"Large ZIP64 directory is unsupported");std::vector<Member> result;
    for(unsigned i=0;i<total;i++){
        auto h=readAt(in,at,46);require(u32(h,0)==0x02014b50,"Invalid ZIP central directory");
        auto flags=u16(h,8),nl=u16(h,28),extra=u16(h,30),comment=u16(h,32);auto raw=readAt(in,at+46,nl);std::string name(raw.begin(),raw.end());
        require(!(flags&1),"Encrypted ZIP is unsupported");require((u32(h,38)>>16&0170000)!=0120000,"ZIP symbolic links are rejected");
        std::wstring wide;if(flags&0x800)wide=decode(name);else{try{wide=decode(name);}catch(const std::exception&){wide=decode(name,949);}}
        bool directory=!wide.empty()&&(wide.back()==L'/'||wide.back()==L'\\');if(directory)wide.pop_back();auto relative=safeRelative(wide);
        if(!directory){require(u32(h,24)!=0xffffffffu,"ZIP64 data member is unsupported");result.push_back({relative,utf8(wide),u32(h,24),u32(h,16)});}
        at+=46ull+nl+extra+comment;
    }
    require(!result.empty(),"ZIP has no files");std::set<std::wstring> seen;for(const auto& m:result)require(seen.insert(lower(m.relative.wstring())).second,"Duplicate ZIP destination");return result;
}
static void normalizeSplit(const std::vector<fs::path>& parts,const fs::path& output){
    require(!fs::exists(output),"Normalization output exists");std::vector<uint64_t> offsets;uint64_t size=0;
    {std::ofstream out(output,std::ios::binary);Bytes buffer(8*1024*1024);for(const auto& p:parts){offsets.push_back(size);std::ifstream in(p,std::ios::binary);require(bool(in),"Split part missing");while(in){in.read(reinterpret_cast<char*>(buffer.data()),static_cast<std::streamsize>(buffer.size()));auto n=in.gcount();if(n>0)out.write(reinterpret_cast<char*>(buffer.data()),n);}require(in.eof()&&bool(out),"Failed to join split ZIP");size+=fs::file_size(p);}}
    std::fstream f(output,std::ios::binary|std::ios::in|std::ios::out);auto end=zipEnd(f,size);
    auto cdDisk=u16(end.b,6),total=u16(end.b,10);require(cdDisk<offsets.size()&&total!=0xffff,"Unsupported split directory");
    uint64_t cd=offsets[cdDisk]+u32(end.b,16),cursor=cd;require(cd<=0xffffffffu,"Large ZIP64 offsets unsupported");
    for(unsigned i=0;i<total;i++){
        auto h=readAt(f,cursor,46);require(u32(h,0)==0x02014b50,"Invalid split ZIP directory");auto disk=u16(h,34);auto local=u32(h,42);
        require(disk<offsets.size()&&local!=0xffffffffu&&offsets[disk]+local<=0xffffffffu,"Unsupported split ZIP location");
        put16(h,34,0);put32(h,42,static_cast<uint32_t>(offsets[disk]+local));writeAt(f,cursor,h);cursor+=46ull+u16(h,28)+u16(h,30)+u16(h,32);
    }
    put16(end.b,4,0);put16(end.b,6,0);put16(end.b,8,total);put16(end.b,10,total);put32(end.b,16,static_cast<uint32_t>(cd));writeAt(f,end.at,end.b);
    if(end.at>=20){auto loc=readAt(f,end.at-20,20);if(u32(loc,0)==0x07064b50){auto disk=u32(loc,4);require(disk<offsets.size(),"Invalid ZIP64 volume");auto absolute=offsets[disk]+u64(loc,8);put32(loc,4,0);put64(loc,8,absolute);put32(loc,16,1);writeAt(f,end.at-20,loc);auto z=readAt(f,absolute,56);require(u32(z,0)==0x06064b50,"ZIP64 end record missing");put32(z,16,0);put32(z,20,0);put64(z,48,cd);writeAt(f,absolute,z);}}
}
static int runTar(const fs::path& archive,const fs::path& dest){
    wchar_t windows[MAX_PATH];require(GetWindowsDirectoryW(windows,MAX_PATH)>0,"Cannot find Windows directory");fs::path tar=fs::path(windows)/L"System32/tar.exe";require(fs::is_regular_file(tar),"Windows tar.exe not found");
    std::wstring command=L"\""+tar.wstring()+L"\" -xf \""+archive.wstring()+L"\" -C \""+dest.wstring()+L"\"";
    std::vector<wchar_t> writable(command.begin(),command.end());writable.push_back(0);STARTUPINFOW si{};si.cb=sizeof(si);PROCESS_INFORMATION pi{};
    require(CreateProcessW(tar.c_str(),writable.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,nullptr,nullptr,&si,&pi)!=0,"Cannot execute Windows tar.exe");
    WaitForSingleObject(pi.hProcess,INFINITE);DWORD code=1;GetExitCodeProcess(pi.hProcess,&code);CloseHandle(pi.hThread);CloseHandle(pi.hProcess);return static_cast<int>(code);
}
static void extractChecked(const fs::path& archive,const fs::path& dest,const std::vector<Member>& members){
    fs::create_directories(dest);require(fs::is_empty(dest),"Extraction directory must be empty");
    for(const auto& m:members)require(within(dest/m.relative,dest),"ZIP extraction escapes destination");
    require(runTar(archive,dest)==0,"ZIP decompression failed");std::set<std::wstring> expected;
    for(const auto& m:members){auto p=dest/m.relative;require(fs::is_regular_file(p)&&within(p,dest)&&!fs::is_symlink(p)&&fs::file_size(p)==m.size,"Missing/invalid extracted member: "+display(p));expected.insert(lower(m.relative.wstring()));}
    for(const auto& e:fs::recursive_directory_iterator(dest)){require(!e.is_symlink()&&within(e.path(),dest),"Extracted link/path rejected");if(e.is_regular_file())require(expected.count(lower(fs::relative(e.path(),dest).wstring()))==1,"Unexpected extracted file");}
}
struct Command{std::string y,t,kind,repeat;};
static Command commandFor(const fs::path& p){
    auto stem=p.stem().u8string();auto lowered=lower(p.stem().wstring());
    if(lowered.rfind(L"center",0)==0){std::string repeat;for(char c:stem)if(c>='0'&&c<='9')repeat+=c;return {"0","0","center",repeat.empty()?"1":repeat};}
    std::smatch y,t,r;bool hy=std::regex_search(stem,y,std::regex("Y\\s*([+-][0-9]+(?:\\.[0-9]+)?)",std::regex::icase));bool ht=std::regex_search(stem,t,std::regex("T\\s*([+-][0-9]+(?:\\.[0-9]+)?)",std::regex::icase));
    require(hy||ht,"Cannot parse command filename: "+stem);std::regex_search(stem,r,std::regex("\\(([0-9]+)\\)"));return {hy?y[1].str():"0",ht?t[1].str():"0",hy&&ht?"mixed":hy?"Y":"T",r.empty()?"1":r[1].str()};
}
static Row sidecarFor(const fs::path& image,const fs::path& base){
    auto p=image;p.replace_extension(L".txt");auto second=fs::path(p.wstring()+L".txt");bool one=fs::is_regular_file(p),two=fs::is_regular_file(second);
    Row r{{"sidecar_path",""},{"sidecar_Y_Actual_raw",""},{"sidecar_T_Actual_raw",""},{"sidecar_encoding",""}};
    require(!(one&&two),"Ambiguous sidecar files");if(!one&&!two)return r;if(two)p=second;
    auto s=text(p);std::string encoding="UTF-8/ASCII";if(s.compare(0,3,"\xef\xbb\xbf")==0)s.erase(0,3);
    try{decode(s);}catch(const std::exception&){s=utf8(decode(s,949));encoding="CP949";}
    std::istringstream in(s);std::string a,b,y,t,extra;in>>a>>b>>y>>t;require(a=="Y_Actual"&&b=="T_Actual"&&!y.empty()&&!t.empty()&&!(in>>extra),"Invalid sidecar: "+display(p));
    size_t used=0;double yv=std::stod(y,&used);require(used==y.size()&&std::isfinite(yv),"Invalid Y_Actual");double tv=std::stod(t,&used);require(used==t.size()&&std::isfinite(tv),"Invalid T_Actual");
    r["sidecar_path"]=relativeName(fs::relative(p,base));r["sidecar_Y_Actual_raw"]=y;r["sidecar_T_Actual_raw"]=t;r["sidecar_encoding"]=encoding;return r;
}
static Row imageRow(const fs::path& image,const fs::path& base,const Bytes& b,const std::string& hash,const std::string& member){
    auto im=cv::imdecode(b,cv::IMREAD_UNCHANGED);require(!im.empty()&&im.cols==3840&&im.rows==2160,"Image decode/dimensions failed: "+display(image));
    auto c=commandFor(image);auto r=sidecarFor(image,base);
    r.insert({{"relative_path",relativeName(fs::relative(image,base))},{"source_member",member},{"bytes",std::to_string(b.size())},{"sha256",hash},{"width",std::to_string(im.cols)},{"height",std::to_string(im.rows)},{"channels",std::to_string(im.channels())},{"filename_Y_command",c.y},{"filename_T_command",c.t},{"capture_repeat",c.repeat},{"condition",c.kind},{"matches_source_archive_bytes","True"}});return r;
}
struct Config{fs::path repo,output;std::string date="all";bool verifyOnly=false,selfTest=false;};
static const std::map<std::string,size_t> expectedImages{{"20261014",281},{"20261020",60}};
static void verifyExisting(const Config& cfg,const std::string& date){
    auto target=cfg.output/(L"clean_"+decode(date));auto raw=cfg.repo/L"data/raw"/(L"zip_"+decode(date)+L".zip");auto meta=target/L"_metadata";
    require(fs::is_regular_file(meta/L"provenance.json")&&fs::is_regular_file(meta/L"file_inventory.csv")&&fs::is_regular_file(meta/L"manifest.csv"),"Existing clean lacks metadata; refusing overwrite");
    auto provenance=text(meta/L"provenance.json");require(jsonString(provenance,"source_archive_sha256")==shaFile(raw),"Current raw ZIP differs from existing clean");
    auto inventory=readCsv(meta/L"file_inventory.csv"),images=readCsv(meta/L"manifest.csv");require(images.size()==expectedImages.at(date),"Unexpected image count");
    std::map<std::string,Row> manifest;for(const auto& r:images)require(manifest.emplace(r.at("relative_path"),r).second,"Duplicate manifest path");
    std::set<std::wstring> expected;size_t n=0,bmp=0,txt=0;
    for(const auto& r:inventory){
        auto relative=safeRelative(decode(r.at("relative_path")));auto p=target/relative;
        require(within(p,target)&&!fs::is_symlink(p)&&expected.insert(lower(relative.wstring())).second,"Invalid/duplicate data path");
        auto b=bytes(p);auto hash=sha(b);require(b.size()==std::stoull(r.at("bytes"))&&hash==r.at("sha256"),"Changed data file: "+display(p));
        auto ci=r.find("zip_crc32");if(ci!=r.end())require(crcHex(crc32(b))==ci->second,"ZIP CRC mismatch: "+display(p));
        if(lower(p.extension().wstring())==L".bmp"){
            auto i=manifest.find(r.at("relative_path"));require(i!=manifest.end(),"BMP missing from manifest");auto fresh=imageRow(p,target,b,hash,r.at("source_member"));
            for(const auto& key:{"sha256","bytes","width","height","channels","filename_Y_command","filename_T_command","capture_repeat","condition","sidecar_path","sidecar_Y_Actual_raw","sidecar_T_Actual_raw","sidecar_encoding"})require(fresh.at(key)==i->second.at(key),"Image/label metadata mismatch: "+display(p));bmp++;
        }else if(lower(p.extension().wstring())==L".txt")txt++;
        if(++n%60==0)std::cout<<date<<" 기존 파일 검증 "<<n<<"/"<<inventory.size()<<std::endl;
    }
    require(bmp==images.size()&&txt==(date=="20261020"?60u:0u),"BMP/TXT count mismatch");
    std::set<std::wstring> actual;for(const auto& e:fs::recursive_directory_iterator(target)){
        require(!e.is_symlink()&&within(e.path(),target),"Existing clean contains a link");if(!e.is_regular_file())continue;auto rel=fs::relative(e.path(),target);
        if(*rel.begin()==L"_metadata"||rel==L"README.txt")continue;actual.insert(lower(rel.wstring()));
    }
    require(actual==expected,"Existing file inventory differs from directory");std::cout<<date<<" 검증 완료: BMP "<<bmp<<", TXT "<<txt<<" (기존 파일 변경 없음)"<<std::endl;
}
struct Scratch{
    fs::path path,root;
    explicit Scratch(const fs::path& base):root(fs::weakly_canonical(base)){
        for(unsigned n=0;n<100;n++){path=root/(L".prepare_cpp_"+std::to_wstring(GetCurrentProcessId())+L"_"+std::to_wstring(GetTickCount64())+L"_"+std::to_wstring(n));if(fs::create_directory(path))return;}throw std::runtime_error("Cannot create private scratch directory");
    }
    ~Scratch(){try{if(!path.empty()&&within(path,root)&&path.parent_path()==root&&path.filename().wstring().rfind(L".prepare_cpp_",0)==0){std::error_code ec;fs::remove_all(path,ec);}}catch(...) {}}
};
static void createClean(const Config& cfg,const std::string& date){
    auto raw=cfg.repo/L"data/raw"/(L"zip_"+decode(date)+L".zip");auto target=cfg.output/(L"clean_"+decode(date));auto before=shaFile(raw);
    fs::create_directories(cfg.output);Scratch scratch(cfg.output);auto outer=scratch.path/L"outer",stage=scratch.path/L"dataset",inner=scratch.path/L"inner";fs::create_directories(stage);
    std::cout<<date<<" 원본 ZIP/분할 압축 해제"<<std::endl;auto original=zipMembers(raw);extractChecked(raw,outer,original);
    std::vector<std::pair<int,fs::path>> segments;fs::path last;std::vector<Row> partRows;
    for(const auto& m:original){auto p=outer/m.relative;auto b=bytes(p);require(crc32(b)==m.crc,"Raw split member CRC mismatch");
        partRows.push_back({{"member",m.name},{"bytes",std::to_string(m.size)},{"crc32",crcHex(m.crc)},{"sha256",sha(b)}});
        if(lower(p.extension().wstring())==L".zip"){require(last.empty(),"More than one nested ZIP");last=p;}else{std::smatch match;auto extension=p.extension().u8string();require(std::regex_match(extension,match,std::regex("\\.z([0-9]+)",std::regex::icase)),"Unexpected outer ZIP member");segments.push_back({std::stoi(match[1]),p});}}
    require(!last.empty(),"Nested ZIP final part missing");std::sort(segments.begin(),segments.end());std::vector<fs::path> parts;
    for(size_t i=0;i<segments.size();i++){require(segments[i].first==static_cast<int>(i+1),"Split part sequence has gaps");require(segments[i].second.parent_path()==last.parent_path()&&segments[i].second.stem()==last.stem(),"Split part stem mismatch");parts.push_back(segments[i].second);}parts.push_back(last);
    auto joined=scratch.path/L"joined.zip";if(parts.size()>1)normalizeSplit(parts,joined);else fs::copy_file(last,joined);
    const auto joinedHash=shaFile(joined);auto members=zipMembers(joined);extractChecked(joined,inner,members);
    std::wstring prefix=members.front().relative.begin()->wstring();bool strip=true;
    for(const auto& m:members)if(m.relative.begin()->wstring()!=prefix||std::distance(m.relative.begin(),m.relative.end())<2)strip=false;
    std::vector<Row> files,images;std::map<std::string,size_t> conditions,hashes;size_t txt=0;
    for(const auto& m:members){
        fs::path relative=m.relative;if(strip){relative.clear();auto it=m.relative.begin();++it;for(;it!=m.relative.end();++it)relative/=*it;}
        require(*relative.begin()!=L"_metadata"&&relative!=L"README.txt","Archive collides with generated metadata");auto source=inner/m.relative,dest=stage/relative;
        require(within(source,inner)&&within(dest,stage),"Source/destination escapes private scratch");auto b=bytes(source);require(b.size()==m.size&&crc32(b)==m.crc,"Inner ZIP CRC/size mismatch");auto hash=sha(b);
        fs::create_directories(dest.parent_path());fs::rename(source,dest);require(shaFile(dest)==hash,"Moved member bytes changed");
        files.push_back({{"relative_path",relativeName(relative)},{"source_member",m.name},{"bytes",std::to_string(m.size)},{"sha256",hash},{"zip_crc32",crcHex(m.crc)}});
        if(lower(dest.extension().wstring())==L".txt")txt++;
        if(files.size()%40==0)std::cout<<date<<" 원본 파일 검증 "<<files.size()<<"/"<<members.size()<<std::endl;
    }
    // All sidecars must be in place before image manifest generation.
    for(const auto& r:files){auto p=stage/safeRelative(decode(r.at("relative_path")));if(lower(p.extension().wstring())!=L".bmp")continue;auto b=bytes(p);images.push_back(imageRow(p,stage,b,r.at("sha256"),r.at("source_member")));conditions[images.back().at("condition")]++;hashes[r.at("sha256")]++;}
    require(images.size()==expectedImages.at(date)&&txt==(date=="20261020"?60u:0u),"Extracted image/TXT count mismatch");require(shaFile(raw)==before,"Raw ZIP changed during preparation");
    auto meta=stage/L"_metadata";fs::create_directories(meta);
    writeCsv(meta/L"file_inventory.csv",{"relative_path","source_member","bytes","sha256","zip_crc32"},files);
    writeCsv(meta/L"manifest.csv",{"relative_path","source_member","bytes","sha256","width","height","channels","filename_Y_command","filename_T_command","capture_repeat","condition","sidecar_path","sidecar_Y_Actual_raw","sidecar_T_Actual_raw","sidecar_encoding","matches_source_archive_bytes"},images);
    size_t duplicates=0;for(const auto& h:hashes)if(h.second>1)duplicates++;
    std::ostringstream j;j<<"{\n  \"dataset\": "<<quoted(date)<<",\n  \"status\": \"verified\",\n  \"preparation_tool\": \"tool/01_prepare_clean.cpp\",\n  \"source_archive\": "<<quoted("data/raw/zip_"+date+".zip")<<",\n  \"source_archive_path_basis\": \"repository_root\",\n  \"source_archive_sha256\": "<<quoted(before)<<",\n  \"source_archive_bytes\": "<<fs::file_size(raw)<<",\n  \"source_archive_unchanged\": true,\n  \"normalized_joined_sha256\": "<<quoted(joinedHash)<<",\n  \"removed_common_wrapper\": "<<(strip?quoted(utf8(prefix)):"null")<<",\n  \"images\": "<<images.size()<<",\n  \"sidecar_txt\": "<<txt<<",\n  \"all_source_files\": "<<files.size()<<",\n  \"conditions\": {";
    bool first=true;for(const auto& c:conditions){if(!first)j<<',';first=false;j<<quoted(c.first)<<':'<<c.second;}j<<"},\n  \"duplicate_image_hash_groups\": "<<duplicates<<",\n  \"duplicate_images_retained\": true,\n  \"all_image_and_sidecar_bytes_match_source\": true,\n  \"all_images_decode\": true,\n  \"image_size\": [3840,2160],\n  \"zip_member_crc32_verified\": true,\n  \"nested_parts\": [";
    for(size_t i=0;i<partRows.size();i++){if(i)j<<',';const auto& r=partRows[i];j<<"\n    {\"member\":"<<quoted(r.at("member"))<<",\"bytes\":"<<r.at("bytes")<<",\"crc32\":"<<quoted(r.at("crc32"))<<",\"sha256\":"<<quoted(r.at("sha256"))<<'}';}
    j<<"\n  ],\n  \"transformation\": \"Extract exact original BMP/TXT bytes. Remove one common wrapper. No pixel processing, filtering, deduplication, label conversion or dataset splitting.\",\n  \"labels\": \"Filename commands and sidecar Y_Actual/T_Actual are separate raw fields. Units/origin are not inferred. Never use labels as image inference inputs.\",\n  \"evaluation_status\": \"Previously exposed development data, not new independent test data.\"\n}\n";writeText(meta/L"provenance.json",j.str());
    std::ostringstream readme;readme<<"clean_"<<date<<"\n\nC++ 원본 정리 도구: tool/01_prepare_clean.cpp\n입력: data/raw/zip_"<<date<<".zip\nBMP "<<images.size()<<"장 / 동봉 TXT "<<txt<<"개\n\nBMP/TXT 바이트와 촬영별 폴더명을 그대로 보존했습니다.\n공통 최상위 포장 폴더 한 개만 제거했습니다. 사진 가공/제외/중복 제거는 없습니다.\n\n_metadata/manifest.csv: 사진별 해시/크기/파일명 명령/동봉 TXT 원시값\n_metadata/file_inventory.csv: 원본 ZIP 멤버와 전체 파일의 해시/CRC\n_metadata/provenance.json: 원본 ZIP/분할 조각/검증 내역\n\n파일명의 명령과 TXT의 Y_Actual/T_Actual은 서로 다른 필드입니다. 좌표계/원점/단위는 변환하지 않습니다.\n파일명/TXT/정답 명령은 영상 추론 입력이 아닙니다.\n기준 영상은 clean_20261014/center.bmp입니다. clean_20261020/기준 위치/center1~3.bmp로 자동 교체하지 않습니다.\n이 자료는 이미 개발에서 사용한 자료이며 독립 신규 시험 자료가 아닙니다.\n";writeText(stage/L"README.txt",readme.str());
    // Never remove nonempty user data. Both resolved paths are checked first.
    require(within(stage,scratch.path)&&within(target,cfg.output)&&target.filename()==L"clean_"+decode(date),"Invalid publication path");
    if(fs::exists(target)){require(fs::is_directory(target)&&fs::is_empty(target),"Destination exists; refusing overwrite");fs::remove(target);}fs::rename(stage,target);
    std::cout<<date<<" 생성 완료: "<<display(target)<<" (BMP "<<images.size()<<", TXT "<<txt<<")"<<std::endl;
}
static fs::path executable(){std::vector<wchar_t> name(32768);DWORD n=GetModuleFileNameW(nullptr,name.data(),static_cast<DWORD>(name.size()));require(n>0&&n<name.size(),"Executable path unavailable");return fs::path(std::wstring(name.data(),n));}
static bool repoAt(const fs::path& p){return fs::is_directory(p/L"data/raw")&&fs::is_regular_file(p/L"tool/01_prepare_clean.cpp");}
static fs::path findRepo(){
    for(auto start:{executable().parent_path(),fs::current_path()})for(auto p=start;!p.empty();){if(repoAt(p))return fs::canonical(p);auto parent=p.parent_path();if(parent==p)break;p=parent;}
#ifdef PREPARE_CLEAN_SOURCE_DIR
    auto source=fs::path(PREPARE_CLEAN_SOURCE_DIR).parent_path();if(repoAt(source))return fs::canonical(source);
#endif
    throw std::runtime_error("Repository not found. Pass --repo-root <repository>.");
}
static void selfTest(){
    Bytes abc{'a','b','c'},crc{'1','2','3','4','5','6','7','8','9'};require(sha(abc)=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256 test vector failed");require(crcHex(crc32(crc))=="cbf43926","CRC32 test vector failed");
    for(const auto& s:{L"../escape.bmp",L"C:/escape.bmp",L"/escape.bmp",L"dir/../../escape.bmp",L"CON.bmp"}){bool rejected=false;try{safeRelative(s);}catch(const std::exception&){rejected=true;}require(rejected,"Unsafe path accepted");}
    auto c=commandFor(L"Y+0.05 T+1.00 (3).bmp");require(c.y=="+0.05"&&c.t=="+1.00"&&c.repeat=="3"&&c.kind=="mixed","Command parser failed");require(commandFor(L"Y-0.05.bmp").t=="0","Single-axis parser failed");require(commandFor(L"center2.bmp").repeat=="2","Center parser failed");require(csvValue("a,\"b")=="\"a,\"\"b\"","CSV quoting failed");std::cout<<"C++ SHA256/CRC32/경로/명령값/CSV 자체 검사 통과"<<std::endl;
}
int wmain(int argc,wchar_t** argv){
    SetConsoleOutputCP(CP_UTF8);
    try{
        Config cfg;
        for(int i=1;i<argc;i++){
            std::wstring a=argv[i];
            if(a==L"--help"||a==L"-h"){std::cout<<"01_prepare_clean.exe [--date all|20261014|20261020] [--verify-only]\n                     [--repo-root PATH] [--output-root PATH] [--self-test]\n\n기본 입력: 저장소/data/raw/zip_날짜.zip\n기본 출력: 저장소/data/clean/clean_날짜\n기존 clean은 검증만 수행합니다. 원본/기존 데이터를 덮어쓰지 않습니다.\nPython 없이 C++/OpenCV/Windows tar.exe/BCrypt로 실행합니다.\n";return 0;}
            if(a==L"--verify-only")cfg.verifyOnly=true;
            else if(a==L"--self-test")cfg.selfTest=true;
            else if(a==L"--date"||a==L"--repo-root"||a==L"--output-root"){
                require(i+1<argc,"Option value missing");std::wstring value=argv[++i];if(a==L"--date")cfg.date=utf8(value);else if(a==L"--repo-root")cfg.repo=value;else cfg.output=value;
            }else throw std::runtime_error("Unknown argument: "+utf8(a));
        }
        if(cfg.selfTest){selfTest();return 0;}
        require(cfg.date=="all"||expectedImages.count(cfg.date),"Unsupported date");if(cfg.repo.empty())cfg.repo=findRepo();cfg.repo=fs::weakly_canonical(cfg.repo);require(repoAt(cfg.repo),"Invalid repository root");
        if(cfg.output.empty())cfg.output=cfg.repo/L"data/clean";cfg.output=fs::weakly_canonical(cfg.output);
        for(const auto& entry:expectedImages){if(cfg.date!="all"&&cfg.date!=entry.first)continue;auto date=entry.first;auto raw=cfg.repo/L"data/raw"/(L"zip_"+decode(date)+L".zip"),target=cfg.output/(L"clean_"+decode(date));require(fs::is_regular_file(raw),"Raw ZIP missing: "+display(raw));
            if(fs::exists(target)){require(fs::is_directory(target),"Clean destination is not a directory");if(!fs::is_empty(target)){verifyExisting(cfg,date);continue;}}
            require(!cfg.verifyOnly,"No clean data to verify: "+display(target));createClean(cfg,date);
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<"정제 실패: "<<e.what()<<std::endl;return 1;}
}
