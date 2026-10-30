#pragma once
#include "io.hpp"
#include <regex>
#include <sstream>
#include <cmath>
namespace semes::ooxml {
    inline std::string escape(const std::string& value){
        std::string out;
        for(char c:value){
            if(c=='&')out+="&amp;";
            else if(c=='<')out+="&lt;";
            else if(c=='>')out+="&gt;";
            else if(c=='\"')out+="&quot;";
            else out+=c;
        }
        return out;
    }
    inline std::string unescape(const std::string& value){
        std::string out;
        Json::Parser parser{
            value
        };
        for(size_t i=0; i<value.size(); ++i){
            if(value[i]!='&'){
                out+=value[i];
                continue;
            }
            size_t end=value.find(';',i);
            if(end==std::string::npos)throw std::runtime_error("Invalid XML entity");
            std::string code=value.substr(i+1,end-i-1);
            if(code=="amp")out+='&';
            else if(code=="lt")out+='<';
            else if(code=="gt")out+='>';
            else if(code=="quot")out+='\"';
            else if(code=="apos")out+='\'';
            else if(code.size()>1&&code[0]=='#'){
                unsigned point=std::stoul(code.substr(code[1]=='x'?2:1),nullptr,code[1]=='x'?16:10);
                parser.utf8(out,point);
            }
            else throw std::runtime_error("Unsupported XML entity");
            i=end;
        }
        return out;
    }
    inline std::string attribute(const std::string& attributes,const std::string& key){
        std::smatch m;
        if(std::regex_search(attributes,m,std::regex("(?:^|\\s)"+key+"=\"([^\"]*)\"")))return unescape(m[1].str());
        return "";
    }
    inline std::string texts(const std::string& xml){
        std::string result;
        std::regex re("<t(?:\\s[^>]*)?>([\\s\\S]*?)</t>");
        for(std::sregex_iterator i(xml.begin(),xml.end(),re),end; i!=end; ++i)result+=unescape((*i)[1]);
        return result;
    }
    inline std::string column(int zero_based){
        std::string s;
        for(int i=zero_based+1; i; i=(i-1)/26)s=char('A'+(i-1)%26)+s;
        return s;
    }
    struct Cell{
        std::string value,formula,type;
    };
    struct Reader {
        std::map<std::string,std::vector<unsigned char>> entries;
        std::vector<std::string> shared;
        explicit Reader(const fs::path& path):entries(read_zip(path)){
            if(entries.count("xl/sharedStrings.xml")){
                auto xml=entry("xl/sharedStrings.xml");
                std::regex re("<si(?:\\s[^>]*)?>([\\s\\S]*?)</si>");
                for(std::sregex_iterator i(xml.begin(),xml.end(),re),end; i!=end; ++i)shared.push_back(texts((*i)[1]));
            }
        }
        std::string entry(const std::string& name)const{
            auto& data=entries.at(name);
            return std::string(data.begin(),data.end());
        }
        std::map<std::string,Cell> sheet(int one_based)const{
            auto xml=entry("xl/worksheets/sheet"+std::to_string(one_based)+".xml");
            std::map<std::string,Cell> cells;
            std::regex re("<c\\s([^>]*?)(?:/>|>([\\s\\S]*?)</c>)");
            for(std::sregex_iterator i(xml.begin(),xml.end(),re),end; i!=end; ++i){
                std::string attrs=(*i)[1],body=(*i)[2];
                Cell cell;
                cell.type=attribute(attrs,"t");
                std::smatch match;
                if(std::regex_search(body,match,std::regex("<f(?:\\s[^>]*)?>([\\s\\S]*?)</f>")))cell.formula=unescape(match[1]);
                if(cell.type=="inlineStr")cell.value=texts(body);
                else if(std::regex_search(body,match,std::regex("<v>([\\s\\S]*?)</v>"))){
                    cell.value=unescape(match[1]);
                    if(cell.type=="s")cell.value=shared.at(std::stoul(cell.value));
                }
                cells.emplace(attribute(attrs,"r"),cell);
            }
            return cells;
        }
    };
    inline std::string numeric(double value){
        if(!std::isfinite(value))throw std::runtime_error("Nonfinite workbook number");
        std::ostringstream out;
        out<<std::setprecision(17)<<value;
        return out.str();
    }
    struct Sheet {
        std::string name;
        std::map<int,std::map<int,std::string>> rows;
        std::vector<double> widths;
        std::vector<std::string> merges;
        std::map<int,double> row_heights;
        int frozen_rows=0,frozen_columns=0,header_row=1;
        void text(int row,int col,const std::string& value,int style=0){
            rows[row][col]="<c r=\""+column(col)+std::to_string(row)+"\" s=\""+std::to_string(style)+"\" t=\"inlineStr\"><is><t xml:space=\"preserve\">"+escape(value)+"</t></is></c>";
        }
        void number(int row,int col,double value,int style=0,const std::string& formula=""){
            rows[row][col]="<c r=\""+column(col)+std::to_string(row)+"\" s=\""+std::to_string(style)+"\">"+(formula.empty()?"":"<f>"+escape(formula)+"</f>")+"<v>"+numeric(value)+"</v></c>";
        }
        std::string xml()const{
            std::ostringstream out;
            out<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?><worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetViews><sheetView workbookViewId=\"0\" showGridLines=\"0\">";
            if(frozen_rows||frozen_columns)out<<"<pane xSplit=\""<<frozen_columns<<"\" ySplit=\""<<frozen_rows<<"\" topLeftCell=\""<<column(frozen_columns)<<frozen_rows+1<<"\" activePane=\"bottomRight\" state=\"frozen\"/>";
            out<<"</sheetView></sheetViews><sheetFormatPr defaultRowHeight=\"20\"/><cols>";
            for(size_t i=0; i<widths.size(); ++i)out<<"<col min=\""<<i+1<<"\" max=\""<<i+1<<"\" width=\""<<widths[i]<<"\" customWidth=\"1\"/>";
            out<<"</cols><sheetData>";
            for(auto& row:rows){
                auto height=row_heights.find(row.first);
                out<<"<row r=\""<<row.first<<"\" ht=\""<<(height!=row_heights.end()?height->second:(row.first==header_row?60:20))<<"\" customHeight=\"1\">";
                for(auto& cell:row.second)out<<cell.second;
                out<<"</row>";
            }
            out<<"</sheetData>";
            if(!merges.empty()){
                out<<"<mergeCells count=\""<<merges.size()<<"\">";
                for(auto& range:merges)out<<"<mergeCell ref=\""<<escape(range)<<"\"/>";
                out<<"</mergeCells>";
            }
            out<<"</worksheet>";
            return out.str();
        }
    };
    inline std::string styles(){
        std::ostringstream out;
        out<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?><styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><fonts count=\"4\"><font><sz val=\"10\"/><name val=\"Arial\"/></font><font><b/><sz val=\"10\"/><name val=\"Arial\"/></font><font><b/><sz val=\"10\"/><color rgb=\"FFFFFFFF\"/><name val=\"Arial\"/></font><font><b/><sz val=\"14\"/><name val=\"Arial\"/></font></fonts><fills count=\"4\"><fill><patternFill patternType=\"none\"/></fill><fill><patternFill patternType=\"gray125\"/></fill><fill><patternFill patternType=\"solid\"><fgColor rgb=\"FF1F4E78\"/></patternFill></fill><fill><patternFill patternType=\"solid\"><fgColor rgb=\"FFFCE4D6\"/></patternFill></fill></fills><borders count=\"1\"><border/></borders><cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs><cellXfs count=\"11\">";
        // 0 normal, 1 bold, 2 header, 3/4 fixed3, 5/6 fixed1, 7/8 percent,
        // 9 orange unavailable, 10 title. Numeric formats are built-in Excel IDs.
        const int fonts[]={
            0,1,2,0,1,0,1,0,1,1,3
        },fills[]={
            0,0,2,0,0,0,0,0,0,3,0
        },formats[]={
            0,0,0,164,164,165,165,166,166,0,0
        };
        for(int i=0; i<11; ++i)out<<"<xf numFmtId=\""<<formats[i]<<"\" fontId=\""<<fonts[i]<<"\" fillId=\""<<fills[i]<<"\" borderId=\"0\" xfId=\"0\" applyNumberFormat=\"1\" applyFont=\"1\" applyFill=\"1\"><alignment vertical=\"center\""<<(i==2?" horizontal=\"center\" wrapText=\"1\"":"")<<"/></xf>";
        out<<"</cellXfs><cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles></styleSheet>";
        auto result=out.str();
        result.insert(result.find("<fonts"),"<numFmts count=\"3\"><numFmt numFmtId=\"164\" formatCode=\"0.000\"/><numFmt numFmtId=\"165\" formatCode=\"0.0\"/><numFmt numFmtId=\"166\" formatCode=\"0.0%\"/></numFmts>");
        return result;
    }
    inline void save(const fs::path& path,const std::vector<Sheet>& sheets,const std::string& style_xml=styles()){
        std::map<std::string,std::string> entries;
        std::ostringstream types,workbook,rels;
        types<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/><Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>";
        workbook<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?><workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets>";
        rels<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">";
        for(size_t i=0; i<sheets.size(); ++i){
            auto n=std::to_string(i+1);
            entries["xl/worksheets/sheet"+n+".xml"]=sheets[i].xml();
            types<<"<Override PartName=\"/xl/worksheets/sheet"<<n<<".xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>";
            workbook<<"<sheet name=\""<<escape(sheets[i].name)<<"\" sheetId=\""<<n<<"\" r:id=\"rId"<<n<<"\"/>";
            rels<<"<Relationship Id=\"rId"<<n<<"\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet"<<n<<".xml\"/>";
        }
        types<<"</Types>";
        workbook<<"</sheets><calcPr calcId=\"191029\" fullCalcOnLoad=\"1\"/></workbook>";
        rels<<"<Relationship Id=\"styles\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/></Relationships>";
        entries["[Content_Types].xml"]=types.str();
        entries["xl/workbook.xml"]=workbook.str();
        entries["xl/_rels/workbook.xml.rels"]=rels.str();
        entries["xl/styles.xml"]=style_xml;
        entries["_rels/.rels"]="<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"officeDocument\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/></Relationships>";
        write_zip(path,entries);
    }
}
