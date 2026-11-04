// 지식베이스 생성기
//
//   읽는 것
//     (A) 문헌연구/**/요약.md        114건. 제목·저자·서지·우리와의 관계를 긁는다. **손으로 안 적는다.**
//     (B) 지식베이스/사실/*.md       우리가 알아낸 것. 머리말로 관계를 적는다.
//
//   쓰는 것
//     (1) 지식베이스/문헌목록.md     찾은 논문·특허 전체 표 (주제별, 관계 등급순)
//     (2) 지식베이스/지식지도.md     사실 표 색인
//     (3) 대상 .canvas               Obsidian Canvas. **손으로 만든 노드는 보존**
//
//   캔버스 읽는 순서 (위에서 아래로)
//     1단  찾은 논문 목록 (주제별)
//     2단  찾은 특허 목록 (주제별)
//     3단  그중 우리가 실제로 쓴 문헌
//     4단  우리 실측값 -> 결론 -> 가정 -> 미해결 -> 촬영 요청
//
//   캔버스 보존 규칙
//     - id 가 "kb-" 로 시작하는 것만 생성기가 관리한다.
//     - 그 밖의 노드·연결은 원문 그대로 둔다.
//     - 이미 있는 kb- 노드는 x, y 를 유지한다. 옮겨 둔 배치가 보존된다.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) { if (!trim(cur).empty()) out.push_back(trim(cur)); cur.clear(); }
        else cur += c;
    }
    if (!trim(cur).empty()) out.push_back(trim(cur));
    return out;
}
std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c == '\r') {}
        else o += c;
    }
    return o;
}
std::string cut(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    size_t e = n;                               // UTF-8 경계에서 자른다
    while (e > 0 && (static_cast<unsigned char>(s[e]) & 0xC0) == 0x80) --e;
    return s.substr(0, e) + "...";
}
// "- 저자: 홍길동" 같은 줄에서 값을 꺼낸다
std::string tagged(const std::string& text, const std::string& key) {
    size_t k = text.find("- " + key + ":");
    if (k == std::string::npos) {
        k = text.find(key + ":");
        if (k == std::string::npos) return "";
    }
    size_t a = text.find(':', k) + 1;
    size_t e = text.find('\n', a);
    return trim(text.substr(a, e == std::string::npos ? std::string::npos : e - a));
}

// ---------- (A) 문헌 ----------
struct Lit {
    std::string group;   // 원인분석 / 유사연구 / QR분석 / Y·T분리 / 오차보정
    std::string kind;    // 논문 / 특허
    std::string folder;  // 폴더 이름 (01_Adiv1989_...)
    std::string path;    // 저장소 기준 경로
    std::string title, author, cite, rel_mark, rel_text;
    int rel = 0;          // 3 = 매우 높음, 2 = 높음, 1 = 보통, 0 = 표시 없음
    bool has_trans = false, has_src = false;
    int pdfs = 0;
};
int rel_of(const std::string& m) {
    if (m == u8"◎") return 3;
    if (m == u8"○") return 2;
    if (m == u8"△") return 1;
    return 0;
}
const char* rel_name(int r) {
    return r == 3 ? u8"◎ 매우 높음" : r == 2 ? u8"○ 높음" : r == 1 ? u8"△ 보통" : u8"- 표시 없음";
}

// ---------- (B) 사실 ----------
struct Fact {
    std::string id, title, kind, state, source, value, lead, file, litpath;
    std::vector<std::string> affects, blocked;
    int col = 0, row = 0;
};
std::string color_of(const Fact& f) {
    if (f.state == u8"반증") return "1";
    if (f.kind == u8"문헌") return "6";
    if (f.kind == u8"측정") return "5";
    if (f.kind == u8"결론") return "4";
    if (f.kind == u8"가정") return "3";
    if (f.kind == u8"미해결") return "2";
    if (f.kind == u8"요청") return "1";
    return "";
}
int column_of(const std::string& k) {
    if (k == u8"문헌") return 0;
    if (k == u8"측정") return 1;
    if (k == u8"결론") return 2;
    if (k == u8"가정") return 3;
    if (k == u8"미해결") return 4;
    if (k == u8"요청") return 5;
    return 6;
}

// ---------- 캔버스 파서 ----------
std::vector<std::string> split_objects(const std::string& body) {
    std::vector<std::string> out;
    int depth = 0;
    bool instr = false, esc = false;
    size_t start = 0;
    for (size_t i = 0; i < body.size(); ++i) {
        char c = body[i];
        if (esc) { esc = false; continue; }
        if (c == '\\') { esc = true; continue; }
        if (c == '"') { instr = !instr; continue; }
        if (instr) continue;
        if (c == '{') { if (depth == 0) start = i; ++depth; }
        else if (c == '}') { if (--depth == 0) out.push_back(body.substr(start, i - start + 1)); }
    }
    return out;
}
std::string array_body(const std::string& s, const std::string& key) {
    size_t k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    size_t a = s.find('[', k);
    if (a == std::string::npos) return "";
    int depth = 0;
    bool instr = false, esc = false;
    for (size_t i = a; i < s.size(); ++i) {
        char c = s[i];
        if (esc) { esc = false; continue; }
        if (c == '\\') { esc = true; continue; }
        if (c == '"') { instr = !instr; continue; }
        if (instr) continue;
        if (c == '[') ++depth;
        else if (c == ']') { if (--depth == 0) return s.substr(a + 1, i - a - 1); }
    }
    return "";
}
std::string field(const std::string& obj, const std::string& key) {
    size_t k = obj.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    size_t c = obj.find(':', k);
    size_t i = obj.find_first_not_of(" \t", c + 1);
    if (i == std::string::npos) return "";
    if (obj[i] == '"') {
        std::string o;
        bool esc = false;
        for (size_t j = i + 1; j < obj.size(); ++j) {
            if (esc) { o += obj[j]; esc = false; continue; }
            if (obj[j] == '\\') { esc = true; o += obj[j]; continue; }
            if (obj[j] == '"') break;
            o += obj[j];
        }
        return o;
    }
    size_t e = obj.find_first_of(",}", i);
    return trim(obj.substr(i, e - i));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        fs::path root = fs::u8path(KB_DIR);
        // 인자: [캔버스경로] [relayout]
        //   relayout 을 주면 저장된 kb- 노드 위치를 버리고 배치를 다시 잡는다.
        //   (생성기의 배치 규칙을 바꿨을 때 쓴다. 손으로 옮긴 배치는 사라진다.)
        fs::path canvas = fs::u8path(CANVAS_PATH);
        bool relayout = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "relayout") relayout = true;
            else canvas = fs::u8path(a);
        }
        fs::path litroot = fs::u8path(LIT_DIR);

        // ================= (A) 문헌 스캔 =================
        struct Area { const char* rel; const char* group; };
        const Area areas[] = {
            {u8"01_주제_관련/원인분석", u8"원인분석"},
            {u8"01_주제_관련/유사연구", u8"유사연구"},
            {u8"02_QR_분석", u8"QR 분석"},
            {u8"03_Y_T_분리", u8"Y·T 분리"},
            {u8"04_Y_T_오차_보정", u8"오차 보정"},
        };
        std::vector<Lit> L;
        for (auto& a : areas) {
            for (const char* kind : {u8"논문", u8"특허"}) {
                fs::path dir = litroot / fs::u8path(a.rel) / fs::u8path(kind);
                if (!fs::exists(dir)) continue;
                std::vector<fs::path> items;
                for (auto& e : fs::directory_iterator(dir))
                    if (e.is_directory() && !e.path().filename().u8string().empty() &&
                        isdigit(static_cast<unsigned char>(e.path().filename().u8string()[0])))
                        items.push_back(e.path());
                std::sort(items.begin(), items.end());
                for (auto& p : items) {
                    fs::path sum = p / fs::u8path(u8"요약.md");
                    if (!fs::exists(sum)) continue;
                    std::ifstream in(sum);
                    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                    Lit l;
                    l.group = a.group;
                    l.kind = kind;
                    l.folder = p.filename().u8string();
                    l.path = std::string(u8"docs/hye/문헌연구/") + a.rel + "/" + kind + "/" + l.folder;
                    size_t h = text.find("# ");
                    if (h != std::string::npos) l.title = trim(text.substr(h + 2, text.find('\n', h) - h - 2));
                    l.author = tagged(text, u8"저자");
                    l.cite = tagged(text, u8"서지");
                    l.has_trans = fs::exists(p / fs::u8path(u8"번역.md"));
                    l.has_src = fs::exists(p / fs::u8path(u8"원문.md"));
                    for (auto& g : fs::directory_iterator(p)) if (g.path().extension() == ".pdf") ++l.pdfs;
                    l.rel_text = tagged(text, u8"우리와의 관계");
                    if (!l.rel_text.empty()) {
                        l.rel_mark = l.rel_text.substr(0, std::min<size_t>(3, l.rel_text.size()));
                        l.rel = rel_of(l.rel_mark);
                        if (l.rel) l.rel_text = trim(l.rel_text.substr(l.rel_mark.size()));
                        else l.rel_mark = "";
                    }
                    L.push_back(std::move(l));
                }
            }
        }
        std::cout << u8"문헌 " << L.size() << u8"건 스캔\n";

        // ================= (B) 사실 읽기 =================
        fs::path facts = root / fs::u8path(u8"사실");
        std::vector<Fact> F;
        std::vector<std::string> problems;
        if (fs::exists(facts)) {
            for (auto& e : fs::directory_iterator(facts)) {
                if (e.path().extension() != ".md") continue;
                std::ifstream in(e.path());
                std::string line;
                if (!std::getline(in, line) || trim(line) != "---") {
                    problems.push_back(e.path().filename().u8string() + u8": 머리말이 --- 로 시작하지 않는다");
                    continue;
                }
                Fact f;
                f.file = e.path().filename().u8string();
                while (std::getline(in, line)) {
                    if (trim(line) == "---") break;
                    size_t p = line.find(':');
                    if (p == std::string::npos) continue;
                    std::string k = trim(line.substr(0, p)), v = trim(line.substr(p + 1));
                    if (k == "id") f.id = v;
                    else if (k == u8"제목") f.title = v;
                    else if (k == u8"종류") f.kind = v;
                    else if (k == u8"상태") f.state = v;
                    else if (k == u8"출처") f.source = v;
                    else if (k == u8"값") f.value = v;
                    else if (k == u8"문헌경로") f.litpath = v;
                    else if (k == u8"영향") f.affects = split(v, ',');
                    else if (k == u8"막힘") f.blocked = split(v, ',');
                }
                while (std::getline(in, line)) { if (!trim(line).empty()) { f.lead = trim(line); break; } }
                if (f.id.empty()) { problems.push_back(f.file + u8": id 가 없다"); continue; }
                if (column_of(f.kind) == 6) problems.push_back(f.file + u8": 종류가 이상하다 (" + f.kind + ")");
                F.push_back(std::move(f));
            }
        }
        std::sort(F.begin(), F.end(), [](const Fact& a, const Fact& b) { return a.id < b.id; });
        std::map<std::string, int> idx;
        for (size_t i = 0; i < F.size(); ++i) idx[F[i].id] = int(i);
        for (auto& f : F) {
            for (auto& a : f.affects) if (!idx.count(a)) problems.push_back(f.id + u8" 의 영향 대상 없음: " + a);
            for (auto& a : f.blocked) if (!idx.count(a)) problems.push_back(f.id + u8" 의 막힘 대상 없음: " + a);
            if (!f.litpath.empty() && !fs::exists(fs::u8path(REPO_DIR) / fs::u8path(f.litpath)))
                problems.push_back(f.id + u8" 의 문헌경로를 찾을 수 없다: " + f.litpath);
        }
        std::cout << u8"사실 " << F.size() << u8"개\n";

        // ================= (1) 문헌목록.md =================
        std::ostringstream lm;
        lm << u8"# 찾은 논문·특허 전체 목록\n\n"
           << u8"**이 파일은 생성기가 `문헌연구/**/요약.md` 를 읽어 다시 쓴다. 손으로 고치지 않는다.**\n\n";
        {
            int np = 0, nt = 0;
            std::map<int, int> byrel;
            for (auto& l : L) { (l.kind == u8"논문" ? np : nt)++; byrel[l.rel]++; }
            lm << u8"- 논문·기술자료 **" << np << u8"건**, 특허 **" << nt << u8"건**, 합계 **" << L.size() << u8"건**\n"
               << u8"- 관계 등급: ◎ 매우 높음 " << byrel[3] << u8"건 · ○ 높음 " << byrel[2] << u8"건 · △ 보통 "
               << byrel[1] << u8"건 · 표시 없음 " << byrel[0] << u8"건\n"
               << u8"- 등급은 각 `요약.md` 의 \"우리와의 관계\" 줄에서 그대로 가져온다\n\n";
        }
        for (auto& a : areas) {
            for (const char* kind : {u8"논문", u8"특허"}) {
                std::vector<const Lit*> v;
                for (auto& l : L) if (l.group == a.group && l.kind == kind) v.push_back(&l);
                if (v.empty()) continue;
                std::stable_sort(v.begin(), v.end(), [](const Lit* x, const Lit* y) { return x->rel > y->rel; });
                lm << u8"## " << a.group << " · " << kind << " (" << v.size() << u8"건)\n\n"
                   << u8"| 관계 | 문헌 | 저자 | 서지 | 가진 자료 | 우리와의 관계 |\n|---|---|---|---|---|---|\n";
                for (auto* l : v) {
                    std::string mat = l->pdfs ? u8"PDF " + std::to_string(l->pdfs) : u8"**PDF 없음**";
                    if (l->has_trans) mat += u8" · 번역";
                    if (l->has_src) mat += u8" · 원문";
                    lm << "| " << rel_name(l->rel) << u8" | [" << cut(l->title, 70) << "](../../../" << l->path
                       << u8"/요약.md) | " << l->author << " | " << cut(l->cite, 60) << " | " << mat << " | "
                       << cut(l->rel_text, 160) << " |\n";
                }
                lm << "\n";
            }
        }
        { std::ofstream o(root / fs::u8path(u8"문헌목록.md"), std::ios::binary); o << lm.str(); }

        // ================= 캔버스 =================
        std::vector<std::string> keep_nodes, keep_edges;
        std::map<std::string, std::pair<std::string, std::string>> old_xy;
        if (fs::exists(canvas)) {
            std::ifstream in(canvas, std::ios::binary);
            std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            for (auto& o : split_objects(array_body(s, "nodes"))) {
                std::string id = field(o, "id");
                if (id.rfind("kb-", 0) == 0) { if (!relayout) old_xy[id] = {field(o, "x"), field(o, "y")}; }
                else keep_nodes.push_back(o);
            }
            for (auto& o : split_objects(array_body(s, "edges")))
                if (field(o, "id").rfind("kb-", 0) != 0) keep_edges.push_back(o);
            std::cout << u8"기존 캔버스: 손으로 만든 노드 " << keep_nodes.size() << u8"개 보존\n";
        }

        const int X0 = -640, GX = 820, W = 620;
        const int Y_PAPER = 900, Y_PATENT = 2750, Y_EXCL = 4600, Y_FACT = 6000;
        std::ostringstream c;
        c << "{\n\t\"nodes\":[\n";
        bool first = true;
        auto comma = [&] { if (!first) c << ",\n"; first = false; };
        auto node = [&](const std::string& id, const std::string& text, int dx, int dy, int w, int h, const std::string& color) {
            comma();
            std::string x = std::to_string(dx), y = std::to_string(dy);
            if (old_xy.count(id)) { x = old_xy[id].first; y = old_xy[id].second; }
            c << "\t\t{\"id\":\"" << id << "\",\"type\":\"text\",\"text\":\"" << json_escape(text) << "\",\"x\":" << x
              << ",\"y\":" << y << ",\"width\":" << w << ",\"height\":" << h;
            if (!color.empty()) c << ",\"color\":\"" << color << "\"";
            c << "}";
        };
        for (auto& n : keep_nodes) { comma(); c << "\t\t" << n; }

        // --- 1단 / 2단: 찾은 논문·특허 목록 ---
        for (int pass = 0; pass < 2; ++pass) {
            const char* kind = pass == 0 ? u8"논문" : u8"특허";
            int ybase = pass == 0 ? Y_PAPER : Y_PATENT;
            int total = 0;
            for (auto& l : L) if (l.kind == kind) ++total;
            node(std::string("kb-sec") + (pass ? "2" : "1"),
                 std::string(u8"# ") + (pass == 0 ? u8"1단. 찾은 논문" : u8"2단. 찾은 특허") + u8" — 전체 " +
                     std::to_string(total) + u8"건\n\n" + u8"주제별로 나눴다. ◎ ○ △ 는 각 요약.md 의 \"우리와의 관계\" 등급이다.\n\n" +
                     u8"전체 표: `docs/hye/지식베이스/문헌목록.md`",
                 X0, ybase - 220, 1600, 170, "");
            int col = 0;
            for (auto& a : areas) {
                std::vector<const Lit*> v;
                for (auto& l : L) if (l.group == a.group && l.kind == kind) v.push_back(&l);
                if (v.empty()) continue;
                std::stable_sort(v.begin(), v.end(), [](const Lit* x, const Lit* y) { return x->rel > y->rel; });
                std::ostringstream t;
                t << "## " << a.group << " " << kind << " (" << v.size() << u8"건)\n\n| | 문헌 |\n|---|---|\n";
                for (auto* l : v) t << "| " << (l->rel ? l->rel_mark : "-") << " | " << cut(l->title, 60) << " |\n";
                t << u8"\n`" << a.rel << "/" << kind << "`";
                node(std::string("kb-cat-") + (pass ? "p" : "a") + std::to_string(col), t.str(),
                     X0 + col * GX, ybase, 740, 1400, pass ? "2" : "6");
                ++col;
            }
        }

        // --- 2.5단: 쓰지 않는 문헌 ---
        {
            fs::path ex = root / fs::u8path(u8"제외와격리.md");
            std::string t = u8"# 2.5단. 쓰지 않는 문헌과 그 이유\n\n";
            if (fs::exists(ex)) {
                std::ifstream in(ex);
                std::string line;
                int n = 0;
                while (std::getline(in, line) && n < 60) {
                    if (line.rfind("# ", 0) == 0) continue;
                    t += line + "\n";
                    ++n;
                }
                t += u8"\n(이어서: `docs/hye/지식베이스/제외와격리.md`)";
            } else t += u8"제외와격리.md 가 없다";
            node("kb-exclude", t, X0, Y_EXCL, 1600, 1000, "1");
        }

        // --- 3단 이후: 사실 ---
        node("kb-sec3",
             std::string(u8"# 3단. 그중 우리가 실제로 쓴 것 → 알아낸 것 → 남은 것\n\n") +
                 u8"왼쪽에서 오른쪽으로 **문헌 → 실측값 → 결론 → 가정 → 미해결 → 촬영 요청** 순서다.\n\n" +
                 u8"검은 화살표는 영향, 빨간 화살표는 \"이게 풀려야 확정\" 이다.",
             X0, Y_FACT - 220, 1600, 170, "");
        const char* heads[] = {u8"문헌 — 우리가 쓴 것", u8"실측값", u8"결론", u8"가정 (주의)", u8"미해결", u8"촬영 요청"};
        std::map<int, int> next_row;
        for (auto& f : F) f.col = column_of(f.kind);
        // 0열(문헌)은 id 순. 1열부터는 "나를 가리키는 노드들의 평균 행"으로 정렬해 선 꼬임을 줄인다.
        {
            std::map<int, std::vector<int>> bycol;
            for (size_t i = 0; i < F.size(); ++i) bycol[F[i].col].push_back(int(i));
            for (int c = 0; c < 6; ++c) {
                auto& v = bycol[c];
                if (c > 0) {
                    std::vector<double> key(F.size(), 1e9);
                    for (int i : v) {
                        double sum = 0;
                        int n = 0;
                        for (size_t j = 0; j < F.size(); ++j)
                            for (auto& a : F[j].affects)
                                if (a == F[i].id && F[j].col < c) { sum += F[j].row; ++n; }
                        key[i] = n ? sum / n : 1e9;
                    }
                    std::stable_sort(v.begin(), v.end(), [&](int x, int y) { return key[x] < key[y]; });
                }
                for (size_t r = 0; r < v.size(); ++r) F[v[r]].row = int(r);
            }
        }
        for (int k = 0; k < 6; ++k)
            node("kb-head" + std::to_string(k), std::string("## ") + heads[k], X0 + k * GX, Y_FACT - 20, W, 70, "");
        for (auto& f : F) {
            std::ostringstream t;
            t << "### `" << f.id << "` " << f.title << "\n\n";
            if (!f.value.empty()) t << u8"**값** " << f.value << "\n\n";
            t << u8"**출처** " << f.source << u8"  ·  **상태** " << f.state << "\n\n";
            if (!f.lead.empty()) t << cut(f.lead, 240) << "\n\n";
            if (!f.litpath.empty()) t << u8"원문: `" << f.litpath << u8"/요약.md`\n\n";
            t << "`" << u8"docs/hye/지식베이스/사실/" << f.file << "`";
            node("kb-" + f.id, t.str(), X0 + f.col * GX, Y_FACT + 90 + f.row * 330, W, 270, color_of(f));
        }

        c << "\n\t],\n\t\"edges\":[\n";
        first = true;
        int eid = 0;
        for (auto& e : keep_edges) { comma(); c << "\t\t" << e; }
        for (auto& f : F) {
            for (auto& a : f.affects) {
                if (!idx.count(a)) continue;
                comma();
                bool right = F[idx[a]].col >= f.col;
                c << "\t\t{\"id\":\"kb-e" << (++eid) << "\",\"fromNode\":\"kb-" << f.id << "\",\"fromSide\":\""
                  << (right ? "right" : "left") << "\",\"toNode\":\"kb-" << a << "\",\"toSide\":\""
                  << (right ? "left" : "right") << "\"}";
            }
            for (auto& a : f.blocked) {
                if (!idx.count(a)) continue;
                comma();
                c << "\t\t{\"id\":\"kb-e" << (++eid) << "\",\"fromNode\":\"kb-" << a
                  << "\",\"fromSide\":\"bottom\",\"toNode\":\"kb-" << f.id << "\",\"toSide\":\"bottom\",\"color\":\"1\",\"label\":\""
                  << u8"이게 풀려야 확정" << "\"}";
            }
        }
        c << "\n\t]\n}\n";
        fs::create_directories(canvas.parent_path());
        { std::ofstream o(canvas, std::ios::binary); o << c.str(); }

        // ================= (2) 지식지도.md =================
        std::ostringstream m;
        m << u8"# 지식지도 — 우리가 알아낸 것\n\n"
          << u8"**생성기가 다시 쓴다. 손으로 고치지 않는다.** 찾은 문헌 전체 목록은 [문헌목록.md](문헌목록.md) 에 있다.\n\n"
          << u8"- 사실 " << F.size() << u8"개, 연결 " << eid << u8"개\n"
          << u8"- 그림: `" << canvas.u8string() << u8"`\n\n";
        const char* kinds[] = {u8"문헌", u8"측정", u8"결론", u8"가정", u8"미해결", u8"요청"};
        const char* mh[] = {u8"## 우리가 실제로 쓴 문헌", u8"## 실측값", u8"## 결론",
                            u8"## 가정 (주의해서 쓸 것)", u8"## 미해결", u8"## 촬영 요청"};
        for (int k = 0; k < 6; ++k) {
            m << mh[k] << u8"\n\n| ID | 제목 | 값 | 출처 | 상태 | 영향 |\n|---|---|---|---|---|---|\n";
            for (auto& f : F) {
                if (f.kind != kinds[k]) continue;
                m << "| `" << f.id << u8"` | [" << f.title << u8"](사실/" << f.file << ") | " << f.value << " | "
                  << f.source << " | " << f.state << " | ";
                for (size_t i = 0; i < f.affects.size(); ++i) m << (i ? ", " : "") << "`" << f.affects[i] << "`";
                m << " |\n";
            }
            m << "\n";
        }
        std::map<std::string, std::vector<std::string>> back;
        for (auto& f : F) for (auto& a : f.affects) if (idx.count(a)) back[a].push_back(f.id);
        m << u8"## 거꾸로 보기 — 무엇이 이것을 떠받치는가\n\n| 사실 | 근거가 되는 것 |\n|---|---|\n";
        for (auto& f : F) {
            if (!back.count(f.id)) continue;
            m << "| `" << f.id << "` " << f.title << " | ";
            auto& v = back[f.id];
            for (size_t i = 0; i < v.size(); ++i) m << (i ? ", " : "") << "`" << v[i] << "`";
            m << " |\n";
        }
        m << u8"\n## 막혀 있는 것과 푸는 방법\n\n| 막힌 것 | 풀려면 | 그러면 확정되는 것 |\n|---|---|---|\n";
        for (auto& f : F) {
            if (f.blocked.empty()) continue;
            m << "| `" << f.id << "` " << f.title << " | ";
            for (size_t i = 0; i < f.blocked.size(); ++i)
                m << (i ? ", " : "") << "`" << f.blocked[i] << "` "
                  << (idx.count(f.blocked[i]) ? F[idx[f.blocked[i]]].title : std::string("?"));
            m << " | ";
            for (size_t i = 0; i < f.affects.size(); ++i) m << (i ? ", " : "") << "`" << f.affects[i] << "`";
            m << " |\n";
        }
        m << "\n";
        if (!problems.empty()) {
            m << u8"## 고쳐야 할 것\n\n";
            for (auto& p : problems) m << "- " << p << "\n";
            m << "\n";
        }
        { std::ofstream o(root / fs::u8path(u8"지식지도.md"), std::ios::binary); o << m.str(); }

        std::cout << u8"연결 " << eid << u8"개\n";
        if (problems.empty()) std::cout << u8"문제 없음\n";
        else { std::cout << u8"문제 " << problems.size() << u8"개:\n"; for (auto& p : problems) std::cout << "  " << p << "\n"; }
        std::cout << u8"문헌목록.md / 지식지도.md / " << canvas.filename().u8string() << u8" 갱신\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << u8"오류: " << e.what() << "\n";
        return 1;
    }
}
