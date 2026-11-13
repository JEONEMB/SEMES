CSS = """
:root{--ink:#14213D;--sub:#4A5568;--line:#E3E6E4;--bg:#F7F8F5;--card:#FFFFFF;--acc:#2F6FED;--warn:#C2410C;--ok:#15803D}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 'Noto Sans KR','Malgun Gothic',sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 20px 80px}h1{font-size:28px;margin:0 0 6px}h2{font-size:22px;margin:40px 0 12px;padding-top:12px;border-top:2px solid var(--ink)}
h3{font-size:17px;margin:24px 0 8px}p,li{color:var(--sub)}b,strong{color:var(--ink)}.meta{color:var(--sub);font-size:13px}
table{border-collapse:collapse;width:100%;margin:10px 0 18px;background:var(--card);font-size:13.5px}th,td{border:1px solid var(--line);padding:6px 9px;text-align:right;vertical-align:top}
th{background:#EEF1F6;text-align:center;color:var(--ink)}td.l,th.l{text-align:left}tr.hl td{background:#FFF7E6}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:12px 0}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:700;margin-right:4px}
.t-ok{background:#DCFCE7;color:#166534}.t-warn{background:#FFEDD5;color:#9A3412}.t-info{background:#DBEAFE;color:#1E40AF}
.good{color:var(--ok);font-weight:700}.bad{color:var(--warn);font-weight:700}code{background:#EEF1F6;padding:1px 5px;border-radius:4px;font-size:12.5px}
.toc a{color:var(--acc);text-decoration:none}.small{font-size:12.5px}img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
@media (max-width:700px){table{font-size:12px}th,td{padding:4px 5px}}
"""
def page(title, body, sub=''):
    return f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{title}</title><style>{CSS}</style></head><body><main><h1>{title}</h1><p class='meta'>{sub}</p>{body}</main></body></html>"
def table(head, rows, cls=None, left=1):
    h = ''.join(f"<th{' class=l' if i < left else ''}>{x}</th>" for i, x in enumerate(head))
    out = []
    for j, r in enumerate(rows):
        c = f" class='{cls[j]}'" if cls and cls[j] else ''
        out.append(f"<tr{c}>" + ''.join(f"<td{' class=l' if i < left else ''}>{x}</td>" for i, x in enumerate(r)) + "</tr>")
    return f"<table><tr>{h}</tr>{''.join(out)}</table>"
