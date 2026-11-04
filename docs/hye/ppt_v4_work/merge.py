import zipfile,posixpath as pp,xml.etree.ElementTree as E
from pathlib import Path
b=Path(__file__).parent
P='http://schemas.openxmlformats.org/presentationml/2006/main'; R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'; PK='http://schemas.openxmlformats.org/package/2006/relationships'; CT='http://schemas.openxmlformats.org/package/2006/content-types'
E.register_namespace('p',P);E.register_namespace('r',R)
with zipfile.ZipFile(b.parent/'2026-10-29_중간_v3.pptx') as z: orig={n:z.read(n) for n in z.namelist()}
with zipfile.ZipFile(b/'additions.pptx') as z: add={n:z.read(n) for n in z.namelist()}
mapping={n:pp.join(pp.dirname(n),'v4_'+pp.basename(n)) for n in add if n!='[Content_Types].xml' and '/_rels/' not in n and not n.startswith('_rels/')}
out=dict(orig)
for n,new in mapping.items():out[new]=add[n].replace(b'typeface="Aptos"',b'typeface="Noto Sans KR"') if n.endswith('.xml') else add[n]
for n,data in add.items():
 if '/_rels/' not in n:continue
 owner=n.replace('/_rels/','/')[:-5]
 if owner not in mapping:continue
 newowner=mapping[owner]; newrel=pp.join(pp.dirname(newowner),'_rels',pp.basename(newowner)+'.rels'); tree=E.fromstring(data)
 for rel in tree:
  if rel.get('TargetMode')=='External':continue
  target=rel.get('Target'); resolved=pp.normpath(pp.join(pp.dirname(owner),target)) if not target.startswith('/') else target[1:]
  if resolved in mapping:rel.set('Target',pp.relpath(mapping[resolved],pp.dirname(newowner)))
 out[newrel]=E.tostring(tree,encoding='utf-8',xml_declaration=True)
E.register_namespace('',CT)
ct=E.fromstring(orig['[Content_Types].xml']); existing={x.get('Extension') for x in ct if x.tag.endswith('Default')}
for x in E.fromstring(add['[Content_Types].xml']):
 if x.tag.endswith('Override') and x.get('PartName')[1:] in mapping:
  x.set('PartName','/'+mapping[x.get('PartName')[1:]]);ct.append(x)
 elif x.tag.endswith('Default') and x.get('Extension') not in existing:ct.append(x);existing.add(x.get('Extension'))
out['[Content_Types].xml']=E.tostring(ct,encoding='utf-8',xml_declaration=True)
pres=E.fromstring(orig['ppt/presentation.xml']); rels=E.fromstring(orig['ppt/_rels/presentation.xml.rels']); slides=pres.find('{'+P+'}sldIdLst')
addedpres=E.fromstring(add['ppt/presentation.xml']); addedrels={r.get('Id'):r.get('Target') for r in E.fromstring(add['ppt/_rels/presentation.xml.rels'])}; maxid=max(int(x.get('id')) for x in slides)
for i,sl in enumerate(addedpres.find('{'+P+'}sldIdLst')):
 rid='rIdV4Added'+str(i);target=pp.normpath(pp.join('ppt',addedrels[sl.get('{'+R+'}id')])).lstrip('/')
 E.SubElement(rels,'{'+PK+'}Relationship',{'Id':rid,'Type':R+'/slide','Target':pp.relpath(mapping[target],'ppt')})
 slides.insert(11+i,E.Element('{'+P+'}sldId',{'id':str(maxid+i+1),'{'+R+'}id':rid}))
out['ppt/presentation.xml']=E.tostring(pres,encoding='utf-8',xml_declaration=True);out['ppt/_rels/presentation.xml.rels']=E.tostring(rels,encoding='utf-8',xml_declaration=True)
with zipfile.ZipFile(b/'candidate.pptx','w',zipfile.ZIP_DEFLATED) as z:
 for n,d in out.items():z.writestr(n,d)
assert len(slides)==19
assert all(out[n]==d for n,d in orig.items() if n not in ['ppt/presentation.xml','ppt/_rels/presentation.xml.rels','[Content_Types].xml'])
print('19 slides; original slide parts preserved byte-for-byte; inserted 5 after slide 11')


