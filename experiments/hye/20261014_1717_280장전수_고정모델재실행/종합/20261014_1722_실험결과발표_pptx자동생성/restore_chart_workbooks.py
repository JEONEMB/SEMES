"""Preserve unchanged source charts/workbooks after the PPTX importer drops them."""
from pathlib import Path
import posixpath
import zipfile
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
SOURCE=BASE.parent/'presentation/QR_YT_이동추정_실험결과.pptx'
EDITED=BASE/'candidate_full280.pptx'
OUT=BASE/'candidate_full280_preserved.pptx'
R='{http://schemas.openxmlformats.org/package/2006/relationships}'
CT='{http://schemas.openxmlformats.org/package/2006/content-types}'
ET.register_namespace('', 'http://schemas.openxmlformats.org/package/2006/content-types')

def owners(z):
    result={}
    for name in z.namelist():
        if name.startswith('ppt/slides/_rels/slide') and name.endswith('.xml.rels'):
            targets=[]
            for r in ET.fromstring(z.read(name)):
                if r.attrib.get('Type','').endswith('/chart'):
                    targets.append(posixpath.normpath(posixpath.join('ppt/slides',r.attrib['Target'])))
            if targets:result[name]=targets
    return result

with zipfile.ZipFile(SOURCE) as source,zipfile.ZipFile(EDITED) as edited:
    assert owners(source)==owners(edited),'Charts changed owner or order; restoration would be ambiguous.'
    preserve={name:source.read(name) for name in source.namelist() if name.startswith(('ppt/slides/charts/','ppt/embeddings/'))}
    srcct=ET.fromstring(source.read('[Content_Types].xml'))
    dstct=ET.fromstring(edited.read('[Content_Types].xml'))
    keys={(e.tag,e.attrib.get('PartName',e.attrib.get('Extension'))) for e in dstct}
    for elem in srcct:
        name=elem.attrib.get('PartName','').lstrip('/')
        if name in preserve or elem.attrib.get('Extension')=='xlsx':
            key=(elem.tag,elem.attrib.get('PartName',elem.attrib.get('Extension')))
            if key not in keys:dstct.append(elem);keys.add(key)
    with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as result:
        for name in edited.namelist():
            if name in preserve or name=='[Content_Types].xml':continue
            result.writestr(name,edited.read(name))
        for name,data in preserve.items():result.writestr(name,data)
        result.writestr('[Content_Types].xml',ET.tostring(dstct,encoding='utf-8',xml_declaration=True))
with zipfile.ZipFile(SOURCE) as source,zipfile.ZipFile(OUT) as restored:
    assert all(source.read(name)==restored.read(name) for name in preserve)
print(f'Restored {len(preserve)} original chart/workbook parts byte-for-byte: {OUT}')
