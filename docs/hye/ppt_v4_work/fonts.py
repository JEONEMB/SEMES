import zipfile
with zipfile.ZipFile('native_candidate.pptx') as z, zipfile.ZipFile('native_fixed.pptx','w',zipfile.ZIP_DEFLATED) as o:
 for n in z.namelist():
  d=z.read(n)
  if n.endswith('.xml'):d=d.replace(b'typeface="Aptos"',b'typeface="Noto Sans KR"')
  o.writestr(n,d)
