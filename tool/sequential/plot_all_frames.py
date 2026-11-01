"""Fresh four-process Fourier extraction -> frozen-model inference -> all-frame charts."""
import argparse,concurrent.futures,hashlib,html,json,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image,ImageDraw,ImageFont
from sequential import estimate

def render(rows,out):
    good=rows[rows.status=='ok'];n=len(rows);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    im=Image.new('RGB',(1600,1000),'white');draw=ImageDraw.Draw(im);svgs=[]
    colors={'train':'#2469ac','holdout':'#e98520','reference':'#222222'}
    for j,(field,title) in enumerate([('y_error','Y error (mm)'),('theta_error','Theta error (deg)')]):
        x0,y0,w,h=100,80+j*455,1430,330;limit=max(.11,float(abs(good[field]).max())*1.12)
        draw.text((x0,y0-40),title+' | all images, frozen model',font=font,fill='black')
        parts=[f'<svg viewBox="0 0 1600 450" role="img"><text x="100" y="25">{title}</text><rect x="100" y="50" width="1430" height="330" fill="white" stroke="#aaa"/>']
        for v in np.linspace(-limit,limit,5):
            yy=y0+h/2-v/limit*h/2;draw.line((x0,yy,x0+w,yy),fill='#dddddd');draw.text((12,yy-9),f'{v:.3f}',font=small,fill='black')
            sy=50+h/2-v/limit*h/2;parts.append(f'<line x1="100" x2="1530" y1="{sy}" y2="{sy}" stroke="#ddd"/><text x="5" y="{sy}">{v:.3f}</text>')
        for v in [-.09,0,.09]:
            yy=y0+h/2-v/limit*h/2;draw.line((x0,yy,x0+w,yy),fill='#cf3333' if v else '#888888',width=2)
            sy=50+h/2-v/limit*h/2;parts.append(f'<line x1="100" x2="1530" y1="{sy}" y2="{sy}" stroke="{"#cf3333" if v else "#888"}"/>')
        for idx,r in rows.iterrows():
            xx=x0+idx/max(1,n-1)*w
            if idx%25==0:
                draw.text((xx-10,y0+h+8),str(idx),font=small,fill='black');parts.append(f'<text x="{xx}" y="405">{idx}</text>')
            if r.status!='ok':continue
            yy=y0+h/2-r[field]/limit*h/2;color=colors[r.partition]
            draw.ellipse((xx-3,yy-3,xx+3,yy+3),fill=color)
            tip=html.escape(f'{idx}: {r.filename} | {r.partition} | label Y={r.label_y:.3f} mm, theta={r.label_theta:.3f} deg | estimate Y={r.y_mm:.4f}, theta={r.theta_deg:.4f} | errors Y={r.y_error:.4f}, theta={r.theta_error:.4f}')
            sy=50+h/2-r[field]/limit*h/2;parts.append(f'<circle cx="{xx}" cy="{sy}" r="4" fill="{color}"><title>{tip}</title></circle>')
        draw.text((x0,y0+h+38),'Image index: center, Y (- to +), Theta (- to +). Red lines: +/-0.09.',font=small,fill='black');parts.append('</svg>');svgs.append(''.join(parts))
    draw.text((100,978),'Blue = training images | Orange = heldout images | Black = center reference',font=small,fill='black');im.save(out/'all_image_errors.png')
    columns=['filename','partition','label_y','label_theta','y_mm','theta_deg','y_error','theta_error','joint_pass','status']
    table=rows[columns].to_html(index=False,float_format=lambda x:f'{x:.5f}',escape=True)
    document='<!doctype html><meta charset="utf-8"><title>All-image Y/theta errors</title><style>body{font:16px sans-serif;margin:24px}svg{width:100%;max-height:480px}circle:hover{r:7;stroke:black}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}th{position:sticky;top:0;background:#eee}</style><h1>전체 영상 Y·θ 오차</h1><p>점 위에 마우스를 올리면 파일명·라벨·추정값·오차가 표시됩니다. 파랑=학습, 주황=보류, 검정=기준. 빨강=±0.09. 파일명을 명목 라벨로 사용하며 세션 보정값은 알려진 상태입니다.</p><p><a href="all_predictions.csv">전체 CSV</a></p>'+''.join(svgs)+'<h2>사진별 결과</h2>'+table
    (out/'all_image_errors.html').write_text(document,encoding='utf-8')

def main(repo,work,out,modelpath):
    repo=repo.resolve();work=work.resolve();out=out.resolve();work.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    paths=[repo/'data/SSAFY Test Image Set/center.bmp']+sorted((repo/'data/SSAFY Test Image Set/Theta').glob('*.bmp'))+sorted((repo/'data/SSAFY Test Image Set/Y').glob('*.bmp'))
    jobs=[];start=time.perf_counter()
    for k in range(4):
        d=work/f'worker_{k}';d.mkdir(exist_ok=True);lst=d/'images.txt';lst.write_text('\n'.join(map(str,paths[k::4])),encoding='utf-8');jobs.append((k,d,lst))
    def collect(job):
        k,d,lst=job
        with (d/'log.txt').open('w') as log:subprocess.run([str(repo/'tool/qr-subpixel/build/Release/validate.exe'),str(lst),str(d),'fourier'],stdout=log,stderr=log,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(collect,jobs))
    collection_seconds=time.perf_counter()-start;tables=[]
    for k,d,_ in jobs:
        t=pd.read_csv(d/'points.csv');t['frame']=t.frame*4+k;tables.append(t)
    points=pd.concat(tables);points=points[points.valid==1].copy();points['key']=points.payload+':'+points.id.astype(str)
    maps={i:{r.key:[r.x,r.y] for r in g.itertuples()} for i,g in points.groupby('frame')};model=json.loads(modelpath.read_text(encoding='utf-8'))
    if model.get('split')!='blocked':raise ValueError('fixed report partition expects blocked model')
    records=[]
    for i,path in enumerate(paths):
        name=path.stem
        if name=='center':records.append(dict(frame=i,filename=path.name,axis='C',nominal=0.,group=None,label_y=0.,label_theta=0.,partition='reference'));continue
        axis=name[0];v=float(name[1:]);group=axis+('+' if v>0 else '-')+(str(int(round(abs(v)*100))%10) if axis=='Y' else '')
        records.append(dict(frame=i,filename=path.name,axis=axis,nominal=v,group=group,label_y=v if axis=='Y' else 0.,label_theta=v if axis=='T' else 0.))
    for group in model['groups']:
        subset=sorted([r for r in records if r['group']==group],key=lambda r:r['nominal']);n=len(subset)
        for j,r in enumerate(subset):r['partition']='holdout' if n//3<=j<2*n//3 else 'train'
    def infer(r):
        try:
            result=estimate(model,maps.get(r['frame'],{}),r['group']);r.update(result,status='ok');r['y_error']=result['y_mm']-r['label_y'];r['theta_error']=result['theta_deg']-r['label_theta'];r['joint_pass']=abs(r['y_error'])<=.09 and abs(r['theta_error'])<=.09
        except ValueError as e:r.update(status=str(e),joint_pass=False,y_mm=np.nan,theta_deg=np.nan,y_error=np.nan,theta_error=np.nan)
        return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(infer,records))
    rows=pd.DataFrame(records);rows['order']=rows.axis.map({'C':0,'Y':1,'T':2});rows=rows.sort_values(['order','nominal']).reset_index(drop=True);rows.to_csv(out/'all_predictions.csv',index_label='image_index')
    summary={}
    for label,g in [('all_nonreference',rows[rows.partition!='reference']),('train',rows[rows.partition=='train']),('holdout',rows[rows.partition=='holdout']),('reference',rows[rows.partition=='reference'])]:
        ok=g[g.status=='ok'];summary[label]={'images':len(g),'success':len(ok),'y_rmse':float(np.sqrt(np.mean(ok.y_error**2))),'theta_rmse':float(np.sqrt(np.mean(ok.theta_error**2))),'y_max_abs':float(abs(ok.y_error).max()),'theta_max_abs':float(abs(ok.theta_error).max()),'joint_pass':int(ok.joint_pass.sum())}
    summary['execution']={'images':len(paths),'workers':4,'collection_seconds':collection_seconds,'total_seconds':time.perf_counter()-start,'frozen_model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'model':str(modelpath),'known_session_offsets':True,'filename_labels':True,'real_mixed_validation':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));render(rows,out);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'));p.add_argument('--work',type=Path,default=Path('docs/validation-runs/all-image-errors'));p.add_argument('--output',type=Path,default=Path('docs/sequential-y-theta/all-image-errors'));p.add_argument('--model',type=Path,default=Path('docs/sequential-y-theta/fourier-recheck/fourier_experimental_model.json'));a=p.parse_args();main(a.repo,a.work,a.output,a.model)
