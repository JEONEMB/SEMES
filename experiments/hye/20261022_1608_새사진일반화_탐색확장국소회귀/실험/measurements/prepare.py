"""Verify sources and assemble fixed-ID measurements without refitting models."""
from pathlib import Path
import hashlib,json,numpy as np,pandas as pd
H=Path(__file__).resolve().parent; ROOT=H.parents[2]
CACHE=ROOT/'motion_web_20261022/runs/72a03bae6d1c47858525d6ae278d8cef'
EXP=ROOT/'joint_center_630_20261022/results'
SRC=ROOT/'codex_experiments_20261014/evaluate_archives_20261022'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    verified=pd.read_csv(SRC/'source_manifest.csv').sort_values('index').reset_index(drop=True)
    em=pd.read_csv(EXP/'manifest.csv');z=np.load(EXP/'measurements.npz')
    assert len(verified)==341 and len(em)==341 and em.success.all()
    by_hash={r.sha256:r for r in em.itertuples()};assert len(by_hash)==341
    D=[];quality=[];valid=[];compiled=[];rows=[];deltas=[]
    points=z['points'];ids=z['ids'];regions=z['regions']
    for r in verified.itertuples():
        e=by_hash[r.image_sha256];i=int(r.index);j=int(e.index)
        assert str(Path(r.extracted_path).resolve()).casefold()==str(Path(e.path).resolve()).casefold()
        assert r.image_sha256==sha(Path(e.path))
        assert np.allclose([r.Y,r.T],[e.Y,e.T]) and r.scope==e.scope
        path=CACHE/f'result_{i}.json';strict=path.exists();recovered=H/f'recovered_{i}.npz'
        if strict:
            assert sha(path)==r.cache_sha256
            t=json.loads(path.read_text(encoding='utf-8'))['trace']
            assert np.array_equal(ids,t['ids']) and np.array_equal(points,t['points'])
            assert t['measurement']=='blurred_grid_boundary'
            d=np.array(t['displacements']);q=np.array(t['quality']);v=np.array(t['valid'])
            source='strict_scipy_cache';source_sha=sha(path)
            diff=z['D'][j]-d;deltas.append(diff)
        elif recovered.exists():
            rr=np.load(recovered);d=rr['D'];q=rr['quality'];v=rr['valid']
            assert np.array_equal(ids,rr['ids']) and np.array_equal(points,rr['points'])
            source='expanded_scipy_same_objective';source_sha=sha(recovered)
        else:
            d=z['D'][j];q=z['quality'][j];v=z['valid'][j]
            source='expanded_compiled_cache_provisional';source_sha=sha(EXP/'measurements.npz')
        D.append(d);quality.append(q);valid.append(v);compiled.append(z['D'][j])
        rows.append(dict(index=i,dataset=int(r.dataset),file=r.name,path=r.extracted_path,sha256=r.image_sha256,Y=r.Y,T=r.T,scope=r.scope,condition=f'Y{r.Y:+.4f}_T{r.T:+.4f}',source=source,source_sha256=source_sha,strict_success=strict,success=True,pad=int(e.pad),valid=int(np.sum(v)),ncc_min=float(e.ncc_min),expanded_cache_index=j))
    out=pd.DataFrame(rows);out.to_csv(H/'manifest.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(H/'features.npz',D=np.array(D),quality=np.array(quality),valid=np.array(valid),points=points,ids=ids,regions=regions,D_compiled=np.array(compiled),quality_compiled=z['quality'][out.expanded_cache_index],valid_compiled=z['valid'][out.expanded_cache_index],Y=out[['Y','T']].to_numpy(),index=out['index'].to_numpy(),dataset=out.dataset.to_numpy(),strict_success=out.strict_success.to_numpy(),success=out.success.to_numpy())
    delta=np.array(deltas);a=dict(images=341,success=341,strict_success=329,recovered=12,expanded_pad=260,source_verified_all_sha256=True,source_manifest_sha256=sha(SRC/'source_manifest.csv'),source_expanded_npz_sha256=sha(EXP/'measurements.npz'),source_expanded_manifest_sha256=sha(EXP/'manifest.csv'),points_exactly_equal=True,ids_exactly_equal=True,source_counts=out.source.value_counts().to_dict(),compiled_vs_scipy329=dict(mean_abs=float(abs(delta).mean()),max_abs=float(abs(delta).max()),p99_abs=float(np.quantile(abs(delta),.99))),notes=['Inference inputs are original fixed center pixels and current image; file labels are evaluation metadata only.','D keeps the original strict SciPy measurement for all329 successful images.','The12 missing images use the same SciPy boundary objective with only NCC search expanded once recovered files exist.','D_compiled is the separately marked compiled-LM sensitivity comparison, not mixed into strict329.'])
    (H/'audit.json').write_text(json.dumps(a,indent=2),encoding='utf-8');print(json.dumps(a,indent=2),flush=True)

if __name__=='__main__':main()
