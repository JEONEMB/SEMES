"""Check numerical identities of the exported PPT protocol dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate(out):
    p=pd.read_csv(out/'displacements.csv.gz')
    means=pd.read_csv(out/'mean_displacements.csv')
    fits=pd.read_csv(out/'linear_fits.csv')
    fr=pd.read_csv(out/'frame_linear_residuals.csv')
    pr=pd.read_csv(out/'point_linear_residuals.csv.gz')
    checks={}
    def check(name, condition):
        checks[name]=bool(condition)
        if not condition: raise AssertionError(name)
    check('341 frames, each 755 unique corners',len(p)==341*755 and p.groupby('path').size().eq(755).all() and not p.duplicated(['path','key']).any())
    check('630 non-diagonal corners per frame',p[~p.diagonal].groupby('path').size().eq(630).all())
    check('center1 is exact displacement origin',np.allclose(p.loc[p.path.eq('기준 위치/center1.bmp'),['dx','dy']],0))
    direct=p[~p.diagonal].groupby('path')[['dx','dy']].mean().sort_index()
    published=means[means.method.eq('no_diagonal630')].set_index('path')[['dx','dy']].sort_index()
    check('published 630 mean equally weights every corner',np.allclose(direct,published,atol=1e-12))
    sigma=fr.groupby(['group','method']).residual_dx.std()
    expected=fits.set_index(['group','method']).residual_sigma_dx
    check('linear residual sigma is sample std ddof1',np.allclose(sigma.sort_index(),expected.sort_index(),atol=1e-12))
    common=pr[~pr.diagonal].groupby(['group','path']).residual_dx.mean()
    reference=fr[fr.method.eq('no_diagonal630')].set_index(['group','path']).residual_dx
    check('mean point OLS residual equals mean-frame OLS residual',np.allclose(common.sort_index(),reference.sort_index(),atol=1e-10))
    check('individual equals raw residual minus common',np.allclose(pr.individual_dx,pr.residual_dx-pr.common_dx,atol=1e-12))
    check('diagnostic failure retained for diagonal corners',not p[p.diagonal].accepted.any())
    result={'status':'pass','checks':checks}
    (out/'numerical_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True)
    validate(a.parse_args().output)
