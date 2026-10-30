from pathlib import Path
import numpy as np,json
H=Path(__file__).resolve().parent
rng=np.random.default_rng(21);R=rng.normal(size=(12,25));X=rng.normal(size=(4,25));R/=np.sqrt(len(R))
_,s,V=np.linalg.svd(R,full_matrices=False);alpha=.2
Y=X/np.sqrt(alpha)+(X@V.T*(1/np.sqrt(alpha+(1-alpha)*s*s)-1/np.sqrt(alpha)))@V
precision=np.linalg.inv(alpha*np.eye(25)+(1-alpha)*R.T@R)
error=float(abs(np.sum(Y*Y,axis=1)-np.einsum('ni,ij,nj->n',X,precision,X)).max());assert error<1e-9
import run as E
fold=json.loads((E.B/'multiref_4fold_20261021/extended/splits.json').read_text())[0];tr=np.array([E.ix[n] for n in fold['train']]);te=np.array([E.ix[n] for n in fold['test']]);c=E.configs[0]
one=E.fit(c,tr)[2];saved=E.features[c[0]][te].copy();E.features[c[0]][te]+=999
two=E.fit(c,tr)[2];E.features[c[0]][te]=saved
assert all(np.allclose(one[k],two[k]) for k in ['std','V','adjust'])
(H/'numerical_checks.json').write_text(json.dumps(dict(whitened_cost_vs_dense_precision_max_error=error,heldout_feature_perturbation_does_not_change_training_covariance=True),indent=2));print('verified',error)
