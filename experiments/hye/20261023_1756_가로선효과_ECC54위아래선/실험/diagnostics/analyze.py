"""Read-only, train-fold response diagnosis of the four permitted observations.

This is not a new decoder, model selection, or independent validation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
ROOT = H.parent
NAMES = ['qr', 'upper', 'lower', 'both']
LABELS = ['QR only', 'QR + upper', 'QR + lower', 'QR + both']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slices(name):
    result = [('qr', slice(0, 54))]
    if name in ('upper', 'both'):
        result.append(('upper', slice(54, 78)))
    if name == 'lower':
        result.append(('lower', slice(54, 78)))
    if name == 'both':
        result.append(('lower', slice(78, 102)))
    return result


def weights(name, n, metric):
    w = np.ones(n)
    if metric == 'family_equal_rms':
        ss = slices(name)
        for _, s in ss:
            w[s] = np.sqrt(n / (len(ss) * (s.stop - s.start)))
    return w


def main():
    inputs = [ROOT / 'features.npz', ROOT / 'manifest.csv',
              ROOT / 'lines/features.npz', ROOT / 'lines/protocol.json']
    protected = list(ROOT.glob('*.joblib')) + inputs
    before = {str(p): sha(p) for p in protected}
    a = np.load(ROOT / 'features.npz')
    lines = np.load(ROOT / 'lines/features.npz')
    m = pd.read_csv(ROOT / 'manifest.csv')
    assert np.array_equal(m['index'], np.arange(341))
    y = m[['Y', 'T']].to_numpy()
    records, jac_rows, projections, folds, response_rows, contrast_rows = [], [], [], [], [], []
    for fold in range(4):
        tr = np.flatnonzero((m.fold.to_numpy() != fold) & (m.dataset.to_numpy() == 20261020))
        te = np.flatnonzero((m.fold.to_numpy() == fold) & (m.dataset.to_numpy() == 20261020))
        keys = [tuple(v) for v in y[tr]]
        groups = sorted(set(keys))
        masks = [np.array([k == g for k in keys]) for g in groups]
        gy = np.array(groups)
        assert not set(m.iloc[tr].cv_group) & set(m.iloc[te].cv_group)
        design = np.c_[np.ones(len(gy)), gy]
        assert np.linalg.matrix_rank(design) == 3
        folds.append(dict(fold=fold, training_indices=tr.tolist(),
                          excluded_indices=te.tolist(), conditions=gy.tolist(),
                          design_rank=3, design_condition=float(np.linalg.cond(design))))
        for name in NAMES:
            X = a['X_' + name][tr]
            good = a['success_' + name][tr]
            # No partial-feature substitution or filtering is allowed here.
            assert good.all() and np.isfinite(X).all()
            gx = np.array([X[mask].mean(axis=0) for mask in masks])
            coef = np.linalg.lstsq(design, gx, rcond=None)[0]
            J = coef[1:]
            repeat = X.copy()
            for g, mask in enumerate(masks):
                repeat[mask] -= gx[g]
            mean_fit_residual = gx - design @ coef
            for feature in range(X.shape[1]):
                jac_rows.append(dict(fold=fold, observation=name, feature=feature,
                                     intercept_px=coef[0, feature],
                                     slope_Y_px_per_command=J[0, feature],
                                     slope_T_px_per_command=J[1, feature]))
            for family, s in slices(name):
                selections = [('all', np.arange(s.start, s.stop))]
                if family == 'qr':
                    selections += [('dx', np.arange(0, 54, 2)), ('dy', np.arange(1, 54, 2))]
                else:
                    selections += [('left', np.arange(s.start, s.start+12)), ('right', np.arange(s.start+12, s.stop))]
                for part, ix in selections:
                    response_rows.append(dict(fold=fold, observation=name, family=family,
                        part=part, mean_slope_Y=float(J[0, ix].mean()),
                        mean_slope_T=float(J[1, ix].mean()),
                        rms_slope_Y=float(np.sqrt(np.mean(J[0, ix]**2))),
                        rms_slope_T=float(np.sqrt(np.mean(J[1, ix]**2)))))
                if name == 'both' and family != 'qr':
                    # Fixed descriptive contrast, never a new fitted decoder.
                    c = np.zeros(X.shape[1]); c[s.start:s.start+12] = 1/12; c[s.start+12:s.stop] = -1/12
                    r = repeat @ c
                    contrast_rows.append(dict(fold=fold,line=family,
                        definition='mean(left12) - mean(right12)',
                        Y_response_px_per_command=float(J[0]@c),
                        T_response_px_per_command=float(J[1]@c),
                        repeat_contrast_SD_px=float(np.sqrt(np.sum(r*r)/(len(tr)-len(groups)))),
                        condition_mean_contrast_fit_RMS_px=float(np.sqrt(np.mean((mean_fit_residual@c)**2)))))
            for metric in ['raw_coordinate', 'family_equal_rms']:
                w = weights(name, X.shape[1], metric)
                jw = J * w
                cosine = float(jw[0] @ jw[1] / (np.linalg.norm(jw[0])*np.linalg.norm(jw[1])))
                normalized = jw / np.linalg.norm(jw, axis=1, keepdims=True)
                tcomp = -.09 * float(jw[0] @ jw[1]) / float(jw[1] @ jw[1])
                residual = .09*jw[0] + tcomp*jw[1]
                norm = np.linalg.norm(residual)
                unit = residual / norm
                projected = (repeat*w) @ unit
                # Each group's sample mean removes one degree of freedom.
                dof = len(tr) - len(groups)
                noise = float(np.sqrt(np.sum(projected**2)/dof))
                systematic = (mean_fit_residual*w) @ unit
                row = dict(fold=fold, observation=name, metric=metric,
                    feature_count=X.shape[1], training_images=len(tr), training_conditions=len(groups),
                    repeat_noise_dof=dof, cosine=cosine,
                    axis_angle_degrees=float(np.degrees(np.arccos(np.clip(cosine,-1,1)))),
                    normalized_jacobian_condition=float(np.linalg.cond(normalized)),
                    Y_009_compensating_T=tcomp,
                    remaining_coordinate_RMS_px=float(np.sqrt(np.mean(residual**2))),
                    remaining_projected_signal_px=float(norm),
                    repeat_projected_noise_SD_px=noise,
                    repeat_projected_signal_to_noise=float(norm/noise),
                    condition_mean_projected_fit_RMS_px=float(np.sqrt(np.mean(systematic**2))),
                    train_condition_mean_fit_RMS_px=float(np.sqrt(np.mean((mean_fit_residual*w)**2))))
                records.append(row)
                for idx, projection in zip(tr, projected):
                    projections.append(dict(fold=fold, observation=name, metric=metric,
                                            index=int(idx), projected_repeat_residual_px=float(projection)))
    df = pd.DataFrame(records)
    df.to_csv(H/'fold_separation.csv', index=False)
    pd.DataFrame(jac_rows).to_csv(H/'fold_response_coefficients.csv', index=False)
    pd.DataFrame(response_rows).to_csv(H/'fold_family_response.csv', index=False)
    pd.DataFrame(contrast_rows).to_csv(H/'fold_line_left_right_contrast.csv', index=False)
    pd.DataFrame(projections).to_csv(H/'training_repeat_projections.csv', index=False)
    (H/'training_fold_audit.json').write_text(json.dumps(folds, indent=2))

    new = m.dataset.to_numpy() == 20261020
    groups = sorted(set(map(tuple, y[new])))
    raw = np.c_[lines['X_upper'], lines['X_lower']]
    line_rows, stats_rows, group_means = [], [], []
    for Y, T in groups:
        ix = np.flatnonzero(new & (y[:,0] == Y) & (y[:,1] == T))
        assert len(ix) == 3
        mu = raw[ix].mean(axis=0)
        sd = raw[ix].std(axis=0, ddof=1)
        group_means.append(mu)
        for b in range(48):
            line_rows.append(dict(Y=Y, T=T, block=str(lines['block_names'][b]),
                line=str(lines['line_names'][b]), side='left' if b%24<12 else 'right',
                x_px=float(lines['x_centers'][b]), mean_vertical_shift_px=float(mu[b]),
                repeat_SD_px=float(sd[b]), count=3))
        for line, off in [('upper',0),('lower',24)]:
            for side, k in [('left',0),('right',12)]:
                sl = slice(off+k, off+k+12)
                xx = lines['x_centers'][sl]
                slope = np.polyfit(xx-xx.mean(), mu[sl], 1)[0]
                stats_rows.append(dict(Y=Y,T=T,line=line,side=side,
                    mean_vertical_shift_px=float(mu[sl].mean()),
                    spatial_slope_px_per_1000px=float(1000*slope),
                    repeat_SD_of_segment_mean_px=float(raw[ix,sl].mean(axis=1).std(ddof=1))))
    pd.DataFrame(line_rows).to_csv(H/'observed_line_condition_means.csv',index=False)
    pd.DataFrame(stats_rows).to_csv(H/'observed_line_segment_statistics.csv',index=False)

    plt.rcParams.update({'font.size':10,'figure.dpi':150})
    fig, axes = plt.subplots(2,2,figsize=(11.5,7.3),constrained_layout=True)
    columns = [('normalized_jacobian_condition','Condition number of unit axis directions'),
               ('remaining_coordinate_RMS_px','Residual after Y=0.09 and best T compensation\nRMS pixels across weighted coordinates'),
               ('repeat_projected_noise_SD_px','Repeated-image noise along separating direction\nProjected SD (weighted pixels)'),
               ('repeat_projected_signal_to_noise','Y=0.09 separating signal / repeat noise SD')]
    for ax,(column,title) in zip(axes.ravel(),columns):
        for metric,color,offset in [('raw_coordinate','#d55e00',-.1),('family_equal_rms','#0072b2',.1)]:
            for j,name in enumerate(NAMES):
                values=df[(df.observation==name)&(df.metric==metric)][column].to_numpy()
                ax.scatter(j+offset+np.linspace(-.025,.025,4),values,s=20,color=color,
                    label=metric.replace('_',' ') if j==0 else None)
                ax.plot([j+offset-.07,j+offset+.07],[values.mean()]*2,color=color,lw=2)
        ax.set_xticks(range(4),LABELS)
        ax.set_title(title,fontsize=10)
        ax.set_yscale('log')
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Train-fold response geometry: 20261020 condition means only\nEach dot is one outer training fold; no decoder or threshold was changed',fontsize=12)
    fig.savefig(H/'separation_geometry.png')
    plt.close(fig)

    # Actual measured condition means; centering highlights line shape, not model output.
    gm=np.array(group_means)
    fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    names=[f'Y={Y:g}, T={T:g}' for Y,T in groups]
    for col,(line,off) in enumerate([('Upper',0),('Lower',24)]):
        vals=gm[:,off:off+24]
        for row in range(2):
            shown=vals if row==0 else vals-vals.mean(axis=1,keepdims=True)
            lim=np.max(abs(shown))
            im=axes[row,col].imshow(shown,aspect='auto',cmap='coolwarm',vmin=-lim,vmax=lim,interpolation='nearest')
            axes[row,col].axvline(11.5,color='black',lw=2)
            axes[row,col].set_yticks(range(20),names,fontsize=8)
            xx=lines['x_centers'][off:off+24]
            axes[row,col].set_xticks([5.5,17.5],[f'Left x={int(xx[0])}-{int(xx[11])}',f'Right x={int(xx[12])}-{int(xx[23])}'],fontsize=8)
            axes[row,col].set_xlabel('12 blocks per side; block-center x coordinates (pixels)')
            axes[row,col].set_title(f'{line}: '+('measured vertical displacement' if row==0 else 'same line mean subtracted (spatial shape)'))
            fig.colorbar(im,ax=axes[row,col],label='pixels',shrink=.8)
    fig.suptitle('Observed line shifts for every 20261020 pose condition (3-image mean)\nRows are command-sorted conditions, not a robot trajectory; lower panels subtract only each row mean',fontsize=12)
    fig.savefig(H/'observed_line_shifts.png')
    plt.close(fig)

    summary=[]
    for (name,metric),g in df.groupby(['observation','metric']):
        r=dict(observation=name,metric=metric)
        for col in ['cosine','axis_angle_degrees','normalized_jacobian_condition',
                    'Y_009_compensating_T','remaining_coordinate_RMS_px',
                    'remaining_projected_signal_px','repeat_projected_noise_SD_px',
                    'repeat_projected_signal_to_noise','condition_mean_projected_fit_RMS_px']:
            r[col+'_mean']=float(g[col].mean())
            r[col+'_min']=float(g[col].min())
            r[col+'_max']=float(g[col].max())
        summary.append(r)
    pd.DataFrame(summary).to_csv(H/'separation_summary.csv',index=False)
    protocol=dict(
        status='Retrospective development diagnosis. No model change or new candidate; no independent or real-robot validation.',
        inputs={str(p):sha(p) for p in inputs},
        excluded='Prior forbidden external five-ROI features/models are never loaded or imported.',
        fitting='For each existing outer fold, only 20261020 training images are used. Each 3-repeat command condition contributes one equally weighted mean. Fit X = intercept + Y*JY + T*JT by least squares.',
        inference_boundary='Command targets Y/T are used only for the offline train-fold forward-response fit and descriptive plots. No predictor/model is written or modified.',
        raw_metric='Every coordinate has equal weight: QR is 27 projected locations * dx/dy; each line is 24 normal-displacement samples. Coordinates are correlated and are not independent measurements.',
        family_equal_rms='Fixed, non-learned coordinate weights sqrt(F/(K*n_family)); QR, upper, lower each contribute equal family RMS. This is a sensitivity diagnostic, not a selected performance model. No per-column variance normalization is applied.',
        axis_normalization='Only the two fitted axis-response vectors are each normalized to unit length for cosine/condition number, avoiding arbitrary Y/T command scale in this diagnostic.',
        compensated_signal='For dY=+0.09, choose dT=-0.09*(JY dot JT)/(JT dot JT) under the stated metric. Residual is 0.09*JY+dT*JT. RMS is over coordinates; projected signal is its Euclidean norm.',
        noise='For each training image subtract its own training condition mean, project onto the unit compensated-residual direction; SD=sqrt(sum(projection^2)/(N-G)). Preserves empirical cross-feature correlation. Repeats may include mechanical pose variability, not just extraction noise.',
        SNR_limit='Signal/noise ratio describes this fitted direction under observed repeats; it is not a probability of satisfying 0.09, an error bound, or an independent-performance estimate.',
        linearization_limit='A single global forward linear response is fitted to available conditions; nonlinear/condition-dependent behavior and date drift may differ. A condition-mean projected fit residual is provided separately.',
        geometry_limit='Observed differences support complementary image sensitivity; unknown camera calibration, depths, rigidity, lighting effects, and T physical unit prohibit claiming a proven depth-based causal mechanism. Line horizontal motion is aperture-ambiguous.',
        figures='Observed-line figure uses all 20 condition means for descriptive visualization only. Upper/lower raw displacement and within-line mean-removed shapes are separated. No spatial feature selection is performed.',
        preserved={str(p):sha(p)==before[str(p)] for p in protected})
    assert all(protocol['preserved'].values())
    (H/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(df[df.metric=='raw_coordinate'].groupby('observation')[
        ['cosine','normalized_jacobian_condition','remaining_coordinate_RMS_px',
         'repeat_projected_noise_SD_px','repeat_projected_signal_to_noise']].mean().to_string())
    print('Protected files unchanged:',len(protected))


if __name__=='__main__':
    main()
