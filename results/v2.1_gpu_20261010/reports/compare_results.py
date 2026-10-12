"""Compare saved v2/v2.1 validation predictions; no model or test inference."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / 'recomputed'
OLD = ROOT / 'results/v2_baseline_20261009/toy_runs/hs_unet'
NEW = SCRIPT_DIR.parent / 'run'

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, values):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)

def aggregate(records):
    n=sum(r['n'] for r in records)
    result={'cases':len(records),'n':n}
    for v in ['old','new','baseline']:
        sse=sum(r[v+'_sse'] for r in records)
        result[v+'_rmse']=float(np.sqrt(sse/n))
    for v in ['old','new']:
        bias2=sum(r['n']*r[v+'_bias']**2 for r in records)/n
        mse=result[v+'_rmse']**2
        result[v+'_bias']=sum(r['n']*r[v+'_bias'] for r in records)/n
        result[v+'_case_bias_share']=bias2/mse
        result[v+'_oracle_centered_rmse']=float(np.sqrt(max(0,mse-bias2)))
        result[v+'_spatial_r_median']=float(np.median([r[v+'_spatial_r'] for r in records]))
        result[v+'_skill']=1-mse/result['baseline_rmse']**2
    result['rmse_improvement_pct']=100*(1-result['new_rmse']/result['old_rmse'])
    result['wins_vs_old']=sum(r['new_sse']<r['old_sse'] for r in records)
    result['wins_vs_baseline']=sum(r['new_sse']<r['baseline_sse'] for r in records)
    return result

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-archive',type=Path,help='Optional original dataset archive for region diagnostics')
    parser.add_argument('--output',type=Path,default=OUT,help='Recomputed artifacts directory')
    args=parser.parse_args()
    OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    old={r['case_id']:r for r in rows(OLD/'evaluation/validation/metrics_by_case.csv')}
    new={r['case_id']:r for r in rows(NEW/'evaluation/validation/metrics_by_case.csv')}
    assert old.keys()==new.keys()
    source={}
    if args.data_archive is not None:
        with tarfile.open(args.data_archive,'r|gz') as tar:
            for member in tar:
                cid=Path(member.name).stem
                if member.isfile() and member.name.endswith('.npz') and cid in new:
                    with np.load(io.BytesIO(tar.extractfile(member).read()),allow_pickle=False) as z:
                        source[cid]={'x':z['x'],'y':z['y'],'mask':z['mask']}
    checks={'cases':len(new),'target_max_difference_m':0.,'target_vs_original_dataset_max_difference_m':0. if source else None,'mask_mismatches':0,'csv_rmse_max_difference_m':0.,'csv_baseline_rmse_max_difference_m':0.}
    records=[]; fields={}; region_buffers={}
    for cid, nr in new.items():
        orow=old[cid]
        assert all(nr[k]==orow[k] for k in ['terrain_id','forcing_id','split','generalization'])
        with np.load(OLD/'evaluation/validation/predictions'/f'{cid}.npz',allow_pickle=False) as z:
            op,ot,om=z['prediction_hs_m'],z['target_hs_m'],z['valid_mask']
        with np.load(NEW/'evaluation/validation/predictions'/f'{cid}.npz',allow_pickle=False) as z:
            p,t,m=z['prediction_hs_m'],z['target_hs_m'],z['valid_mask'].astype(bool)
        ny,nx=p.shape;data=source.get(cid);b=data['x'][4,:ny,:nx]*5 if data is not None else None
        checks['target_max_difference_m']=max(checks['target_max_difference_m'],float(np.max(np.abs(t-ot))))
        if data is not None:
            checks['target_vs_original_dataset_max_difference_m']=max(checks['target_vs_original_dataset_max_difference_m'],float(np.max(np.abs(t-data['y'][0,:ny,:nx]*5))))
        checks['mask_mismatches']+=int(np.count_nonzero(m!=om))+(int(np.count_nonzero(m!=data['mask'][0,:ny,:nx])) if data is not None else 0)
        r={k:nr[k] for k in ['case_id','terrain_id','forcing_id','generalization']};r['n']=int(m.sum())
        for v,pred in [('old',op),('new',p)]+([('baseline',b)] if b is not None else []):
            err=pred[m].astype(float)-t[m]
            r[v+'_sse']=float(np.sum(err**2));r[v+'_rmse']=float(np.sqrt(np.mean(err**2)))
            if v!='baseline':
                r[v+'_bias']=float(err.mean());r[v+'_spatial_r']=float(np.corrcoef(pred[m],t[m])[0,1])
                checks['csv_rmse_max_difference_m']=max(checks['csv_rmse_max_difference_m'],abs(r[v+'_rmse']-float((nr if v=='new' else orow)['rmse_m'])))
        if b is None:
            assert float(nr['baseline_rmse_m']) == float(orow['baseline_rmse_m'])
            r['baseline_rmse']=float(nr['baseline_rmse_m'])
            r['baseline_sse']=r['baseline_rmse']**2*r['n']
        checks['csv_baseline_rmse_max_difference_m']=max(checks['csv_baseline_rmse_max_difference_m'],abs(r['baseline_rmse']-float(nr['baseline_rmse_m'])))
        records.append(r);fields[cid]=(op,p,t,m,b)
        if data is not None:
            depth=10*np.expm1(data['x'][0,:ny,:nx].astype(float)*np.log(11))
            iy,ix=np.indices(m.shape);edge=np.minimum.reduce([iy,ix,ny-1-iy,nx-1-ix])<5
            regions={'depth_lt20':m&(depth<20),'depth20to80':m&(depth>=20)&(depth<80),'depth_ge80':m&(depth>=80),'edge5':m&edge,'interior':m&~edge}
            for group in ['all',r['generalization'],r['forcing_id']]:
                for reg,mask in regions.items():
                    region_buffers.setdefault((group,reg),[]).append((op[mask],p[mask],t[mask],b[mask]))
    groups=[]
    for field in ['generalization','forcing_id']:
        for name in sorted({r[field] for r in records}):
            groups.append({'kind':field,'group':name,**aggregate([r for r in records if r[field]==name])})
    total_new=sum(r['new_sse'] for r in records);gain=sum(r['old_sse']-r['new_sse'] for r in records)
    for r in groups:
        selected=[c for c in records if c[r['kind']]==r['group']]
        r['new_sse_share']=sum(c['new_sse'] for c in selected)/total_new
        r['total_sse_reduction_share']=sum(c['old_sse']-c['new_sse'] for c in selected)/gain
    regions=[]
    for (group,reg),chunks in region_buffers.items():
        o,p,t,b=[np.concatenate([c[i] for c in chunks]).astype(float) for i in range(4)]
        regions.append({'group':group,'region':reg,'n':int(t.size),**{v+'_rmse':float(np.sqrt(np.mean((a-t)**2))) for v,a in [('old',o),('new',p),('baseline',b)]}})
    hs={}
    for name,folder in [('old',OLD),('new',NEW)]:
        h=rows(folder/'history.csv'); vals=np.array([float(r['validation_rmse_m']) for r in h]);train=np.array([5*np.sqrt(float(r['train_loss'])) for r in h])
        hs[name]={'epochs':len(h),'best_epoch':int(np.argmin(vals)+1),'best_rmse':float(vals.min()),'last_rmse':float(vals[-1]),'last_online_train_rmse':float(train[-1]),'last20_val_mean':float(vals[-20:].mean()),'last20_val_min':float(vals[-20:].min()),'last20_val_max':float(vals[-20:].max()),'epochs_better_than_baseline':int(np.sum(vals<aggregate(records)['baseline_rmse']))}
    summary=json.loads((NEW/'evaluation/validation/summary.json').read_text())
    checks['best_checkpoint_sha256_matches']=hashlib.sha256((NEW/'best.pt').read_bytes()).hexdigest()==summary['checkpoint']['sha256']
    configs=[json.loads((f/'run_config.json').read_text()) for f in [OLD,NEW]]
    checks['saved_model_config_equal']=configs[0]['model']==configs[1]['model']
    checks['saved_training_config_equal']=configs[0]['training']==configs[1]['training']
    checks['new_npz_count']=len(list((NEW/'evaluation/validation/predictions').glob('*.npz')))
    result={'checks':checks,'all':aggregate(records),'groups':groups,'history':hs,'regions':regions,'raw_dataset_verified':bool(source),'note':'Oracle debiasing uses targets; no new predictions or test evaluation. Without --data-archive, baseline metrics come from saved CSV and raw-depth regions are not recomputed.'}
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv('paired_cases.csv',records);write_csv('groups.csv',groups);
    if regions:write_csv('regions.csv',regions)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name,folder in [('v2',OLD),('v2.1',NEW)]:
        h=rows(folder/'history.csv');axes[0,0].plot([int(r['epoch']) for r in h],[float(r['validation_rmse_m']) for r in h],label=name)
    axes[0,0].axhline(result['all']['baseline_rmse'],ls='--',color='gray',label='boundary baseline');axes[0,0].set(title='Validation learning curves',xlabel='Epoch',ylabel='RMSE (m)');axes[0,0].legend()
    for ax,names,title in [(axes[0,1],['new_island','new_forcing','new_both'],'Generalization groups'),(axes[1,0],['F013','F014','F015'],'Unseen forcing IDs')]:
        lookup={r['group']:r for r in groups};xx=np.arange(len(names))
        for i,(v,label) in enumerate([('old','v2'),('new','v2.1'),('baseline','boundary baseline')]):
            ax.bar(xx+(i-1)*.25,[lookup[n][v+'_rmse'] for n in names],.25,label=label)
        ax.set(xticks=xx,xticklabels=names,title=title,ylabel='RMSE (m)');ax.legend(fontsize=8)
    for group,color in [('new_island','#277da1'),('new_forcing','#f8961e'),('new_both','#f94144')]:
        chosen=[r for r in records if r['generalization']==group]
        axes[1,1].scatter([r['old_rmse'] for r in chosen],[r['new_rmse'] for r in chosen],label=group,s=18,color=color)
    axes[1,1].plot([0,1],[0,1],'k--');axes[1,1].set(title='Matched cases: below diagonal is better',xlabel='v2 RMSE (m)',ylabel='v2.1 RMSE (m)');axes[1,1].legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    chosen=['IT000329','IT000338','IT000339'];fig,axes=plt.subplots(3,4,figsize=(14,10),layout='constrained')
    for i,cid in enumerate(chosen):
        o,p,t,m,b=fields[cid];vmax=max(t[m].max(),o[m].max(),p[m].max());emax=max(abs(p[m]-t[m]).max(),.1)
        for j,(a,label) in enumerate([(t,'SWAN'),(o,'v2'),(p,'v2.1'),(p-t,'v2.1 error')]):
            im=axes[i,j].imshow(np.ma.masked_where(~m,a),origin='lower',vmin=-emax if j==3 else 0,vmax=emax if j==3 else vmax,cmap='RdBu_r' if j==3 else 'viridis')
            axes[i,j].set_title(label);axes[i,j].set_xticks([]);axes[i,j].set_yticks([]);fig.colorbar(im,ax=axes[i,j],shrink=.8,label='m')
        axes[i,0].set_ylabel(cid+' / '+new[cid]['forcing_id'])
    fig.savefig(OUT/'matched_spatial.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='regions'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
