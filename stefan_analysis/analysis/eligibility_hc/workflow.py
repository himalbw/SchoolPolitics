from pathlib import Path
import os,re,json,hashlib,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.special import ndtr
import joblib

ROOT=Path(__file__).resolve().parent
RAW=Path(os.environ.get('STAT236_RAW_DIR', Path(__file__).resolve().parents[2] / 'data' / 'raw')).expanduser().resolve()
LDA=ROOT.parent/'lda_2018_2023'
SEED=236
YEARS=[2018,2019,2022,2023]

def save(name,obj):
    (ROOT/name).write_text(json.dumps(obj,indent=2,default=str))

def audit():
    m=pd.read_csv(LDA/'cleaning_manifest.csv',dtype={'leaid':str,'id':str})
    m=m[m.status.eq('retained')].copy();m['leaid']=m.leaid.str.zfill(7)
    raw=pd.read_parquet(RAW/'meta_districtview.parquet',columns=['id','channel_id'])
    m=m.merge(raw,on='id',validate='one_to_one')
    title=m.title.str.lower().fillna('')
    school=r'school|education|\bboe\b|\bboard of ed\b|\b[a-z]*isd\b|\b[a-z]+usd\b|\b[a-z]+csd\b|\brsu\s*\d|\bmsad\s*\d|\bsau\s*\d'
    body=r'board|committee|\bboe\b|council|governance'
    other=r'zoning board|zoning advisory|zoning (?:and|&) planning|planning board|planning commission|city council|town council|borough council|selectmen|select board|board of (?:finance|health|elections|assessors|supervisors|commissioners|adjustments|parks)|county commission|conservation commission|water board|sewer board|library board|housing authority|land use committee|charter review committee'
    event=r'graduation|commencement|concert|football game|basketball game|soccer game|promotional'
    m['explicit_school_title']=title.str.contains(school,regex=True)&title.str.contains(body,regex=True)
    m['other_body_title']=title.str.contains(other,regex=True)
    m['event_title']=title.str.contains(event,regex=True)&~title.str.contains(body,regex=True)
    m['generic_governance_title']=title.str.contains(body,regex=True)
    m['clip_or_campaign_title']=title.str.contains(r'candidate.{0,30}(?:debate|forum)|(?:debate|forum).{0,30}candidate',regex=True)|(title.str.contains(r'report|update|presentation',regex=True)&~title.str.contains(r'meeting|session|hearing|committee',regex=True))
    channel=m.groupby('channel_id').agg(videos=('id','size'),explicit=('explicit_school_title','sum'),other=('other_body_title','sum'))
    channel['explicit_fraction']=channel.explicit/channel.videos
    m=m.join(channel[['explicit','explicit_fraction','other']],on='channel_id',rsuffix='_channel')
    prefixes={}
    edwords=['school','schools','student','students','teacher','teachers','education','superintendent','curriculum']
    for p in sorted((LDA/'chunks').glob('*.parquet')):
        for b in pq.ParquetFile(p).iter_batches(batch_size=4096,columns=['id','chunk','text']):
            f=b.to_pandas();f=f[f.chunk.eq(0)]
            for r in f.itertuples():
                words=r.text.split()[:600];cnt={w:words.count(w) for w in edwords}
                prefixes[r.id]=(sum(cnt.values()),sum(v>0 for v in cnt.values()),' '.join(words)[:3500])
        print('Audit prefix scan',p.name,flush=True)
    m['education_mentions']=m.id.map(lambda x:prefixes[x][0]);m['education_distinct_terms']=m.id.map(lambda x:prefixes[x][1])
    m['direct_school_body_prefix']=m.id.map(lambda x:bool(re.search(r'school board|board of education|school committee|school district.{0,50}board|public schools.{0,40}board|board.{0,50}school district',prefixes[x][2])))
    corroborated=m.generic_governance_title&(m.explicit>=10)&(m.explicit_fraction>=.5)&(m.education_mentions>=3)&(m.education_distinct_terms>=2)
    corroborated=corroborated|(m.generic_governance_title&m.direct_school_body_prefix)
    m['audit_status']=np.select([m.clip_or_campaign_title,m.other_body_title&m.explicit_school_title,m.other_body_title,m.event_title,m.explicit_school_title,corroborated],['review_possible_clip_or_campaign','review_joint_bodies','exclude_other_body','exclude_standalone_event','include_explicit_title','include_corroborated'],default='review_ambiguous')
    m['primary_eligible']=m.audit_status.str.startswith('include_');m['strict_eligible']=m.audit_status.eq('include_explicit_title')
    m.to_csv(ROOT/'eligibility_audit.csv',index=False)
    channel.to_csv(ROOT/'channel_audit.csv')
    # Reproducible stratified manual evidence review, not a census of manual labels.
    sizes={'include_explicit_title':20,'include_corroborated':20,'review_possible_clip_or_campaign':10,'exclude_other_body':15,'exclude_standalone_event':10,'review_joint_bodies':10,'review_ambiguous':25}
    samples=[]
    for status,n in sizes.items():
        f=m[m.audit_status.eq(status)].sample(n=min(n,int(m.audit_status.eq(status).sum())),random_state=SEED+1)
        for r in f.to_dict('records'):
            samples.append({k:r[k] for k in ['id','title','leaid','year','audit_status']}|{'prefix':prefixes[r['id']][2]})
    save('manual_review_candidates.json',samples)
    save('audit_summary.json',{'records':len(m),'statuses':m.audit_status.value_counts().to_dict(),'primary_eligible':int(m.primary_eligible.sum()),'strict_eligible':int(m.strict_eligible.sum()),'scope':'Automated title/channel/transcript-prefix screening of every previously retained 2018–2023 video, plus a reproducible stratified manual text review. Not full manual adjudication. Unresolved cases are excluded from primary inference.'})
    print(m.audit_status.value_counts().to_string(),flush=True)

def aggregate():
    m=pd.read_csv(ROOT/'eligibility_audit.csv',dtype={'id':str,'leaid':str});m=m[m.year.isin(YEARS)&m.primary_eligible].copy()
    m['date']=pd.to_datetime(m.meeting_date);m['quarter']=m.date.dt.quarter
    # Require at least 4 videos and 3 covered quarters in every comparison year.
    def panel(f):
        a=f.groupby(['leaid','year']).agg(n=('id','size'),quarters=('quarter','nunique'))
        good=a[(a.n>=4)&(a.quarters>=3)].reset_index().groupby('leaid').year.nunique()
        return sorted(good[good.eq(4)].index)
    primary=panel(m);strict=panel(m[m.strict_eligible]);ids=set(primary)|set(strict)
    m=m[m.leaid.isin(ids)];m.to_csv(ROOT/'analysis_videos.csv',index=False)
    vec=joblib.load(LDA/'vectorizer.joblib');terms=vec.get_feature_names_out().astype(str);p=len(terms)
    keys=sorted(set(zip(m.leaid,m.year)));lookup={k:i for i,k in enumerate(keys)}
    sums={mode:np.zeros((len(keys),p),dtype=np.float64) for mode in ['primary','strict']}
    nums={mode:np.zeros(len(keys),dtype=np.int64) for mode in sums}
    mi=m.set_index('id');allowed=set(mi.index);empty=[]
    for file in sorted((LDA/'chunks').glob('*.parquet')):
        rows=[];vids=[]
        for b in pq.ParquetFile(file).iter_batches(batch_size=4096,columns=['id','text']):
            f=b.to_pandas();f=f[f.id.isin(allowed)]
            if len(f):rows.append(vec.transform(f.text));vids.extend(f.id)
        if not rows:continue
        X=sp.vstack(rows,format='csr');unique,inv=np.unique(vids,return_inverse=True)
        G=sp.csr_matrix((np.ones(len(inv)),(inv,np.arange(len(inv)))),shape=(len(unique),len(inv)))
        V=G@X;total=np.asarray(V.sum(axis=1)).ravel();keep=total>0
        empty.extend(unique[~keep]);unique=unique[keep];V=V[keep];total=total[keep]
        # Equal video weight within year; composition over fixed modeled vocabulary.
        V=sp.diags(1/total)@V
        meta=mi.loc[unique];ix=np.array([lookup[(r.leaid,r.year)] for r in meta.itertuples()])
        for mode in sums:
            ok=np.ones(len(unique),bool) if mode=='primary' else meta.strict_eligible.to_numpy(bool)
            H=sp.csr_matrix((np.ones(ok.sum()),(ix[ok],np.where(ok)[0])),shape=(len(keys),len(unique)))
            sums[mode]+=(H@V).toarray();np.add.at(nums[mode],ix[ok],1)
        print('Aggregated',file.name,'videos',len(unique),flush=True)
    # Write sufficient statistics for compact, fast notebook reruns.
    out={'terms':terms,'district_year_keys':np.array([f'{d}_{y}' for d,y in keys])}
    for mode,districts in [('primary',primary),('strict',strict)]:
        avg=sums[mode]/np.maximum(nums[mode][:,None],1)
        early=np.array([(avg[lookup[(d,2018)]]+avg[lookup[(d,2019)]])/2 for d in districts])
        late=np.array([(avg[lookup[(d,2022)]]+avg[lookup[(d,2023)]])/2 for d in districts])
        presence=np.array([sum(sums[mode][lookup[(d,y)]] for y in YEARS)>0 for d in districts])
        out.update({mode+'_early':early,mode+'_late':late,mode+'_presence':presence,mode+'_districts':np.array(districts)})
    assert not empty, 'Recompute coverage after removing zero-vocabulary videos'
    for mode in ['primary','strict']:
        assert np.allclose(out[mode+'_early'].sum(axis=1),1) and np.allclose(out[mode+'_late'].sum(axis=1),1)
    np.savez_compressed(ROOT/'district_profiles.npz',**out)
    m.groupby(['year','strict_eligible']).agg(videos=('id','size'),districts=('leaid','nunique')).to_csv(ROOT/'panel_coverage.csv')
    save('aggregation.json',{'primary_districts':len(primary),'strict_districts':len(strict),'video_records':len(m),'empty_videos':empty,'years':YEARS,'weighting':'equal videos within district-year, equal years within period, equal districts in contrasts','panel_rule':'At least 4 primary/strict-eligible videos and 3 distinct calendar quarters in each of 2018, 2019, 2022, 2023','vocabulary':'Fixed existing LDA training vocabulary of 12000 terms; no new outcome-based vocabulary tuning. Not an independent confirmation of earlier exploratory topic discoveries.'})

def stats(Z):
    Z=np.atleast_2d(Z);p=Z.shape[1];pv=2*ndtr(-np.abs(Z));s=np.sort(pv,axis=1);i=np.arange(1,p+1)/p
    mask=(s>1/p)&(i<=.5)
    hc=np.sqrt(p)*(i-s)/np.sqrt(np.maximum(s*(1-s),1e-300))
    hc=np.max(np.where(mask,hc,0),axis=1)
    return np.column_stack([hc,np.max(np.abs(Z),axis=1),np.sum(Z**2,axis=1)])

def compare(B=1999):
    data=np.load(ROOT/'district_profiles.npz');terms=data['terms'];outputs=[]
    for mode in ['primary','strict','primary_common_panel','primary_artifact_sensitivity']:
        base='strict' if mode=='strict' else 'primary'
        select=np.isin(data['primary_districts'],data['strict_districts']) if mode=='primary_common_panel' else np.ones(len(data[base+'_districts']),bool)
        a=data[base+'_early'][select];b=data[base+'_late'][select];D=b-a;n=len(D)
        assert n>=30, 'Insufficient independent district units'
        presence=data[base+'_presence'][select]
        sd=D.std(axis=0,ddof=1)
        # Frequency filter is symmetric in periods, and excludes zero-variance words.
        keep=(presence.sum(axis=0)>=max(30,int(np.ceil(.1*n))))&(sd>1e-10)&~np.isin(terms,['th','nd','rd','st'])
        if mode=='primary_artifact_sensitivity':keep &= ~np.isin(terms,['foreign','music','applause','inaudible','laughter'])
        D=D[:,keep];a=a[:,keep];b=b[:,keep];words=terms[keep];sd=sd[keep]
        z=D.mean(axis=0)/(sd/np.sqrt(n));R=(D-D.mean(axis=0))/sd
        observed=stats(z)[0];rng=np.random.default_rng(SEED)
        null=[]
        for start in range(0,B,100):
            g=rng.standard_normal((min(100,B-start),n))
            null.append(stats((g@R)/np.sqrt(n-1)))
        null=np.vstack(null);pvals=(1+(null>=observed).sum(axis=0))/(B+1)
        for j,name in enumerate(['Higher Criticism (HC+)','Minimum p-value / max |Z|','Sum of squared Z']):
            outputs.append(dict(analysis=mode,test=name,districts=n,words=len(words),statistic=observed[j],bootstrap_p=pvals[j],critical_95=np.quantile(null[:,j],.95),replicates=B))
        pd.DataFrame({'word':words,'early_frequency':a.mean(axis=0),'late_frequency':b.mean(axis=0),'change_per_10000':D.mean(axis=0)*10000,'se_per_10000':sd/np.sqrt(n)*10000,'z':z,'normal_approx_p':2*ndtr(-np.abs(z))}).sort_values('z',key=lambda s:s.abs(),ascending=False).to_csv(ROOT/f'word_changes_{mode}.csv',index=False)
        np.savez_compressed(ROOT/f'test_null_{mode}.npz',statistics=null,z=z,words=words)
        if mode=='primary':
            # Independent conditional-Gaussian replicates: a diagnostic, not proof for real data.
            sims=[];rng2=np.random.default_rng(SEED+1);crit=np.quantile(null,.95,axis=0)
            nrep=300;noise=(rng2.standard_normal((nrep,n))@R)/np.sqrt(n-1)
            for signals in [0,10,100,1000]:
                for strength in ([0] if signals==0 else [1,2,3]):
                    zz=noise.copy();indices=rng2.choice(len(words),size=min(signals,len(words)),replace=False)
                    if signals:zz[:,indices]+=strength*rng2.choice([-1,1],len(indices))
                    power=(stats(zz)>crit).mean(axis=0)
                    for j,name in enumerate(['HC+','Min-P','Sum squares']):sims.append(dict(signals=len(indices),effect_z=strength,test=name,rejection_rate=power[j],replicates=nrep))
            pd.DataFrame(sims).to_csv(ROOT/'gaussian_power_simulation.csv',index=False)
        print(mode,'n',n,'p',len(words),'p-values',pvals,flush=True)
    pd.DataFrame(outputs).to_csv(ROOT/'test_comparison.csv',index=False)
    save('inference_settings.json',{'seed':SEED,'bootstrap_replicates':B,'calibration':'Null-centered Gaussian district multiplier bootstrap; one multiplier shared across every word in a district; plug-in sample covariance, unit marginal variance. Fixed standard errors. Conditional Gaussian calibration is approximate for the actual district differences.','assumptions':'District contrast vectors behave approximately as independent sampling units with adequate moments and sufficient district support for retained words. Within-district dependence and cross-word covariance are retained in the contrast vector. Cross-district shared shocks are not fully accommodated. Results are descriptive period associations, not causal effects.','HC':'Two-sided normal marginal p-values; ranks up to p/2 and p_(i)>1/p; negative/empty HC maximum set to zero. Bootstrap repeats exactly the same statistic.','family':'Three global tests of the same predeclared period contrast; report all, do not select whichever rejects. Strict-title panel is a sensitivity analysis.'})

if __name__=='__main__':
    import sys
    {'audit':audit,'aggregate':aggregate,'compare':compare}[sys.argv[1]]()
