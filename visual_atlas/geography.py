"""Exact-ID geographic joins, audited exclusions, and political-context figures."""
import sys,io,zipfile,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
import statsmodels.api as sm
from prepare import ROOT,PROJECT
sys.path.insert(0,str(ROOT/'vendor'))
import shapefile
import figures as v
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize,LogNorm,TwoSlopeNorm
import matplotlib.pyplot as plt
T=ROOT/'tables';E=ROOT/'external'

def shapes():
    z=zipfile.ZipFile(E/'counties.zip');base=next(n[:-4] for n in z.namelist() if n.endswith('.shp'))
    r=shapefile.Reader(**{k:io.BytesIO(z.read(base+'.'+k)) for k in ['shp','shx','dbf']})
    polygons=[];codes=[];names=[]
    for sr in r.iterShapeRecords():
        rec=sr.record.as_dict();names.append(rec)
        if rec['STATEFP'] in ['02','15','60','66','69','72','78']:continue
        pts=np.asarray(sr.shape.points);ends=list(sr.shape.parts)+[len(pts)]
        for a,b in zip(ends[:-1],ends[1:]):
            if b-a<3:continue
            # Spherical Albers equal-area conic; coordinates in radians / Earth-radius units.
            lon,lat=np.deg2rad(pts[a:b,0]),np.deg2rad(pts[a:b,1]);p1,p2=np.deg2rad([29.5,45.5]);n=(np.sin(p1)+np.sin(p2))/2
            C=np.cos(p1)**2+2*n*np.sin(p1);rho=np.sqrt(C-2*n*np.sin(lat))/n;rho0=np.sqrt(C-2*n*np.sin(np.deg2rad(37.5)))/n;theta=n*(lon-np.deg2rad(-96))
            polygons.append(np.c_[rho*np.sin(theta),rho0-rho*np.cos(theta)]);codes.append(rec['GEOID'])
    return polygons,codes,pd.DataFrame(names)

def map_ax(ax,values,polys,codes,cmap,norm):
    cm=plt.get_cmap(cmap);colors=[cm(norm(values[k])) if k in values and pd.notna(values[k]) else '#E4E6E7' for k in codes]
    pc=PolyCollection(polys,facecolors=colors,edgecolors='white',linewidths=.12);ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal');ax.axis('off')
    return plt.cm.ScalarMappable(norm=norm,cmap=cm)

def join():
    _,d=v.load()
    x=pd.read_csv(E/'district_county_2020.csv',dtype={'LEAID':str,'STCOUNTY':str}).rename(columns={'LEAID':'leaid','STCOUNTY':'county_fips'})
    x['leaid']=x.leaid.str.zfill(7);x['county_fips']=x.county_fips.str.zfill(5)
    x=x.drop_duplicates(['leaid','county_fips']);n=x.groupby('leaid').county_fips.nunique()
    one=x[x.leaid.isin(n[n.eq(1)].index)]
    assert one.leaid.is_unique
    audit=d[['leaid']].drop_duplicates().merge(n.rename('county_count'),on='leaid',how='left')
    audit['status']=np.select([audit.county_count.isna(),audit.county_count.gt(1)],['No 2020 crosswalk match','Multiple counties'],default='Single county')
    audit.to_csv(T/'geography_match_audit.csv',index=False);x.to_csv(T/'district_county_all_relationships.csv',index=False)
    e=pd.read_csv(E/'county_returns_2000_2016.csv',dtype={'FIPS':str,'fips':str})
    e.columns=e.columns.str.lower();e=e[e.year.eq(2016)&e.party.str.lower().isin(['democrat','republican'])].copy()
    e['county_fips']=e.fips.str.replace(r'\.0$','',regex=True).str.zfill(5)
    e['candidatevotes']=pd.to_numeric(e.candidatevotes,errors='raise')
    wide=e.pivot_table(index='county_fips',columns='party',values='candidatevotes',aggfunc='sum')
    wide['rep_share_2016']=wide['republican']/(wide['republican']+wide['democrat'])
    wide=wide[wide.index.str.fullmatch(r'\d{5}',na=False)&wide.rep_share_2016.notna()]
    assert wide.rep_share_2016.between(0,1).all()
    one=one.merge(wide[['rep_share_2016']],on='county_fips',how='left',validate='many_to_one')
    one['state_fips']=one.county_fips.str[:2]
    one.to_csv(T/'single_county_district_elections.csv',index=False)
    polys,codes,names=shapes();state_names=names[['STATEFP','STUSPS']].drop_duplicates().set_index('STATEFP').STUSPS
    dif=pd.read_csv(T/'district_changes_v2.csv',dtype={'leaid':str});dif=dif[dif['mode'].eq('all')]
    linked=dif.merge(one,on='leaid',validate='many_to_one')
    linked.to_csv(T/'district_policy_election_changes.csv',index=False)
    stats={'all_eligible_districts':d.leaid.nunique(),'geography_status':audit.status.value_counts().to_dict(),'period_panel_districts':dif.leaid.nunique(),'single_county_period_districts':linked.leaid.nunique(),'single_county_period_counties':linked.county_fips.nunique(),'period_districts_with_elections':linked[linked.rep_share_2016.notna()].leaid.nunique(),'crosswalk_year':2020,'election_year':2016}
    (T/'geography_summary.json').write_text(json.dumps(stats,indent=2))
    return d,x,one,linked,polys,codes,state_names,wide

def maps(d,one,linked,polys,codes,wide):
    counts=d.merge(one[['leaid','county_fips']],on='leaid').groupby('county_fips').id.size()
    fig,axes=plt.subplots(2,1,figsize=(13,12))
    im=map_ax(axes[0],counts.to_dict(),polys,codes,'viridis',LogNorm(1,counts.max()));axes[0].set_title('Eligible recordings represented by single-county districts');fig.colorbar(im,ax=axes[0],shrink=.65,label='Recordings, 2018–2023 (log scale)')
    im=map_ax(axes[1],wide.rep_share_2016.to_dict(),polys,codes,'RdBu_r',Normalize(0,1));axes[1].set_title('2016 Republican share of Democratic + Republican presidential votes');fig.colorbar(im,ax=axes[1],shrink=.65,label='Republican two-party vote share')
    v.save(fig,'06a_coverage_and_voting_maps','Geographic coverage and baseline political context','Sources: v2 / NCES 2020 district–county links / Census 2020 boundaries / MEDSL 2016 returns. Contiguous US shown. Gray = no linked observations or missing votes; not zero. County votes are not district or speaker votes.')
    a=linked.groupby(['county_fips','theme']).agg(change=('change','mean'),districts=('leaid','nunique')).reset_index();a.to_csv(T/'county_theme_changes.csv',index=False)
    fig,axes=plt.subplots(3,2,figsize=(15,14))
    for ax,k in zip(axes.flat,v.PRIMARY):
        z=a[a.theme.eq(k)].set_index('county_fips');lim=max(abs(z.change).max(),.001)
        im=map_ax(ax,z.change.to_dict(),polys,codes,'RdBu_r',TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim));ax.set_title(v.LABELS[k]);fig.colorbar(im,ax=ax,shrink=.65,label='Change: mentions / 10,000 core tokens')
    v.save(fig,'06b_county_subject_changes','Where subject mentions changed in the observed panel','Source: v2 matched-period panel; 2022–2023 minus 2018–2019; mean across linked single-county districts. Contiguous US; gray = no eligible panel district. Different color scales; no population weighting or county-wide representativeness.')

def state_changes(d,x,state_names):
    # Multi-county districts can enter this state summary if all linked counties are in one state.
    a=x.assign(state_fips=x.county_fips.str[:2])[['leaid','state_fips']].drop_duplicates();n=a.groupby('leaid').state_fips.nunique();a=a[a.leaid.isin(n[n.eq(1)].index)]
    dif=pd.read_csv(T/'district_changes_v2.csv',dtype={'leaid':str});dif=dif[dif['mode'].eq('all')].merge(a,on='leaid',validate='many_to_one')
    rows=[]
    for (s,k),z in dif.groupby(['state_fips','theme']):
        n=len(z);se=z.change.std()/np.sqrt(n);ci=t.ppf(.975,n-1)*se if n>1 else np.nan
        rows.append(dict(state=state_names.get(s,s),theme=k,n=n,change=z.change.mean(),lo=z.change.mean()-ci,hi=z.change.mean()+ci))
    a=pd.DataFrame(rows);a.to_csv(T/'state_theme_changes.csv',index=False)
    keep=a.groupby('state').n.min();states=keep[keep>=3].index.tolist()
    fig,axes=plt.subplots(2,3,figsize=(14,11))
    for ax,k in zip(axes.flat,v.PRIMARY):
        z=a[a.theme.eq(k)&a.state.isin(states)].set_index('state').reindex(states)
        ax.errorbar(z.change,range(len(z)),xerr=[z.change-z.lo,z.hi-z.change],fmt='o',color=v.BLUE,ms=3,capsize=2);ax.axvline(0,color=v.GRAY,lw=.8);ax.set_yticks(range(len(z)),[f'{s} (n={int(z.loc[s,"n"])})' for s in states],fontsize=8);ax.set_title(v.LABELS[k]);ax.set_xlabel('Change in mentions / 10,000 tokens');ax.grid(axis='x',alpha=.15)
    v.save(fig,'06c_state_subject_changes','Within-state summaries of the observed district changes','Source: v2, 2022–2023 minus 2018–2019; NCES 2020 state linkage. States with ≥3 panel districts shown. Equal district weight; pointwise t intervals, no adjustment for shared state shocks; not state population estimates.')

def scatter(linked):
    z=linked.dropna(subset=['rep_share_2016']).copy();rows=[]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,k in zip(axes.flat,v.PRIMARY):
        a=z[z.theme.eq(k)];ax.scatter(a.rep_share_2016*100,a.change,s=18,color=v.BLUE,alpha=.45,edgecolors='none')
        X=sm.add_constant(a.rep_share_2016)
        fit=sm.OLS(a.change,X).fit(cov_type='cluster',cov_kwds={'groups':a.state_fips},use_t=True)
        xx=np.linspace(a.rep_share_2016.min(),a.rep_share_2016.max(),100);xp=np.c_[np.ones(100),xx];y=xp@fit.params;se=np.sqrt(np.einsum('ij,jk,ik->i',xp,fit.cov_params(),xp));crit=t.ppf(.975,a.state_fips.nunique()-1)
        ax.plot(xx*100,y,color=v.ORANGE);ax.fill_between(xx*100,y-crit*se,y+crit*se,color=v.ORANGE,alpha=.15);ax.axhline(0,color=v.GRAY,lw=.8)
        ax.set_title(v.LABELS[k]);ax.set_xlabel('2016 county Republican two-party vote (%)');ax.set_ylabel('Change in mentions / 10,000 core tokens');ax.grid(alpha=.12)
        ci=fit.conf_int().iloc[1];rows.append(dict(theme=k,n_districts=len(a),n_counties=a.county_fips.nunique(),n_states=a.state_fips.nunique(),slope_per_10_percentage_points=fit.params.iloc[1]*.1,ci_low=ci.iloc[0]*.1,ci_high=ci.iloc[1]*.1,p=fit.pvalues.iloc[1]))
    pd.DataFrame(rows).to_csv(T/'political_associations_state_clustered.csv',index=False)
    v.save(fig,'07_political_context_scatter','Do changes differ with the surrounding political environment?',f'Source: v2 early/late contrasts + MEDSL 2016; {z.leaid.nunique()} single-county districts. Points = districts; OLS mean bands use state-clustered 95% intervals. Unadjusted exploratory associations; not causal or speaker-level ideology.')

def political_time(d,one):
    ids=pd.read_csv(T/'complete_quarter_panel.csv',dtype=str).leaid
    z=d[d.leaid.isin(ids)].merge(one[['leaid','county_fips','state_fips','rep_share_2016']],on='leaid',validate='many_to_one').dropna(subset=['rep_share_2016'])
    z['political_group']=pd.cut(z.rep_share_2016,[-.001,.4,.6,1.001],labels=['Below 40% Republican','40–60% Republican','Above 60% Republican'],right=False)
    qs=pd.period_range('2018Q1','2023Q4',freq='Q').astype(str);colors=[v.BLUE,v.GRAY,v.ORANGE];fig,axes=plt.subplots(2,3,figsize=(15,9));rows=[]
    groupcounts=z.groupby('political_group',observed=True).leaid.nunique()
    for ax,k in zip(axes.flat,v.PRIMARY):
        for (g,color) in zip(z.political_group.cat.categories,colors):
            a=z[z.political_group.eq(g)].groupby(['leaid','state_fips','quarter'])[k+'_all_rate'].mean().reset_index()
            if a.empty:continue
            wide=a.pivot(index=['leaid','state_fips'],columns='quarter',values=k+'_all_rate').reindex(columns=qs)
            assert wide.notna().all().all()
            bystate=wide.groupby(level='state_fips');sums=bystate.sum();ns=bystate.size();S=len(ns);rng=np.random.default_rng(236);weights=rng.multinomial(S,np.full(S,1/S),size=499)
            boot=(weights@sums.to_numpy())/(weights@ns.to_numpy())[:,None];lo,hi=np.quantile(boot,[.025,.975],axis=0);mean=wide.mean().to_numpy()
            ax.plot(range(24),mean,color=color,label=f'{g} (n={len(wide)})',lw=1.8);ax.fill_between(range(24),lo,hi,color=color,alpha=.09)
            rows.extend(dict(theme=k,group=str(g),quarter=q,mean=mean[i],lo=lo[i],hi=hi[i],districts=len(wide),states=S) for i,q in enumerate(qs))
        ax.set_title(v.LABELS[k]);ax.set_xlabel('Recorded meeting quarter');ax.set_ylabel('Mentions / 10,000 core tokens');ax.set_xticks(range(0,24,4),[x[:4] for x in qs[::4]]);ax.grid(axis='y',alpha=.15)
    axes.flat[0].legend(frameon=False,fontsize=8)
    pd.DataFrame(rows).to_csv(T/'quarterly_political_groups.csv',index=False)
    v.save(fig,'08_political_trajectories','Subject trajectories by pre-period county voting',f'Source: v2 / MEDSL 2016; {z.leaid.nunique()} linked districts present every quarter, 2018–2023. Fixed groups; equal district weight; 499 within-group state-cluster bootstrap draws, pointwise 95% bands. Descriptive, no covariate adjustment.')

def main():
    v.CATALOG.clear()
    d,x,one,linked,polys,codes,state_names,wide=join()
    maps(d,one,linked,polys,codes,wide);state_changes(d,x,state_names);scatter(linked);political_time(d,one)
    old=json.loads((ROOT/'figure_catalog.json').read_text());old=[a for a in old if not a['name'].startswith(('06','07','08'))]
    (ROOT/'figure_catalog.json').write_text(json.dumps(old+v.CATALOG,indent=2));print('GEOGRAPHIC FIGURES COMPLETE',flush=True)

if __name__=='__main__':main()
