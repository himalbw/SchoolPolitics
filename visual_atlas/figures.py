"""Reproducible scientific figures; run after prepare.py and geography.py."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/stat236-mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,TwoSlopeNorm
from scipy.stats import t
from prepare import ROOT,PROJECT,THEMES
T=ROOT/'tables'; F=ROOT/'figures';F.mkdir(exist_ok=True)
LABELS={'equity_inclusion':'Equity / inclusion','race_racism':'Race / racism','book_restrictions':'Book restrictions','parents_rights':"Parents’ rights",'school_choice_vouchers':'School choice / vouchers','school_closures_reopening':'Closures / reopening'}
PRIMARY=list(LABELS); BLUE='#27647B'; ORANGE='#B45C36'; GRAY='#8B9195'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':12,'figure.facecolor':'white','axes.labelcolor':'#30383C','text.color':'#263238','savefig.facecolor':'white','pdf.fonttype':42})
CATALOG=[]
def save(fig,name,title,caption):
    fig.suptitle(title,x=.06,ha='left',fontsize=17,fontweight='bold',y=.99)
    fig.text(.06,.014,caption,ha='left',va='bottom',fontsize=8,color='#525C61',wrap=True)
    fig.tight_layout(rect=[.025,.085,.98,.94])
    for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=180)
    plt.close(fig);CATALOG.append(dict(name=name,title=title,caption=caption))

def load():
    c=pd.read_parquet(T/'cohort.parquet'); d=c.merge(pd.read_parquet(T/'phrase_counts_v2.parquet'),on='id',validate='one_to_one')
    d=d[~d.title_year_conflict].copy()
    for mode in ['all','no_covid']:
        for k in PRIMARY:d[k+'_'+mode+'_rate']=d[k+'_'+mode]/d['words_'+mode].replace(0,np.nan)*10000
    return c,d

def eligible_period(d):
    z=d[d.year.isin([2018,2019,2022,2023])]
    g=z.groupby(['leaid','year']).agg(n=('id','size'),q=('quarter','nunique'))
    ids=g[(g.n>=4)&(g.q>=3)].reset_index().groupby('leaid').year.nunique()
    return ids[ids.eq(4)].index

def paired(d,ids):
    out=[];diffs=[]
    for mode in ['all','no_covid']:
        z=d[d.leaid.isin(ids)&d.year.isin([2018,2019,2022,2023])&(d['words_'+mode]>0)]
        # Recheck coverage after COVID-block exclusion, avoiding missing-year averages.
        valid=eligible_period(z);z=z[z.leaid.isin(valid)]
        for k in PRIMARY:
            a=z.groupby(['leaid','year'])[k+'_'+mode+'_rate'].mean().unstack().reindex(columns=[2018,2019,2022,2023]).dropna()
            early=a[[2018,2019]].mean(1);late=a[[2022,2023]].mean(1);delta=late-early;n=len(delta)
            se=delta.std(ddof=1)/np.sqrt(n);crit=t.ppf(.975,n-1)
            out.append(dict(mode=mode,theme=k,n=n,early=early.mean(),late=late.mean(),change=delta.mean(),lo=delta.mean()-crit*se,hi=delta.mean()+crit*se))
            diffs.extend(dict(leaid=i,theme=k,mode=mode,early=early[i],late=late[i],change=delta[i]) for i in a.index)
    pd.DataFrame(out).to_csv(T/'theme_changes_v2.csv',index=False)
    pd.DataFrame(diffs).to_csv(T/'district_changes_v2.csv',index=False)
    return pd.DataFrame(out)

def coverage(c,d):
    qs=pd.period_range('2018Q1','2023Q4',freq='Q').astype(str)
    a=d.groupby(['leaid','quarter']).size().unstack(fill_value=0).reindex(columns=qs,fill_value=0)
    a=a.loc[a.sum(1).sort_values(ascending=False).index]
    a.to_csv(T/'district_quarter_coverage.csv')
    fig,(ax,bx)=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[4,1]})
    im=ax.imshow(np.ma.masked_equal(a.values,0),aspect='auto',interpolation='nearest',cmap='viridis',norm=LogNorm(vmin=1,vmax=a.values.max()))
    ax.set_facecolor('#E6E8EA');ax.set_xticks(range(0,24,4),[x[:4] for x in qs[::4]]);ax.set_ylabel(f'{len(a):,} districts, ordered by recording count');ax.set_xlabel('Recorded meeting quarter');fig.colorbar(im,ax=ax,label='Eligible recordings per district-quarter (log scale)',shrink=.8)
    bx.plot(range(24),(a>0).sum(),color=BLUE,marker='o',ms=3);bx.set_xticks(range(0,24,4),[x[:4] for x in qs[::4]]);bx.set_ylabel('Districts observed');bx.set_xlabel('Recorded meeting quarter');bx.grid(axis='y',alpha=.2)
    save(fig,'01_coverage','Where the six-year record is complete—and where it is not',f'Source: v2 primary cohort, 2018–2023. {len(d):,} recordings after {int(c.title_year_conflict.sum()):,} title-year conflicts excluded. Gray = no eligible recording; not zero discussion.')

def changes(s):
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,k in zip(axes.flat,PRIMARY):
        z=s[s.theme.eq(k)].set_index('mode')
        for i,(mode,color) in enumerate([('all',BLUE),('no_covid',ORANGE)]):
            r=z.loc[mode];ax.errorbar(r.change,i,xerr=[[r.change-r.lo],[r.hi-r.change]],fmt='o',color=color,capsize=4)
        ax.axvline(0,color=GRAY,lw=1);ax.set_yticks([0,1],['All passages','Outside COVID chunks']);ax.set_ylim(-.6,1.6);ax.set_title(LABELS[k]);ax.set_xlabel('Change in mentions / 10,000 core tokens');ax.grid(axis='x',alpha=.15)
    save(fig,'02_theme_changes','Which policy phrases became more frequent?',f'Source: v2; 2022–2023 minus 2018–2019; {int(s.n.min())}–{int(s.n.max())} eligible districts. Equal meetings/year/district weighting; pointwise 95% t intervals. Mentions, not stance; panels use different x scales.')

def trajectories(d):
    qs=pd.period_range('2018Q1','2023Q4',freq='Q').astype(str)
    counts=d.groupby(['leaid','quarter']).size().unstack().reindex(columns=qs)
    ids=counts.dropna().index
    z=d[d.leaid.isin(ids)]
    rows=[];fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,k in zip(axes.flat,PRIMARY):
        a=z.groupby(['leaid','quarter'])[k+'_all_rate'].mean().unstack().reindex(columns=qs)
        mean=a.mean();se=a.std()/np.sqrt(a.count());ci=t.ppf(.975,len(a)-1)*se
        ax.plot(range(24),mean,color=BLUE,lw=2);ax.fill_between(range(24),mean-ci,mean+ci,color=BLUE,alpha=.15)
        ax.set_title(LABELS[k]);ax.set_xticks(range(0,24,4),[x[:4] for x in qs[::4]]);ax.set_xlabel('Recorded meeting quarter');ax.set_ylabel('Mentions / 10,000 core tokens');ax.grid(axis='y',alpha=.2)
        for q in qs:rows.append(dict(theme=k,quarter=q,mean=mean[q],lo=mean[q]-ci[q],hi=mean[q]+ci[q],districts=len(a)))
    pd.DataFrame(rows).to_csv(T/'quarterly_trajectories.csv',index=False)
    pd.Series(ids,name='leaid').to_csv(T/'complete_quarter_panel.csv',index=False)
    save(fig,'03_quarterly_subjects','Six subjects across all 24 quarters',f'Source: v2, 2018–2023; fixed panel of {len(ids)} districts with ≥1 eligible recording every quarter. Equal district weight; pointwise district-t 95% intervals; shared shocks not modeled.')
    return ids

def cleaning():
    a=pd.read_csv(PROJECT/'analysis/lda_cleaning_comparison/comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,col,label in zip(axes,['npmi_mean','top10_diversity'],['Mean top-10 NPMI coherence','Unique fraction of top-10 topic words']):
        for arm,color,marker in [('before',GRAY,'o'),('after',BLUE,'s')]:
            z=a[a.arm.eq(arm)];ax.plot(z.k,z[col],marker=marker,color=color,label='Earlier preprocessing' if arm=='before' else 'Version-2 cleaning')
        ax.set_xticks([10,20,30]);ax.set_xlabel('Number of topics');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    save(fig,'04_cleaning_metrics','Cleaning changed the topics, but did not improve coherence','Source: completed paired LDA fits; same 83,366 recordings and split, 2018–2023. Shared-reference NPMI; one seed / training pass. Higher values indicate greater coherence or diversity.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,arm in zip(axes,['before','after']):
        z=a[a.arm.eq(arm)];ax.plot(z.k,z.completion_perplexity,'o-',color=BLUE);ax.set_xticks([10,20,30]);ax.set_xlabel('Number of topics');ax.set_ylabel('Held-out completion perplexity');ax.set_title('Earlier preprocessing' if arm=='before' else 'Version-2 cleaning');ax.grid(axis='y',alpha=.2)
    save(fig,'04b_perplexity','More topics improve prediction within each representation','Source: paired LDA, 2018–2023. Lower is better WITHIN each panel; vocabularies differ, so cross-panel perplexity levels do not measure the effect of cleaning.')
    m=pd.read_csv(PROJECT/'analysis/lda_cleaning_comparison/topic_matches.csv');m=m[m.k.eq(30)].sort_values('shared_vocab_cosine')
    fig,ax=plt.subplots(figsize=(12,11));ax.barh(range(len(m)),m.shared_vocab_cosine,color=BLUE)
    ax.set_yticks(range(len(m)),[f"Before {r.before_topic} → after {r.after_topic}: "+', '.join(r.after_terms.split(', ')[:3]) for r in m.itertuples()],fontsize=8)
    ax.set_xlim(0,1);ax.set_xlabel('Cosine similarity over shared vocabulary');ax.grid(axis='x',alpha=.15)
    save(fig,'04c_topic_matching','Some topics survive cleaning; others change substantially','Source: 30-topic paired fits. Forced one-to-one matching is descriptive; low similarity is not evidence of topic equivalence. Food-topic examples include ads; technology examples include repetition.')

def vocabulary():
    a=pd.read_csv(PROJECT/'analysis/eligibility_hc_2018_2019_vs_2022_2023/word_changes_primary.csv')
    defs=json.loads((PROJECT/'analysis/eligibility_hc_2018_2019_vs_2022_2023/cleaning_followup/definitions.json').read_text())
    covid=set(defs['expanded_covid']);art=set(defs['numeric_artifact']);policy={'equity','racism','racial','parental','banning','vouchers','diversity','inclusive'}
    fig,ax=plt.subplots(figsize=(12,7))
    a['category']=np.select([a.word.isin(covid),a.word.isin(art),a.word.isin(policy)],['Pandemic vocabulary','Numbers / possible artifacts','Selected policy terms'],default='Other words')
    for group,color,size in [('Other words','#B8BFC2',7),('Pandemic vocabulary',BLUE,28),('Numbers / possible artifacts',ORANGE,28),('Selected policy terms','#66558C',30)]:
        z=a[a.category.eq(group)];ax.scatter(z.change_per_10000,z.z.abs(),s=size,c=color,label=group,alpha=.65)
    # Limited labels, displaced deterministically; full values saved in source CSV.
    z=a[a.word.isin(['pandemic','covet','equity','racism','parental','foreign','zoom','vaccine'])]
    for i,r in enumerate(z.itertuples()):ax.annotate(r.word,(r.change_per_10000,abs(r.z)),xytext=(6,5+(i%3)*9),textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','lw':.5,'color':GRAY})
    ax.set_xscale('symlog',linthresh=.2);ax.axvline(0,color=GRAY,lw=.8);ax.set_xlabel('2022–2023 minus 2018–2019: modeled-word share / 10,000 (symmetric-log axis)');ax.set_ylabel('Absolute standardized change |Z|');ax.legend(frameon=False);ax.grid(alpha=.12)
    save(fig,'05_vocabulary_change','The vocabulary shift includes policy, pandemic and transcription signals','Source: EARLIER preprocessing, 273 matched districts, 11,804 tested words. Exploratory standardized evidence, not FDR-controlled discoveries. This is not a v2 HC rerun.')
    a.to_csv(T/'vocabulary_plot_data.csv',index=False)
    tests=pd.read_csv(PROJECT/'analysis/eligibility_hc_2018_2019_vs_2022_2023/test_comparison.csv')
    fig,ax=plt.subplots(figsize=(11,4));z=tests[tests.analysis.eq('primary')];rat=z.statistic/z.critical_95
    ax.barh(z.test,rat,color=BLUE);ax.axvline(1,color=ORANGE,ls='--',label='95% bootstrap threshold');ax.set_xscale('log');ax.set_xlabel('Observed statistic / its own 95% bootstrap critical value (log scale)');ax.legend(frameon=False)
    save(fig,'05b_global_tests','All three global tests detect vocabulary change','Source: earlier pipeline, 2018–2019 vs 2022–2023, 273 districts; 1,999 bootstrap draws. All p values reach 0.0005 floor. Ratios do not rank test power.')

def main():
    CATALOG.clear()
    c,d=load();coverage(c,d);s=paired(d,eligible_period(d));changes(s);trajectories(d);cleaning();vocabulary()
    (ROOT/'figure_catalog.json').write_text(json.dumps(CATALOG,indent=2))
    print('BASE FIGURES COMPLETE',len(CATALOG),flush=True)

if __name__=='__main__':main()
