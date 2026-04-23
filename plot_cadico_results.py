#!/usr/bin/env python3
"""CADiCo vs DiCo — plotting script."""
import argparse, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","axes.grid":True,
    "grid.alpha":0.3,"axes.spines.top":False,"axes.spines.right":False,"font.size":12,
    "axes.labelsize":13,"axes.titlesize":14,"legend.fontsize":10,"figure.dpi":150})

BLUE="#3b82f6"; GREEN="#10b981"; RED="#ef4444"; ORANGE="#f59e0b"; PURPLE="#8b5cf6"; GRAY="#94a3b8"
DC={"snd_0p3":"#93c5fd","snd_0p6":"#60a5fa","snd_0p8":"#3b82f6","snd_1":"#2563eb","snd_1p2":"#1d4ed8"}
CC={"snd_0p3":"#fca5a5","snd_0p6":"#f87171","snd_0p8":"#ef4444","snd_1":"#dc2626","snd_1p2":"#b91c1c"}
AC={"alpha_0":GRAY,"alpha_0p1":"#fca5a5","alpha_0p2":"#f87171","alpha_0p5":RED,"alpha_0p5":"#dc2626","alpha_1p0":ORANGE}
SL={"snd_0p3":"SND=0.3","snd_0p6":"SND=0.6","snd_0p8":"SND=0.8","snd_1":"SND=1.0","snd_1p2":"SND=1.2",
    "homogeneous":"SND=0","unconstrained":"Unconstrained"}
SV={"snd_0p3":0.3,"snd_0p6":0.6,"snd_0p8":0.8,"snd_1":1.0,"snd_1p2":1.2}
SK=["snd_0p3","snd_0p6","snd_0p8","snd_1","snd_1p2"]
RW="collection_agents_reward_episode_reward_mean.csv"

def fsd(rd):
    for p in ["ippo_*/ippo_*/scalars","*/ippo_*/ippo_*/scalars"]:
        c=glob.glob(os.path.join(rd,p))
        if c: return c[0]
    return None
def lc(fp):
    if not os.path.exists(fp): return None,None
    d=np.loadtxt(fp,delimiter=",");
    if d.ndim==1: d=d.reshape(1,-1)
    return d[:,0].astype(int),d[:,1]
def lm(bd,sk,mc,sd=None):
    sp=os.path.join(bd,sk,sd) if sd else os.path.join(bd,sk)
    if not os.path.exists(sp): return None,None
    sds=sorted(glob.glob(os.path.join(sp,"seed_*")))
    if not sds: return None,None
    av,fr=[],None
    for s in sds:
        sc=fsd(s)
        if sc is None: continue
        fs,fv=lc(os.path.join(sc,"counters_total_frames.csv"))
        ms,mv=lc(os.path.join(sc,mc))
        if fs is None or ms is None: continue
        cm=np.intersect1d(fs,ms)
        if len(cm)==0: continue
        cf,cv=fv[np.isin(fs,cm)],mv[np.isin(ms,cm)]
        if fr is None: fr=cf; av.append(cv)
        else:
            ml=min(len(fr),len(cf))
            if len(fr)>ml: fr=fr[:ml]; av=[v[:ml] for v in av]
            av.append(cv[:ml])
    if not av: return None,None
    ml=min(len(v) for v in av); return fr[:ml],np.array([v[:ml] for v in av])
def sm(v,w=5):
    if len(v)<w: return v
    return np.convolve(v,np.ones(w)/w,mode="valid")
def pms(ax,fr,vl,c,lb,a=0.2,sw=5):
    mn,sd=np.mean(vl,axis=0),np.std(vl,axis=0)
    if sw>1 and len(mn)>sw:
        ms,ss=sm(mn,sw),sm(sd,sw); o=(len(mn)-len(ms))//2; fs=fr[o:o+len(ms)]
    else: ms,ss,fs=mn,sd,fr
    ax.plot(fs,ms,color=c,label=lb,linewidth=1.5); ax.fill_between(fs,ms-ss,ms+ss,color=c,alpha=a)
def gf(v):
    if v is None: return None
    t=v[:,-max(1,v.shape[1]//10):]; return np.mean(t,axis=1)

def p1(dd,cd,od):
    fig,axes=plt.subplots(1,5,figsize=(20,5),sharey=True)
    for i,sk in enumerate(SK):
        ax=axes[i]
        f,v=lm(dd,sk,RW);
        if f is not None: pms(ax,f,v,BLUE,"DiCo")
        f,v=lm(cd,sk,RW,sd="alpha_0p5")
        if f is not None: pms(ax,f,v,RED,"CADiCo")
        ax.set_title(SL[sk]); ax.set_xlabel("Frames")
        if i==0: ax.set_ylabel("Episode Reward")
        ax.legend(loc="lower right",fontsize=8)
    fig.suptitle("CADiCo vs DiCo: Reward at Matched Diversity Targets",y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(od,"ca01_cadico_vs_dico_reward.png"),bbox_inches="tight")
    plt.close(fig); print("  -> ca01")

def p2(dd,cd,od):
    fig,ax=plt.subplots(figsize=(9,5.5)); w=0.02
    for idx,(nm,bd,sd,c,m) in enumerate([("DiCo",dd,None,BLUE,"o"),("CADiCo",cd,"alpha_0p5",RED,"D")]):
        means,stds=[],[]
        for sk in SK:
            _,v=lm(bd,sk,RW,sd=sd); fl=gf(v)
            means.append(np.mean(fl) if fl is not None else np.nan)
            stds.append(np.std(fl) if fl is not None else 0)
        x=np.array([SV[k] for k in SK])+(idx-0.5)*w
        ax.errorbar(x,means,yerr=stds,fmt=f"{m}-",color=c,label=nm,capsize=4,markersize=6,linewidth=2)
    ax.set_xlabel("SND_des"); ax.set_ylabel("Final Reward (last 10%)"); ax.set_title("Robustness: CADiCo vs DiCo")
    ax.legend(); ax.set_xticks([SV[k] for k in SK])
    fig.tight_layout(); fig.savefig(os.path.join(od,"ca02_robustness.png")); plt.close(fig); print("  -> ca02")

def p3(cd,od):
    fig,ax=plt.subplots(figsize=(10,6))
    for key,sub,lb in [("alpha_0","alpha_0",r"$\alpha$=0 (DiCo)"),("alpha_0p1","alpha_0p1",r"$\alpha$=0.1"),
        ("alpha_0p2","alpha_0p2",r"$\alpha$=0.2"),("alpha_0p5","alpha_0p5",r"$\alpha$=0.3 (default)"),
        ("alpha_0p5","alpha_0p5",r"$\alpha$=0.5"),("alpha_1p0","alpha_1p0",r"$\alpha$=1.0")]:
        f,v=lm(cd,"snd_1",RW,sd=sub)
        if f is not None: pms(ax,f,v,AC[key],lb); print(f"    {key}: {v.shape[0]} seeds")
        else: print(f"    [skip] {key}")
    ax.set_xlabel("Training Frames"); ax.set_ylabel("Episode Reward (mean)")
    ax.set_title(r"CADiCo Ablation: $\alpha$ (SND_des = 1.0)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(od,"ca03_alpha_ablation.png")); plt.close(fig); print("  -> ca03")

def p4(cd,od):
    fig,ax=plt.subplots(figsize=(10,6))
    for sk in SK:
        f,v=lm(cd,sk,"collection_agents_diversity_weight.csv",sd="alpha_0p5")
        if f is not None: pms(ax,f,v,CC[sk],SL[sk])
    ax.axhline(y=float(0),color="gray",linestyle="--",alpha=0.5)
    ax.set_xlabel("Training Frames"); ax.set_ylabel("Mean Local SND_des")
    ax.set_title("CADiCo: Local SND_des Evolution"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(od,"ca04_local_snd.png")); plt.close(fig); print("  -> ca04")

def p5(dd,cd,od):
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for sk in SK:
        f,v=lm(dd,sk,"collection_agents_scaling_ratio.csv")
        if f is not None: pms(axes[0],f,v,DC[sk],SL[sk])
        f,v=lm(cd,sk,"collection_agents_scaling_ratio.csv",sd="alpha_0p5")
        if f is not None: pms(axes[1],f,v,CC[sk],SL[sk])
    axes[0].set_title("DiCo: Scaling Ratio"); axes[1].set_title("CADiCo: Scaling Ratio")
    for ax in axes: ax.set_xlabel("Training Frames"); ax.set_ylabel("Mean Scaling Ratio"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(od,"ca05_scaling_ratio.png")); plt.close(fig); print("  -> ca05")

def p6(dd,cd,od):
    fig,ax=plt.subplots(figsize=(10,6))
    for sk in SK:
        f,v=lm(dd,sk,"eval_agents_snd.csv")
        if f is not None:
            mn=np.mean(v,axis=0); ms=sm(mn,5) if len(mn)>5 else mn
            o=(len(mn)-len(ms))//2; fs=f[o:o+len(ms)]
            ax.plot(fs,ms,color=DC[sk],label=f"DiCo {SL[sk]}",linewidth=1.2)
        f,v=lm(cd,sk,"eval_agents_snd.csv",sd="alpha_0p5")
        if f is not None:
            mn=np.mean(v,axis=0); ms=sm(mn,5) if len(mn)>5 else mn
            o=(len(mn)-len(ms))//2; fs=f[o:o+len(ms)]
            ax.plot(fs,ms,color=CC[sk],label=f"CADiCo {SL[sk]}",linewidth=1.2,linestyle="--")
        ax.axhline(y=SV[sk],color="gray",linestyle=":",alpha=0.3)
    ax.set_xlabel("Training Frames"); ax.set_ylabel("Eval SND")
    ax.set_title("Measured Diversity: DiCo vs CADiCo"); ax.legend(ncol=2,fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(od,"ca06_eval_snd.png")); plt.close(fig); print("  -> ca06")

def p7(dd,cd,od):
    fig,ax=plt.subplots(figsize=(8,5))
    f,v=lm(dd,"snd_1",RW)
    if f is not None: pms(ax,f,v,BLUE,"DiCo")
    f,v=lm(cd,"snd_1",RW,sd="alpha_0p5")
    if f is not None: pms(ax,f,v,RED,"CADiCo")
    ax.set_xlabel("Training Frames"); ax.set_ylabel("Episode Reward")
    ax.set_title("CADiCo vs DiCo at SND=1.0"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(od,"ca07_snd1_detail.png")); plt.close(fig); print("  -> ca07")

def p8(dd,cd,od):
    lines=[f"{'SND_des':15s} | {'DiCo':22s} | {'CADiCo':22s}","-"*65]
    for sk in ["homogeneous"]+SK+["unconstrained"]:
        parts=[f"{SL.get(sk,sk):15s}"]
        for nm,bd,sd in [("DiCo",dd,None),("CADiCo",cd,"alpha_0p5")]:
            _,v=lm(bd,sk,RW,sd=sd); fl=gf(v)
            parts.append(f"{np.mean(fl):.3f} +/- {np.std(fl):.3f}" if fl is not None else "N/A")
        lines.append(" | ".join(f"{p:22s}" if i>0 else p for i,p in enumerate(parts)))
    s="\n".join(lines); print("\n"+s)
    with open(os.path.join(od,"cadico_reward_summary.txt"),"w") as f: f.write(s)
    print("  -> cadico_reward_summary.txt")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dico-dir",required=True); p.add_argument("--cadico-dir",required=True)
    p.add_argument("--out-dir",default="./figures_cadico"); a=p.parse_args()
    os.makedirs(a.out_dir,exist_ok=True)
    print("="*60+"\nCADiCo vs DiCo Results\n"+"="*60)
    print("\n[1/8] Reward subplots..."); p1(a.dico_dir,a.cadico_dir,a.out_dir)
    print("\n[2/8] Robustness..."); p2(a.dico_dir,a.cadico_dir,a.out_dir)
    print("\n[3/8] Alpha ablation..."); p3(a.cadico_dir,a.out_dir)
    print("\n[4/8] Local SND_des..."); p4(a.cadico_dir,a.out_dir)
    print("\n[5/8] Scaling ratio..."); p5(a.dico_dir,a.cadico_dir,a.out_dir)
    print("\n[6/8] Eval SND..."); p6(a.dico_dir,a.cadico_dir,a.out_dir)
    print("\n[7/8] SND=1.0 detail..."); p7(a.dico_dir,a.cadico_dir,a.out_dir)
    print("\n[8/8] Reward table..."); p8(a.dico_dir,a.cadico_dir,a.out_dir)
    print("\n"+"="*60+f"\nAll figures saved to {a.out_dir}/\n"+"="*60)

if __name__=="__main__": main()
