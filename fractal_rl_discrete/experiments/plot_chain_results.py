from __future__ import annotations
import os, json
import numpy as np
import matplotlib.pyplot as plt

def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _load():
    root=_repo_root()
    path=os.path.join(root,"results","results_chain.json")
    with open(path,"r") as f: d=json.load(f)
    return root,d

def _get_run(d, i=0):
    return d["runs"][i]

def main():
    root,d=_load()
    out_dir=os.path.join(root,"figures")
    os.makedirs(out_dir, exist_ok=True)

    run=_get_run(d,0)
    vb=np.array(run["baseline"]["V_end"], dtype=float)
    vf=np.array(run["fractional"]["V_end"], dtype=float)

    # Try to pull intermediate snapshots if present; otherwise synthesize from v_start->v_end is impossible.
    # So we read deltas only and plot final profile (still useful).
    n=len(vb)
    x=np.arange(n)

    plt.figure()
    plt.plot(x, vb, label="baseline")
    plt.plot(x, vf, label="fractional")
    plt.xlabel("state index")
    plt.ylabel("V(s)")
    plt.title("Chain: final value profile")
    plt.legend()
    plt.tight_layout()
    p=os.path.join(out_dir,"diag_chain_V_profile_final.png")
    plt.savefig(p); plt.close()
    print("[wrote]", p)

    # Convergence curves
    db=run["baseline"]["hist"]["deltas"]
    df=run["fractional"]["hist"]["deltas"]
    L=max(len(db),len(df))
    def pad(a):
        if not a: return np.zeros(L)
        a=np.array(a,float)
        if len(a)<L: a=np.concatenate([a, np.full(L-len(a), a[-1])])
        return a[:L]
    db,df=pad(db),pad(df)
    it=np.arange(1,L+1)

    plt.figure()
    plt.plot(it, db, label="baseline")
    plt.plot(it, df, label="fractional")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("max |V_{k+1}-V_k|")
    plt.title("Chain: convergence")
    plt.legend()
    plt.tight_layout()
    p=os.path.join(out_dir,"diag_chain_deltas.png")
    plt.savefig(p); plt.close()
    print("[wrote]", p)

if __name__=="__main__":
    main()
