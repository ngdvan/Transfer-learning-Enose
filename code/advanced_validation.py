from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import co_ethylene_session_transfer as co
import same_device_session_transfer as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "advanced_validation"
VARIANTS = {
    "full": dict(use_sensor_ids=True, use_attention=True, temporal_mask=True, sensor_masking=True),
    "no_sensor_ids": dict(use_sensor_ids=False, use_attention=True, temporal_mask=True, sensor_masking=True),
    "no_attention": dict(use_sensor_ids=True, use_attention=False, temporal_mask=True, sensor_masking=True),
    "no_sensor_mask": dict(use_sensor_ids=True, use_attention=True, temporal_mask=True, sensor_masking=False),
    "no_temporal_mask": dict(use_sensor_ids=True, use_attention=True, temporal_mask=False, sensor_masking=True),
}


def configure(dataset: str):
    if dataset == "nh3_h2":
        core.N_SENSORS, core.TARGETS = 20, ["NH3", "H2"]
        core.TARGET_SCALE = np.asarray([500., 500.], np.float32)
    else:
        core.N_SENSORS, core.TARGETS = 16, ["CO", "Ethylene"]
        core.TARGET_SCALE = np.asarray([533.33, 20.], np.float32)


def validation_split(groups: np.ndarray, seed: int):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed); rng.shuffle(unique)
    count = max(1, int(round(.2 * len(unique))))
    val_groups = unique[:count]
    return ~np.isin(groups, val_groups), np.isin(groups, val_groups), val_groups


def metrics(y, pred, gases):
    out = {}
    for j, gas in enumerate(gases):
        err = np.abs(y[:, j] - pred[:, j])
        denom = np.sum((y[:, j] - y[:, j].mean()) ** 2)
        out[gas] = {
            "mae_ppm": float(err.mean()),
            "rmse_ppm": float(np.sqrt(np.mean((y[:, j] - pred[:, j]) ** 2))),
            "r2": float(1 - np.sum((y[:, j] - pred[:, j]) ** 2) / denom) if denom else float("nan"),
            "median_abs_error_ppm": float(np.median(err)),
            "q90_abs_error_ppm": float(np.quantile(err, .9)),
            "q95_abs_error_ppm": float(np.quantile(err, .95)),
        }
    return out


def conformal_quantiles(model, val_x, val_y, alpha=.1):
    pred = core.predict(model, val_x)
    n = len(val_y); level = min(1., np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(np.abs(val_y - pred), level, axis=0, method="higher")


def fit_eval(train_x, train_y, train_groups, test_x, test_y, initial, frozen,
             seed, epochs, variant):
    tr, va, val_groups = validation_split(train_groups, seed)
    model, history = core.fit_neural_with_history(
        train_x[tr], train_y[tr], initial, frozen, epochs, seed,
        val_x=train_x[va], val_y=train_y[va],
        use_sensor_ids=variant["use_sensor_ids"], use_attention=variant["use_attention"],
    )
    pred = core.predict(model, test_x)
    q = conformal_quantiles(model, train_x[va], train_y[va])
    lower = np.maximum(0, pred - q); upper = np.minimum(core.TARGET_SCALE, pred + q)
    coverage = np.mean((test_y >= lower) & (test_y <= upper), axis=0)
    width = np.mean(upper - lower, axis=0)
    return model, history, pred, q, coverage, width, val_groups


def load_fixed(dataset):
    configure(dataset)
    if dataset == "nh3_h2":
        data, audit = core.load_exposures(core.DATA)
        pre_idx=np.flatnonzero(data["sessions"]<=4); adapt_idx=np.flatnonzero(data["sessions"]==5); test_idx=np.flatnonzero(data["sessions"]==6)
        pre_raw,_,_=core.make_windows(data,pre_idx); test_raw,test_y,test_groups=core.make_windows(data,test_idx,offsets=[0])
        scaler=core.SensorScaler().fit(pre_raw)
        return data,audit,scaler,scaler.transform(pre_raw),adapt_idx,scaler.transform(test_raw),test_y,test_groups
    data,audit=co.load_data(co.CACHE)
    pre=data["blocks"]<=3; adapt0=(data["blocks"]==4)&(data["offsets"]==0); test=(data["blocks"]==5)&(data["offsets"]==0)
    scaler=core.SensorScaler().fit(data["x"][pre])
    return data,audit,scaler,scaler.transform(data["x"][pre]),adapt0,scaler.transform(data["x"][test]),data["y"][test],data["groups"][test]


def adaptation(dataset,data,adapt_selector,scaler,budget,seed):
    if dataset=="nh3_h2":
        order=adapt_selector[core.farthest_subset(data["targets"][adapt_selector],57,seed)][:budget]
        x,y,g=core.make_windows(data,order); return scaler.transform(x),y,g
    x0=data["x"][adapt_selector]; y0=data["y"][adapt_selector]; g0=data["groups"][adapt_selector]
    order=co.farthest_order(y0,seed); chosen=g0[order][:budget]
    m=(data["blocks"]==4)&np.isin(data["groups"],chosen)
    return scaler.transform(data["x"][m]),data["y"][m],data["groups"][m]


def run_ablation(dataset,args):
    configure(dataset); data,audit,scaler,pre_x,adapt_sel,test_x,test_y,test_groups=load_fixed(dataset)
    gases=list(core.TARGETS); budgets=[args.low_budget,57 if dataset=="nh3_h2" else 24]
    rows=[]; preds=[]; histories={}; pretraining={}
    for name,variant in VARIANTS.items():
        state,hist=core.pretrain(pre_x,args.pretrain_epochs,args.seed,**variant)
        pretraining[name]=hist
        for repeat in range(args.repeats):
            rseed=args.seed+100*repeat
            for budget in budgets:
                tx,ty,tg=adaptation(dataset,data,adapt_sel,scaler,budget,rseed)
                _,history,pred,q,cov,width,val_groups=fit_eval(tx,ty,tg,test_x,test_y,state,False,rseed+budget,args.epochs,variant)
                key=f"{name}/r{repeat+1}/b{budget}"; histories[key]=history
                row={"variant":name,"repeat":repeat+1,"seed":rseed,"budget":budget,"metrics":metrics(test_y,pred,gases),
                     "conformal_90":{g:{"quantile_ppm":float(q[j]),"coverage":float(cov[j]),"mean_width_ppm":float(width[j])} for j,g in enumerate(gases)},
                     "validation_groups":list(map(int,val_groups))}
                rows.append(row); print(json.dumps({"stage":"ablation","dataset":dataset,"variant":name,"repeat":repeat+1,"budget":budget}),flush=True)
                for group,y,p in zip(test_groups,test_y,pred):
                    item={"variant":name,"repeat":repeat+1,"budget":budget,"group":int(group)}
                    for j,g in enumerate(gases): item.update({f"{g}_true":float(y[j]),f"{g}_pred":float(p[j])})
                    preds.append(item)
    out=OUT/dataset;out.mkdir(parents=True,exist_ok=True)
    result={"protocol":{"variants":VARIANTS,"budgets":budgets,"repeats":args.repeats,"pretrain_epochs":args.pretrain_epochs,"supervised_epochs":args.epochs,
                        "validation":"20% adaptation groups only; test never selects checkpoints"},"data_audit":audit,"pretraining_histories":pretraining,"supervised_histories":histories,"runs":rows}
    (out/"ablation_results.json").write_text(json.dumps(result,indent=2));pd.DataFrame(preds).to_csv(out/"ablation_predictions.csv",index=False)


def fold_dataset_nh():
    configure("nh3_h2"); return core.load_exposures(core.DATA)


def run_nh_folds(args):
    configure("nh3_h2"); data,audit=fold_dataset_nh(); rows=[];preds=[]; histories={}
    for test_session in range(1,7):
        adapt_session=6 if test_session==1 else test_session-1
        pre_sessions=[s for s in range(1,7) if s not in (test_session,adapt_session)]
        pre_idx=np.flatnonzero(np.isin(data["sessions"],pre_sessions)); adapt_idx=np.flatnonzero(data["sessions"]==adapt_session); test_idx=np.flatnonzero(data["sessions"]==test_session)
        pre_raw,_,_=core.make_windows(data,pre_idx); scaler=core.SensorScaler().fit(pre_raw); state,ph=core.pretrain(scaler.transform(pre_raw),args.pretrain_epochs,args.seed+test_session)
        test_raw,test_y,test_g=core.make_windows(data,test_idx,offsets=[0]);test_x=scaler.transform(test_raw)
        for repeat in range(args.repeats):
            seed=args.seed+100*repeat+test_session; order=adapt_idx[core.farthest_subset(data["targets"][adapt_idx],57,seed)]
            for budget in [5,10,20,40,57]:
                raw,y,g=core.make_windows(data,order[:budget]);x=scaler.transform(raw)
                for policy,initial,frozen in [("scratch",None,False),("transfer",state,False)]:
                    _,hist,pred,q,cov,width,val_groups=fit_eval(x,y,g,test_x,test_y,initial,frozen,seed+budget+(policy=="transfer"),args.epochs,VARIANTS["full"])
                    histories[f"s{test_session}/r{repeat+1}/b{budget}/{policy}"]=hist
                    rows.append({"fold":test_session,"test_session":test_session,"adapt_session":adapt_session,"pretrain_sessions":pre_sessions,"repeat":repeat+1,"budget":budget,"model":policy,"metrics":metrics(test_y,pred,core.TARGETS),
                                 "conformal_90":{gas:{"quantile_ppm":float(q[j]),"coverage":float(cov[j]),"mean_width_ppm":float(width[j])} for j,gas in enumerate(core.TARGETS)}})
                    for group,yt,yp in zip(test_g,test_y,pred): preds.append({"fold":test_session,"repeat":repeat+1,"budget":budget,"model":policy,"group":int(group),"NH3_true":float(yt[0]),"H2_true":float(yt[1]),"NH3_pred":float(yp[0]),"H2_pred":float(yp[1])})
                print(json.dumps({"stage":"nh_fold","fold":test_session,"repeat":repeat+1,"budget":budget}),flush=True)
    out=OUT/"nh3_h2";out.mkdir(parents=True,exist_ok=True);(out/"session_cv_results.json").write_text(json.dumps({"data_audit":audit,"runs":rows,"supervised_histories":histories},indent=2));pd.DataFrame(preds).to_csv(out/"session_cv_predictions.csv",index=False)


def run_co_folds(args):
    configure("co_ethylene");data,audit=co.load_data(co.CACHE); unique=np.unique(data["groups"]);chunks=np.array_split(unique,7); block={int(g):i+1 for i,c in enumerate(chunks) for g in c}; b=np.array([block[int(g)] for g in data["groups"]]);rows=[];preds=[];histories={}
    for fold,start in enumerate([1,2,3],1):
        pre_blocks=[start,start+1,start+2];adapt_block=start+3;test_block=start+4
        pre=np.isin(b,pre_blocks);adapt0=(b==adapt_block)&(data["offsets"]==0);test=(b==test_block)&(data["offsets"]==0)
        scaler=core.SensorScaler().fit(data["x"][pre]);state,ph=core.pretrain(scaler.transform(data["x"][pre]),args.pretrain_epochs,args.seed+fold);test_x=scaler.transform(data["x"][test]);test_y=data["y"][test];test_g=data["groups"][test]
        x0,y0,g0=data["x"][adapt0],data["y"][adapt0],data["groups"][adapt0];maxb=len(np.unique(y0,axis=0))
        for repeat in range(args.repeats):
            seed=args.seed+100*repeat+fold; chosen=g0[co.farthest_order(y0,seed)]
            for budget in sorted(set([5,10,20,maxb])):
                use=chosen[:budget];m=(b==adapt_block)&np.isin(data["groups"],use);x=scaler.transform(data["x"][m]);y=data["y"][m];g=data["groups"][m]
                for policy,initial in [("scratch",None),("transfer",state)]:
                    _,hist,pred,q,cov,width,val_groups=fit_eval(x,y,g,test_x,test_y,initial,False,seed+budget+(policy=="transfer"),args.epochs,VARIANTS["full"]);histories[f"f{fold}/r{repeat+1}/b{budget}/{policy}"]=hist
                    rows.append({"fold":fold,"pretrain_blocks":pre_blocks,"adapt_block":adapt_block,"test_block":test_block,"repeat":repeat+1,"budget":budget,"model":policy,"metrics":metrics(test_y,pred,core.TARGETS),"conformal_90":{gas:{"quantile_ppm":float(q[j]),"coverage":float(cov[j]),"mean_width_ppm":float(width[j])} for j,gas in enumerate(core.TARGETS)}})
                    for group,yt,yp in zip(test_g,test_y,pred):preds.append({"fold":fold,"repeat":repeat+1,"budget":budget,"model":policy,"group":int(group),"CO_true":float(yt[0]),"Ethylene_true":float(yt[1]),"CO_pred":float(yp[0]),"Ethylene_pred":float(yp[1])})
                print(json.dumps({"stage":"co_fold","fold":fold,"repeat":repeat+1,"budget":budget}),flush=True)
    out=OUT/"co_ethylene";out.mkdir(parents=True,exist_ok=True);(out/"rolling_cv_results.json").write_text(json.dumps({"data_audit":audit,"rolling_blocks":7,"runs":rows,"supervised_histories":histories},indent=2));pd.DataFrame(preds).to_csv(out/"rolling_cv_predictions.csv",index=False)


def bootstrap_file(path:Path,gases,outpath:Path,seed=42,n_boot=5000):
    df=pd.read_csv(path); key="fold" if "fold" in df else None; rng=np.random.default_rng(seed); results={}
    budgets=sorted(df.budget.unique())
    for budget in budgets:
        d=df[df.budget==budget]
        if not {"scratch","transfer"}.issubset(set(d.model.unique())): continue
        merge=d[d.model=="scratch"].merge(d[d.model=="transfer"],on=[c for c in [key,"repeat","budget","group"] if c],suffixes=("_scratch","_transfer"))
        results[str(int(budget))]={}
        for gas in gases:
            delta=np.abs(merge[f"{gas}_true_scratch"]-merge[f"{gas}_pred_scratch"])-np.abs(merge[f"{gas}_true_transfer"]-merge[f"{gas}_pred_transfer"])
            vals=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(n_boot)])
            results[str(int(budget))][gas]={"mean_paired_mae_improvement_ppm":float(delta.mean()),"bootstrap_95_ci_ppm":list(map(float,np.quantile(vals,[.025,.975]))),"n_paired_predictions":len(delta)}
    outpath.write_text(json.dumps(results,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["smoke","ablations","folds","stats","all"],default="all");p.add_argument("--pretrain-epochs",type=int,default=60);p.add_argument("--epochs",type=int,default=120);p.add_argument("--repeats",type=int,default=3);p.add_argument("--seed",type=int,default=42);p.add_argument("--low-budget",type=int,default=5);args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);started=time.time()
    if args.mode=="smoke": args.pretrain_epochs,args.epochs,args.repeats=1,2,1;run_ablation("nh3_h2",args);run_ablation("co_ethylene",args)
    if args.mode in ("ablations","all"): run_ablation("nh3_h2",args);run_ablation("co_ethylene",args)
    if args.mode in ("folds","all"): run_nh_folds(args);run_co_folds(args)
    if args.mode in ("stats","all"):
        bootstrap_file(OUT/"nh3_h2/session_cv_predictions.csv",["NH3","H2"],OUT/"nh3_h2/bootstrap_intervals.json",args.seed)
        bootstrap_file(OUT/"co_ethylene/rolling_cv_predictions.csv",["CO","Ethylene"],OUT/"co_ethylene/bootstrap_intervals.json",args.seed)
    print(json.dumps({"complete":args.mode,"elapsed_seconds":time.time()-started}),flush=True)


if __name__=="__main__": main()
