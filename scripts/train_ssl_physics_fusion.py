"""Extract inference-aligned SSL/physics features and train a leakage-safe TOI fusion head.

This research cohort is follow-up selected. Calibrated outputs are cohort scores,
not population-wide planet probabilities.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True, type=Path)
    ap.add_argument('--download-manifest', required=True, type=Path)
    ap.add_argument('--split-manifest', required=True, type=Path)
    ap.add_argument('--ssl-checkpoint', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--max-events', type=int, default=0, help='Optional smoke subset; 0 processes all eligible events.')
    args=ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from astrovet.lightcurves.reader import read_tess_fits_lightcurve
    from astrovet.lightcurves.quality import filter_quality_cadences
    from astrovet.lightcurves.normalization import segment_and_normalize_lightcurve
    from astrovet.views.folding import extract_transit_views
    from astrovet.physics.diagnostics import run_physics_diagnostics
    from astrovet.ssl.neural_encoder import SelfSupervisedTransitEncoder
    from astrovet.models.fusion import AstrovetFusionModel
    from astrovet.models.calibration import PlattCalibrator, compute_calibration_metrics

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort=pd.read_csv(args.cohort, low_memory=False)
    dl=json.loads(args.download_manifest.read_text(encoding='utf-8'))
    split=json.loads(args.split_manifest.read_text(encoding='utf-8'))['metadata']['host_ids_by_split']
    files={}
    for item in dl.get('files',[]):
        if item.get('status') not in ('DOWNLOADED','CACHED'): continue
        path=Path(item['path'])
        if path.is_file(): files[(int(item['tic_id']),int(item['sector']))]=path
    encoder=SelfSupervisedTransitEncoder.load_checkpoint(args.ssl_checkpoint)
    if not encoder.is_fitted or str(getattr(encoder,'data_source','')).upper()=='SYNTHETIC_DEMO':
        raise ValueError('A fitted real-data SSL checkpoint is required.')
    fusion=AstrovetFusionModel()
    rows=[]; failures=[]
    limit=args.max_events if args.max_events>0 else len(cohort)
    for idx,row in cohort.iloc[:limit].iterrows():
        tic=int(row['ticid'])
        sec_match=re.search(r'\d+',str(row['sectors']))
        sector=int(sec_match.group()) if sec_match else -1
        path=files.get((tic,sector))
        if not path:
            failures.append({'row':int(idx),'ticid':tic,'sector':sector,'error':'No matching FITS in download manifest'})
            continue
        try:
            raw=read_tess_fits_lightcurve(path)
            t,f,fe,q=filter_quality_cadences(raw.time,raw.flux,raw.flux_err,raw.quality)
            if len(t)<100: raise ValueError(f'Only {len(t)} usable cadences after quality filtering')
            period=float(row['tce_period']); epoch=float(row['tce_time0bt']); duration=float(row['tce_duration'])
            norm=segment_and_normalize_lightcurve(t,f,fe,ephemeris_period_days=period,ephemeris_epoch_btjd=epoch,ephemeris_duration_hours=duration)
            primary,full,odd,even,stamps=extract_transit_views(norm.time,norm.flux_detrended,norm.flux_err_norm,period,epoch,duration)
            diag,_=run_physics_diagnostics(primary,full,stamps,norm.time,norm.flux_detrended,norm.flux_err_norm,period,epoch,duration,float(row['tce_depth']),periodogram_results={})
            emb=encoder.encode(primary.flux.reshape(1,-1))[0]
            vec,names=fusion.build_feature_vector(None,diag,emb)
            keep=[i for i,n in enumerate(names) if not n.startswith('baseline_')]
            rows.append({'row':int(idx),'ticid':tic,'sector':sector,'target':int(row['target']),'disposition':str(row['disposition']),'features':vec[keep].astype(float).tolist(),'feature_names':[names[i] for i in keep],'path':str(path)})
        except Exception as e:
            failures.append({'row':int(idx),'ticid':tic,'sector':sector,'error':f'{type(e).__name__}: {e}'[:500]})
        if (idx+1)%25==0:
            print(f'features {idx+1}/{limit}; usable={len(rows)} failed={len(failures)}',flush=True)
    if len(rows)<100: raise RuntimeError(f'Only {len(rows)} events produced features; refusing to train.')
    names=rows[0]['feature_names']
    inconsistent=[r['row'] for r in rows if r['feature_names']!=names]
    if inconsistent: raise RuntimeError(f'Inconsistent feature order on rows: {inconsistent[:10]}')
    X=np.asarray([r['features'] for r in rows],dtype=np.float32)
    y=np.asarray([r['target'] for r in rows],dtype=np.int64)
    hosts=np.asarray([r['ticid'] for r in rows],dtype=np.int64)
    feature_file=args.output_dir/'planet_fusion_features.npz'
    np.savez_compressed(feature_file,X=X,y=y,hosts=hosts,row_ids=np.asarray([r['row'] for r in rows]),feature_names=np.asarray(names))
    audit={'created_utc':datetime.now(timezone.utc).isoformat(),'cohort_file':str(args.cohort.resolve()),'cohort_sha256':sha256(args.cohort),'download_manifest':str(args.download_manifest.resolve()),'ssl_checkpoint':str(args.ssl_checkpoint.resolve()),'ssl_checkpoint_sha256':sha256(args.ssl_checkpoint),'events_attempted':int(limit),'events_with_features':len(rows),'failures':failures,'feature_names':names,'feature_matrix':str(feature_file.resolve())}
    (args.output_dir/'planet_fusion_feature_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    host_splits={k:set(map(int,v)) for k,v in split.items()}
    split_idx={k:np.asarray([i for i,h in enumerate(hosts) if int(h) in hs],dtype=int) for k,hs in host_splits.items()}
    unseen=set(map(int,hosts))-set().union(*host_splits.values())
    if unseen: raise RuntimeError(f'{len(unseen)} hosts absent from split manifest; do not train with implicit split.')
    for k,ix in split_idx.items():
        if len(ix)<20 or len(np.unique(y[ix]))<2: raise RuntimeError(f'Insufficient two-class {k} set after FITS extraction: n={len(ix)}')
    scaler=StandardScaler().fit(X[split_idx['train']])
    Xs=scaler.transform(X)
    best=None
    for C in [0.01,0.1,1.0,10.0]:
        clf=LogisticRegression(C=C,max_iter=3000,class_weight='balanced',random_state=42)
        clf.fit(Xs[split_idx['train']],y[split_idx['train']])
        prob=clf.predict_proba(Xs[split_idx['validation']])[:,1]
        metric=float(average_precision_score(y[split_idx['validation']],prob))
        item=(metric,C,clf)
        if best is None or item[0]>best[0]:best=item
    val_ap,best_C,clf=best
    cal_raw=clf.predict_proba(Xs[split_idx['calibration']])[:,1]
    calibrator=PlattCalibrator().fit(cal_raw,y[split_idx['calibration']])
    calibrator.provenance='Held-out calibration hosts from selected TOI follow-up cohort; score is not population calibrated.'
    test_ix=split_idx['test']; raw=clf.predict_proba(Xs[test_ix])[:,1]; calibrated=calibrator.calibrate_batch(raw)
    metrics={'test_events':int(len(test_ix)),'test_hosts':int(len(set(hosts[test_ix]))),'positive_prevalence':float(y[test_ix].mean()),'roc_auc_raw':float(roc_auc_score(y[test_ix],raw)),'pr_auc_raw':float(average_precision_score(y[test_ix],raw)),'raw_calibration':compute_calibration_metrics(y[test_ix],raw),'calibrated_calibration':compute_calibration_metrics(y[test_ix],calibrated),'threshold_0_5':{'precision':float(precision_score(y[test_ix],calibrated>=.5,zero_division=0)),'recall':float(recall_score(y[test_ix],calibrated>=.5,zero_division=0)),'f1':float(f1_score(y[test_ix],calibrated>=.5,zero_division=0)),'confusion_matrix':confusion_matrix(y[test_ix],calibrated>=.5).tolist()},'validation_pr_auc_for_selected_C':val_ap,'selected_C':best_C,'split_counts':{k:{'events':int(len(ix)),'hosts':int(len(set(hosts[ix]))),'positives':int(y[ix].sum()),'negatives':int(len(ix)-y[ix].sum())} for k,ix in split_idx.items()},'interpretation':'Selected-TOI-cohort calibrated planet-like score; not population calibrated and not a confirmation.'}
    model=AstrovetFusionModel(random_state=42);model.classifier=clf;model.feature_scaler=scaler;model.feature_names=names;model.calibrator=calibrator;model.is_fitted=True
    model.provenance={'model_type':'ssl_physics_late_fusion_logistic','target_is_planet_identification':True,'label_mapping':{'CP/KP':1,'FP/FA':0,'PC/APC/UNSET':'excluded'},'training_cohort':'selected NASA Exoplanet Archive TOIs matched to MAST TESS TCE light curves','training_events':int(len(rows)),'training_hosts':int(len(set(hosts[split_idx['train']]))),'feature_names':names,'ssl_checkpoint_sha256':audit['ssl_checkpoint_sha256'],'cohort_sha256':audit['cohort_sha256'],'split_manifest':str(args.split_manifest.resolve()),'split_seed':42,'selected_C':best_C,'validation_pr_auc':val_ap,'calibration_scope':'Held-out calibration partition from selected TOI follow-up cohort only; do not interpret as population prevalence probability.','limitations':['Selected TOI follow-up sample; distribution shift likely.','Physics diagnostics are approximate screening measurements.','Baseline ExoMiner score omitted to keep inference independent of catalog lookup.','Human review remains required.']}
    artifact=args.output_dir/'astrovet_ssl_physics_fusion_research.joblib';model.save(artifact)
    (args.output_dir/'astrovet_ssl_physics_fusion_evaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    split_out={k:{'events':int(len(ix)),'hosts':sorted(map(int,set(hosts[ix]))),'row_ids':list(map(int,np.asarray([r['row'] for r in rows])[ix]))} for k,ix in split_idx.items()}
    (args.output_dir/'astrovet_ssl_physics_fusion_splits.json').write_text(json.dumps(split_out,indent=2),encoding='utf-8')
    print(json.dumps({'artifact':str(artifact),'feature_rows':len(rows),'failed':len(failures),'metrics':metrics},indent=2),flush=True)

if __name__=='__main__':main()
