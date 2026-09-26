# =========================================================
# File        : build_v2_hearingaid_task12_manifest.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Build v2 hearingaid task12 manifest.
#
# Input       :
#   - Function/CLI arguments, frozen manifests, and project-relative data paths as documented in README.md.
#
# Output      :
#   - Derived CSV/JSON evidence or evaluation summaries as defined by the CLI entry point.
#
# Used in paper:
#   - Hearing-Aid fallback-safety extension reported in the manuscript and Supplementary Material.
#
# Main parameters:
#   - Frozen manuscript parameters and manifests; see README.md and documentation/CODE_GUIDE.md.
#
# Software:
#   - Python 3.x; dependencies listed in requirements.txt/environment.yml.
#
# Author      : Xin Lu
# Last update : 2026-09-25
# =========================================================
from __future__ import annotations

import argparse, gc, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

CODE_DIR=Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path: sys.path.insert(0,str(CODE_DIR))

from sgf_amusic_route_b.active_speech_gate import (
    ActiveSpeechGateParameters, activity_window_stats, band_energy_cumulative_from_audio,
    build_activity_index, estimate_full_recording_inactive_noise_floor_dbfs,
    qualify_active_speech_window, window_band_rms,
)
from sgf_amusic_route_b.b0_r2_protocol import preregistered_window_gate
from sgf_amusic_route_b.locata_geometry import audit_locata_root, infer_locata_task, load_broadband_windows, load_recording_geometry
from sgf_amusic_route_b.frozen_hearingaid_task12_extension_protocol import *
from build_r8_final_eigenmike_manifest import _activity_bounds, _recover_full_audio

PREFIX='V2_HA12'
KNOWN={'dicit','eigenmike','benchmark2'}


def detect_hearingaid_array(root: Path) -> tuple[str, list[dict]]:
    root=Path(root)
    counts={}
    sample={}
    for task in REQUIRED_TASKS:
        td=root/task
        if not td.is_dir(): continue
        for rec in sorted(td.glob('recording*')):
            if not rec.is_dir(): continue
            for child in sorted(p for p in rec.iterdir() if p.is_dir()):
                name=child.name
                if name.lower() in KNOWN: continue
                if not list(child.glob('audio_array*.wav')): continue
                if not list(child.glob('position_array*.txt')): continue
                counts[name]=counts.get(name,0)+1; sample.setdefault(name,rec)
    diag=[]; valid=[]
    for name,count in sorted(counts.items(), key=lambda kv:(-kv[1],kv[0])):
        try:
            g=load_recording_geometry(sample[name], array_name=name)
            channels=int(g['audio_channels'])
            diag.append({'array_name':name,'recordings_seen':int(count),'channels':channels,'status':'usable'})
            if channels==EXPECTED_CHANNEL_COUNT: valid.append((count,name))
        except Exception as e:
            diag.append({'array_name':name,'recordings_seen':int(count),'channels':None,'status':f'error:{type(e).__name__}:{e}'})
    if not valid:
        raise RuntimeError(f'No unseen {EXPECTED_CHANNEL_COUNT}-channel LOCATA array found. Candidates={diag}')
    valid.sort(key=lambda x:(-x[0],x[1]))
    if len(valid)>1 and valid[0][0]==valid[1][0]:
        raise RuntimeError(f'Ambiguous unseen 4-channel array candidates: {valid}; diagnostics={diag}')
    return valid[0][1], diag


def prepare_recording(recording: Path, array_name: str, gate_params: ActiveSpeechGateParameters):
    task=infer_locata_task(recording)
    geometry=load_recording_geometry(recording,array_name=array_name)
    if int(geometry['audio_channels'])!=EXPECTED_CHANNEL_COUNT:
        raise RuntimeError(f'Expected {EXPECTED_CHANNEL_COUNT} channels, got {geometry["audio_channels"]}')
    if geometry.get('microphone_positions_local_m') is None:
        raise RuntimeError('Hearing-aid local microphone coordinates unavailable')
    windows,_=load_broadband_windows(recording,array_name=array_name,window_duration_s=WINDOW_DURATION_S,max_windows=0,window_anchor_mode='required_time_causal',activity_aware_max_windows=False)
    if not windows: raise RuntimeError('No required-time windows')
    activity_index=build_activity_index(geometry)
    if not activity_index or str(geometry.get('source_activity_source',''))!='official_array_aligned_vad':
        raise RuntimeError('Official array-aligned VAD required')
    full_audio,recovery=_recover_full_audio(windows,geometry)
    full_audio=np.asarray(full_audio,dtype=float)
    band=band_energy_cumulative_from_audio(full_audio,windows[0]['sample_rate_hz'],frequency_min_hz=gate_params.localization_band_min_hz,frequency_max_hz=gate_params.localization_band_max_hz,filter_order=gate_params.filter_order)
    noise_floor,noise_audit,_=estimate_full_recording_inactive_noise_floor_dbfs(
        band, geometry['audio_t_abs_sec'], activity_index, windows[0]['sample_rate_hz'],
        window_duration_s=WINDOW_DURATION_S, hop_duration_s=gate_params.noise_grid_hop_s,
        inactive_fraction_max=gate_params.inactive_fraction_max, minimum_noise_windows=gate_params.minimum_noise_windows,
        lower_envelope_fraction=gate_params.lower_envelope_fraction)
    rows=[]; eligible=[]; timestamps=[]; ids=[]
    for i,frame in enumerate(windows):
        center=preregistered_window_gate(frame,hard_silence_floor_dbfs=-80.0)
        start,end=_activity_bounds(frame)
        stats=activity_window_stats(activity_index,start,end,center.active_source_index)
        brms=window_band_rms(band,int(frame['audio_window_start_sample']),int(frame['audio_window_end_exclusive_sample']))
        speech=qualify_active_speech_window(
            center_official_exactly_one_active=bool(center.official_activity_available and center.exactly_one_source_active),
            active_source_index=center.active_source_index, activity_stats=stats, band_rms=brms,
            recording_noise_floor_dbfs=noise_floor, window_duration_s=WINDOW_DURATION_S,
            relative_gate_required=bool(noise_audit['physical_noise_floor_claimed']),params=gate_params)
        si=int(center.active_source_index) if center.active_source_index is not None else -1
        truth=np.asarray(frame['theta_true_deg'],dtype=float).reshape(-1)
        az=float(truth[si]) if 0<=si<len(truth) and np.isfinite(truth[si]) else float('nan')
        other_ok=bool(np.isfinite(stats.maximum_other_source_fraction) and stats.maximum_other_source_fraction<=MAXIMUM_OTHER_SOURCE_ACTIVITY_FRACTION+1e-12)
        truth_ok=bool(np.isfinite(az) and THETA_MIN_DEG<=az<=THETA_MAX_DEG)
        ok=bool(speech.eligible and other_ok and truth_ok)
        if ok: eligible.append(i)
        timestamps.append(float(frame['timestamp_abs_s'])); ids.append(int(frame['required_time_row_index']))
        rows.append({'release':RELEASE,'protocol_id':PROTOCOL_ID,'task':task,'recording':str(recording),'recording_name':recording.name,
            'required_time_row_index':ids[-1],'timestamp_abs_s':timestamps[-1],'active_source_index':si,
            'true_azimuth_deg_for_domain_check_only':az,'target_activity_duration_s':float(speech.target_activity_duration_s),
            'target_activity_fraction_full_window':float(stats.target_fraction),'maximum_other_source_activity_fraction':float(stats.maximum_other_source_fraction),
            'strict_50pct_pass':bool(speech.strict_50pct_pass),'sensitivity_80pct_pass':bool(speech.sensitivity_80pct_pass),
            'eligible_before_sampling':ok,'truth_used_for_estimation':False,'error_used_for_selection':False,'r8_output_used_for_selection':False})
    selected=deterministic_time_spread_subset(eligible,timestamps,ids)
    sset=set(selected); ranks={j:k for k,j in enumerate(selected)}
    for i,r in enumerate(rows): r.update(selected_for_holdout=i in sset,selection_rank=int(ranks.get(i,-1)),selection_rule='uniform_time_spread_among_preregistered_eligible_anchors' if i in sset else '')
    manifest=[]
    for i in selected:
        r=rows[i]
        manifest.append({k:r[k] for k in ['release','protocol_id','task','recording','recording_name','required_time_row_index','timestamp_abs_s','active_source_index','target_activity_duration_s','target_activity_fraction_full_window','maximum_other_source_activity_fraction','strict_50pct_pass','sensitivity_80pct_pass','selection_rank','selection_rule','truth_used_for_estimation','error_used_for_selection','r8_output_used_for_selection']})
    sufficient=len(selected)>=MIN_SELECTED_PER_RECORDING
    status={'task':task,'recording_name':recording.name,'recording':str(recording),'eligible_windows':len(eligible),'selected_windows':len(selected),'coverage_sufficient':sufficient,'audio_channels':int(geometry['audio_channels']),'audio_recovery':recovery}
    del windows,full_audio,band; gc.collect()
    return pd.DataFrame(manifest),pd.DataFrame(rows),status


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    array_name,detect_diag=detect_hearingaid_array(args.root)
    audit,audit_status=audit_locata_root(args.root,args.output/'geometry_audit',array_name=array_name)
    usable=[Path(x) for x in audit.loc[audit['usable'].astype(bool),'recording'].tolist() if infer_locata_task(Path(x)) in REQUIRED_TASKS]
    gate=ActiveSpeechGateParameters(); manifests=[]; quals=[]; stats=[]; errors=[]
    for rec in usable:
        try:
            m,q,s=prepare_recording(rec,array_name,gate); manifests.append(m); quals.append(q); stats.append(s)
            print(f'[manifest {infer_locata_task(rec)}/{rec.name}] eligible={s["eligible_windows"]} selected={s["selected_windows"]}',flush=True)
        except Exception as e:
            errors.append({'recording':str(rec),'error':f'{type(e).__name__}: {e}'})
    manifest=pd.concat(manifests,ignore_index=True) if manifests else pd.DataFrame(); qual=pd.concat(quals,ignore_index=True) if quals else pd.DataFrame()
    manifest.to_csv(args.output/f'{PREFIX}_FROZEN_MANIFEST.csv',index=False); qual.to_csv(args.output/f'{PREFIX}_QUALIFICATION.csv',index=False)
    pd.DataFrame(stats).to_csv(args.output/f'{PREFIX}_RECORDING_STATUS.csv',index=False)
    sufficient=[s for s in stats if s['coverage_sufficient']]
    tasks=sorted({s['task'] for s in sufficient}); total=len(manifest)
    census_count=len(stats)+len(errors)
    gates={'hearingaid_array_detected':bool(array_name),'exactly_4_channels':True,'census_exactly_26_recordings':census_count==EXPECTED_CENSUS_RECORDINGS,
           'at_least_6_sufficient_recordings':len(sufficient)>=MIN_RECORDING_COUNT,
           'task1_and_task2_represented':all(t in tasks for t in REQUIRED_TASKS),'at_least_72_windows':total>=MIN_TOTAL_SELECTED,'no_recording_errors':len(errors)==0}
    frozen=verify_frozen_hashes(Path(__file__).resolve().parent.parent)
    lock={'release':RELEASE,'protocol_id':PROTOCOL_ID,'status':'FROZEN' if all(gates.values()) and frozen['status']=='PASS' else 'PREFLIGHT_FAIL',
          'array_name':array_name,'native_channel_indices_zero_based':list(range(EXPECTED_CHANNEL_COUNT)),'census_recording_count':census_count,'manifest_rows':total,
          'manifest_canonical_sha256':canonical_manifest_sha256(manifest.to_dict('records')) if total else None,'represented_tasks':tasks,
          'sufficient_recording_count':len(sufficient),'gates':gates,'array_detection_diagnostics':detect_diag,'audit_status':audit_status,
          'recording_errors':errors,'scientific_identity':'outcome-unseen post-freeze cross-task extension within LOCATA','frozen_estimator':{'method':FINAL_METHOD,'assignment_strength':FROZEN_ASSIGNMENT_STRENGTH,'anchor_strength':FROZEN_ANCHOR_STRENGTH,'evidence_change_guard':True},
          'frozen_hash_audit':frozen,'holdout_errors_inspected_before_freeze':False,'holdout_used_for_tuning':False,'manual_exclusion_permitted':False}
    (args.output/f'{PREFIX}_MANIFEST_LOCK.json').write_text(json.dumps(lock,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(lock,indent=2,ensure_ascii=False)); return 0 if lock['status']=='FROZEN' else 2
if __name__=='__main__': raise SystemExit(main())
