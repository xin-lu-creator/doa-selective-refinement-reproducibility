# =========================================================
# File        : run_v2_hearingaid_task12_once.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Run v2 hearingaid task12 once.
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
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
CODE_DIR=Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path: sys.path.insert(0,str(CODE_DIR))
from sgf_amusic_route_b.baseline_extension import _circular_difference_deg
from sgf_amusic_route_b.baseline_forensic import BaselineForensicParameters
from sgf_amusic_route_b.causal_multiscale_hodge import CausalMultiscaleHodgeParameters
from sgf_amusic_route_b.locata_geometry import audit_locata_root,infer_locata_task,load_broadband_windows
from sgf_amusic_route_b.mechanism_corrected_r8 import prepare_r8m_state_banks
from sgf_amusic_route_b.mechanism_corrected_r8_multimethod_evidence_change import MultiMethodParameters,rescore_prepared_r8m_multimethod_banks
from sgf_amusic_route_b.frozen_hearingaid_task12_extension_protocol import *
PREFIX='V2_HA12'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--manifest-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--resume',action='store_true'); a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=True); per=a.output/'per_recording'; per.mkdir(exist_ok=True)
 lock=json.loads((a.manifest_dir/f'{PREFIX}_MANIFEST_LOCK.json').read_text(encoding='utf-8'))
 if lock.get('status')!='FROZEN': raise SystemExit('Manifest is not frozen')
 if verify_frozen_hashes(Path(__file__).resolve().parent.parent)['status']!='PASS': raise SystemExit('Frozen estimator hash audit failed')
 manifest=pd.read_csv(a.manifest_dir/f'{PREFIX}_FROZEN_MANIFEST.csv'); array_name=str(lock['array_name']); channels=np.arange(EXPECTED_CHANNEL_COUNT,dtype=int)
 if canonical_manifest_sha256(manifest.to_dict('records'))!=lock['manifest_canonical_sha256']: raise SystemExit('Manifest SHA mismatch')
 audit,_=audit_locata_root(a.root,a.output/'geometry_audit',array_name=array_name); usable=audit.loc[audit['usable'].astype(bool),'recording'].tolist(); recmap={(infer_locata_task(Path(x)),Path(x).name):Path(x) for x in usable}
 r8=CausalMultiscaleHodgeParameters(); fp=BaselineForensicParameters(); mp=MultiMethodParameters(FROZEN_ASSIGNMENT_STRENGTH,FROZEN_ANCHOR_STRENGTH); allrows=[]
 for (task,rn),sub in manifest.groupby(['task','recording_name'],sort=True):
  rec=recmap.get((str(task),str(rn))); 
  if rec is None: raise SystemExit(f'Missing frozen recording {(task,rn)}')
  out=per/f'{PREFIX}_TRIALS_{task}_{rn}.csv'; stat=per/f'{PREFIX}_STATUS_{task}_{rn}.json'
  if a.resume and out.is_file() and stat.is_file():
   df=pd.read_csv(out)
   if len(df)==len(sub): allrows.append(df); continue
  windows,_=load_broadband_windows(rec,array_name=array_name,window_duration_s=max(r8.scales_s),max_windows=0,window_anchor_mode='required_time_causal',activity_aware_max_windows=False)
  wm={int(w['required_time_row_index']):w for w in windows}; rows=[]
  for pos,q in enumerate(sub.sort_values('selection_rank').itertuples(),1):
   f=wm[int(q.required_time_row_index)]; si=int(q.active_source_index); truth=float(np.asarray(f['theta_true_deg'])[si]); audio=np.asarray(f['audio'],float)[channels]; mic=np.asarray(f['microphone_positions_m'],float)[channels]
   t=time.perf_counter(); banks=prepare_r8m_state_banks(audio,f['sample_rate_hz'],mic,r8_params=r8,forensic_params=fp)
   res=rescore_prepared_r8m_multimethod_banks(banks,structural_variant=FINAL_METHOD,r8m_params=mp,r8_params=r8); est=float(res['estimate_deg']); e0=float(res['e0_estimate_deg']); se=float(_circular_difference_deg(est,truth)); e0se=float(_circular_difference_deg(e0,truth))
   rows.append({'release':RELEASE,'protocol_id':PROTOCOL_ID,'task':task,'recording_name':rn,'required_time_row_index':int(q.required_time_row_index),'active_source_index':si,'true_azimuth_deg':truth,
    'e0_estimate_deg':e0,'estimated_azimuth_deg':est,'e0_signed_error_deg':e0se,'signed_error_deg':se,'e0_absolute_error_deg':abs(e0se),'absolute_error_deg':abs(se),'e0_squared_error_deg2':e0se*e0se,'squared_error_deg2':se*se,
    'adopt':bool(res['adopt_joint_path']),'pre_guard_adopt':bool(res['pre_guard_adopt_joint_path']),'guard_triggered':bool(res['no_reassignment_guard_triggered']),'reassigned_pair_count':int(res['selected_long_reassigned_pair_count']),'model_selection_gain':float(res['model_selection_gain']),
    'method':FINAL_METHOD,'assignment_strength':FROZEN_ASSIGNMENT_STRENGTH,'anchor_strength':FROZEN_ANCHOR_STRENGTH,'truth_used_for_estimation':False,'task_identity_used_for_estimation':False,'holdout_used_for_tuning':False})
   print(f'[holdout {task}/{rn}] {pos}/{len(sub)} row={q.required_time_row_index} adopt={res["adopt_joint_path"]} time={time.perf_counter()-t:.2f}s',flush=True)
  df=pd.DataFrame(rows); df.to_csv(out,index=False); stat.write_text(json.dumps({'status':'completed','task':task,'recording_name':rn,'windows':len(df)},indent=2)+'\n'); allrows.append(df)
 trials=pd.concat(allrows,ignore_index=True); trials.to_csv(a.output/f'{PREFIX}_TRIALS.csv',index=False)
 status={'release':RELEASE,'protocol_id':PROTOCOL_ID,'status':'completed','array_name':array_name,'total_windows':len(trials),'recordings':int(trials[['task','recording_name']].drop_duplicates().shape[0]),'method':FINAL_METHOD,'assignment_strength':FROZEN_ASSIGNMENT_STRENGTH,'anchor_strength':FROZEN_ANCHOR_STRENGTH,'scientific_identity':'outcome-unseen post-freeze cross-task extension within LOCATA','independent_cross_array_holdout':False,'independent_corpus_claimed':False,'holdout_used_for_tuning':False}
 (a.output/f'{PREFIX}_EXECUTION_STATUS.json').write_text(json.dumps(status,indent=2)+'\n'); print(json.dumps(status,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
