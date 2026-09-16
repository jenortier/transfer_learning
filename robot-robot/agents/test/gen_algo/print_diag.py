from pathlib import Path
import pickle, numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "data" / "models").exists())
DATA_ROOT = ROOT / "data"

p = DATA_ROOT / "DIRECT_GEN_ALGO" / "diagnostics_30.pkl"
with open(p,'rb') as f:
    d=pickle.load(f)
res=d['results']
print('Episodes run:', len(res['episodes']))
print('Successes:', sum(1 for e in res['episodes'] if e['success']))
print('Pred eff min dist median:', np.nanmedian(res['pred_eff_to_target_m']))
print('True eff min dist median:', np.nanmedian(res['true_eff_to_target_m']))
medians=[x['median_deg'] for x in res['angle_errors_deg'] if x['median_deg'] is not None]
print('Angle median of medians (deg):', np.median(medians) if medians else None)
print('Angle 90th median:', np.median([x['90th_deg'] for x in res['angle_errors_deg'] if x['90th_deg'] is not None]))
print('\nShowing up to 2 failed example summaries (first/last 3 steps):')
for idx, fe in enumerate(d['failed_examples'][:2]):
    print('\nFailed episode:', fe['episode'], 'steps:', len(fe['steps']))
    steps=fe['steps']
    n=len(steps)
    show_indices=list(range(min(3,n)))+list(range(max(0,n-3),n)) if n>6 else list(range(n))
    for i in show_indices:
        s=steps[i]
        print(f" step {s['step']:3d} | pred_to_target={s['pred_to_target_m']:.3f}m | true_to_target={s['true_to_target_m']:.3f}m | angle_err(deg)={s['angle_err_deg']}")
        print('   raw angles deg:', (np.array(s['state_raw'][:2])*180.0/np.pi).round(1).tolist())
        print('   proj angles deg:', (np.array(s['state_proj'][:2])*180.0/np.pi).round(1).tolist())
        print('   ik_sol deg   :', (np.array(s['ik_sol'])*180.0/np.pi).round(1).tolist())
        print('   act2:', s['act2'].tolist(), 'act3:', np.round(s['act3'],3).tolist())
