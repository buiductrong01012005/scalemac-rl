from __future__ import annotations
import argparse
from pathlib import Path
from scalemac_rl.dimensionwise_ppo_analysis import build_dimensionwise_analysis

def main():
    p=argparse.ArgumentParser(); p.add_argument('--round-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--diagnostic-slots',type=int,default=512); a=p.parse_args()
    for k,v in build_dimensionwise_analysis(round_dir=a.round_dir,output_dir=a.output_dir,device=a.device,diagnostic_slots=a.diagnostic_slots).items(): print(f'{k}: {v}')
if __name__=='__main__': main()
