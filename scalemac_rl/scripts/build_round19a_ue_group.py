from __future__ import annotations
import argparse
from pathlib import Path
from scalemac_rl.ue_group_ppo_analysis import build_ue_group_analysis

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--round-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--diagnostic-slots',type=int,default=512)
    a=p.parse_args()
    for k,v in build_ue_group_analysis(round_dir=a.round_dir,output_dir=a.output_dir,device=a.device,diagnostic_slots=a.diagnostic_slots).items():
        print(f'{k}: {v}')

if __name__=='__main__': main()
