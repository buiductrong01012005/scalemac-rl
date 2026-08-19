from __future__ import annotations
import argparse
from pathlib import Path
from scalemac_rl.controlled_feature_oracle_analysis import build_controlled_feature_oracle_report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--docs-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    print(build_controlled_feature_oracle_report(docs_dir=a.docs_dir,output=a.output))
if __name__=='__main__': main()
