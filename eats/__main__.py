"""Command-line entry point for the paper reproduction package."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'

import argparse
import json
from eats.io import verify_integrity

def positive(value):
    number=int(value)
    if number<1: raise argparse.ArgumentTypeError('must be positive')
    return number

def main():
    parser=argparse.ArgumentParser(description='Reproduce the e-ATS paper: saved-result analysis or exact seeded simulation.')
    sub=parser.add_subparsers(dest='command',required=True)
    for name,help_text in [('verify','Check integrity, all suite censuses, and paper claims'),('reproduce','Recompute tables and figures from the frozen per-run results')]:
        p=sub.add_parser(name,help=help_text)
        p.add_argument('--bootstrap',action='store_true',help='Recompute all paper confidence intervals (50,000 resamples)')
        if name=='reproduce':
            p.add_argument('--output',default='results/paper')
            p.add_argument('--no-figures',action='store_true')
    p=sub.add_parser('run',help='Execute policies and compare each trace with frozen results; resumable')
    p.add_argument('suite',choices=['registered','replay','calibration','null'])
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true',help='Small deterministic subset at the original horizons')
    mode.add_argument('--full',action='store_true',help='All frozen groups; can take substantial compute')
    p.add_argument('--workers',type=positive,default=1)
    p.add_argument('--policies',nargs='+',default=['all'],help='all, e-ats, open-ramp, ots, certified-glr, practical-glr, calibrated-glr, or baseline name (e.g. ts, sw-ucb; see docs/ALGORITHMS.md)')
    p.add_argument('--k',type=int,choices=[4,8,16,32,64])
    p.add_argument('--family',help='Exact registered mechanism or replay family name')
    p.add_argument('--limit',type=positive,help='First N selected groups; explicitly a partial rerun')
    p.add_argument('--output')
    args=parser.parse_args()
    files=verify_integrity()
    if args.command=='run':
        from eats.experiments.runner import run
        result=run(args)
    else:
        from eats.analysis import compute, verify_paper, write_report
        result=compute(bootstrap=args.bootstrap)
        verify_paper(result)
        if args.command=='reproduce':
            out=write_report(result,args.output)
            if not args.no_figures:
                from eats.figures import build
                build(out/'figures')
            print(f'Wrote verified results to {out}')
        result={'status':'passed','integrity_files':files,'counts':result['counts'],'bootstrap_recomputed':args.bootstrap}
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
