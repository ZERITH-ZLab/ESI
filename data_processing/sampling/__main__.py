"""Inspect/sample a frozen index without loading images or starting training."""
import argparse
from pathlib import Path
import numpy as np
from data_processing.subtasks.io import write, require
from data_processing.subtasks.pipeline import batch_lock
from .sampler import HierarchicalSampler, load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',required=True,help='Completed training_index directory')
    parser.add_argument('--config',help='Defaults: GPT .5, random .25, successful episode .7, seed 0')
    parser.add_argument('--draws',type=int,default=100000,help='Diagnostic draws only, not training steps')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    require(args.draws>0,'draws must be positive')
    index,output=Path(args.index).resolve(),Path(args.output).resolve()
    require(not output.is_relative_to(index),'Do not write into the frozen index')
    require(not output.is_relative_to(Path(__file__).resolve().parents[2]),'Use mounted disk or /tmp for diagnostic outputs')
    with batch_lock(output):
        require(not any(p.name!='.lock' for p in output.iterdir()),'Output already contains a preview; use a new directory')
        sampler=HierarchicalSampler(index,load_config(args.config))
        write(output/'plan.json',sampler.plan())
        write(output/'exposure.json',sampler.exposure(args.draws))
        indices=sampler.draw_indices(args.draws)
        # Audit/replay draw order without duplicating images/actions.
        np.savez_compressed(output/'draws.npz',row_index=indices,sample_id=sampler.sample_ids[indices])
        write(output/'sampler_state.json',sampler.state_dict())
        report=sampler.report(indices)
        write(output/'report.json',report)
        write(output/'manifest.json',dict(status='completed',fingerprint=sampler.fingerprint,
                                         index=str(index),draws=args.draws))
        import json
        print(json.dumps(dict(output=str(output),fingerprint=sampler.fingerprint,
              fallback_groups=len(sampler.fallbacks),**{k:report[k] for k in
              ('draws','unique_windows','fractions','supervised_steps')}),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
