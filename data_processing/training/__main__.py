import argparse
import json
import os
from .config import load_config


def main():
    parser=argparse.ArgumentParser(description='Indexed pi05 full fine-tuning with masked temporal loss')
    parser.add_argument('command',choices=['show-config','check-data','run'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--mode', choices=['task', 'subtask'], help='Override text conditioning mode')
    parser.add_argument('--stop-after',type=int,help='Stop and save after this many additional updates; schedule unchanged')
    args=parser.parse_args()
    cfg=load_config(args.config, mode=args.mode)
    print(json.dumps(dict(event='resolved_training_config', config=cfg,
                         xla_client_mem_fraction=os.environ.get('XLA_CLIENT_MEM_FRACTION'),
                         per_device_batch_size=cfg['training']['micro_batch_size']//cfg['parallelism']['num_devices'],
                         gradient_accumulation_steps=cfg['training']['batch_size']//cfg['training']['micro_batch_size']),
                     ensure_ascii=False), flush=True)
    if args.command == 'show-config':
        if args.resume or args.stop_after is not None: parser.error('resume/stop-after only apply to run')
        return
    from .runner import run,check_data
    if args.command=='check-data':
        if args.resume or args.stop_after is not None: parser.error('resume/stop-after only apply to run')
        check_data(cfg)
    else:
        run(cfg,resume=args.resume,stop_after=args.stop_after)
        import jax
        if jax.distributed.is_initialized():
            jax.distributed.shutdown()


if __name__=='__main__':
    main()
