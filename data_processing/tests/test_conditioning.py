import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from data_processing.openpi_loader.conditioning import use_subtask, format_prompt
from data_processing.openpi_loader.dataset import IndexedDataset
from data_processing.subtasks.io import write
from data_processing.training.config import load_config
from data_processing.training_index.build import checksum
from data_processing.tests.test_openpi_loader import loader_index


@pytest.mark.parametrize('length,last_enabled', [(80,55),(50,25),(49,24),(30,14),(15,7),(10,4),(1,0)])
def test_instruction_boundaries(length,last_enabled):
    actual=[t for t in range(length) if use_subtask(100,100+length,100+t)]
    assert actual==list(range(last_enabled+1))


def test_config_modes_overrides_batch_derivation_and_typos():
    path=Path(__file__).parents[1]/'training/config.8gpu_train.json'
    cfg=load_config(path,environ={})
    assert cfg['conditioning']['mode']=='task'
    assert '_task_' in cfg['output']
    env=dict(TRAIN_MODE='subtask',TRAIN_PER_DEVICE_BATCH_SIZE='2',
             TRAIN_GRADIENT_ACCUMULATION_STEPS='3',TRAIN_JAX_PREALLOCATE='false',
             TRAIN_OVERRIDES_JSON='{"optimizer":{"learning_rate":0.00003},"sampling":{"random_ratio":0.5}}')
    cfg=load_config(path,environ=env)
    assert cfg['conditioning']['mode']=='subtask' and '_subtask_' in cfg['output']
    assert cfg['training']['micro_batch_size']==16 and cfg['training']['batch_size']==48
    assert cfg['runtime']['jax_preallocate'] is False
    assert cfg['optimizer']['learning_rate']==3e-5 and cfg['sampling']['random_ratio']==.5
    assert load_config(path,environ=env,mode='task')['conditioning']['mode']=='task'
    for overrides,match in [({'TRAIN_MODE':'bad'},'conditioning.mode'),
                            ({'TRAIN_BATCHSIZE':'2'},'Unknown training environment'),
                            ({'TRAIN_JAX_PREALLOCATE':'False'},'JSON number or boolean'),
                            ({'TRAIN_NUM_WORKERS':'"32"'},'Invalid training.num_workers'),
                            ({'TRAIN_PER_DEVICE_BATCH_SIZE':'2','TRAIN_GRADIENT_ACCUMULATION_STEPS':'4',
                              'TRAIN_BATCH_SIZE':'5'},'conflicts'),
                            ({'TRAIN_PER_DEVICE_BATCH_SIZE':'2','TRAIN_BATCH_SIZE':'5'},'Set TRAIN_GRADIENT'),
                            ({'TRAIN_OVERRIDES_JSON':'{"optimizer":{"typo":1}}'},'Unknown override')]:
        with pytest.raises(ValueError,match=match):load_config(path,environ=overrides)


def add_annotation(index):
    annotation=index/'annotation.json'
    write(annotation,dict(version='test',input_hash='input',total_steps=100,segments=[dict(
        segment_id='s',start_step=0,end_step=100,outcome='success',needs_review=False,
        training_eligibility=dict(low_level=True,high_level=False),
        instruction=dict(text='Left arm: Transport object to destination.',label_origin='retrospective'))]))
    with sqlite3.connect(index/'index.sqlite') as db:
        db.execute('update episodes set annotation_path=?,annotation_version=?,input_hash=?',
                   (str(annotation),'test','input'))
        db.execute('insert into sources values (?,?)',(str(annotation),checksum(annotation)))
    manifest=json.loads((index/'manifest.json').read_text())
    manifest['files_sha256']['index.sqlite']=checksum(index/'index.sqlite')
    write(index/'manifest.json',manifest)
    return annotation


def test_prompt_only_change_real_tail_and_retrospective_labels(loader_index):
    index,_,_=loader_index
    add_annotation(index)
    base=IndexedDataset(index)
    instruct=IndexedDataset(index,instruction_mode='subtask')
    assert instruct.conditioning.enabled.tolist()==[True,True,False]
    for i in range(3):
        a,b=base[i],instruct[i]
        for key in a:
            if key!='prompt':np.testing.assert_array_equal(a[key],b[key])
        assert b['prompt']==format_prompt('Move object','Left arm: Transport object to destination.' if i<2 else None)
    assert format_prompt(' Task\noriginal ')== ' Task\noriginal '
    import pickle
    assert pickle.loads(pickle.dumps(instruct))[0]['prompt']==instruct[0]['prompt']
    with (index/'annotation.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='Annotation changed'):IndexedDataset(index,instruction_mode='subtask')


def test_strict_tokenizer_matches_openpi_and_rejects_overflow():
    pytest.importorskip('openpi')
    from openpi.models.tokenizer import PaligemmaTokenizer
    from data_processing.openpi_loader.conditioning import StrictPaligemmaTokenizer
    tokenizer=PaligemmaTokenizer(200)
    strict=StrictPaligemmaTokenizer(tokenizer)
    for state in [None,np.linspace(-2,2,14),np.zeros(14)]:
        prompt='Move digits.\nCurrent subtask: Left arm: Grasp digit_5.'
        tokens,mask=tokenizer.tokenize(prompt,state)
        assert strict.length(prompt,state)==mask.sum()
        a,b=strict.tokenize(prompt,state)
        np.testing.assert_array_equal(a,tokens);np.testing.assert_array_equal(b,mask)
    with pytest.raises(ValueError,match='refusing silent truncation'):
        strict.tokenize('long instruction '*300,np.zeros(14))


def test_online_conditioning_last_decision_half_and_null(loader_index):
    from data_processing.online_subtask_outcomes import VERSION as ONLINE_VERSION
    index,_,_=loader_index
    path=add_annotation(index)
    ann=json.loads(path.read_text());ann.update(version=ONLINE_VERSION,route='online_subtask_outcome')
    s=ann['segments'][0]
    s['online_conditioning']=dict(subtask={'skill_type':'transport'},subtask_text='Left arm: Transport object to destination.',
        episode_instruction='Move object',effective_prompt='Move object\nCurrent subtask: Left arm: Transport object to destination.')
    def update():
        write(path,ann)
        with sqlite3.connect(index/'index.sqlite') as db:
            db.execute('update episodes set annotation_version=?',(ONLINE_VERSION,))
            db.execute('update sources set sha256=? where path=?',(checksum(path),str(path)))
        manifest=json.loads((index/'manifest.json').read_text())
        manifest['files_sha256']['index.sqlite']=checksum(index/'index.sqlite');write(index/'manifest.json',manifest)
    update()
    ds=IndexedDataset(index,instruction_mode='subtask')
    assert ds.conditioning.enabled.tolist()==[True,True,False]  # Starts 0,40,90; cutoff is 50.
    assert ds[2]['prompt']=='Move object'
    s['online_conditioning'].update(subtask=None,subtask_text=None,effective_prompt='Move object')
    s['instruction']['text']=None
    update()
    ds=IndexedDataset(index,instruction_mode='subtask')
    assert ds.conditioning.enabled.tolist()==[False,False,False]
    assert all(ds[i]['prompt']=='Move object' for i in range(3))


def test_online_cutoffs_keep_repeated_decisions_and_stop_at_null():
    from data_processing.openpi_loader.conditioning import online_instruction_cutoffs
    def s(i,a,b,text):
        return dict(segment_id=str(i),start_step=a,end_step=b,
                    online_conditioning=dict(subtask_text=text))
    segments=[s(0,0,15,'grasp'),s(1,15,30,'grasp'),s(2,30,40,'grasp'),
              s(3,40,45,None),s(4,45,50,'grasp'),s(5,50,50,'grasp'),s(6,50,51,'place')]
    assert online_instruction_cutoffs(segments)=={'0':35,'1':35,'2':35,'3':43,'4':48,'6':51}
    with pytest.raises(ValueError,match='overlapping'):
        online_instruction_cutoffs([s(0,0,15,'x'),s(1,10,20,'x')])
