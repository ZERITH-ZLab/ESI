import assert from 'node:assert/strict';
import {decisionAtFrame,frameIndex,groups,matches,label} from './static/logic.mjs';
const rows=[{id:0,start:0,end:2,executed_steps:2,state:'null',control:'STUDENT'},
{id:1,start:2,end:2,executed_steps:0,state:'subtask',text:'Grasp 5',skill:'grasp',control:'SHORT',failure:'planning_failed'},
{id:2,start:2,end:4,executed_steps:2,state:'subtask',text:'Grasp 5',skill:'grasp',control:'STUDENT'},
{id:3,start:4,end:6,executed_steps:2,state:'subtask',text:'Grasp 5',skill:'grasp',control:'STUDENT'}];
assert.equal(decisionAtFrame(rows,0).id,0);assert.equal(decisionAtFrame(rows,2).id,0);
assert.equal(decisionAtFrame(rows,3).id,2);assert.equal(decisionAtFrame(rows,4).id,2);
assert.equal(decisionAtFrame(rows,6).id,3);assert.equal(decisionAtFrame(rows,7),null);
assert.equal(frameIndex(.12,.04,6),3);assert.equal(frameIndex(100,.04,6),6);
assert.equal(groups(rows).length,2);assert.deepEqual(groups(rows)[1].ids,[2,3]);
assert(matches(rows[1],'zero','grasp'));assert(!matches(rows[2],'zero',''));
assert(matches(rows[1],'failure',''));assert(!matches(rows[0],'gpt',''));
assert.notEqual(label({state:'null'}),label({state:'legacy'}));
console.log('Policy review JS: ACK frame mapping, zero steps, grouping and filtering passed');
const annotated={...rows[2],annotation:{outcome:'failure',needs_review:true}};
assert(matches(annotated,'outcome_failure',''));
assert(!matches(annotated,'outcome_success',''));
assert(matches(annotated,'review',''));
const freeform={...rows[2],text:'Move the brush to the blocks, then sweep toward the pan.',
  skill:null,arm:null,instruction_format:'free_text'};
assert.equal(label(freeform),freeform.text);
assert(matches(freeform,'free_text','brush'));
assert(matches(freeform,'subtask','sweep'));
assert(!matches(freeform,'custom',''));
assert(!matches(rows[0],'free_text',''));
assert.equal(groups([freeform,{...freeform,id:3,start:4,end:6}]).length,1);
assert.equal(groups([freeform,{...freeform,id:3,start:4,end:6,instruction_format:'structured'}]).length,2);
