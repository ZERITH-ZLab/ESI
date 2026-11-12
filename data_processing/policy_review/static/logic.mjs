export const COLORS={STUDENT:'#3889b8',EDIT:'#b39024',SHORT:'#ba8a2f',LONG:'#a16cc4'};
export const LABELS={STUDENT:'pi05',EDIT:'GPT edit',SHORT:'GPT short',LONG:'GPT long'};
export function label(d){return d.state==='subtask'?d.text:d.state==='null'?'仅 episode 指令 · 无 subtask':d.state==='legacy'?'旧路线 · 未启用子任务指导':'指令记录缺失 · 不等同于 null';}
export function frameIndex(time,dt,total){return Math.min(total,Math.max(0,Math.floor(time/dt+1e-6)));}
// A video frame k is the observation AFTER action k; frame zero previews the first decision.
export function decisionAtFrame(rows,frame){if(frame===0)return rows[0]||null;return rows.find(d=>d.start<frame&&frame<=d.end)||null;}
export function groups(rows){const result=[];for(const d of rows){if(d.executed_steps===0)continue;const key=JSON.stringify([d.state,d.text,d.instruction_format]);const last=result.at(-1);if(last&&last.end===d.start&&last.key===key){last.end=d.end;last.ids.push(d.id);}else result.push({key,start:d.start,end:d.end,ids:[d.id],text:label(d),state:d.state,skill:d.skill,instruction_format:d.instruction_format});}return result;}
export function matches(d,filter,query){const pass=filter==='all'||filter===d.state||(filter==='free_text'&&d.state==='subtask'&&d.instruction_format==='free_text')||(filter==='custom'&&d.skill==='custom')||(filter==='gpt'&&d.control!=='STUDENT')||(filter==='zero'&&d.executed_steps===0)||(filter==='failure'&&!!d.failure)||(filter.startsWith('outcome_')&&d.annotation?.outcome===filter.slice(8))||(filter==='review'&&d.annotation?.needs_review);return pass&&JSON.stringify([d.text,d.skill,d.arm,d.reason,d.assessment,d.conditioning?.current_intent,d.annotation]).toLowerCase().includes(query.toLowerCase());}

export const OUTCOMES={success:'#23886a',failure:'#bd5354',unknown:'#89939d',interrupted:'#b58a32'};
export function outcomeLabel(d){const a=d.annotation;return a?`${a.outcome} · ${a.outcome_basis}${a.needs_review?' · 待复核':''}`:'—';}
