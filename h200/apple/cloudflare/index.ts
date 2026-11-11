import { DurableObject } from 'cloudflare:workers';

const MODELS=['va_k96','va_k128','convnextv2_atto','tinyvim_s','parc_net_s'];
export const JOBS=new Set(MODELS.flatMap(m=>[501,509,521].map(s=>`${m}-${s}`)));
const K96_COCO_JOBS=new Set(['k96coco-521-8k']);
const VA64_OPT_JOBS=new Set(['apple70m-data-preflight','apple70m-train','apple70m-vonly','apple70m-vonly-checkpoint']);
const CHUNK=256*1024;
const VA64_MAX_ARTIFACT_BYTES=384*1024**2;
const VA64_MAX_ARTIFACT_CHUNKS=Math.ceil(VA64_MAX_ARTIFACT_BYTES/CHUNK);
type Obj=Record<string,any>;
type ControlState={stop:boolean,force:boolean,ready_jobs:string[],finished_jobs:string[],observer_seen:number};
type ArtifactMeta={sha:string,bytes:number,chunks:number,epoch:number,complete:boolean,verified:boolean,created:number,workers:number,chunk_bytes?:number,global_step?:number,code_sha?:string};
const reply=(value:unknown,status=200)=>Response.json(value,{status,headers:{'Cache-Control':'no-store'}});
async function body(req:Request,limit=65536):Promise<Obj> {
  if(Number(req.headers.get('Content-Length')||0)>limit)throw Error('body_limit');
  const reader=req.body?.getReader();let n=0;const parts:Uint8Array[]=[];
  if(!reader)throw Error('empty_body');
  for(;;){const r=await reader.read();if(r.done)break;n+=r.value.length;if(n>limit){await reader.cancel();throw Error('body_limit');}parts.push(r.value);}
  const bytes=new Uint8Array(n);let at=0;for(const part of parts){bytes.set(part,at);at+=part.length;}
  const value=JSON.parse(new TextDecoder().decode(bytes));
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error('object_required');return value;
}
async function digest(value:string){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value)))).map(x=>x.toString(16).padStart(2,'0')).join('');}

export class Va64Control extends DurableObject<Env> {
  constructor(ctx:DurableObjectState,env:Env){super(ctx,env);
    ctx.storage.sql.exec('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)');
    ctx.storage.sql.exec('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, seq INTEGER, value TEXT, UNIQUE(session,seq))');
    // Bound cursor polling reads: UNIQUE(session,seq) otherwise scans the full session.
    ctx.storage.sql.exec('CREATE INDEX IF NOT EXISTS events_session_id ON events(session,id)');
  }
  read(key:string):Obj|null {const rows=this.ctx.storage.sql.exec<{value:string}>('SELECT value FROM meta WHERE key=?',key).toArray();return rows.length?JSON.parse(rows[0].value):null;}
  write(key:string,value:unknown){this.ctx.storage.sql.exec('INSERT OR REPLACE INTO meta VALUES (?,?)',key,JSON.stringify(value));}
  enroll(value:Obj){
    const old=this.read('session');
    if(old&&!old.ended){
      if(old.pod===value.pod&&old.tokenHash===value.token_hash&&old.code===value.code_sha)return {session_id:old.id,chunk_bytes:CHUNK};
      throw Error('active_session_requires_owner_release');
    }
    if(this.read('control')?.stop)throw Error('campaign_stopped');
    const session={id:crypto.randomUUID(),pod:value.pod,tokenHash:value.token_hash,code:value.code_sha,
      created:Date.now(),lastSeen:Date.now(),ended:false};
    this.write('session',session);const c=this.control();c.ready_jobs=[];c.finished_jobs=[];this.write('control',c);return {session_id:session.id,chunk_bytes:CHUNK};
  }
  auth(id:string,hash:string){const s=this.read('session');return !!s&&s.id===id&&s.tokenHash===hash;}
  control():ControlState{return {stop:false,force:false,ready_jobs:[],finished_jobs:[],observer_seen:0,...(this.read('control')||{})};}
  command(action:string,job?:string,scope='in10'){
    const c=this.control();
    const jobs=scope==='va64g'?VA64_OPT_JOBS:scope==='k96coco'?K96_COCO_JOBS:JOBS;
    if(action==='stop'){c.stop=true;c.force=false;}
    else if(action==='force'){c.stop=true;c.force=true;}
    else if(action==='ready'&&job&&jobs.has(job)){c.ready_jobs=Array.from(new Set([...c.ready_jobs,job]));}
    else if(action==='finish'&&job&&jobs.has(job)){c.finished_jobs=Array.from(new Set([...c.finished_jobs,job]));}
    else if(action==='heartbeat'){c.observer_seen=Date.now();}
    else if(action==='release'){const s=this.read('session');if(s){s.ended=true;this.write('session',s);}}
    else if(action==='arm'){const s=this.read('session');if(s&&!s.ended)throw Error('active_session');c.stop=false;c.force=false;c.ready_jobs=[];c.finished_jobs=[];}
    else throw Error('invalid_command');
    this.write('control',c);return c;
  }
  append(id:string,value:Obj){
    const s=this.read('session');if(!s||s.id!==id)throw Error('wrong_session');
    this.ctx.storage.sql.exec('INSERT OR IGNORE INTO events(session,seq,value) VALUES (?,?,?)',id,value.seq,JSON.stringify(value));
    this.ctx.storage.sql.exec('DELETE FROM events WHERE id < (SELECT MAX(id)-3000 FROM events)');
    s.lastSeen=Date.now();if(value.status?.ended)s.ended=true;this.write('session',s);
    return {ok:true,seq:value.seq};
  }
  snapshot(after:number){const s=this.read('session');return {session:s?{id:s.id,pod:s.pod,code:s.code,lastSeen:s.lastSeen,ended:s.ended}:null,
    control:this.control(),events:this.ctx.storage.sql.exec('SELECT id,session,value FROM events WHERE id>? AND session=? ORDER BY id LIMIT 100',after,s?.id||'').toArray()};}
}

export class Va64Artifact extends DurableObject<Env> {
  constructor(ctx:DurableObjectState,env:Env){super(ctx,env);
    ctx.storage.sql.exec('CREATE TABLE IF NOT EXISTS versions (sha TEXT PRIMARY KEY,value TEXT)');
    ctx.storage.sql.exec('CREATE TABLE IF NOT EXISTS chunks (sha TEXT,part INTEGER,data BLOB,PRIMARY KEY(sha,part))');
  }
  private meta(sha:string):ArtifactMeta|null {const r=this.ctx.storage.sql.exec<{value:string}>('SELECT value FROM versions WHERE sha=?',sha).toArray();return r.length?JSON.parse(r[0].value):null;}
  begin(m:Obj,retentionLimit=3){
    const existing=this.meta(m.sha);if(existing){
      if(existing.chunks!==m.chunks||existing.bytes!==m.bytes||existing.epoch!==m.epoch
        ||existing.global_step!==m.global_step||existing.code_sha!==m.code_sha)throw Error('artifact_identity_changed');
      return existing;
    }
    const rows=this.ctx.storage.sql.exec<{sha:string,value:string}>('SELECT sha,value FROM versions').toArray();
    const created=Math.max(Date.now(),...rows.map(row=>(JSON.parse(row.value).created??0)+1));
    for(const row of rows){
      const prior=JSON.parse(row.value);
      const older=prior.epoch<m.epoch||(prior.epoch===m.epoch&&(
        (prior.global_step??0)<(m.global_step??0)
        ||((prior.global_step??0)===(m.global_step??0)&&(prior.created??0)<created)));
      if(!prior.complete&&older){this.ctx.storage.sql.exec('DELETE FROM chunks WHERE sha=?',row.sha);this.ctx.storage.sql.exec('DELETE FROM versions WHERE sha=?',row.sha);}
    }
    // Keep only verified recovery points plus the incoming version. Old verified
    // versions are removed only after the controller confirms a newer download.
    const count=this.ctx.storage.sql.exec<{n:number}>('SELECT COUNT(*) AS n FROM versions').one().n;
    if(count>=retentionLimit)throw Error('artifact_queue_full');
    const value:ArtifactMeta={sha:m.sha,bytes:m.bytes,chunks:m.chunks,epoch:m.epoch,workers:m.workers??0,
      chunk_bytes:m.chunk_bytes??CHUNK,global_step:m.global_step,code_sha:m.code_sha,complete:false,verified:false,created};
    this.ctx.storage.sql.exec('INSERT INTO versions VALUES (?,?)',m.sha,JSON.stringify(value));return value;
  }
  put(sha:string,part:number,data:ArrayBuffer){
    const m=this.meta(sha);if(!m||m.complete||part>=m.chunks)throw Error('invalid_chunk');
    const size=m.chunk_bytes??CHUNK;
    const expected=part===m.chunks-1?m.bytes-size*part:size;
    if(data.byteLength!==expected)throw Error('chunk_size');
    this.ctx.storage.sql.exec('INSERT OR REPLACE INTO chunks VALUES (?,?,?)',sha,part,data);return {ok:true};
  }
  finish(sha:string){const m=this.meta(sha);if(!m)throw Error('missing_artifact');
    const r=this.ctx.storage.sql.exec<{n:number,bytes:number}>('SELECT COUNT(*) AS n,SUM(length(data)) AS bytes FROM chunks WHERE sha=?',sha).one();
    if(r.n!==m.chunks||r.bytes!==m.bytes)throw Error('incomplete_artifact');
    m.complete=true;this.ctx.storage.sql.exec('UPDATE versions SET value=? WHERE sha=?',JSON.stringify(m),sha);return m;
  }
  pending(){return this.ctx.storage.sql.exec<{sha:string,value:string}>('SELECT sha,value FROM versions').toArray().map(row=>({
    ...JSON.parse(row.value),uploaded_chunks:this.ctx.storage.sql.exec<{n:number}>('SELECT COUNT(*) AS n FROM chunks WHERE sha=?',row.sha).one().n}));}
  latest():ArtifactMeta|null {const rows:ArtifactMeta[]=this.ctx.storage.sql.exec<{value:string}>('SELECT value FROM versions').toArray().map(r=>JSON.parse(r.value));
    return rows.filter(r=>r.complete).sort((a,b)=>b.epoch-a.epoch||(b.global_step??0)-(a.global_step??0)||b.created-a.created)[0]||null;}
  get(sha:string,part:number){const r=this.ctx.storage.sql.exec<{data:ArrayBuffer}>('SELECT data FROM chunks WHERE sha=? AND part=?',sha,part).toArray();return r.length?r[0].data:null;}
  ack(sha:string){const m=this.meta(sha);if(!m?.complete)throw Error('not_complete');m.verified=true;
    this.ctx.storage.sql.exec('UPDATE versions SET value=? WHERE sha=?',JSON.stringify(m),sha);
    const old=this.ctx.storage.sql.exec<{sha:string,value:string}>('SELECT sha,value FROM versions').toArray();
    for(const row of old){const prior:ArtifactMeta=JSON.parse(row.value);
      const older=prior.epoch<m.epoch||(prior.epoch===m.epoch&&(
        (prior.global_step??0)<(m.global_step??0)
        ||((prior.global_step??0)===(m.global_step??0)&&(prior.created??0)<m.created)));
      if(row.sha!==sha&&older){
        this.ctx.storage.sql.exec('DELETE FROM chunks WHERE sha=?',row.sha);this.ctx.storage.sql.exec('DELETE FROM versions WHERE sha=?',row.sha);}}
    return m;}
}

export default {
  async fetch(req:Request,env:Env):Promise<Response>{try{
    const url=new URL(req.url),k96=false,va64g=true;
    if(!url.pathname.startsWith('/va64g/'))return reply({error:'wrong_namespace'},404);
    const prefix='/va64g';
    const route=url.pathname.slice(prefix.length);
    if(route==='/health')return reply({ok:true,campaign:'apple70m-h200-vonly-v1',jobs:4});
    const token=(req.headers.get('Authorization')||'').replace(/^Bearer /,'');
    const owner=!!env.IN10_OWNER_TOKEN&&token===env.IN10_OWNER_TOKEN;
    const test=url.searchParams.get('test')==='1';if(test&&!owner)return reply({error:'test_requires_owner'},403);
    const controlName=k96?(test?'k96-coco521-test':'k96-coco521-v1')
      :va64g?(test?'va64g-test':'va64g-v1')
      :(test?'simclr-in1k10-test':'simclr-in1k10-v1');
    const control=env.IN10_CONTROL.getByName(controlName);
    const ip=req.headers.get('CF-Connecting-IP')||'';
    const allowed=!!env.ALLOWED_EGRESS_IPS&&env.ALLOWED_EGRESS_IPS.split(',').map(s=>s.trim()).includes(ip);
    if(!owner&&!allowed)return reply({error:'source_not_allowed'},403);
    const rateScope=k96?'k96coco:':va64g?'va64g:':'in10:';
    if(!await env.RELAY_RATE_LIMITER.limit({key:rateScope+ip+':'+String(owner)}).then(r=>r.success))return reply({error:'rate_limited'},429);
    if(route==='/enroll'&&req.method==='POST'){
      const m=await body(req,2048);
      if(!/^job-daehwa00-\d+-[a-z0-9]+$/.test(m.pod)||! /^[a-f0-9]{64}$/.test(m.token_hash)||! /^[a-f0-9]{40}$/.test(m.code_sha))return reply({error:'enrollment_identity'},400);
      if(!env.EXPECTED_CODE_SHA || m.code_sha!==env.EXPECTED_CODE_SHA)return reply({error:'code_not_allowed'},403);
      return reply(await control.enroll(m));}
    const session=req.headers.get('X-Session-ID')||'';
    if(!owner&&(!/^[a-f0-9]{64}$/.test(token)||!await control.auth(session,await digest(token))))return reply({error:'unauthorized'},401);
    if(route==='/control'&&req.method==='GET')return reply(await control.control());
    if(route==='/events'&&req.method==='POST'){
      const m=await body(req);if(!Number.isSafeInteger(m.seq)||m.seq<0||typeof m.logs!=='string'||m.logs.length>24000||!m.status)return reply({error:'event_shape'},400);
      return reply(await control.append(session,m));}
    if(owner&&route==='/snapshot'&&req.method==='GET'){
      const after=Number(url.searchParams.get('after')||0);if(!Number.isSafeInteger(after)||after<0)return reply({error:'cursor'},400);
      return reply(await control.snapshot(after));}
    if(owner&&route==='/command'&&req.method==='POST'){
      const m=await body(req,2048);
      return reply(await control.command(m.action,m.job,va64g?'va64g':k96?'k96coco':'in10'));}
    const parts=route.split('/').filter(Boolean);
    const artifactJobs=k96?K96_COCO_JOBS:va64g?VA64_OPT_JOBS:JOBS;
    if(parts[0]==='artifact'&&artifactJobs.has(parts[1]||'')){
      const artifactPrefix=k96?'k96coco/':va64g?'va64g/':'';
      const artifact=env.IN10_ARTIFACT.getByName(`${artifactPrefix}${test?'test':'v1'}/${parts[1]}`);
      if(owner&&parts[2]==='pending'&&req.method==='GET')return reply(await artifact.pending());
      if(parts[2]==='latest'&&req.method==='GET')return reply(await artifact.latest());
      if(parts[2]==='begin'&&req.method==='POST'){
        const m=await body(req,4096);
        const chunkBytes=k96?8192:CHUNK;
        const maxBytes=va64g?VA64_MAX_ARTIFACT_BYTES:192*1024**2;
        const maxEpoch=100;
        const minEpoch=va64g?0:1;
        const maxChunks=va64g?VA64_MAX_ARTIFACT_CHUNKS:Math.ceil(maxBytes/chunkBytes);
        if(!/^[a-f0-9]{64}$/.test(m.sha)||!Number.isInteger(m.epoch)||m.epoch<minEpoch||m.epoch>maxEpoch
          ||!Number.isInteger(m.bytes)||m.bytes<1||m.bytes>maxBytes||m.chunks!==Math.ceil(m.bytes/chunkBytes)
          ||m.chunks>maxChunks
          ||(va64g&&(!Number.isSafeInteger(m.global_step)||m.global_step<0||! /^[a-f0-9]{40}$/.test(m.code_sha||''))))
          return reply({error:'artifact_shape'},400);
        m.chunk_bytes=chunkBytes;
        return reply(await artifact.begin(m,va64g?2:3));}
      const sha=parts[2];if(!/^[a-f0-9]{64}$/.test(sha||''))return reply({error:'sha'},400);
      if(parts[3]==='finish'&&req.method==='POST')return reply(await artifact.finish(sha));
      if(parts[3]==='ack'&&req.method==='POST')return owner?reply(await artifact.ack(sha)):reply({error:'owner_required'},403);
      const maxPart=va64g?VA64_MAX_ARTIFACT_CHUNKS-1:6143;
      const part=Number(parts[3]);if(!/^\d+$/.test(parts[3]||'')||!Number.isSafeInteger(part)||part>maxPart)return reply({error:'part'},400);
      if(req.method==='PUT'||(k96&&owner&&req.method==='POST')){
        const length=Number(req.headers.get('Content-Length')||0);
        if(!Number.isInteger(length)||length<1||length>CHUNK*1.4)return reply({error:'chunk_size'},413);
        let bytes:ArrayBuffer;
        if(req.headers.get('Content-Type')==='application/json'){
          const value=await body(req,Math.ceil(CHUNK*1.4));
          if(typeof value.base64!=='string')return reply({error:'chunk_encoding'},400);
          bytes=Uint8Array.from(atob(value.base64),c=>c.charCodeAt(0)).buffer;
        }else bytes=await req.arrayBuffer();
        if(bytes.byteLength>CHUNK)return reply({error:'chunk_size'},413);
        return reply(await artifact.put(sha,part,bytes));}
      if(req.method==='GET'){
        const value=await artifact.get(sha,part);if(!value)return reply({error:'missing_chunk'},404);
        if(req.headers.get('Accept')==='application/json'){
          const bytes=new Uint8Array(value);let binary='';
          for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
          return reply({base64:btoa(binary)});
        }
        return new Response(value,{headers:{'Cache-Control':'no-store','Content-Type':'application/octet-stream','X-Content-Type-Options':'nosniff'}});}
    }
    return reply({error:'forbidden_route'},403);
  }catch(error){return reply({error:error instanceof Error?error.message:'request_failed'},409);}}
};
