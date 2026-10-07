import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Button } from '@heroui/react';
import { Box, Camera, Check, CircleHelp, Download, ImagePlus, Layers3, Maximize2, MousePointer2, Rotate3D, Settings2, Sparkles, X } from 'lucide-react';
import Preview from './Preview';
import './style.css';

const VIEWS = [
  { key: 'front', label: 'ด้านหน้า', icon: '↑' }, { key: 'right', label: 'ด้านขวา', icon: '→' },
  { key: 'back', label: 'ด้านหลัง', icon: '↓' }, { key: 'left', label: 'ด้านซ้าย', icon: '←' },
];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const API_BASE = import.meta.env.DEV ? 'http://127.0.0.1:8008' : '';
async function api(path, options = {}) {
  const response = await fetch(API_BASE + path, options);
  let data;
  try { data = await response.json(); } catch { throw new Error('ตัวรัน AI ยังไม่เปิด กรุณาเปิด Open FourView Studio.cmd'); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'ตัวรัน AI ตอบกลับผิดพลาด');
  return data;
}

function App() {
  const [images, setImages] = useState({});
  const [busy, setBusy] = useState(false), [model, setModel] = useState(null);
  const [quality, setQuality] = useState('draft'), [color, setColor] = useState(true);
  const [backend, setBackend] = useState(null), [job, setJob] = useState(null);
  const [format, setFormat] = useState('glb'), [message, setMessage] = useState('');
  const [reset, setReset] = useState(0);
  const [surface, setSurface] = useState('texture');
  const [inputMode, setInputMode] = useState('sheet');
  const refs = useRef({}), objectUrls = useRef(new Set()), mounted = useRef(true);
  const imageRevision = useRef(0);
  const singleImage = inputMode !== 'multiview';
  const requiredViews = singleImage ? VIEWS.slice(0, 1) : VIEWS;
  const ready = requiredViews.every(view => images[view.key]);

  const restoreReferences = async (latest, active = () => mounted.current) => {
    const revision = imageRevision.current;
    const saved = await Promise.all((latest.input_mode === 'sheet' ? [{ key: 'front', source: 'source' }] : latest.input_mode === 'front' ? VIEWS.slice(0, 1) : VIEWS).map(async view => {
      const response = await fetch(API_BASE + '/api/jobs/' + latest.id + '/images/' + (view.source || view.key));
      if (!response.ok) throw new Error('โหลดภาพอ้างอิงไม่สำเร็จ');
      return [view.key, new File([await response.blob()], view.key + '.png', { type: 'image/png' })];
    }));
    if (!active() || revision !== imageRevision.current) return;
    const restored = {};
    for (const [key, file] of saved) {
      const url = URL.createObjectURL(file); objectUrls.current.add(url);
      restored[key] = { file, url, generated: latest.input_mode === 'front' && key !== 'front' };
    }
    setImages(restored);
  };

  const changeInputMode = mode => {
    if (busy || mode === inputMode) return;
    imageRevision.current++;
    setInputMode(mode); setModel(null); setJob(null);
    if ((mode === 'sheet') !== (inputMode === 'sheet')) setImages({});
    else if (mode === 'front') setImages(previous => previous.front ? { front: previous.front } : {});
  };

  useEffect(() => {
    mounted.current = true;
    let active = true;
    const status = async () => {
      try { const result = await api('/api/health', { signal: AbortSignal.timeout(5000) }); if (active) setBackend(result); }
      catch { if (active) setBackend({ ready: false, reason: 'ตัวรัน AI ยังไม่เปิด' }); }
    };
    status();
    api('/api/latest').then(async latest => {
      if (!active || !latest || imageRevision.current) return;
      setModel({ ...latest, preview: API_BASE + latest.preview }); setJob(latest);
      setQuality(latest.quality); setColor(Boolean(latest.color));
      setInputMode(latest.input_mode || 'multiview');
      await restoreReferences(latest, () => active);
    }).catch(() => {});
    const interval = setInterval(status, 10000);
    return () => { active = false; mounted.current = false; clearInterval(interval); for (const url of objectUrls.current) URL.revokeObjectURL(url); };
  }, []);

  const addFiles = (files, slot) => {
    if (busy) return;
    if (singleImage && slot && slot !== 'front') return;
    imageRevision.current++;
    const selected = [...files].filter(file => ['image/png', 'image/jpeg', 'image/webp'].includes(file.type));
    if (!selected.length) { setMessage('เลือกไฟล์ JPG, PNG หรือ WebP'); return; }
    const next = {};
    for (let i = 0; i < Math.min(selected.length, slot || singleImage ? 1 : 4); i++) {
      const file = selected[i];
      if (file.size > 12 * 1024 * 1024) { setMessage('ใช้ภาพขนาดไม่เกิน 12 MB ต่อภาพ'); continue; }
      const key = slot || VIEWS[i].key, url = URL.createObjectURL(file);
      objectUrls.current.add(url); next[key] = { file, url };
    }
    if (!Object.keys(next).length) return;
    setImages(previous => singleImage ? next : ({ ...previous, ...next }));
    setModel(null); setJob(null);
  };

  const generate = async (resumeId = null, refineId = null) => {
    if ((!ready && !resumeId && !refineId) || busy) return;
    setBusy(true); setModel(null); setJob({ id: resumeId, stage: refineId ? 'กำลังเพิ่มรายละเอียดผิวจากภาพต้นฉบับ…' : resumeId ? 'กำลังกู้รูปทรงที่บันทึกไว้…' : 'กำลังส่งภาพให้ตัวรันในเครื่อง…' });
    try {
      const health = await api('/api/health', { signal: AbortSignal.timeout(10000) });
      setBackend(health);
      if (!health.ready) throw new Error(health.reason);
      const form = new FormData();
      if (!resumeId && !refineId) requiredViews.forEach(view => form.append(view.key, images[view.key].file));
      form.append('input_mode', inputMode);
      form.append('quality', quality); form.append('color', String(color));
      const { id } = refineId
        ? await api('/api/jobs/' + refineId + '/refine-texture', { method: 'POST' })
        : resumeId
        ? await api('/api/jobs/' + resumeId + '/resume', { method: 'POST' })
        : await api('/api/jobs', { method: 'POST', body: form });
      let failures = 0, generatedRestored = false;
      while (mounted.current) {
        await delay(2000);
        let result;
        try { result = await api('/api/jobs/' + id, { signal: AbortSignal.timeout(10000) }); failures = 0; }
        catch (error) { if (++failures >= 3) throw error; continue; }
        if (!mounted.current) return;
        setJob(result);
        if (result.generated_views && result.input_mode === 'front' && !generatedRestored) {
          await restoreReferences(result); generatedRestored = true;
        }
        if (result.status === 'failed') throw new Error(result.error || 'สร้างโมเดลไม่สำเร็จ');
        if (result.status === 'completed') {
          setModel({ ...result, preview: API_BASE + result.preview });
          if (result.input_mode === 'front' && !generatedRestored) await restoreReferences(result);
          setMessage('สร้างผิวโมเดลแล้ว · ' + result.faces.toLocaleString() + ' triangles'); break;
        }
      }
    } catch (error) {
      if (mounted.current) { setMessage(error.message); setJob(previous => ({ ...previous, status: 'failed', error: error.message })); }
    } finally { if (mounted.current) setBusy(false); }
  };

  const exportModel = async () => {
    if (!model) return;
    try {
      const artifactFormat = format === 'obj' && model.texture ? 'obj-zip' : format;
      const response = await fetch(API_BASE + '/api/jobs/' + model.id + '/artifacts/' + artifactFormat + (surface === 'shape' ? '?surface=shape' : ''));
      if (!response.ok) throw new Error('ดาวน์โหลดไม่สำเร็จ กรุณาเปิดตัวรัน AI ไว้');
      const url = URL.createObjectURL(await response.blob()), anchor = document.createElement('a');
      anchor.href = url; anchor.download = 'fourview-' + model.id.slice(0, 8) + '.' + (artifactFormat === 'obj-zip' ? 'zip' : format);
      anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (error) { setMessage(error.message); }
  };

  return <div className="app" onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); addFiles(event.dataTransfer.files); }}>
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><Box size={19} /></div><span>มิติ</span><span className="brand-tag">FOURVIEW STUDIO</span></div>
      <div className="top-center"><span className={'status-dot ' + (backend?.ready ? '' : 'offline')} />{backend?.ready ? 'Local AI พร้อมใช้งาน' : 'กำลังเตรียม Local AI'}</div>
      <div className="top-actions"><button className="icon-button" aria-label="วิธีใช้งาน" onClick={() => setMessage('เพิ่มภาพหน้า ขวา หลัง ซ้าย เปิดตัวรัน AI แล้วกดสร้างโมเดล GLB เก็บสีได้ ส่วน OBJ/STL ใช้สำหรับรูปทรง')}><CircleHelp size={18} /></button><div className="avatar">ม</div></div>
    </header>
    <main>
      <section className="welcome">
        <div><div className="eyebrow"><Sparkles size={13} /> 2D → 3D WORKSPACE</div><h1>เปลี่ยนภาพถ่าย<br /><span>ให้กลายเป็นรูปทรง</span></h1><p>สร้างโมเดลจากภาพเดียวหรือภาพ 4 มุม ด้วย GPU ในเครื่อง</p></div>
        <div className="steps"><div className={'step ' + (!model && !busy ? 'active' : '')}><b>01</b><span>เพิ่มภาพ</span></div><i /><div className={'step ' + (busy ? 'active' : '')}><b>02</b><span>สร้างโมเดล</span></div><i /><div className={'step ' + (model ? 'active' : '')}><b>03</b><span>ส่งออก</span></div></div>
      </section>
      <div className="workspace">
        <aside className="panel inputs-panel">
          <div className="panel-heading"><div><div className="panel-kicker">ขั้นตอนที่ 1</div><h2>ภาพอ้างอิง</h2></div><span className={'count ' + (ready ? 'done' : '')}>{requiredViews.filter(view => images[view.key]).length} / {requiredViews.length} {ready && <Check size={13} />}</span></div>
          <div className="input-mode-tabs" role="group" aria-label="เลือกวิธีเพิ่มภาพ">
            <Button className={inputMode === 'multiview' ? 'selected' : ''} aria-pressed={inputMode === 'multiview'} isDisabled={busy} onPress={() => changeInputMode('multiview')}>อัปโหลด 4 มุม</Button>
            <Button className={singleImage ? 'selected' : ''} aria-pressed={singleImage} isDisabled={busy} onPress={() => changeInputMode('sheet')}>อัปโหลดภาพเดียว</Button>
          </div>
          {singleImage && <select className="engine-select" aria-label="รูปแบบภาพเดียว" value={inputMode} disabled={busy} onChange={event => changeInputMode(event.target.value)}><option value="sheet">ภาพรวม 4 มุมในไฟล์เดียว (2×2)</option><option value="front">ภาพด้านหน้าเดียว · AI สร้างมุมเพิ่ม</option></select>}
          <p className="panel-copy">{inputMode === 'sheet' ? 'อัปโหลดภาพรวมแบบรูปวัว: บนซ้าย หน้า · บนขวา ซ้าย · ล่างซ้าย ขวา · ล่างขวา หลัง ระบบแยกมุมแล้วสร้าง 3D ทันที' : inputMode === 'front' ? 'เพิ่มภาพด้านหน้า ระบบจะเจนซ้าย ขวา และหลัง แล้วสร้างโมเดลต่ออัตโนมัติ' : 'ใช้ภาพตัวละครเดียวกันจาก 4 มุม และสัดส่วนตรงกัน'}</p>
          <div className={'view-grid' + (singleImage ? ' single-image-grid' : '')}>{requiredViews.map(view => <div key={view.key}>
            <button className={'view-card ' + (images[view.key] ? 'has-image' : '')} disabled={busy} aria-label={singleImage ? 'อัปโหลดภาพเดียว' : 'เพิ่มภาพ' + view.label} onClick={() => refs.current[view.key]?.click()} onDrop={event => { event.preventDefault(); event.stopPropagation(); addFiles(event.dataTransfer.files, view.key); }}>
              {images[view.key] ? <><img src={images[view.key].url} alt={singleImage ? 'ภาพอ้างอิงต้นฉบับ' : view.label} /><span className="view-card-label">{singleImage ? 'ภาพอ้างอิงต้นฉบับ · คลิกเพื่อเปลี่ยนภาพ' : view.label}</span><span className="replace"><ImagePlus size={12} /></span></> : <><span className="view-angle">{singleImage ? <ImagePlus size={30} /> : view.icon}</span><span className="view-label">{singleImage ? 'คลิกหรือลากภาพมาวางที่นี่' : view.label}</span><span className="add-image">{singleImage ? '1 ภาพ' : <ImagePlus size={15} />}</span></>}
            </button>
            <input ref={input => { refs.current[view.key] = input; }} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={event => { addFiles(event.target.files, view.key); event.target.value = ''; }} />
          </div>)}</div>
          <div className="upload-tip"><Camera size={15} /><span>ตัวละครครบทั้งตัว ไม่ถูกตัด ฉากหลังเรียบ<br /><small>JPG, PNG หรือ WebP · ไม่เกิน 12 MB/ภาพ</small></span></div>
          <div className="divider" />
          <div className="panel-heading compact"><div><div className="panel-kicker">ขั้นตอนที่ 2</div><h2>ตั้งค่า AI</h2></div><Settings2 size={17} className="muted-icon" /></div>
          <label className="field-label" htmlFor="quality">คุณภาพรูปทรง</label>
          <select id="quality" className="engine-select" value={quality} disabled={busy} onChange={event => setQuality(event.target.value)}>
            <option value="draft">ร่าง · 256 · แนะนำสำหรับ GPU 8 GB</option>
            <option value="balanced">สมดุล · 384 · รายละเอียดมากขึ้น</option>
            <option value="detail">ละเอียด · 512 · ใช้หน่วยความจำมากขึ้น</option>
          </select>
          <label className="color-option"><input type="checkbox" checked={color} disabled={busy} onChange={event => setColor(event.target.checked)} /><span>รายละเอียดผิวจากภาพต้นฉบับ · Texture 4K</span></label>
          <div className={'engine-status ' + (backend?.ready ? 'connected' : '')}><span className={'status-dot ' + (backend?.ready ? '' : 'offline')} /><div><b>Hunyuan3D-2mv</b><small>{backend?.ready ? backend.gpu + ' · ' + backend.vram_gb + ' GB' : backend?.reason || 'กำลังเชื่อมต่อตัวรัน…'}</small></div></div>
          {!backend?.ready && <p className="setup-hint">เปิด <code>Open FourView Studio.cmd</code> ในโฟลเดอร์ FourViewAI</p>}
          {inputMode === 'front' && <p className="panel-copy">{backend?.view_generator_ready ? 'Wonder3D พร้อม · มุมที่ AI สร้างมีขนาด 256×256 และอาจคาดเดารายละเอียดต่างจากต้นฉบับ' : 'กำลังติดตั้งตัวสร้างมุมภาพ Wonder3D บนไดรฟ์ D'}</p>}
          <Button className="generate-btn" isDisabled={!ready || busy || (inputMode === 'front' && !backend?.view_generator_ready)} onPress={() => generate()}>{busy ? <span className="spinner" /> : <Sparkles size={16} />} {busy ? 'กำลังสร้างโมเดล AI…' : inputMode === 'front' ? 'เจน 3 มุมและสร้าง 3D' : 'สร้างโมเดล 3D'}<span className="button-kbd">LOCAL AI</span></Button>
          {model && <Button className="export-btn" onPress={() => generate(null, model.id)}><Sparkles size={15} />เพิ่มรายละเอียดผิวโมเดลเดิม</Button>}
          <div className="privacy-note"><span className="privacy-dot" />รูปภาพประมวลผลในเครื่อง · ไม่เสียโทเคน</div>
          {job && <div role="status" className={'job-status ' + (job.status === 'failed' ? 'error' : '')}><span>{job.error || job.stage}</span>{job.steps && busy && <progress value={job.step || 0} max={job.steps} />}</div>}
          {job?.status === 'failed' && job.id && <Button className="export-btn" isDisabled={busy || !backend?.ready} onPress={() => generate(job.id)}>สร้างต่อจากงานเดิม</Button>}
        </aside>
        <section className="panel viewer-panel">
          <div className="viewer-head"><div><div className="panel-kicker">หน้าต่างแสดงผล</div><h2>{model ? 'โมเดลที่สร้างด้วย AI' : 'พื้นที่แสดงโมเดล'}</h2></div><div className="viewer-tools"><select className="export-format" value={surface} onChange={event => setSurface(event.target.value)} aria-label="การแสดงผิว"><option value="texture">สีภาพ</option><option value="shape">รูปทรงสีเทา</option></select><button className="tool-btn active" title="หมุนโมเดล" onClick={() => setReset(value => value + 1)}><Rotate3D size={16} /></button><button className="tool-btn" title="คืนมุมมองเริ่มต้น" onClick={() => setReset(value => value + 1)}><Maximize2 size={15} /></button><div className="tool-separator" /><select className="export-format" value={format} onChange={event => setFormat(event.target.value)} aria-label="รูปแบบส่งออก"><option value="glb">GLB</option><option value="obj">OBJ</option><option value="stl">STL</option></select><Button className="export-btn" isDisabled={!model} onPress={exportModel}><Download size={15} />ส่งออก</Button></div></div>
          <div className="viewer"><Preview model={model} reset={reset} surface={surface} onError={setMessage} />{!model && <div className="empty-state"><div className="empty-illustration"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="cube-icon"><Box size={45} strokeWidth={1.1} /></div><span className="spark s1">✳</span><span className="spark s2">✦</span></div><h3>{busy ? 'AI กำลังสร้างผิวโมเดล' : 'โมเดลของคุณจะปรากฏที่นี่'}</h3><p>{busy ? job?.stage : 'เลือกภาพเดียวหรือ 4 มุม แล้วกดสร้างโมเดล 3D'}<br />{busy ? 'ภาพถูกประมวลผลด้วย GPU ในเครื่อง' : 'หมุนดูผิวโมเดลและส่งออกได้เมื่อสร้างเสร็จ'}</p></div>}<div className="viewer-badge"><span className="live-dot" />{model ? 'AI MESH · SMOOTH SURFACE' : '3D VIEWPORT'}</div><div className="viewport-controls"><span><MousePointer2 size={13} />ลากเพื่อหมุน</span><span>เลื่อนเพื่อซูม</span></div></div>
          <div className="viewer-footer"><div className="model-meta"><div className="meta-icon"><Layers3 size={16} /></div><div><div className="meta-title">{model ? model.faces.toLocaleString() + ' triangles' : 'รอสร้างโมเดล'}</div><div className="meta-sub">{model ? model.vertices.toLocaleString() + ' vertices · ' + model.seconds + ' วินาที · Hunyuan3D-2mv' : 'ใช้ภาพทั้งสี่มุมสร้างผิวโมเดลด้วย AI'}</div></div></div><div className="orientation"><span>X</span><span>Y</span><span>Z</span></div></div>
          {model?.simplified && <p className="meta-sub">ผิวสีใช้ {model.texture_faces.toLocaleString()} สามเหลี่ยมเพื่อลดเวลาคลี่ UV · เลือกรูปทรงสีเทาเพื่อดูและส่งออกต้นฉบับเต็ม {model.original_faces.toLocaleString()} สามเหลี่ยม</p>}
        </section>
      </div>
      <section className="info-banner"><div className="info-icon"><Sparkles size={16} /></div><div><b>AI ในเครื่อง · ภาพหลายมุม · รายละเอียดผิว Texture 4K</b><span>GLB เก็บผิวครบ · OBJ ดาวน์โหลดพร้อมภาพผิวและวัสดุใน ZIP · STL เก็บรูปทรง · รายละเอียดที่ถูกบังและรอยต่อยังอาจต่างจากภาพต้นฉบับ</span></div></section>
      <footer><span>มิติ · FOURVIEW STUDIO</span><span>HUNYUAN3D MULTIVIEW · OBJ / GLB / STL</span></footer>
    </main>
    {message && <div role="status" className="toast"><span>{message}</span><button aria-label="ปิดข้อความ" onClick={() => setMessage('')}><X size={15} /></button></div>}
  </div>;
}
createRoot(document.getElementById('root')).render(<App />);


