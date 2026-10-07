import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Button } from '@heroui/react';
import { Box, Camera, Check, CircleHelp, Download, ImagePlus, Layers3, Maximize2, MousePointer2, Rotate3D, Settings2, Sparkles, Trash2, X } from 'lucide-react';
import Preview from './Preview';
import AlignmentOverlay from './AlignmentOverlay';
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
  const [backdrop, setBackdrop] = useState('dark'), [showGrid, setShowGrid] = useState(false);
  const [inputMode, setInputMode] = useState('image');
  const [engine, setEngine] = useState('hunyuan-single');
  const [referenceSymmetry, setReferenceSymmetry] = useState(false);
  const [alignment, setAlignment] = useState({ auto: true, views: {} });
  const [alignmentView, setAlignmentView] = useState('front');
  const [alignmentOpen, setAlignmentOpen] = useState(false);
  const updateAlignment = (key, value) => setAlignment(previous => ({ ...previous,
    views: { ...previous.views, [alignmentView]: { ...previous.views[alignmentView], [key]: Number(value) } } }));
  const refs = useRef({}), objectUrls = useRef(new Set()), mounted = useRef(true);
  const imageRevision = useRef(0);
  const singleImage = inputMode !== 'multiview';
  const requiredViews = singleImage ? VIEWS.slice(0, 1) : VIEWS;
  const ready = requiredViews.every(view => images[view.key]);
  const aiSingle = inputMode === 'image' && engine === 'hunyuan-single';
  const nativeSingle = inputMode === 'image' && engine === 'trellis2';
  const engineReady = Boolean(backend?.ready && (nativeSingle ? backend.trellis_ready : (!aiSingle ||
    (backend.single_image_ready && (!color || backend.paint_ready)))));

  const restoreReferences = async (latest, active = () => mounted.current) => {
    const revision = imageRevision.current;
    const saved = await Promise.all((latest.input_mode === 'sheet' ? [{ key: 'front', source: 'source' }] : ['front', 'image'].includes(latest.input_mode) ? VIEWS.slice(0, 1) : VIEWS).map(async view => {
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
    else if (mode !== 'multiview') setImages(previous => previous.front ? { front: previous.front } : {});
    if (mode === 'image') setAlignmentView('front');
  };
  const removeImage = key => {
    if (busy) return;
    imageRevision.current++;
    setImages(previous => { const next = { ...previous }; delete next[key]; return next; });
  };
  // One bar for the whole job: diffusion steps fill it, the other stages only animate.
  const diffusing = busy && job?.steps > 0;

  useEffect(() => {
    mounted.current = true;
    let active = true;
    const status = async () => {
      try { const result = await api('/api/health', { signal: AbortSignal.timeout(5000) }); if (active) setBackend(result); }
      catch { if (active) setBackend({ ready: false, reason: 'ตัวรัน AI ยังไม่เปิด' }); }
    };
    status();
    api('/api/active').then(running => running && active ? follow(running) : api('/api/latest')).then(async latest => {
      if (!active || !latest || imageRevision.current) return;
      setModel({ ...latest, preview: API_BASE + latest.preview }); setJob(latest);
      setQuality(latest.quality); setColor(Boolean(latest.color));
      setInputMode(latest.input_mode || 'multiview');
      setEngine(latest.engine || 'hunyuan-single'); setReferenceSymmetry(Boolean(latest.reference_symmetry));
      setAlignment(latest.alignment || { auto: true, views: {} });
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
    // Manual offsets were tuned against the picture being replaced.
    setAlignment(previous => ({ ...previous, views: Object.fromEntries(Object.entries(previous.views).filter(([key]) => !next[key])) }));
    setImages(previous => singleImage ? next : ({ ...previous, ...next }));
    setModel(null); setJob(null);
  };

  // Follow a job until it ends; throws when it fails.
  const track = async id => {
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
  };
  // A reload forgets the job this page started, while the server keeps running it.
  const follow = async running => {
    setBusy(true); setModel(null); setJob(running);
    setQuality(running.quality); setColor(Boolean(running.color));
    setInputMode(running.input_mode || 'multiview');
    setEngine(running.engine || 'legacy'); setReferenceSymmetry(Boolean(running.reference_symmetry));
    setAlignment(running.alignment || { auto: true, views: {} });
    restoreReferences(running).catch(() => {});
    try { await track(running.id); }
    catch (error) {
      if (mounted.current) { setMessage(error.message); setJob(previous => ({ ...previous, status: 'failed', error: error.message })); }
    } finally { if (mounted.current) setBusy(false); }
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
      form.append('engine', inputMode === 'image' ? engine : 'legacy');
      form.append('reference_symmetry', String(aiSingle && referenceSymmetry));
      form.append('quality', quality); form.append('color', String(color));
      form.append('alignment', JSON.stringify(alignment));
      const { id } = refineId
        ? await api('/api/jobs/' + refineId + '/refine-texture', { method: 'POST', body: form })
        : resumeId
        ? await api('/api/jobs/' + resumeId + '/resume', { method: 'POST' })
        : await api('/api/jobs', { method: 'POST', body: form });
      await track(id);
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
            <Button className={singleImage ? 'selected' : ''} aria-pressed={singleImage} isDisabled={busy} onPress={() => changeInputMode('image')}>อัปโหลดภาพเดียว</Button>
          </div>
          {inputMode === 'image' && <><label className="engine-label" htmlFor="single-engine">ตัวสร้างโมเดล</label><select id="single-engine" className="engine-select" value={engine} disabled={busy} onChange={event => setEngine(event.target.value)}><option value="trellis2">TRELLIS.2 · รูปทรงและวัสดุ 3D จากภาพ (ทดลอง)</option><option value="hunyuan-single">Hunyuan3D · รูปทรง AI + ลายผิวและภาพฉาย</option><option value="legacy">แบบเดิม · ฉายสีจากภาพ</option></select><p className="engine-explanation">{nativeSingle ? 'สร้างรูปทรงและวัสดุในพื้นที่ 3D · รายละเอียดที่ภาพบังยังเป็นการคาดเดาของ AI' : aiSingle ? 'รูปทรงกับภาพฉายอาจไม่ตรงกัน · เพิ่มความละเอียดของลายผิวไม่เพิ่มรายละเอียดรูปทรง' : 'ฉายสีจากภาพต้นฉบับ ด้านที่มองไม่เห็นใช้สีประมาณ'}</p></>}
          {singleImage && <select className="engine-select" aria-label="รูปแบบภาพเดียว" value={inputMode} disabled={busy} onChange={event => changeInputMode(event.target.value)}><option value="image">ภาพเดียว · สร้าง 3D ทันที</option><option value="front">ภาพด้านหน้าเดียว · AI สร้างมุมเพิ่มก่อน</option><option value="sheet">ภาพรวม 4 มุมในไฟล์เดียว (2×2)</option></select>}
          <p className="panel-copy">{inputMode === 'image' ? 'เพิ่มภาพตัวละครหรือวัตถุภาพเดียวที่เห็นครบทั้งตัว แล้วกดสร้าง 3D ได้เลย ไม่ต้องเตรียมภาพหลายมุม' : inputMode === 'sheet' ? 'อัปโหลดภาพรวมแบบรูปวัว: บนซ้าย หน้า · บนขวา ซ้าย · ล่างซ้าย ขวา · ล่างขวา หลัง ระบบแยกมุมแล้วสร้าง 3D ทันที' : inputMode === 'front' ? 'เพิ่มภาพด้านหน้า ระบบจะเจนซ้าย ขวา และหลัง แล้วสร้างโมเดลต่ออัตโนมัติ' : 'ใช้ภาพตัวละครเดียวกันจาก 4 มุม และสัดส่วนตรงกัน'}</p>
          <div className={'view-grid' + (singleImage ? ' single-image-grid' : '')}>{requiredViews.map(view => <div key={view.key}>
            <button className={'view-card ' + (images[view.key] ? 'has-image' : '')} disabled={busy} aria-label={singleImage ? 'อัปโหลดภาพเดียว' : 'เพิ่มภาพ' + view.label} onClick={() => refs.current[view.key]?.click()} onDrop={event => { event.preventDefault(); event.stopPropagation(); addFiles(event.dataTransfer.files, view.key); }}>
              {images[view.key] ? <><img src={images[view.key].url} alt={singleImage ? 'ภาพอ้างอิงต้นฉบับ' : view.label} /><span className="view-card-label">{singleImage ? 'ภาพอ้างอิงต้นฉบับ · คลิกเพื่อเปลี่ยนภาพ' : view.label}</span>{!singleImage && <span className="replace"><ImagePlus size={12} /></span>}</> : <><span className="view-angle">{singleImage ? <ImagePlus size={30} /> : view.icon}</span><span className="view-label">{singleImage ? 'คลิกหรือลากภาพมาวางที่นี่' : view.label}</span><span className="add-image">{singleImage ? '1 ภาพ' : <ImagePlus size={15} />}</span></>}
            </button>
            {singleImage && images[view.key] && <button className="remove-image" aria-label="ลบภาพ" title="ลบภาพ" disabled={busy} onClick={() => removeImage(view.key)}><Trash2 size={14} /></button>}
            <input ref={input => { refs.current[view.key] = input; }} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={event => { addFiles(event.target.files, view.key); event.target.value = ''; }} />
          </div>)}</div>
          <div className="upload-tip"><Camera size={15} /><span>ตัวละครครบทั้งตัว ไม่ถูกตัด ฉากหลังเรียบ<br /><small>JPG, PNG หรือ WebP · ไม่เกิน 12 MB/ภาพ</small></span></div>
          <div className="divider" />
          <div className="panel-heading compact"><div><div className="panel-kicker">ขั้นตอนที่ 2</div><h2>ตั้งค่า AI</h2></div><Settings2 size={17} className="muted-icon" /></div>
          <label className="field-label" htmlFor="quality">คุณภาพรูปทรง</label>
          <select id="quality" className="engine-select" value={quality} disabled={busy} onChange={event => setQuality(event.target.value)}>
            <option value="draft">{nativeSingle ? 'ร่าง · รูปทรง 512 · Texture 2K' : 'ร่าง · 256 · แนะนำสำหรับ GPU 8 GB'}</option>
            <option value="balanced">{nativeSingle ? 'สมดุล · รูปทรง 1024 · Texture 4K' : 'สมดุล · 384 · รายละเอียดมากขึ้น'}</option>
            <option value="detail">{nativeSingle ? 'ละเอียด · รูปทรง 1024 · 16 ขั้น · Texture 4K' : 'ละเอียด · 512 · ใช้หน่วยความจำมากขึ้น'}</option>
          </select>
          {aiSingle && <label className="color-option"><input type="checkbox" checked={referenceSymmetry} disabled={busy} onChange={event => setReferenceSymmetry(event.target.checked)} /><span>ภาพด้านข้าง · ฉายลายภาพซ้ำอีกฝั่ง (อาจเกิดลายซ้ำบริเวณอกและปาก)</span></label>}
          <label className="color-option"><input type="checkbox" checked={color} disabled={busy} onChange={event => setColor(event.target.checked)} /><span>{nativeSingle ? 'วัสดุ 3D ที่ AI สร้างจากภาพ' : aiSingle ? ('ลายผิวจากภาพ + AI รอบตัว · Texture ' + (quality === 'detail' ? '4K' : '2K')) : inputMode === 'image' ? 'ใส่สีจากภาพ · ด้านที่ภาพมองไม่เห็นเติมสีโดยประมาณ' : 'ฉายสีจากภาพอ้างอิงทั้งสี่ด้าน · แบบดั้งเดิม'}</span></label>
          {color && !aiSingle && !nativeSingle && <details className="alignment-panel" onToggle={event => setAlignmentOpen(event.currentTarget.open)}>
            <summary>จัดแนวภาพกับโมเดล</summary>
            <label className="color-option"><input type="checkbox" checked={alignment.auto} disabled={busy} onChange={event => setAlignment(previous => ({ ...previous, auto: event.target.checked }))} /><span>จัดมุมกล้องอัตโนมัติ</span></label>
            <p className="panel-copy">ตรวจส่วนที่ถูกบังก่อนฉายสี · ปรับทีละด้าน แล้วกดใส่สีโมเดลเดิมเพื่อดูผล โดยคงรูปทรงเดิม</p>
            <select aria-label="ด้านที่ปรับการฉายสี" className="engine-select" value={alignmentView} disabled={busy} onChange={event => setAlignmentView(event.target.value)}>{(inputMode === 'image' ? VIEWS.slice(0, 1) : VIEWS).map(view => <option key={view.key} value={view.key}>{view.label}</option>)}</select>
            <AlignmentOverlay jobId={model?.id} view={alignmentView} alignment={alignment} apiBase={API_BASE} open={alignmentOpen} />
            {[['x', 'เลื่อนแนวนอน', -20, 20, 1, 0, '%'], ['y', 'เลื่อนแนวตั้ง', -20, 20, 1, 0, '%'], ['scale', 'ขนาดภาพบนโมเดล', .5, 1.5, .01, 1, '×'], ['yaw', 'ปรับมุมด้านข้าง', -30, 30, 1, 0, '°']].map(([key, label, min, max, step, fallback, unit]) => <label className="alignment-field" key={key}><span>{label}<b>{(alignment.views[alignmentView]?.[key] ?? fallback).toFixed(key === 'scale' ? 2 : 0)}{unit}</b></span><input aria-label={label} type="range" min={min} max={max} step={step} disabled={busy} value={alignment.views[alignmentView]?.[key] ?? fallback} onChange={event => updateAlignment(key, event.target.value)} /></label>)}
            <small className="muted-icon">ภาพซ้อนอัปเดตตามค่าที่ปรับ ส่วนสีบนโมเดล 3D จะเปลี่ยนเมื่อกดนำค่าไปใส่สีโมเดลเดิม</small>
            <button className="alignment-reset" disabled={busy} onClick={() => setAlignment(previous => ({ ...previous, views: { ...previous.views, [alignmentView]: {} } }))}>คืนค่าด้านนี้</button>
            {model && <Button className="export-btn" isDisabled={busy} onPress={() => generate(null, model.id)}>นำค่าไปใส่สีโมเดลเดิม</Button>}
          </details>}
          <div className={'engine-status ' + (backend?.ready ? 'connected' : '')}><span className={'status-dot ' + (backend?.ready ? '' : 'offline')} /><div><b>{nativeSingle ? 'TRELLIS.2 FP8 · Built with DINOv3' : aiSingle ? 'Hunyuan3D + Hunyuan3D-Paint' : 'Hunyuan3D-2mv'}</b><small>{nativeSingle && !backend?.trellis_ready ? backend?.trellis_reason || 'กำลังตรวจ TRELLIS.2' : backend?.ready ? backend.gpu + ' · ' + backend.vram_gb + ' GB' : backend?.reason || 'กำลังเชื่อมต่อตัวรัน…'}</small></div></div>
          {nativeSingle && !backend?.trellis_dino_ready && <p className="setup-hint">ขอสิทธิ์โมเดลด้วยบัญชีของคุณ: <a href={backend?.trellis_access_url || "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"} target="_blank" rel="noreferrer">DINOv3 ของ Meta</a></p>}
          {!backend?.ready && <p className="setup-hint">เปิด <code>Open FourView Studio.cmd</code> ในโฟลเดอร์ FourViewAI</p>}
          {inputMode === 'front' && <p className="panel-copy">{backend?.view_generator_ready ? 'Wonder3D พร้อม · มุมที่ AI สร้างมีขนาด 256×256 และอาจคาดเดารายละเอียดต่างจากต้นฉบับ' : 'กำลังติดตั้งตัวสร้างมุมภาพ Wonder3D บนไดรฟ์ D'}</p>}
          <Button className="generate-btn" isDisabled={!ready || busy || !engineReady || (inputMode === 'front' && !backend?.view_generator_ready)} onPress={() => generate()}>{busy ? <span className="spinner" /> : <Sparkles size={16} />} {busy ? 'กำลังสร้างโมเดล AI…' : inputMode === 'front' ? 'เจน 3 มุมและสร้าง 3D' : 'สร้างโมเดล 3D'}<span className="button-kbd">LOCAL AI</span></Button>
          {model && !nativeSingle && <Button className="export-btn" isDisabled={busy || !engineReady} onPress={() => generate(null, model.id)}><Sparkles size={15} />{aiSingle ? 'สร้างลายผิว AI ให้โมเดลเดิม' : 'ใส่สีโมเดลเดิม'}</Button>}
          <div className="privacy-note"><span className="privacy-dot" />รูปภาพประมวลผลในเครื่อง · ไม่เสียโทเคน</div>
          {job && <div role="status" className={'job-status ' + (job.status === 'failed' ? 'error' : '')}><span>{job.error || job.stage}</span>{Boolean(job.steps) && busy && <progress value={job.step || 0} max={job.steps} />}</div>}
          {job?.status === 'failed' && job.id && <Button className="export-btn" isDisabled={busy || !backend?.ready} onPress={() => generate(job.id)}>สร้างต่อจากงานเดิม</Button>}
        </aside>
        <section className="panel viewer-panel">
          <div className="viewer-head"><div><div className="panel-kicker">หน้าต่างแสดงผล</div><h2>{model ? 'โมเดลที่สร้างด้วย AI' : 'พื้นที่แสดงโมเดล'}</h2></div><div className="viewer-tools"><select className="export-format" value={surface} onChange={event => setSurface(event.target.value)} aria-label="การแสดงผิว"><option value="texture">สีภาพ</option><option value="shape">รูปทรงสีเทา</option></select><button className="tool-btn active" title="หมุนโมเดล" onClick={() => setReset(value => value + 1)}><Rotate3D size={16} /></button><button className="tool-btn" title="คืนมุมมองเริ่มต้น" onClick={() => setReset(value => value + 1)}><Maximize2 size={15} /></button><div className="tool-separator" /><select className="export-format" value={format} onChange={event => setFormat(event.target.value)} aria-label="รูปแบบส่งออก"><option value="glb">GLB</option><option value="obj">OBJ</option><option value="stl">STL</option></select><Button className="export-btn" isDisabled={!model} onPress={exportModel}><Download size={15} />ส่งออก</Button></div></div>
          <div className="viewport-options"><span>มุมมองสตูดิโอ</span><button type="button" aria-pressed={backdrop === 'dark'} onClick={() => setBackdrop(backdrop === 'dark' ? 'light' : 'dark')}>{backdrop === 'dark' ? 'พื้นหลังเข้ม' : 'พื้นหลังสว่าง'}</button><button type="button" aria-pressed={showGrid} onClick={() => setShowGrid(value => !value)}>เส้นกริด</button></div><div className={'viewer viewport-' + backdrop}><Preview model={model} reset={reset} surface={surface} backdrop={backdrop} showGrid={showGrid} onError={setMessage} generating={busy && !model} source={images.front?.url} quadrant={inputMode === 'sheet'} />{busy && !model && <div className="generating" role="status"><h3>กำลังสร้าง…</h3><div className={'generating-bar' + (diffusing ? '' : ' indeterminate')}><i style={diffusing ? { width: 6 + 84 * job.step / job.steps + '%' } : undefined} /></div><p>{job?.stage}</p></div>}{!model && !busy && <div className="empty-state"><div className="empty-illustration"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="cube-icon"><Box size={45} strokeWidth={1.1} /></div><span className="spark s1">✳</span><span className="spark s2">✦</span></div><h3>โมเดลของคุณจะปรากฏที่นี่</h3><p>เลือกภาพเดียวหรือ 4 มุม แล้วกดสร้างโมเดล 3D<br />หมุนดูผิวโมเดลและส่งออกได้เมื่อสร้างเสร็จ</p></div>}<div className="viewer-badge"><span className="live-dot" />{model ? (model.native_pbr ? ('TRELLIS.2 · PBR · ' + (model.texture_size === 4096 ? '4K' : '2K')) : model.texture_method === 'hunyuan3d-paint-v2-0' ? ((model.reference_preserved ? (model.reference_symmetry ? 'ภาพต้นฉบับ · สองฝั่งสมมาตร' : 'ภาพต้นฉบับ + AI') : 'AI TEXTURE') + ' · ' + (model.texture_size === 4096 ? '4K' : '2K') + ' · 6 มุม') : model.vertex_color ? 'สีฉายจากภาพอ้างอิง' : '3D MODEL') : '3D VIEWPORT'}</div><div className="viewport-controls"><span><MousePointer2 size={13} />ลากเพื่อหมุน</span><span>เลื่อนเพื่อซูม</span></div></div>
          <div className="viewer-footer"><div className="model-meta"><div className="meta-icon"><Layers3 size={16} /></div><div><div className="meta-title">{model ? model.faces.toLocaleString() + ' triangles' : 'รอสร้างโมเดล'}</div><div className="meta-sub">{model ? model.vertices.toLocaleString() + ' vertices · ' + model.seconds + ' วินาที · ' + (model.engine === 'trellis2' ? 'TRELLIS.2 · ' + (model.tier_used || '') : model.engine === 'hunyuan-single' ? 'Hunyuan3D ภาพเดียว' : 'Hunyuan3D-2mv') : 'สร้างผิวโมเดลจากภาพอ้างอิงด้วย AI'}</div></div></div><div className="orientation"><span>X</span><span>Y</span><span>Z</span></div></div>
          {model?.simplified && <p className="meta-sub">ผิวสีใช้ {model.texture_faces.toLocaleString()} สามเหลี่ยมเพื่อลดเวลาคลี่ UV · เลือกรูปทรงสีเทาเพื่อดูและส่งออกต้นฉบับเต็ม {model.original_faces.toLocaleString()} สามเหลี่ยม</p>}
        </section>
      </div>
      <section className="info-banner"><div className="info-icon"><Sparkles size={16} /></div><div><b>AI ในเครื่อง · ภาพเดียวหรือหลายมุม</b><span>โหมด Hunyuan3D สร้างลายผิวรอบตัวจากภาพเดียว · GLB รวม texture · OBJ พร้อม MTL/PNG · STL เก็บรูปทรง</span></div></section>
      <footer><span>มิติ · FOURVIEW STUDIO</span><span>HUNYUAN3D MULTIVIEW · OBJ / GLB / STL</span></footer>
    </main>
    {message && <div role="status" className="toast"><span>{message}</span><button aria-label="ปิดข้อความ" onClick={() => setMessage('')}><X size={15} /></button></div>}
  </div>;
}
createRoot(document.getElementById('root')).render(<App />);






