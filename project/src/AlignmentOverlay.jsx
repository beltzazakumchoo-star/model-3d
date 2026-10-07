import React, { useEffect, useState } from 'react';

export default function AlignmentOverlay({ jobId, view, alignment, apiBase, open }) {
  const [opacity, setOpacity] = useState(.5), [zoom, setZoom] = useState(1);
  const [url, setUrl] = useState(''), [pending, setPending] = useState(false);
  const [error, setError] = useState(''), [camera, setCamera] = useState(null);
  const [retry, setRetry] = useState(0);
  const settings = JSON.stringify(alignment);
  useEffect(() => {
    if (!open || !jobId) return;
    let active = true, objectUrl;
    const controller = new AbortController();
    setPending(true); setError('');
    const timer = setTimeout(async () => {
      try {
        const form = new FormData();
        form.append('view', view); form.append('alignment', settings); form.append('opacity', String(opacity));
        const response = await fetch(apiBase + '/api/jobs/' + jobId + '/alignment-preview', { method: 'POST', body: form, signal: controller.signal });
        if (!response.ok) {
          const result = await response.json(); throw new Error(result.detail || 'โหลดพรีวิวซ้อนไม่สำเร็จ');
        }
        const blob = await response.blob();
        if (!active) return;
        objectUrl = URL.createObjectURL(blob); setUrl(objectUrl);
        setCamera({ yaw: response.headers.get('X-Preview-Yaw'), pitch: response.headers.get('X-Preview-Pitch'), calibrated: response.headers.get('X-Preview-Calibrated') === 'true' });
      } catch (failure) {
        if (active && failure.name !== 'AbortError') setError(failure.message);
      } finally { if (active) setPending(false); }
    }, 450);
    return () => { active = false; clearTimeout(timer); controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [jobId, view, settings, opacity, open, apiBase, retry]);
  if (!jobId) return <p className="panel-copy">สร้างโมเดลก่อน แล้วพรีวิวจะซ้อนภาพต้นฉบับกับรูปทรงจริงให้ปรับตำแหน่งได้</p>;
  return <div className="alignment-overlay">
    <div className="overlay-legend"><span className="mesh-key">ฟ้า: ขอบโมเดล</span><span className="reference-key">ชมพู: ขอบภาพ</span><span>ม่วง: ขอบตรงกัน</span></div>
    <div className="overlay-frame" aria-busy={pending}>
      {url && !error && <img src={url} alt="ภาพต้นฉบับโปร่งใสซ้อนรูปทรงโมเดลในมุมฉายสีเดียวกัน" style={{ width: `${zoom * 100}%`, opacity: pending ? .35 : 1 }} />}
      {pending && <div role="status" className="overlay-status">กำลังอัปเดตภาพซ้อน…</div>}
      {error && <div role="alert" className="overlay-status">{error}<button onClick={() => setRetry(value => value + 1)}>โหลดใหม่</button></div>}
    </div>
    <label className="alignment-field"><span>ความทึบภาพต้นฉบับ<b>{Math.round(opacity * 100)}%</b></span><input aria-label="ความทึบภาพต้นฉบับ" type="range" min="0" max="1" step=".05" value={opacity} onChange={event => setOpacity(Number(event.target.value))} /></label>
    <label className="alignment-field"><span>ขยายดูรายละเอียด<b>{zoom.toFixed(1)}×</b></span><input aria-label="ขยายดูรายละเอียด" type="range" min="1" max="3" step=".25" value={zoom} onChange={event => setZoom(Number(event.target.value))} /></label>
    {camera && <small>มุมฐาน {camera.yaw}° · ก้ม/เงย {camera.pitch}°{alignment.auto && !camera.calibrated ? ' · ยังไม่มีมุมอัตโนมัติที่บันทึกไว้ ใช้มุมมาตรฐานก่อน' : ''}</small>}
    <p className="panel-copy">ลดความทึบเพื่อดูตา ปาก และเข็มขัดบนรูปทรงสีเทา ปรับให้ตรงกับภาพต้นฉบับ ขอบตรงกันอย่างเดียวไม่ได้แปลว่ารายละเอียดภายในตรงทั้งหมด</p>
  </div>;
}
