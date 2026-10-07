import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

function dispose(object) {
  object.traverse(child => {
    child.geometry?.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) {
      if (!material) continue;
      for (const value of Object.values(material)) if (value?.isTexture) value.dispose();
      material.dispose();
    }
  });
}

// Dust cloud in the shape of the reference, shown while the mesh is generated.
async function silhouettePoints(url, quadrant) {
  const image = new Image();
  image.src = url;
  await image.decode();
  // A 2x2 sheet keeps its front view in the top-left quarter.
  const sourceWidth = image.naturalWidth / (quadrant ? 2 : 1), sourceHeight = image.naturalHeight / (quadrant ? 2 : 1);
  const fit = 150 / Math.max(sourceWidth, sourceHeight);
  const width = Math.max(1, Math.round(sourceWidth * fit)), height = Math.max(1, Math.round(sourceHeight * fit));
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(image, 0, 0, sourceWidth, sourceHeight, 0, 0, width, height);
  const { data } = context.getImageData(0, 0, width, height);
  let cutout = false;
  for (let i = 3; i < data.length; i += 4) if (data[i] < 250) { cutout = true; break; }
  // Opaque uploads are not cut out yet, so treat the corner color as background.
  const solid = i => cutout ? data[i + 3] > 96
    : Math.abs(data[i] - data[0]) + Math.abs(data[i + 1] - data[1]) + Math.abs(data[i + 2] - data[2]) > 60;
  let pixels = [];
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (solid((y * width + x) * 4)) pixels.push([x, y]);
  if (pixels.length < 60 || pixels.length > width * height * .92) {
    pixels = [];
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++)
      if (((x - width / 2) / (width * .32)) ** 2 + ((y - height / 2) / (height * .42)) ** 2 < 1) pixels.push([x, y]);
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of pixels) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
  const unit = 3 / Math.max(maxX - minX, maxY - minY, 1), count = 14000;
  const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) * .82;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const [x, y] = pixels[Math.floor(Math.random() * pixels.length)];
    const scatter = Math.random() < .14 ? .22 : .012;
    positions[i * 3] = (x + Math.random() - (minX + maxX + 1) / 2) * unit + gauss() * scatter;
    positions[i * 3 + 1] = ((minY + maxY + 1) / 2 - y - Math.random()) * unit + gauss() * scatter;
    positions[i * 3 + 2] = gauss() * .28;
  }
  return positions;
}

export default function Preview({ model, reset, surface = 'texture', onError, generating = false, source, quadrant = false }) {
  const container = useRef(null), sceneRef = useRef(null), errorRef = useRef(onError);
  errorRef.current = onError;
  useEffect(() => {
    const element = container.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#f2f4f6');
    const camera = new THREE.PerspectiveCamera(35, 1, .01, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.shadowMap.enabled = false;
    element.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 2;
    controls.maxDistance = 14;
    const home = () => { camera.position.set(3.9, 2.6, 5.3); controls.target.set(0, 0, 0); controls.update(); };
    home();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8793a8, 2.5));
    const light = new THREE.DirectionalLight(0xffffff, 2);
    light.position.set(4, 6, 5);
    scene.add(light);
    const grid = new THREE.GridHelper(6, 24, '#cbd2dc', '#e0e5ec');
    grid.position.y = -1.5;
    scene.add(grid);
    const group = new THREE.Group();
    scene.add(group);
    const resize = () => {
      const width = element.clientWidth, height = element.clientHeight;
      renderer.setSize(width, height);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    resize();
    let frame;
    // Aim below the cloud so it sits above the progress overlay at the bottom.
    const front = () => { camera.position.set(0, -.2, 8.2); controls.target.set(0, -.55, 0); controls.update(); };
    const render = () => {
      frame = requestAnimationFrame(render);
      const cloud = group.children.find(child => child.userData.cloud);
      if (cloud) {
        cloud.rotation.y = Math.sin(performance.now() / 1900) * .55;
        cloud.material.opacity = .62 + Math.sin(performance.now() / 520) * .14;
      }
      controls.update(); renderer.render(scene, camera);
    };
    sceneRef.current = { group, home, front };
    render();
    return () => {
      cancelAnimationFrame(frame); observer.disconnect(); controls.dispose();
      dispose(scene); renderer.dispose(); element.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, []);
  useEffect(() => {
    const { group, home } = sceneRef.current;
    let cancelled = false;
    for (const child of [...group.children]) if (!child.userData.cloud) { dispose(child); group.remove(child); }
    if (!model) return;
    const previewUrl = surface === 'shape'
      ? model.preview + (model.preview.includes('?') ? '&' : '?') + 'surface=shape'
      : model.preview;
    new GLTFLoader().loadAsync(previewUrl).then(gltf => {
      if (cancelled) { dispose(gltf.scene); return; }
      const mesh = gltf.scene;
      const bounds = new THREE.Box3().setFromObject(mesh);
      const size = bounds.getSize(new THREE.Vector3());
      const center = bounds.getCenter(new THREE.Vector3());
      const scale = 3 / Math.max(size.x, size.y, size.z);
      mesh.position.copy(center.multiplyScalar(-scale));
      mesh.scale.setScalar(scale);
      mesh.traverse(node => {
        if (!node.isMesh) return;
        const vertexColors = Boolean(node.geometry.getAttribute('color'));
        const unlit = source => {
          const material = surface === 'shape'
            ? new THREE.MeshStandardMaterial({ color: 0xb8c1ce, roughness: 1, metalness: 0, side: THREE.DoubleSide })
            : new THREE.MeshBasicMaterial({
            color: vertexColors ? 0xffffff : (source?.color ?? 0xffffff),
            vertexColors,
            map: source?.map ?? null,
            alphaMap: source?.alphaMap ?? null,
            transparent: source?.transparent ?? false,
            opacity: source?.opacity ?? 1,
            alphaTest: source?.alphaTest ?? 0,
            side: THREE.DoubleSide,
            toneMapped: false,
          });
          if (surface === 'shape') {
            for (const value of Object.values(source ?? {})) if (value?.isTexture) value.dispose();
          }
          source?.dispose();
          return material;
        };
        node.material = Array.isArray(node.material) ? node.material.map(unlit) : unlit(node.material);
        node.castShadow = false;
        node.receiveShadow = false;
      });
      group.add(mesh); home();
    }).catch(error => { if (!cancelled) errorRef.current?.(`เปิดโมเดลไม่สำเร็จ: ${error.message}`); });
    return () => { cancelled = true; };
  }, [model, surface]);
  useEffect(() => {
    if (!generating || !source) return;
    const { group, front } = sceneRef.current;
    let cancelled = false, cloud;
    silhouettePoints(source, quadrant).then(positions => {
      if (cancelled) return;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      cloud = new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0x6b6fdc, size: .03, transparent: true, opacity: .7, depthWrite: false }));
      cloud.userData.cloud = true;
      group.add(cloud); front();
    }).catch(() => {});
    return () => { cancelled = true; if (cloud) { dispose(cloud); group.remove(cloud); } };
  }, [generating, source, quadrant]);
  useEffect(() => sceneRef.current?.home(), [reset]);
  return <div ref={container} className="preview-canvas" aria-label="แสดงโมเดล 3D ลากเพื่อหมุนและเลื่อนเพื่อซูม" />;
}
