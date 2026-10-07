"""Resumable HTTP range downloader for the official, pinned multiview weights."""
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REVISION = '3a761b539b29fe4ff64714813aa9560fd66f5de0'
FILE = 'hunyuan3d-dit-v2-mv/model.fp16.safetensors'
SIZE = 4928151562
SHA256 = 'd36f5881bcdc56726b73e517cd444c13c60732431622da7268145355c8d38e9c'
CHUNK = 64 * 1024 * 1024
URL = 'https://huggingface.co/tencent/Hunyuan3D-2mv/resolve/' + REVISION + '/' + FILE


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def download_checkpoint(model_root: Path):
    target = model_root / FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    verified = target.with_suffix('.verified.json')
    if target.is_file() and target.stat().st_size == SIZE:
        print('Verifying existing checkpoint...', flush=True)
        if digest(target) == SHA256:
            verified.write_text(json.dumps({'sha256': SHA256, 'revision': REVISION}), encoding='utf-8')
            return
    partial = target.with_suffix('.http-incomplete')
    state_file = target.with_suffix('.download.json')
    count = (SIZE + CHUNK - 1) // CHUNK
    completed = set()
    if state_file.exists() and partial.exists() and partial.stat().st_size == SIZE:
        state = json.loads(state_file.read_text(encoding='utf-8'))
        if state.get('sha256') == SHA256:
            completed = {n for n in state.get('completed', []) if isinstance(n, int) and 0 <= n < count}
    if not partial.exists() or partial.stat().st_size != SIZE:
        with partial.open('wb') as stream:
            stream.truncate(SIZE)
    guard = threading.Lock()

    def persist():
        temp = state_file.with_suffix('.json.tmp')
        temp.write_text(json.dumps({'sha256': SHA256, 'completed': sorted(completed)}), encoding='utf-8')
        temp.replace(state_file)

    def fetch(index):
        start = index * CHUNK
        end = min(SIZE, start + CHUNK) - 1
        expected = end - start + 1
        for attempt in range(4):
            try:
                # A distinct cache key avoids intermediary caches reusing a
                # different byte range from the same large public download.
                response = requests.get(URL + '?download=true&part=' + str(index),
                    headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(15, 90))
                with response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                        raise RuntimeError('Download server returned an unexpected byte range')
                    written = 0
                    with partial.open('r+b') as output:
                        output.seek(start)
                        for block in response.iter_content(1024 * 1024):
                            if written + len(block) > expected:
                                raise RuntimeError('Download chunk exceeded its expected length')
                            output.write(block)
                            written += len(block)
                        output.flush()
                    if written != expected:
                        raise RuntimeError('Download chunk was incomplete')
                with guard:
                    completed.add(index)
                    persist()
                    print(f'Checkpoint: {len(completed)}/{count} chunks complete', flush=True)
                return
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    print(f'Downloading verified multiview checkpoint ({SIZE / 1e9:.2f} GB); resuming {len(completed)}/{count} chunks', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, index) for index in range(count) if index not in completed]
        for result in as_completed(futures):
            result.result()
    print('Checking checkpoint SHA-256...', flush=True)
    if digest(partial) != SHA256:
        completed.clear()
        persist()
        raise RuntimeError('Checkpoint checksum mismatch. Re-run setup to download the file again.')
    partial.replace(target)
    verified.write_text(json.dumps({'sha256': SHA256, 'revision': REVISION}), encoding='utf-8')
    # Only the downloader's own small progress file is removed after success.
    if state_file.exists():
        state_file.unlink()
    print('Checkpoint SHA-256 verified.', flush=True)
