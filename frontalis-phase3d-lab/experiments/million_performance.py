"""Gated extension of the original measurements; never replaces the main run."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import performance as perf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-tag', default='million-performance')
    args = parser.parse_args()
    authorization = perf.require_million_gate()
    if not args.output_tag.startswith('million-performance'):
        raise ValueError('The extension needs a separate million-performance evidence name.')
    report, runtime_path = perf.output_paths(args.output_tag)
    if report.exists() or runtime_path.exists():
        raise FileExistsError('Recorded extension evidence is preserved; use a new output tag.')
    originals = [ROOT / 'reports' / p for p in ('performance.json', 'runtime.json')]
    original_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    torch.set_num_threads(1)
    start = time.perf_counter()
    runtime = {'pythonExecutable': sys.executable, 'pythonVersion': platform.python_version(), 'torchVersion': torch.__version__, 'numpyVersion': np.__version__, 'platform': platform.platform(), 'cpuThreads': torch.get_num_threads(), 'torchCudaVersion': torch.version.cuda, 'cudaAvailable': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'coreSha256': hashlib.sha256((ROOT / 'npnd' / 'core.py').read_bytes()).hexdigest(), 'nvidiaBefore': perf.nvidia_snapshot(), 'authorization': authorization, 'concurrentLoad': 'Four numeric CPU workers may run concurrently. No other work or applications were stopped for this benchmark; small timing differences are uncertain.', 'scope': 'conditional million-parameter extension; original performance/runtime artifacts remain unchanged'}
    if torch.cuda.is_available():
        context_start = time.perf_counter(); a = torch.zeros(1, device='cuda'); torch.cuda.synchronize()
        runtime['firstCudaTensorAllocationAfterDeviceQueryMs'] = (time.perf_counter() - context_start) * 1000.
        del a
    with runtime_path.open('x', encoding='utf8') as f:
        json.dump(runtime, f, ensure_ascii=False, allow_nan=False, indent=2)
    rows = []
    for device, dtype in [('cpu', 'float64'), ('cuda', 'float64'), ('cuda', 'float32')]:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        row = perf.measure_model(width=31250, device=device, dtype=dtype, allow_million=True)
        rows.append(row)
        print(json.dumps({'progress': 'million-benchmark', 'device': device, 'dtype': dtype, 'parameters': row['learnedParameterCount'], 'singleMedianMs': row['singleInference']['medianMs'], 'updateMedianMs': row['oneExperienceUpdate']['medianMs']}), flush=True)
    portability = perf.cpu_gpu_portability(widths=(31250,), allow_million=True)
    verified_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    if original_hashes != verified_hashes:
        raise RuntimeError('Original performance evidence changed during the extension.')
    result = {'formatVersion': 1, 'runtime': runtime, 'models': rows, 'portability': portability, 'nvidiaAfter': perf.nvidia_snapshot(), 'originalEvidenceHashesBefore': original_hashes, 'originalEvidenceHashesAfter': verified_hashes, 'originalEvidenceUnchanged': True, 'elapsedSeconds': time.perf_counter() - start, 'limitations': ['This extension is run after the pre-final-test validation gate and fixed extension protocol; it is reported separately.', 'Four CPU numeric workers may run concurrently; no exclusive machine reservation was used.', 'API latency includes input checks and CPU-visible output conversion; CUDA work is synchronized.', 'OneExperienceUpdate repeats the same generation16 state five times; cloning is outside the timing interval.', 'Process RSS/peak include Torch and CUDA runtime, and peak is cumulative since process startup.', 'PyTorch allocator VRAM omits some driver/context usage; nvidia-smi whole-device usage includes unrelated applications.', 'Float32 independent training is a different numeric condition, not exact-state portability.', 'The original performance.json and runtime.json are hash-checked and never replaced.']}
    with report.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({'complete': str(report), 'elapsedSeconds': result['elapsedSeconds']}), flush=True)

if __name__ == '__main__':
    main()
