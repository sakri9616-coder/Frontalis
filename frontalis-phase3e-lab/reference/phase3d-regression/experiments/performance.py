"""Actual CPU/GPU measurements; no 1M allocation before its separate gate."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import numpy as np
try:
    import psutil
except ModuleNotFoundError:
    # The preserved primary CPU runtime has no psutil. Its output-only worker
    # must not require or install performance-report dependencies there.
    psutil = None
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npnd.core import Personality
from npnd.data import training, held_out
OLD_PYTHON = Path(r'C:\Users\sakur\AppData\Local\hermes\jef\venv\Scripts\python.exe')
WIDTHS = (16, 64, 312, 3125)

def output_paths(tag):
    if not re.fullmatch(r'[A-Za-z0-9-]{1,48}', tag):
        raise ValueError('Use a short alphanumeric/hyphen tag; path components are forbidden.')
    return ROOT / 'reports' / (tag + '.json'), ROOT / 'reports' / ('runtime.json' if tag == 'performance' else tag + '-runtime.json')

def require_million_gate(root=ROOT):
    """Check the fixed extension protocol before allowing any large allocation."""
    gate_path = Path(root) / 'reports' / 'capacity-gate.json'
    protocol = Path(root) / 'docs' / '01b-million-protocol.md'
    if not protocol.is_file() or not protocol.read_text(encoding='utf8').strip():
        raise ValueError('A fixed million-parameter extension protocol is required.')
    if not gate_path.is_file():
        raise ValueError('The pre-final-test capacity gate is missing.')
    gate = json.loads(gate_path.read_text(encoding='utf8'))
    if gate.get('millionParameterGatePassed') is not True or gate.get('finiteAndUnrelatedBranchPreserved') is not True or gate.get('finalTestInspectedAtGate') is not False:
        raise ValueError('The pre-final-test capacity and stability gate did not pass.')
    return {'gate': gate, 'gateSha256': hashlib.sha256(gate_path.read_bytes()).hexdigest(), 'protocolSha256': hashlib.sha256(protocol.read_bytes()).hexdigest()}

def require_benchmark_width(width, allow_million=False):
    if width in WIDTHS:
        return
    if width == 31250 and allow_million is True:
        require_million_gate()
        return
    raise ValueError('1M model remains gated; unsupported benchmark width.')

def timing_summary(seconds):
    if not seconds or any(not np.isfinite(t) or t < 0 for t in seconds):
        raise ValueError('Finite nonnegative time samples required.')
    return {'samples': len(seconds), 'medianMs': float(np.median(seconds) * 1000.), 'p95Ms': float(np.percentile(seconds, 95) * 1000.), 'minimumMs': min(seconds) * 1000.}

def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)

def measured(call, device, repeats=200, warmups=10):
    for _ in range(warmups):
        call()
    synchronize(device)
    values = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        call()
        synchronize(device)
        values.append(time.perf_counter() - start)
    return timing_summary(values)

def process_memory():
    if psutil is None:
        return {'available': False, 'reason': 'psutil unavailable in this runtime'}
    info = psutil.Process().memory_info()
    return {'rssBytes': int(info.rss), 'peakRssBytes': int(info.peak_wset) if hasattr(info, 'peak_wset') else None, 'virtualBytes': int(info.vms), 'privateBytes': int(info.private) if hasattr(info, 'private') else None, 'peakRssMethod': 'Windows peak_wset' if hasattr(info, 'peak_wset') else 'unavailable'}

def nvidia_snapshot():
    binary = shutil.which('nvidia-smi')
    if not binary:
        return {'available': False, 'reason': 'nvidia-smi not found'}
    command = [binary, '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if result.returncode:
        return {'available': False, 'exitCode': result.returncode, 'error': result.stderr.strip()[:1000]}
    return {'available': True, 'columns': ['name', 'memoryUsedMiB', 'memoryTotalMiB', 'gpuUtilizationPercent'], 'rows': [line.strip() for line in result.stdout.splitlines() if line.strip()], 'scope': 'whole device including other applications; not attributable to this experiment alone'}

def state_bytes(model):
    learned = sum(int(v.numel() * v.element_size()) for b in model.branches.values() for v in b.values() if isinstance(v, torch.Tensor))
    fixed = model.common_base.numel() * model.common_base.element_size()
    memory = sum(8 * (8 + 7) for b in model.branches.values() for _ in b.get('memory', []))
    return {'learnedTensorBytes': int(learned), 'fixedTensorBytes': int(fixed), 'memoryNumericPayloadBytes': int(memory), 'note': 'numeric arrays only; excludes Python objects, retained source text, optimizer temporaries, runtime and CUDA context'}

def measure_model(width=16, kind='nn', device='cpu', dtype='float64', allow_million=False):
    require_benchmark_width(width, allow_million)
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    before = process_memory()
    synchronize(device); start = time.perf_counter()
    model = Personality(seed=4311, width=width, kind=kind, loss='bounded_signal', device=device, dtype=dtype)
    synchronize(device); init_seconds = time.perf_counter() - start
    initialized = process_memory()
    events = training(seed=4311, series=0, count=16)
    # Exact same research history for memory and parametric families.
    for e in events:
        model.update(e)
    trained = process_memory()
    query = events[0]['cue']
    batch = np.asarray([r['cue'] for r in held_out(count=96)])
    single = measured(lambda: model.evaluate(query, 'relational'), device)
    multiple = measured(lambda: model.evaluate_batch(batch, 'relational'), device, repeats=100)
    update_times = []
    update_event = training(seed=4311, series=0, count=17)[16]
    for _ in range(5):
        copy_model = model.clone()
        synchronize(device); start = time.perf_counter()
        copy_model.update(update_event)
        synchronize(device); update_times.append(time.perf_counter() - start)
        del copy_model
    actual_devices = sorted({str(model.common_base.device)} | {str(v.device) for b in model.branches.values() for v in b.values() if isinstance(v, torch.Tensor)})
    result = {'kind': kind, 'width': width if kind == 'nn' else None, 'device': device, 'actualTensorDevices': actual_devices, 'dtype': dtype, 'torchVersion': torch.__version__, 'learnedParameterCount': model.parameter_count(), 'updatedParameterCountPerExperience': model.parameter_count() // 2, 'initializationMs': init_seconds * 1000., 'singleInference': single, 'batch96Inference': multiple, 'oneExperienceUpdate': timing_summary(update_times), 'processMemoryBefore': before, 'processMemoryInitialized': initialized, 'processMemoryAfter16Updates': trained, 'processMemoryAfterBenchmark': process_memory(), 'numericStateBytes': state_bytes(model), 'endToEndAPI': True, 'cudaMemory': None, 'generation': model.generation, 'parameterHash': model.parameter_hash()}
    if str(device).startswith('cuda'):
        result['cudaMemory'] = {'allocatedBytes': int(torch.cuda.memory_allocated(device)), 'reservedBytes': int(torch.cuda.memory_reserved(device)), 'peakAllocatedBytes': int(torch.cuda.max_memory_allocated(device)), 'peakReservedBytes': int(torch.cuda.max_memory_reserved(device)), 'scope': 'this process PyTorch allocator since benchmark reset; includes cloned update and optimizer temporaries'}
    del model
    gc.collect()
    return result

def cpu_worker(width):
    torch.set_num_threads(1)
    model = Personality(width=width, device='cpu', dtype='float64', loss='bounded_signal')
    for e in training(seed=4311, series=0, count=16):
        model.update(e)
    queries = held_out(count=96)
    return {'pythonVersion': platform.python_version(), 'torchVersion': torch.__version__, 'numpyVersion': np.__version__, 'width': width, 'stateHash': model.parameter_hash(), 'outputs': [model.evaluate(row['cue'], row['space'])['raw'] for row in queries], 'scope': '16 synthetic experiences, seed4311 series0, bounded_signal, float64 CPU single thread'}

def cross_cpu_versions():
    rows = []
    for width in (16, 64):
        results = []
        for python in (OLD_PYTHON, Path(sys.executable)):
            if not python.exists():
                results.append({'available': False, 'reason': 'Python runtime missing', 'path': str(python)})
                continue
            command = [str(python), str(Path(__file__).resolve()), '--worker', str(width)]
            p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf8', timeout=240, env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
            if p.returncode:
                results.append({'available': False, 'exitCode': p.returncode, 'error': p.stderr[-2000:]})
            else:
                results.append({'available': True, **json.loads(p.stdout)})
        row = {'width': width, 'results': results, 'maxAbsOutputDifference': None, 'meanAbsOutputDifference': None}
        if all(r['available'] for r in results):
            difference = np.abs(np.asarray(results[0]['outputs']) - results[1]['outputs'])
            row['maxAbsOutputDifference'] = float(difference.max()); row['meanAbsOutputDifference'] = float(difference.mean())
        rows.append(row)
    return rows

def cpu_gpu_portability(widths=WIDTHS, allow_million=False):
    for width in widths:
        require_benchmark_width(width, allow_million)
    rows = []
    if not torch.cuda.is_available():
        return {'available': False, 'reason': 'CUDA unavailable', 'rows': []}
    queries = held_out(count=96)
    for width in widths:
        models = {}
        for name, device, dtype in [('cpu64', 'cpu', 'float64'), ('gpu64', 'cuda', 'float64'), ('gpu32', 'cuda', 'float32')]:
            model = Personality(width=width, device=device, dtype=dtype, loss='bounded_signal')
            for e in training(seed=4311, series=0, count=16):
                model.update(e)
            models[name] = model
        outputs = {name: np.asarray([model.evaluate(row['cue'], row['space'])['raw'] for row in queries]) for name, model in models.items()}
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime', prefix='performance-portability-') as folder:
            prefix = Path(folder) / 'gpu64-generation16'
            models['gpu64'].save(prefix)
            restored = Personality.load(prefix, device='cpu')
            restored_output = np.asarray([restored.evaluate(row['cue'], row['space'])['raw'] for row in queries])
            restore_diff = np.abs(restored_output - outputs['gpu64'])
            cpu_gpu = np.abs(outputs['cpu64'] - outputs['gpu64'])
            precision = np.abs(outputs['gpu64'] - outputs['gpu32'])
            rows.append({'width': width, 'parameterCount': models['cpu64'].parameter_count(), 'experiences': 16, 'cpu64VsGpu64': {'maxAbsOutputDifference': float(cpu_gpu.max()), 'meanAbsOutputDifference': float(cpu_gpu.mean()), 'parameterHashesEqual': models['cpu64'].parameter_hash() == models['gpu64'].parameter_hash()}, 'gpu64VsGpu32': {'maxAbsOutputDifference': float(precision.max()), 'meanAbsOutputDifference': float(precision.mean()), 'note': 'changed arithmetic precision and CPU/GPU learning; not exact-state portability'}, 'gpuSnapshotLoadedOnCpu64': {'parameterHashesEqual': restored.parameter_hash() == models['gpu64'].parameter_hash(), 'maxAbsOutputDifference': float(restore_diff.max()), 'meanAbsOutputDifference': float(restore_diff.mean())}, 'outputEvidence': {**{name: output.tolist() for name, output in outputs.items()}, 'gpuSnapshotOnCpu': restored_output.tolist()}})
        del models, restored
        gc.collect(); torch.cuda.empty_cache()
    return {'available': True, 'rows': rows, 'scope': 'same observations and independently updated CPU/GPU models compared with actual saved GPU weights loaded unchanged on CPU; these are distinct tests'}

def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=(16, 64))
    parser.add_argument('--output-tag', default='performance', help='A fresh evidence name; recorded files are never replaced.')
    arguments = parser.parse_args()
    if arguments.worker:
        print(json.dumps(cpu_worker(arguments.worker), allow_nan=False)); return
    report, runtime_path = output_paths(arguments.output_tag)
    if report.exists() or runtime_path.exists():
        raise FileExistsError('Performance evidence exists; preserve the recorded run.')
    start = time.perf_counter()
    runtime = {'pythonExecutable': sys.executable, 'pythonVersion': platform.python_version(), 'torchVersion': torch.__version__, 'numpyVersion': np.__version__, 'platform': platform.platform(), 'cpuThreads': torch.get_num_threads(), 'torchCudaVersion': torch.version.cuda, 'cudaAvailable': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'coreSha256': hashlib.sha256((ROOT / 'npnd' / 'core.py').read_bytes()).hexdigest(), 'nvidiaBefore': nvidia_snapshot(), 'scope': 'performance runtime; primary numeric experiments use the preserved existing CPU runtime'}
    if torch.cuda.is_available():
        context_start = time.perf_counter(); a = torch.zeros(1, device='cuda'); torch.cuda.synchronize()
        runtime['firstCudaTensorAllocationAfterDeviceQueryMs'] = (time.perf_counter() - context_start) * 1000.
        del a
    with runtime_path.open('x', encoding='utf8') as file:
        json.dump(runtime, file, ensure_ascii=False, allow_nan=False, indent=2)
    rows = []
    for device, dtype in [('cpu', 'float64'), ('cuda', 'float64'), ('cuda', 'float32')]:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        for width in WIDTHS:
            row = measure_model(width, device=device, dtype=dtype); rows.append(row)
            print(json.dumps({'progress': 'benchmark', 'device': device, 'dtype': dtype, 'parameters': row['learnedParameterCount'], 'singleMedianMs': row['singleInference']['medianMs'], 'updateMedianMs': row['oneExperienceUpdate']['medianMs']}), flush=True)
    for kind in ('linear', 'quadratic', 'memory'):
        rows.append(measure_model(kind=kind, device='cpu', dtype='float64'))
    results = {'formatVersion': 1, 'runtime': runtime, 'models': rows, 'portability': cpu_gpu_portability(), 'crossCpuRuntimeVersions': cross_cpu_versions(), 'nvidiaAfter': nvidia_snapshot(), 'elapsedSeconds': time.perf_counter() - start, 'limitations': ['RSS is whole-process resident memory, including the Torch runtime and CUDA context; peak is cumulative since process startup.', 'GPU memory from nvidia-smi includes unrelated application usage.', 'API timing includes validation and CPU-visible output conversion; CUDA timings are synchronized.', 'OneExperienceUpdate clones outside the measured interval and reuses the same generation16 state five times.', 'Benchmark initialization follows import and available CUDA context setup; it is not full application cold-start time.', 'All GPU metrics are actual values if CUDA was available; missing metrics remain null or explicitly unavailable.', 'No million-parameter model was allocated by this experiment.']}
    with report.open('x', encoding='utf8') as file:
        json.dump(results, file, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({'complete': str(report), 'elapsedSeconds': results['elapsedSeconds']}), flush=True)

if __name__ == '__main__':
    main()
