#!/usr/bin/env python3
"""Package current MeanFlow v2 source, FK asset and a runnable Colab notebook."""

import argparse
import base64
import io
import hashlib
import json
from pathlib import Path
import tarfile


def notebook(snapshot_b64, snapshot_sha256):
    cells = []
    def md(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})
    def code(source):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": source.splitlines(keepends=True)})

    md("""# OpenArm V1 — MeanFlow v2, hai camera depth

V2: MeanFlow cho 14 khớp tay + nhánh gripper riêng, loss action/TCP/vị trí/hướng,
trọng số pha gắp. Dùng dữ liệu đã có; không cần thu thêm hoặc cài MuJoCo/ACT.
TCP loss/metrics là FK theo mô hình robot, không phải calibration TCP độc lập.

1. Runtime → Change runtime type → chọn GPU.
2. Clone `https://github.com/vanphong864/openarm_act.git`. Notebook kèm snapshot
   code MeanFlow v2 hiện tại để chạy được cả khi GitHub chưa có các thay đổi mới.
3. Đọc HDF5 tại `/content/drive/MyDrive/vrbt/dataset/2026-10-13`, convert rồi train.
   Nếu thư mục nằm ở **Được chia sẻ với tôi**, thêm shortcut vào `MyDrive/vrbt`.
4. Train run mới: checkpoint FlowPolicy/v1 không load vào kiến trúc v2.
""")
    code("""from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os, subprocess, sys
REPO_URL = 'https://github.com/vanphong864/openarm_act.git'
REPO_REF = ''  # để trống dùng default branch; có thể điền branch/tag/commit
BASE = Path('/content/drive/MyDrive/vrbt/meanflow_v2/2026-10-13')
RAW = Path('/content/drive/MyDrive/vrbt/dataset/2026-10-13')
WORK = Path('/content/openarm_meanflow_v2')
DATASET = Path('/content/meanflow_openarm_20261013.zarr')
DATA_ARCHIVE = BASE / 'meanflow_openarm_20261013.zarr.tar'
RUN = BASE / 'checkpoints/meanflow_v2_small'
assert RAW.is_dir(), (
    f'Không thấy {RAW}. Nếu folder nằm trong Được chia sẻ với tôi, '
    'thêm shortcut vào MyDrive/vrbt hoặc sửa RAW theo đường dẫn đã mount.'
)
assert any(RAW.rglob('*.hdf5')), f'Không tìm thấy HDF5 trong {RAW}'
if WORK.exists() and not (WORK / '.git').is_dir():
    from datetime import datetime, timezone
    preserved = WORK.with_name(WORK.name + '_previous_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    WORK.rename(preserved)
    print('Đã giữ lại workspace cũ tại:', preserved)
if not WORK.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(WORK)], check=True)
else:
    assert (WORK / '.git').is_dir(), f'{WORK} không phải git checkout; chọn WORK mới'
    origin = subprocess.check_output(['git', '-C', str(WORK), 'remote', 'get-url', 'origin'], text=True).strip()
    assert origin == REPO_URL, f'Checkout có origin khác: {origin}; chọn WORK mới'
if REPO_REF:
    subprocess.run(['git', '-C', str(WORK), 'fetch', 'origin', REPO_REF], check=True)
    subprocess.run(['git', '-C', str(WORK), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
print('Git commit:', subprocess.check_output(['git', '-C', str(WORK), 'rev-parse', 'HEAD'], text=True).strip())
BASE.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
os.environ['PYTHONPATH'] = str(WORK / 'src')
if str(WORK / 'src') not in sys.path:
    sys.path.insert(0, str(WORK / 'src'))
print('Data:', RAW, '\\nRun:', RUN)
""")
    md("""## Đồng bộ code MeanFlow v2 đi kèm notebook
Snapshot giữ đúng bản code đã kiểm tra tại thời điểm xuất notebook. Nếu code clone
khác snapshot, cell này cập nhật các file train cần thiết trong checkout tạm ở
`/content`. Không sửa repo trên GitHub. Không cần upload file tar riêng lên Drive.
Muốn dùng code GitHub mới hơn, đặt `USE_NOTEBOOK_SNAPSHOT=False`; repo đó phải có
đủ MeanFlow v2 và `openarm_data`. Khởi động lại runtime khi đổi phiên bản code.
""")
    code("""import base64, hashlib, io, json, tarfile
USE_NOTEBOOK_SNAPSHOT = True
SNAPSHOT_SHA256 = """ + repr(snapshot_sha256) + """
SNAPSHOT_B64 = """ + repr(snapshot_b64) + """
if USE_NOTEBOOK_SNAPSHOT:
    payload = base64.b64decode(SNAPSHOT_B64)
    assert hashlib.sha256(payload).hexdigest() == SNAPSHOT_SHA256
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        source_hashes = json.load(archive.extractfile('SOURCE_SHA256.json'))
        changed = [name for name, digest in source_hashes.items()
                   if not (WORK / name).is_file()
                   or hashlib.sha256((WORK / name).read_bytes()).hexdigest() != digest]
        if changed:
            archive.extractall(WORK, filter='data')
            print(f'Đã cập nhật {len(changed)} file từ snapshot MeanFlow v2')
        for name, digest in source_hashes.items():
            assert hashlib.sha256((WORK / name).read_bytes()).hexdigest() == digest, name
    print('Snapshot SHA256:', SNAPSHOT_SHA256)
required = ['scripts/convert_meanflow_dataset.py', 'scripts/train_meanflow.py',
            'scripts/eval_meanflow.py', 'conf/meanflow_v2.yaml',
            'src/openarm_data/episode.py', 'src/openarm_flow/meanflow.py',
            'requirements-colab.txt']
missing = [name for name in required if not (WORK / name).is_file()]
assert not missing, f'Repo thiếu code v2: {missing}. Bật USE_NOTEBOOK_SNAPSHOT.'
""")
    code("""%pip install -q -r requirements-colab.txt

import torch
assert torch.cuda.is_available(), 'Chưa bật GPU trong Runtime → Change runtime type'
print('Torch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))

def run_live(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    if status:
        raise subprocess.CalledProcessError(status, command)
""")
    md("""## Dataset: khôi phục cache hoặc convert
Đọc HDF5 trên Drive có thể chậm; dataset train giữ ở `/content`. Converter chỉ đọc
nguồn, ghi rõ file hỏng vào metadata. Sau convert lưu cache tar về Drive.
Cache riêng cho dataset 2026-10-13; dùng tên mới nếu thay nguồn/preprocessing.
""")
    code("""import json, zarr
if DATASET.exists():
    try:
        complete = bool(zarr.open_group(str(DATASET), mode='r').attrs.get('complete'))
    except (ValueError, KeyError, OSError):
        complete = False
    if not complete:
        from datetime import datetime, timezone
        preserved = DATASET.with_name(DATASET.name + '_incomplete_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
        DATASET.rename(preserved)
        print('Đã giữ lại conversion dở tại:', preserved)
if not DATASET.exists():
    if DATA_ARCHIVE.exists():
        subprocess.run(['tar', '-xf', str(DATA_ARCHIVE), '-C', str(DATASET.parent)], check=True)
    else:
        assert RAW.is_dir(), f'Chưa có thư mục HDF5: {RAW}'
        run_live([sys.executable, 'scripts/convert_meanflow_dataset.py', '--raw', str(RAW),
                  '--output', str(DATASET), '--cameras', 'chest_depth', 'front_depth',
                  '--action-source', 'commands', '--skip-invalid'])
metadata = dict(zarr.open_group(str(DATASET), mode='r').attrs)
assert metadata.get('complete'), 'Dataset convert dở: chọn output/cache mới'
assert metadata.get('format') == 'openarm_meanflow_depth_v1'
assert not metadata.get('debug_subset'), 'Dataset này chỉ là smoke subset'
print('Episodes:', len(metadata['episodes']), '| skipped:', len(metadata['skipped']))
print('Train:', sum(e['split']=='train' for e in metadata['episodes']),
      '| val:', sum(e['split']=='val' for e in metadata['episodes']))
BASE.mkdir(parents=True, exist_ok=True)
(BASE / 'dataset_report.json').write_text(json.dumps(metadata, indent=2))
if not DATA_ARCHIVE.exists():
    temporary = Path(str(DATA_ARCHIVE) + '.tmp')
    subprocess.run(['tar', '-cf', str(temporary), '-C', '/content', DATASET.name], check=True)
    temporary.replace(DATA_ARCHIVE)
print('Cache:', DATA_ARCHIVE)
""")
    md("""## Cấu hình model
Mặc định small khoảng 6.8M, 4 observation, horizon 16, trả 4 action. Có thể thử
`WIDTHS=[128,256,512]` với RUN mới. Nếu thiếu VRAM, giảm batch trước khi train run mới.
Khi resume, giữ cấu hình đã lưu (chỉ tổng steps được đổi).
""")
    code("""import yaml
WIDTHS = [64, 128, 256]
BATCH_SIZE = 16  # thử 8 nếu GPU thiếu VRAM
STEPS = 20_000
CONFIG = WORK / 'conf/meanflow_v2_colab.yaml'
if (RUN / 'last.pt').exists():
    cfg = yaml.safe_load((RUN / 'train_config.yaml').read_text())
    print('Resume: dùng cấu hình đã lưu')
else:
    cfg = yaml.safe_load((WORK / 'conf/meanflow_v2.yaml').read_text())
    cfg['model']['down_dims'] = WIDTHS
    cfg['train']['batch_size'] = BATCH_SIZE
    cfg['train']['steps'] = STEPS
CONFIG.write_text(yaml.safe_dump(cfg))
print(CONFIG.read_text())
from openarm_flow.meanflow import MeanFlowPolicy
preview_model = MeanFlowPolicy(**cfg['model'])
print('Parameters:', f'{sum(p.numel() for p in preview_model.parameters()):,}')
del preview_model
""")
    md("""## GPU smoke trước khi train dài
Hai bước với model thật, batch 2: kiểm tra CUDA/JVP, FK loss, nhánh gripper và
validation. Checkpoint smoke ở ổ tạm, không trộn vào run chính.
""")
    code("""import copy, tempfile
smoke_config = copy.deepcopy(cfg)
smoke_config['train'].update(steps=2, batch_size=2, save_every=1, val_samples_per_episode=2)
SMOKE_CONFIG = WORK / 'conf/meanflow_v2_gpu_smoke.yaml'
SMOKE_CONFIG.write_text(yaml.safe_dump(smoke_config))
SMOKE_RUN = Path(tempfile.mkdtemp(prefix='meanflow_v2_smoke_', dir='/content')) / 'run'
run_live([sys.executable, '-u', 'scripts/train_meanflow.py', '--dataset', str(DATASET),
          '--output', str(SMOKE_RUN), '--config', str(SMOKE_CONFIG), '--device', 'cuda'])
print('GPU smoke passed. Đây chưa phải checkpoint dùng cho robot.')
""")
    md("""## Train / resume
Checkpoint về Drive mỗi 500 steps. `best_release` chọn theo TCP pha tiếp cận,
hướng TCP và gripper; nếu validation không có mẫu tiếp cận thì báo rõ fallback.
Sau runtime reset, chạy lại các cell trên để khôi phục code/dataset/config rồi train tiếp.
""")
    code("""from datetime import datetime, timezone
command = [sys.executable, '-u', 'scripts/train_meanflow.py', '--dataset', str(DATASET),
           '--output', str(RUN), '--config', str(CONFIG), '--device', 'cuda', '--steps', str(STEPS)]
saved_step = 0
if (RUN / 'last.pt').exists():
    checkpoint = torch.load(RUN / 'last.pt', map_location='cpu', weights_only=True)
    saved_step = checkpoint['step']
    del checkpoint
    command.append('--resume')
elif RUN.exists():
    # Preserve a previous interrupted run with no usable checkpoint.
    preserved = RUN.with_name(RUN.name + '_incomplete_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    RUN.rename(preserved)
    print('Đã giữ lại run chưa có checkpoint tại:', preserved)
if saved_step >= STEPS:
    print(f'Đã train đến step {saved_step}; tăng STEPS nếu muốn tiếp tục')
else:
    run_live(command)
""")
    md("""## Đánh giá checkpoint tốt nhất
Các chỉ số TCP là proxy từ FK, không phải đầu kẹp được đo độc lập. Validation được
dùng chọn model, không phải test set độc lập; vẫn cần rollout để đánh giá gắp.
""")
    code("""run_live([sys.executable, 'scripts/eval_meanflow.py', '--dataset', str(DATASET),
          '--checkpoint', str(RUN / 'best_release'), '--device', 'cuda',
          '--samples-per-episode', '128', '--batch-size', str(cfg['train']['batch_size']),
          '--output', str(RUN / 'evaluation.json')])
from IPython.display import Image, display
for filename in ('loss_curve.png', 'action_metrics.png'):
    if (RUN / filename).exists():
        display(Image(filename=str(RUN / filename)))
print('Best release:', RUN / 'best_release')
""")
    md("""## Tải model về máy
ZIP chứa `best_release/{model.pt,config.json,metadata.json}`. Cần code v2 đi kèm.
Python API dùng `OpenArmMeanFlowPolicy`; runner MuJoCo một camera hiện có chưa nhận
release hai camera v2. Notebook này chưa chứng minh chất lượng thao tác được cải thiện.
""")
    code("""import shutil
from google.colab import files
release_zip = shutil.make_archive('/content/meanflow_v2_best_release', 'zip',
                                  root_dir=str(RUN), base_dir='best_release')
files.download(release_zip)
""")
    md("""## Kiểm thử code (tùy chọn)
Các test CPU kiểm tra gradient, mask, phase, train/resume/release. MuJoCo không có
trong gói train nên test so sánh FK với simulator sẽ skip; đã chạy tại máy đóng gói.
""")
    code("""run_live([sys.executable, '-m', 'pytest', 'tests/test_meanflow.py', 'tests/test_meanflow_v2.py',
          'tests/test_flow.py', 'tests/test_episode.py', '-q'])
""")
    return {"cells": cells, "metadata": {"colab": {"name": "OpenArm_MeanFlow_v2_Colab.ipynb"},
            "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=root / "outputs/meanflow_v2_colab")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    requirements = (root / "requirements-colab.txt").read_text()
    (args.out / "requirements-colab.txt").write_text(requirements)
    # Embed runtime sources so a cloned repository can use unpublished local v2 changes.
    runtime = [p for directory in ("src/openarm_flow", "src/openarm_data")
               for p in (root / directory).rglob("*") if p.is_file()
               and (p.suffix in (".py", ".json", ".md") or p.name in ("LICENSE", "OPENARM_LICENSE"))]
    runtime += [root / "scripts" / name for name in
                ("convert_meanflow_dataset.py", "train_meanflow.py", "eval_meanflow.py")]
    runtime += [root / "conf" / name for name in ("meanflow.yaml", "meanflow_v2.yaml", "flowpolicy.yaml")]
    runtime += [root / "tests" / name for name in
                ("conftest.py", "test_meanflow.py", "test_meanflow_v2.py", "test_flow.py", "test_episode.py")]
    runtime += [root / "manifests/episodes.yaml"]
    embedded = {str(p.relative_to(root)): p.read_bytes() for p in sorted(set(runtime))}
    embedded["requirements-colab.txt"] = requirements.encode()
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in embedded.items()}
    embedded["SOURCE_SHA256.json"] = json.dumps(source_hashes, indent=2).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in embedded.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
    payload = buffer.getvalue()
    book = notebook(base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest())
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile("\n".join(line for line in source.splitlines() if not line.startswith("%")), "notebook-cell", "exec")
    (args.out / "OpenArm_MeanFlow_v2_Colab.ipynb").write_text(json.dumps(book, indent=2, ensure_ascii=False))
    notebooks = root / "notebooks"
    notebooks.mkdir(exist_ok=True)
    (notebooks / "OpenArm_MeanFlow_v2_Colab.ipynb").write_text(json.dumps(book, indent=2, ensure_ascii=False))
    selected = [p for p in (root / "src/openarm_flow").rglob("*") if p.is_file()
                and (p.suffix in (".py", ".json", ".md") or p.name in ("LICENSE", "OPENARM_LICENSE"))]
    selected += [root / "src/openarm_data" / n for n in ("__init__.py", "constants.py", "episode.py", "manifest.py")]
    selected += [root / "scripts" / n for n in ("convert_meanflow_dataset.py", "train_meanflow.py", "eval_meanflow.py", "build_meanflow_colab.py")]
    selected += [root / "tests" / n for n in ("conftest.py", "test_meanflow.py", "test_meanflow_v2.py", "test_flow.py", "test_episode.py")]
    selected += [root / "conf" / n for n in ("meanflow.yaml", "meanflow_v2.yaml", "flowpolicy.yaml")]
    selected += [root / "manifests/episodes.yaml", root / "docs/MEANFLOW.md", root / "docs/MEANFLOW_V2.md"]
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(selected))}
    (args.out / "SOURCE_SHA256.json").write_text(json.dumps(manifest, indent=2))
    archive = args.out / "meanflow_v2_colab.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(set(selected)):
            tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
        for name in ("requirements-colab.txt", "OpenArm_MeanFlow_v2_Colab.ipynb", "SOURCE_SHA256.json"):
            tar.add(args.out / name, arcname=name)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.out / "SHA256SUMS.txt").write_text(f"{checksum}  {archive.name}\n")
    print(f"Archive: {archive} ({archive.stat().st_size:,} bytes)")
    print(f"Notebook: {args.out / 'OpenArm_MeanFlow_v2_Colab.ipynb'}")


if __name__ == "__main__":
    main()
