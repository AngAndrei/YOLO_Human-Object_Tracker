"""Embed the tested project modules in the standalone Colab notebook."""

import json
from pathlib import Path
import textwrap


def build(root):
    cells = []

    def add(kind, source, *, hidden=False):
        source = textwrap.dedent(source).strip('\n') + '\n'
        cell = {'cell_type': kind, 'id': f'phase4-{len(cells)+1:02d}',
                'source': source.splitlines(keepends=True), 'metadata': {}}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
            if hidden:
                cell['metadata'] = {'colab': {'cellView': 'form'}, 'jupyter': {'source_hidden': True}}
        cells.append(cell)

    add('markdown', """
    # XIAO Human Tracker Phase 4

    Fine-tune YOLO11n on the frozen XIAO study. A uses deterministic resizing and padding. B adds horizontal flips, HSV changes, translation and scale during training. Both start from the same pretrained checkpoint and use the same 60-image validation split.

    1. In Colab, choose **Runtime → Change runtime type → GPU**.
    2. Upload **datasets/xiao-v1.zip** from the project to **My Drive/Human_Tracker_Phase4** or change the input folder below. You can put the original **models/yolo11n.pt** there too. The notebook fetches the exact official checkpoint if it is absent.
    3. Run through the smoke checks. Inspect both training batches, then run the full A and B cells.

    The setup and pretrained validation also run on CPU. Fine-tuning requires a GPU runtime.
    """)
    add('code', '%pip install -q ultralytics==8.4.215')
    add('code', """
    # @title Mount Drive and choose your paths
    from pathlib import Path
    import os
    import sys
    from google.colab import drive

    drive.mount('/content/drive')
    INPUT_FOLDER = '/content/drive/MyDrive/Human_Tracker_Phase4'  # @param {type:"string"}
    OUTPUT_FOLDER = '/content/drive/MyDrive/Human_Tracker_Phase4/runs'  # @param {type:"string"}
    RUN_GROUP = 'study-v1'  # @param {type:"string"}
    BATCH = 8  # @param {type:"integer"}

    drive_root = Path('/content/drive/MyDrive').resolve()
    inputs = Path(INPUT_FOLDER).resolve()
    outputs = Path(OUTPUT_FOLDER).resolve()
    if not inputs.is_relative_to(drive_root) or not outputs.is_relative_to(drive_root):
        raise ValueError('Choose input and output folders inside mounted My Drive.')
    if not RUN_GROUP or RUN_GROUP in ('.', '..') or Path(RUN_GROUP).name != RUN_GROUP:
        raise ValueError('Choose a single folder name for RUN_GROUP.')
    if BATCH < 1:
        raise ValueError('Set BATCH to a positive integer.')
    output = outputs / RUN_GROUP
    work = Path('/content/xiao-phase4')
    helpers = work / 'helpers'
    package = helpers / 'human_tracker'
    package.mkdir(parents=True, exist_ok=True)
    (package / '__init__.py').touch()
    sys.path.insert(0, str(helpers))
    os.environ.setdefault('YOLO_CONFIG_DIR', str(work / 'ultralytics'))
    """)
    for name in ('dataset.py', 'colab_training.py'):
        source = (root / 'src' / 'human_tracker' / name).read_text(encoding='utf-8')
        add('code', f'%%writefile /content/xiao-phase4/helpers/human_tracker/{name}\n' + source, hidden=True)

    add('markdown', '## Verify CUDA and the frozen study\n\nThe notebook checks the archive, manifest, image-label pairs, split membership, and starting checkpoint.')
    add('code', """
    import json
    from urllib.request import urlopen
    import torch
    from human_tracker.colab_training import (
        baseline, check_holdouts, prepare_dataset, recipe, runtime_info, smoke_check,
        train_recipe, verify_initial_weights, verify_recipes,
    )
    from human_tracker.dataset import write_atomic

    device = None
    if torch.cuda.is_available():
        try:
            torch.empty(1, device='cuda:0')
            device = 0
        except RuntimeError as error:
            print(f'CUDA allocation failed. Use a fresh GPU runtime. {error}')

    print(json.dumps(runtime_info(device if device is not None else 'cpu'), indent=2))
    archive = inputs / 'xiao-v1.zip'
    if not archive.is_file():
        raise FileNotFoundError(f'Upload xiao-v1.zip to {inputs} and run this cell again.')
    study = prepare_dataset(archive, work)
    checkpoint = inputs / 'yolo11n.pt'
    if not checkpoint.is_file():
        checkpoint = work / 'yolo11n.pt'
        if not checkpoint.is_file():
            url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
            with urlopen(url, timeout=60) as response:
                write_atomic(checkpoint, response.read())
    verify_initial_weights(checkpoint)
    verify_recipes(recipe('A', batch=BATCH), recipe('B', batch=BATCH))
    print(f'Verified {len(study.entries)} frames and the original checkpoint.')
    if device is None:
        print('Use a GPU runtime to run the smoke checks and full recipes.')
    """)
    add('markdown', '## Pretrained baseline on validation\n\nEvaluate the original COCO person class at 320 pixels with a confidence floor of 0.001 and NMS IoU of 0.70.')
    add('code', """
    baseline_metrics = baseline(study, checkpoint, output, device if device is not None else 'cpu', BATCH)
    print(json.dumps(baseline_metrics, indent=2))
    """)
    add('markdown', """
    ## Smoke and resume on a small subset

    A trains for one epoch on 16 training and 8 validation images. B trains for two epochs and pauses after the first saved epoch. The check reloads the checkpoint, verifies that training restarts at epoch two, checks finite results and loads the exported checkpoint for prediction.

    Inspect the boxes in both batches below. In B, the boxes should follow people after flipping, shifting or resizing.
    """)
    add('code', """
    from IPython.display import Image, display

    if device is None:
        print('Select a GPU runtime and rerun setup to run the smoke checks.')
    else:
        smoke_results = smoke_check(study, checkpoint, output, device, BATCH)
        print(json.dumps(smoke_results, indent=2))
        for name in ('A', 'B'):
            batch_plot = output / f'smoke-{name}' / 'train_batch0.jpg'
            display(Image(filename=str(batch_plot)))
    """)
    add('markdown', """
    ## Train both full recipes

    Both use seed 42, early-stopping patience 10 and up to 40 epochs. If GPU memory requires a smaller batch, set the same smaller **BATCH** for both, choose a fresh **RUN_GROUP**, and repeat the smoke checks.

    After a disconnect, rerun setup with the same paths, **RUN_GROUP**, and **BATCH**. The affected training cell resumes from the last saved epoch.
    """)
    for name, description in [('A', 'stochastic augmentation disabled'), ('B', 'modest training augmentation')]:
        add('code', f"""
        # @title Recipe {name} ({description})
        if device is None:
            print('Select a GPU runtime before training.')
        else:
            report = train_recipe(study, checkpoint, output, '{name}', device, batch=BATCH)
            print(json.dumps(report, indent=2))
        """)

    add('markdown', """
    ## Verify the runs and locate the Phase 5 files

    Each run saves its Ultralytics configuration and validation metrics. The weights/best.pt file is the candidate for Phase 5. The recovery/last.pt file retains optimizer state.
    """)
    add('code', """
    import yaml

    if device is not None:
        configs = [yaml.safe_load((output / name / 'args.yaml').read_text(encoding='utf-8')) for name in ('A', 'B')]
        verify_recipes(*configs)
        check_holdouts(study)
        for name in ('A', 'B'):
            run = output / name
            print(f'{name} model  {run / "weights" / "best.pt"}')
            print(f'{name} results  {run / "summary.json"}')
        print(f'Baseline results  {output / "baseline-val.json"}')
    """)
    metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'provenance': []}}
    return {'cells': cells, 'metadata': metadata, 'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    notebook = build(folder.parent)
    (folder / 'xiao_phase4_colab.ipynb').write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8',
    )
