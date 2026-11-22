"""Run Phase 5 validation, freeze thresholds and score the held-out test split."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import tempfile

from .colab_training import (CONFIDENCE, DATASET_SHA256, MANIFEST_SHA256, WEIGHTS_SHA256,
                             check_holdouts, digest, prepare_dataset, runtime_info, save_json)
from .dataset import publish_new
from .evaluation import SETTINGS, measure, predict_fold, promotion_checks, select_threshold


def read_colab_runs(notebook):
    document = json.loads(notebook.read_text(encoding='utf-8-sig'))
    runs = {}
    for cell in document['cells']:
        title = re.match(r'# @title Recipe ([AB]) \(', ''.join(cell['source']).lstrip())
        if not title:
            continue
        stream = ''.join(''.join(output.get('text', '')) for output in cell.get('outputs', []))
        reported = list(re.finditer(r'(?m)^\{\s*"epochs_completed"\s*:', stream))
        if reported:
            report, _ = json.JSONDecoder().raw_decode(stream[reported[-1].start():])
            if (report.get('dataset_sha256') != DATASET_SHA256 or report.get('manifest_sha256') != MANIFEST_SHA256
                    or not re.fullmatch(r'[0-9a-f]{64}', report.get('best_sha256', ''))):
                raise ValueError(f'Recipe {title[1]} report has different dataset hashes or a missing best.pt hash.')
            runs[title[1]] = report
    if runs.keys() != {'A', 'B'}:
        raise ValueError('Provide the executed Colab notebook with the reports for both full recipes.')
    return runs


def verify_checkpoints(paths, expected):
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f'Missing checkpoint files {", ".join(missing)}. Download the full-run best.pt for each recipe.')
    for name, path in paths.items():
        if digest(path) != expected[name]:
            raise ValueError(f'{path} does not match the {name} checkpoint hash in the experiment records.')


def read_frozen_validation(folder):
    path = folder / 'validation.json'
    if not path.is_file():
        raise ValueError('Run evaluate validate to freeze the model choice and thresholds before evaluating test.')
    content = path.read_bytes()
    frozen = json.loads(content)
    thresholds = frozen.get('thresholds', {})
    if (frozen.get('split') != 'val' or thresholds.keys() != {'baseline', 'A', 'B'}
            or frozen.get('validation_selected_candidate') not in ('A', 'B')
            or any(not isinstance(value, (float, int)) or not math.isfinite(value)
                   or not CONFIDENCE <= value <= 1 for value in thresholds.values())):
        raise ValueError('Validation results are incomplete or contain invalid frozen thresholds.')
    return sha256(content).hexdigest(), frozen


def run(args):
    mode = args.mode
    frozen_hash, frozen = read_frozen_validation(args.output) if mode == 'test' else (None, None)
    if mode == 'validate' and (args.output / 'validation.json').exists():
        raise ValueError('This comparison is already frozen. Use its validation.json or choose a new output folder.')
    paths = {'baseline': args.baseline}
    expected = {'baseline': WEIGHTS_SHA256}
    if mode != 'baseline':
        colab = read_colab_runs(args.colab_notebook)
        paths.update(A=args.recipe_a, B=args.recipe_b)
        expected.update({name: colab[name]['best_sha256'] for name in ('A', 'B')})
    verify_checkpoints(paths, expected)
    versions = runtime_info('cpu')
    import torch

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    split = 'test' if mode == 'test' else 'val'
    with tempfile.TemporaryDirectory(prefix='xiao-evaluation-') as folder:
        study = prepare_dataset(args.dataset, folder)
        identity = {'dataset_sha256': DATASET_SHA256, 'manifest_sha256': MANIFEST_SHA256,
                    'checkpoints': expected, 'evaluation': SETTINGS, 'versions': versions,
                    'validation_fingerprint': study.holdout_fingerprint['val']}
        if frozen is not None and any(frozen.get(key) != value for key, value in identity.items()):
            raise ValueError('The current dataset, models or evaluation environment differs from frozen validation.')
        thresholds = dict(frozen['thresholds']) if frozen is not None else {}
        scores = {}
        for name, checkpoint in paths.items():
            frames = predict_fold(study, checkpoint, split)
            if frozen is None:
                thresholds[name] = select_threshold(frames)
            scores[name] = measure(frames, thresholds[name])
        check_holdouts(study)
        report = {**identity, 'split': split, 'split_fingerprint': study.holdout_fingerprint[split],
                  'thresholds': thresholds, 'scores': scores}
    if mode == 'validate':
        report['validation_selected_candidate'] = max(('A', 'B'), key=lambda name: scores[name]['map50_95'])
    if frozen is not None:
        selected = frozen['validation_selected_candidate']
        report['validation_sha256'] = frozen_hash
        report['validation_selected_candidate'] = selected
        report['promotion_accuracy_checks'] = promotion_checks(scores['baseline'], scores[selected])
        report['hardware_verification'] = 'pending'
    name = {'baseline': 'baseline-validation', 'validate': 'validation', 'test': 'test'}[mode]
    output = args.output / f'{name}.json'
    if mode == 'validate':
        args.output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output) as staging:
            uncommitted = Path(staging) / output.name
            save_json(uncommitted, report)
            publish_new(uncommitted, output)
    else:
        save_json(output, report)
    print(json.dumps(scores, indent=2))
    print(f'Saved {output}')


def main(arguments):
    parser = argparse.ArgumentParser(prog='python -m human_tracker evaluate')
    parser.add_argument('mode', choices=('baseline', 'validate', 'test'))
    parser.add_argument('--dataset', type=Path, default=Path('datasets/xiao-v1.zip'))
    parser.add_argument('--baseline', type=Path, default=Path('models/yolo11n.pt'))
    parser.add_argument('--recipe-a', type=Path, default=Path('models/phase4-A.pt'))
    parser.add_argument('--recipe-b', type=Path, default=Path('models/phase4-B.pt'))
    parser.add_argument('--colab-notebook', type=Path, default=Path('xiao_phase4_colab_results.ipynb'))
    parser.add_argument('--output', type=Path, default=Path('build/phase5'))
    try:
        run(parser.parse_args(arguments))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f'{error}\n')
    return 0
