"""Train the two YOLO11n recipes on the frozen XIAO study."""

from collections import Counter
import csv
from dataclasses import dataclass
from hashlib import file_digest, sha256
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
from zipfile import ZipFile

from .dataset import validate_dataset, write_atomic


ULTRALYTICS_VERSION = '8.4.215'
DATASET_SHA256 = '7359ed6192778af8a8795236f9c12b17cf494e2ab21d4d5258b947bcdf2916d2'
MANIFEST_SHA256 = '51023b1bd700c3826a82f2a2e47047eb3c91effc7000313f125bcbd439d66e6e'
WEIGHTS_SHA256 = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
SEED = 42
IMAGE_SIZE = 320
CONFIDENCE = 0.001
NMS_IOU = 0.7
AUGMENTED_SETTINGS = {'fliplr': 0.50, 'hsv_h': 0.01, 'hsv_s': 0.15, 'hsv_v': 0.20,
                      'translate': 0.05, 'scale': 0.15}
RUN_FIELDS = {'model', 'resume', 'name', 'save_dir', 'data'}


@dataclass(frozen=True)
class StudyData:
    root: Path
    entries: tuple
    holdout_fingerprint: dict[str, str]

    @property
    def yaml(self):
        return self.root / 'data.yaml'


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def heldout_digests(root, entries):
    hashes = {split: sha256() for split in ('val', 'test')}
    for row, _ in entries:
        if row.split in hashes:
            for path in (row.image_path(root), row.label_path(root)):
                hashes[row.split].update(path.relative_to(root).as_posix().encode('utf-8'))
                hashes[row.split].update(bytes.fromhex(digest(path)))
    return {split: value.hexdigest() for split, value in hashes.items()}


def prepare_dataset(archive, work):
    if digest(archive) != DATASET_SHA256:
        raise ValueError('Dataset ZIP SHA-256 differs from the frozen xiao-v1.zip.')
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(dir=work, prefix='xiao-v1-'))
    with ZipFile(archive) as zipped:
        zipped.extractall(root)
    if digest(root / 'manifest.csv') != MANIFEST_SHA256:
        raise ValueError('Manifest SHA-256 differs from the frozen study.')
    entries = tuple(validate_dataset(root))
    counts = Counter(row.split for row, _ in entries)
    if counts != {'train': 240, 'val': 60, 'test': 60}:
        raise ValueError(f'Unexpected frozen split counts {counts}.')
    return StudyData(root, entries, heldout_digests(root, entries))


def check_holdouts(study):
    if (digest(study.root / 'manifest.csv') != MANIFEST_SHA256
            or heldout_digests(study.root, study.entries) != study.holdout_fingerprint):
        raise ValueError('The frozen manifest or held-out image and label bytes changed.')


def make_smoke_data(study):
    """Use deterministic lists of training and validation images, including negatives."""
    rng = random.Random(SEED)
    config = {'path': str(study.root), 'names': {0: 'person'}}
    for split, positives in (('train', 14), ('val', 6)):
        files = []
        for status, count in ((True, positives), (False, 2)):
            pool = [row.image_path(study.root) for row, present in study.entries
                    if row.split == split and present == status]
            files.extend(rng.sample(pool, count))
        rng.shuffle(files)
        file_list = study.root / f'smoke-{split}.txt'
        file_list.write_text(''.join(path.as_posix() + '\n' for path in files), encoding='utf-8')
        config[split] = file_list.as_posix()
    target = study.root / 'smoke.yaml'
    target.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return target


def recipe(name, *, batch=8, epochs=40):
    if name not in ('A', 'B') or not isinstance(batch, int) or batch < 1 or not 1 <= epochs <= 40:
        raise ValueError('Choose recipe A or B, a positive batch, and 1 to 40 epochs.')
    transforms = AUGMENTED_SETTINGS if name == 'B' else dict.fromkeys(AUGMENTED_SETTINGS, 0.0)
    return {
        'seed': SEED, 'imgsz': IMAGE_SIZE, 'batch': batch, 'epochs': epochs, 'patience': 10,
        'optimizer': 'AdamW', 'lr0': 0.002, 'momentum': 0.9, 'warmup_bias_lr': 0.0,
        'deterministic': True, 'workers': 0, 'amp': False, 'cache': False, 'multi_scale': 0.0,
        'val': True, 'plots': True, 'augment': False, 'save': True, 'pretrained': True,
        'conf': CONFIDENCE, 'iou': NMS_IOU, 'rect': False,
        'mosaic': 0.0, 'mixup': 0.0, 'cutmix': 0.0, 'copy_paste': 0.0, 'close_mosaic': 0,
        'degrees': 0.0, 'perspective': 0.0, 'shear': 0.0, 'flipud': 0.0, 'bgr': 0.0,
        'augmentations': [], **transforms,
    }


def verify_recipes(plain, augmented):
    for key in (plain.keys() | augmented.keys() | AUGMENTED_SETTINGS.keys()) - RUN_FIELDS:
        if key in AUGMENTED_SETTINGS:
            if plain.get(key) != 0.0 or augmented.get(key) != AUGMENTED_SETTINGS[key]:
                raise ValueError(f'Unexpected augmentation setting {key}.')
        elif key not in plain or key not in augmented or plain[key] != augmented[key]:
            raise ValueError(f'Recipes must share setting {key}.')


def runtime_info(device):
    import cv2
    import numpy as np
    import torch
    import ultralytics

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f'Install ultralytics=={ULTRALYTICS_VERSION} and restart the kernel.')
    for key in ('sync', 'wandb', 'mlflow', 'comet', 'neptune', 'clearml', 'tensorboard', 'dvc', 'raytune'):
        if key in ultralytics.settings and ultralytics.settings[key]:
            ultralytics.settings.update({key: False})
    return {'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'ultralytics': ultralytics.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__,
            'device': str(device), 'gpu': torch.cuda.get_device_name(device) if isinstance(device, int) else None}


def verify_initial_weights(path):
    path = Path(path).resolve()
    if digest(path) != WEIGHTS_SHA256:
        raise ValueError('Use the same official yolo11n.pt from Human_Tracker/models.')
    return path


def save_json(path, value):
    write_atomic(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))


def validate_model(weights, data_yaml, run_dir, device, batch):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.names.get(0) != 'person':
        raise ValueError(f'Expected person class 0 in {weights}.')
    results = model.val(
        data=str(data_yaml), split='val', classes=[0], imgsz=IMAGE_SIZE, batch=batch,
        conf=CONFIDENCE, iou=NMS_IOU, device=device, workers=0, augment=False, plots=False,
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
    )
    metrics = {key: float(value) for key, value in results.results_dict.items()}
    if not metrics or not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError('Validation returned missing or non-finite metrics.')
    return metrics


def baseline(study, weights, output, device, batch):
    versions = runtime_info(device)
    weights = verify_initial_weights(weights)
    metrics = validate_model(weights, study.yaml, output / 'baseline-val', device, batch)
    check_holdouts(study)
    save_json(output / 'baseline-val.json', {
        'dataset_sha256': DATASET_SHA256, 'manifest_sha256': MANIFEST_SHA256,
        'checkpoint_sha256': WEIGHTS_SHA256, 'split': 'val', 'class_id': 0, 'seed': SEED,
        'imgsz': IMAGE_SIZE, 'batch': batch, 'conf': CONFIDENCE, 'iou': NMS_IOU,
        'versions': versions, 'metrics': metrics,
    })
    return metrics


def persist_checkpoint(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        staged = Path(stream.name)
    try:
        shutil.copyfile(source, staged)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def saved_bests(folder):
    for path in folder.glob('best-epoch-*.pt'):
        epoch = path.stem.removeprefix('best-epoch-')
        if epoch.isdecimal():
            yield int(epoch), path


def restore_recovery(best, history, recovery, completed, results):
    """Restore the best weights and CSV that belong to the committed last.pt."""
    candidates = dict(saved_bests(recovery))
    best_epoch = max((epoch for epoch in candidates if epoch <= completed), default=0)
    if not best_epoch or results.get('epoch') != list(range(1, completed + 1)):
        raise ValueError(f'The recovery state in {recovery} is incomplete.')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(results)
    writer.writerows(zip(*results.values(), strict=True))
    persist_checkpoint(candidates[best_epoch], best)
    write_atomic(history, stream.getvalue().encode('utf-8'))
    for epoch, path in candidates.items():
        if epoch > completed:
            path.unlink()


def require_matching_batch(trainer, requested_batch):
    if trainer.batch_size != requested_batch:
        raise RuntimeError(f'Training switched to batch size {trainer.batch_size}. '
                           'Choose a smaller shared BATCH and a fresh RUN_GROUP for A and B.')


class PauseAfterCheckpoint(Exception):
    pass


def train_recipe(study, weights, output, name, device, *, batch=8, smoke=False, pause_after=None):
    from ultralytics import YOLO

    versions = runtime_info(device)
    weights = verify_initial_weights(weights)
    epochs = (1 if name == 'A' else 2) if smoke else 40
    config = recipe(name, batch=batch, epochs=epochs)
    if smoke and batch > 16:
        raise ValueError('Set BATCH at or below 16 for the smoke dataset.')
    data_yaml = make_smoke_data(study) if smoke else study.yaml
    run = output / (f'smoke-{name}' if smoke else name)
    best = run / 'weights' / 'best.pt'
    recovery = run / 'recovery'
    last = recovery / 'last.pt'
    history_path = run / 'results.csv'
    provenance = run / 'experiment.json'
    summary = run / 'summary.json'
    experiment = {'dataset_sha256': DATASET_SHA256, 'manifest_sha256': MANIFEST_SHA256,
                  'initial_checkpoint_sha256': WEIGHTS_SHA256, 'ultralytics': ULTRALYTICS_VERSION,
                  'recipe': name, 'smoke': smoke, 'settings': config}
    if provenance.is_file():
        if json.loads(provenance.read_text(encoding='utf-8')) != experiment:
            raise ValueError(f'{run} uses different settings. Choose a new experiment folder.')
    else:
        save_json(provenance, experiment)
    if summary.is_file():
        report = json.loads(summary.read_text(encoding='utf-8'))
        if (digest(best) != report['best_sha256'] or digest(last) != report['last_sha256']
                or digest(run / 'args.yaml') != report['config_sha256']):
            raise ValueError(f'Checkpoint or config checksum mismatch in {run}.')
        return report

    resume_epoch, early_stopped = 0, False
    if last.is_file():
        model = YOLO(str(last))
        state = model.ckpt
        resume_epoch = int(state['epoch']) + 1
        if (not state.get('optimizer') or not 0 < resume_epoch <= epochs
                or state['train_args']['batch'] != batch):
            raise ValueError(f'Expected a resumable checkpoint with batch {batch} at {last}.')
        restore_recovery(best, history_path, recovery, resume_epoch, state['train_results'])
        early_stopped = resume_epoch - state['stopper']['best_epoch'] >= config['patience']
    else:
        history_path.unlink(missing_ok=True)
        for _, path in saved_bests(recovery):
            path.unlink()
        model = YOLO(str(weights))

    def audit_start(trainer):
        expected_device = 'cuda' if isinstance(device, int) else 'cpu'
        if trainer.start_epoch != resume_epoch or trainer.device.type != expected_device:
            raise RuntimeError('Training started on an unexpected device or epoch.')
        for transform in trainer.train_loader.dataset.transforms.transforms:
            if type(transform).__name__ == 'Albumentations':
                pipeline = transform.transform
                if pipeline is not None and pipeline.transforms:
                    raise RuntimeError('Unexpected Albumentations transforms in the training loader.')

    def save_epoch(trainer):
        require_matching_batch(trainer, batch)
        completed = trainer.epoch + 1
        snapshot = None
        if trainer.best_fitness == trainer.fitness:
            snapshot = recovery / f'best-epoch-{completed}.pt'
            persist_checkpoint(trainer.best, snapshot)
        persist_checkpoint(trainer.last, last)
        if snapshot is not None:
            for epoch, path in saved_bests(recovery):
                if epoch < completed:
                    path.unlink()
        if pause_after == completed:
            raise PauseAfterCheckpoint

    if resume_epoch < epochs and not early_stopped:
        model.add_callback('on_train_start', audit_start)
        model.add_callback('on_train_epoch_start', lambda trainer: require_matching_batch(trainer, batch))
        model.add_callback('on_model_save', save_epoch)
        try:
            if resume_epoch:
                model.train(resume=True, data=str(data_yaml), device=device, save_dir=str(run))
            else:
                model.train(data=str(data_yaml), project=str(output), name=run.name,
                            exist_ok=True, device=device, **config)
        except PauseAfterCheckpoint:
            return {'paused_after_epoch': pause_after, 'last_sha256': digest(last)}

    with history_path.open(newline='', encoding='utf-8') as stream:
        history = list(csv.DictReader(stream))
    if not history or not all(math.isfinite(float(value)) for row in history for value in row.values()):
        raise ValueError(f'Missing or non-finite training results in {history_path}.')
    completed = int(float(history[-1]['epoch']))
    if [int(float(row['epoch'])) for row in history] != list(range(1, completed + 1)):
        raise ValueError(f'Training results contain missing or repeated epochs in {history_path}.')
    best_model = YOLO(str(best))
    if best_model.names != {0: 'person'}:
        raise ValueError(f'{best} must have exactly one class, person.')
    metrics = validate_model(best, data_yaml, run / 'best-val', device, batch)
    check_holdouts(study)
    report = {'epochs_completed': completed, 'resumed_from_epoch': resume_epoch,
              'best_sha256': digest(best), 'last_sha256': digest(last),
              'versions': versions, 'metrics': metrics, 'config_sha256': digest(run / 'args.yaml'),
              'dataset_sha256': DATASET_SHA256, 'manifest_sha256': MANIFEST_SHA256}
    save_json(summary, report)
    return report


def smoke_check(study, weights, output, device, batch):
    from ultralytics import YOLO

    plain = train_recipe(study, weights, output, 'A', device, batch=batch, smoke=True)
    resumed = train_recipe(study, weights, output, 'B', device, batch=batch, smoke=True, pause_after=1)
    if 'paused_after_epoch' in resumed:
        resumed = train_recipe(study, weights, output, 'B', device, batch=batch, smoke=True)
    if plain['epochs_completed'] != 1 or resumed['epochs_completed'] != 2 or resumed['resumed_from_epoch'] != 1:
        raise RuntimeError('Expected a one-epoch run and a two-epoch run resumed after epoch one.')
    image = next(row.image_path(study.root) for row, person in study.entries if row.split == 'val' and person)
    prediction = YOLO(str(output / 'smoke-B' / 'weights' / 'best.pt')).predict(
        str(image), device=device, imgsz=IMAGE_SIZE, conf=CONFIDENCE, verbose=False,
    )
    if not prediction or prediction[0].orig_shape != (240, 320):
        raise RuntimeError('The smoke export did not return a prediction for the validation image.')
    return {'A': plain, 'B': resumed, 'prediction_boxes': len(prediction[0].boxes)}
