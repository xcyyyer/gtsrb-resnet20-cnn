"""Audit saved validation-only LR runs; no training or image evaluation."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import sys
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'lr_supplement/supplement_source'
RESULT = ROOT / 'lr_supplement/GTSRB_LR_Sensitivity'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-10), (a, b)

def main():
    protocol = json.loads((RESULT / 'protocol.json').read_text())
    summary = json.loads((RESULT / 'validation_summary.json').read_text())
    assert summary['complete'] and len(summary['per_lr']) == 3
    pins = json.loads((SOURCE / 'SOURCE_SHA256.json').read_text())
    assert pins == protocol['source_sha256']
    for name, digest in pins.items():
        assert sha(SOURCE / name) == digest, name
    for name, digest in protocol['manifest_sha256'].items():
        assert sha(SOURCE / 'splits' / (name + '.csv')) == digest
        assert sha(ROOT / 'study/splits' / (name + '.csv')) == digest
    assert not (SOURCE / 'splits/test.csv').exists()
    sys.path.insert(0, str(SOURCE))
    import models
    audit = []
    torch.set_num_threads(2)
    for row in summary['per_lr']:
        lr = row['initial_lr']
        folder = RESULT / 'runs' / ('ds_lr' + format(lr, 'g').replace('.', 'p') + '_s42')
        config = json.loads((folder / 'config.json').read_text())
        env = json.loads((folder / 'environment.json').read_text())
        for key in ['python', 'torch', 'numpy', 'device', 'gpu']:
            assert env[key] == protocol['environment'][key]
        assert env['source_sha256'] == pins
        assert env['train_val_hashes'] == protocol['manifest_sha256']
        for key, value in dict(variant='ds', seed=42, epochs=80, batch_size=128,
                               size=32, workers=2, lr=lr).items():
            assert config[key] == value
        history = list(csv.DictReader((folder / 'history.csv').open()))
        assert [int(x['epoch']) for x in history] == list(range(1, 81))
        for i, item in enumerate(history):
            close(item['lr'], lr * (1 + math.cos(math.pi * i / 80)) / 2)
            # Validation accuracy must correspond to an integer count / 7,830.
            close(float(item['val_accuracy']) * 7830,
                  round(float(item['val_accuracy']) * 7830))
        best = max(history, key=lambda x: float(x['val_accuracy']))
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        # Saved by this project's trainer and supplied by its owner.
        last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == int(best['epoch']) == row['best_epoch']
        assert last['epoch'] == 79 and len(last['history']) == 80
        assert checkpoint['manifest_hashes'] == last['manifest_hashes'] == protocol['manifest_sha256']
        for key in history[0]:
            for logged, saved in zip(history, last['history']):
                close(logged[key], saved[key])
        for key, valkey in [('accuracy', 'val_accuracy'), ('macro_f1', 'val_macro_f1'), ('loss', 'val_loss')]:
            close(checkpoint['val_metrics'][key], best[valkey])
        close(row['best_val_accuracy_pct'], 100 * float(best['val_accuracy']))
        close(row['val_macro_f1_at_best_accuracy_pct'], 100 * float(best['val_macro_f1']))
        close(row['val_loss_at_best_accuracy'], best['val_loss'])
        close(row['last_val_accuracy_pct'], 100 * float(history[-1]['val_accuracy']))
        close(row['logged_train_minutes'], sum(float(x['epoch_seconds']) for x in history) / 60)
        assert row['checkpoint_sha256'] == sha(folder / 'best.pt')
        assert row['history_sha256'] == sha(folder / 'history.csv')
        model = models.ResNet20('ds').eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 32, 32))
        assert tuple(output.shape) == (2, 43) and torch.isfinite(output).all()
        prof = models.profile(model)
        assert prof['parameters'] == 39915 and prof['conv_linear_macs'] == 6394560
        model.load_state_dict(last['model'], strict=True)
        assert last['scheduler']['last_epoch'] == 80
        close(last['optimizer']['param_groups'][0]['lr'], 0)
        audit.append(dict(initial_lr=lr, completed_epochs=80, best_epoch=int(best['epoch']),
                          best_val_correct=round(float(best['val_accuracy']) * 7830),
                          best_val_accuracy_pct=row['best_val_accuracy_pct'],
                          best_checkpoint_load=True, last_checkpoint_load=True,
                          checkpoint_and_log_metrics_agree=True,
                          resume_state_and_full_history_agree=True,
                          cpu_forward_shape=[2, 43], **prof))
    winner = min(summary['per_lr'], key=lambda x: (-x['best_val_accuracy_pct'], x['initial_lr']))
    assert winner['initial_lr'] == summary['validation_selected_lr'] == 0.1
    result = dict(status='passed', completed_runs=3, completed_epochs=240,
                  validation_selected_lr=0.1, per_run=audit,
                  audit_runtime=dict(python=sys.version.split()[0], torch=torch.__version__),
                  scope='Saved source, manifests, configurations, histories, best/last checkpoints and CPU forward checks. No raw-image validation or test evaluation was performed by this audit.')
    (RESULT / 'audit_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
