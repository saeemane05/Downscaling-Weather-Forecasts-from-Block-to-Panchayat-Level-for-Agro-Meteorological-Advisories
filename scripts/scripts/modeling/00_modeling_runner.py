from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def run(script,args):
    subprocess.run([sys.executable,str(HERE/script),*args],check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('state');ap.add_argument('district');ap.add_argument('block');ap.add_argument('--through',choices=['clean','features','audits','training','baseline','train','evaluate'],default='training');a=ap.parse_args();loc=[a.state,a.district,a.block]
    if a.through in {'clean','features','audits','training','baseline','train','evaluate'}:run(Path('cleaning')/'run_cleaning.py',loc)
    if a.through in {'features','audits','training','baseline','train','evaluate'}:
        run(Path('feature_engg')/'build_feature_dictionary.py',loc);run(Path('feature_engg')/'feature_engineering.py',loc)
    if a.through in {'audits','training','baseline','train','evaluate'}:
        run(Path('audit')/'audit_engineered_features.py',loc);run(Path('audit')/'temporal_target_leakage_audit.py',loc);run(Path('audit')/'information_availability_audit.py',loc);run(Path('training_data')/'00_target_preflight.py',loc)
    if a.through in {'training','baseline','train','evaluate'}:run(Path('training_data')/'01_build_leakage_safe_training_dataset.py',loc);run(Path('training_data')/'02_audit_training_dataset.py',loc)
    if a.through in {'baseline','train','evaluate'}:run(Path('02_baseline.py'),loc)
    if a.through in {'train','evaluate'}:run(Path('03_train_models.py'),loc)
    if a.through=='evaluate':run(Path('04_evaluate.py'),loc)
if __name__=='__main__':main()
