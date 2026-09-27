from pathlib import Path
import subprocess,sys
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    raise SystemExit(subprocess.call([sys.executable,str(HERE/'training_data'/'01_build_leakage_safe_training_dataset.py'),*sys.argv[1:]]))
