"""Restore verified classifier runtime assets from a separately supplied team model folder."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--from-folder',type=Path,required=True);a=p.parse_args()
 expected=json.loads((ROOT/'config/classifier_artifact_sha256.json').read_text(encoding='utf-8'))
 names=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','results.json']
 source=a.from_folder.resolve();target=ROOT/'models/opp115_distilbert_other_f2_experiment'
 for name in names:
  path=source/name
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected[name]:raise ValueError('Missing or different model file: '+name)
 target.mkdir(parents=True,exist_ok=True)
 for name in names:
  if (source/name).resolve()!=(target/name).resolve():shutil.copy2(source/name,target/name)
 print('Restored five verified runtime files. The model folder remains excluded from Git.')
if __name__=='__main__':main()
