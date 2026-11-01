"""Load frozen legal text with verified source metadata, without rebuilding vectors."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
IMMUTABLE={'id','document_id','article_number','language','text','chapter'}

def load_legal_corpus(root=None,check_files=False):
    root=Path(root) if root else ROOT
    base=root/'data/processed/legal_articles.json'
    raw=base.read_bytes()
    records=json.loads(raw.decode('utf-8-sig'))
    metadata=json.loads((root/'data/processed/legal_article_metadata.json').read_text(encoding='utf-8'))
    if hashlib.sha256(raw).hexdigest()!=metadata['base_corpus_sha256']:
        raise ValueError('Corpus changed: metadata must be revalidated before use.')
    patches=metadata['records']
    if len({r['id'] for r in records})!=len(records) or set(patches)!={r['id'] for r in records}:
        raise ValueError('Metadata IDs do not match corpus IDs.')
    merged=[]
    checked=set()
    for record in records:
        patch=patches[record['id']]
        if IMMUTABLE & set(patch):
            raise ValueError('Metadata must not override text, identity, language or chapter.')
        result=dict(record,**patch)
        if check_files:
            for path,expected in [(root/'data/processed'/result['source_file'],result['source_sha256']),
                                  (root/result['source_pdf_path'],result['source_pdf_sha256'])]:
                if path not in checked:
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                        raise ValueError(f'Source file changed: {path.name}')
                    checked.add(path)
        merged.append(result)
    return merged

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-files',action='store_true')
    p.add_argument('--export',type=Path,help='Optional enriched output; never overwrite frozen legal_articles.json.')
    args=p.parse_args()
    records=load_legal_corpus(check_files=args.check_files)
    if args.export:
        if args.export.resolve()==(ROOT/'data/processed/legal_articles.json').resolve():
            raise ValueError('Refusing to overwrite the frozen retrieval corpus.')
        args.export.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Loaded {len(records)} articles with source metadata. Frozen corpus and index unchanged.')

if __name__=='__main__':main()
