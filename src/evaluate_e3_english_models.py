"""English-only development comparison: English E5 vs multilingual E5.

Identical English passage boundaries, queries and article scoring. This is
the English model-comparison arm of E3, not an Arabic monolingual comparison.
"""
import csv
import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

MODELS = ('intfloat/multilingual-e5-base', 'intfloat/e5-base-v2')


def metrics(ranked, gold):
    rank = next((i for i, key in enumerate(ranked[:5], 1) if key in gold), None)
    return dict(recall_at5=len(set(ranked[:5]) & gold)/len(gold),
                recall_at10=len(set(ranked[:10]) & gold)/len(gold),
                mrr_at5=1/rank if rank else 0.0)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parent.parent
    from dotenv import load_dotenv
    load_dotenv(root/'.env', override=False, encoding='utf-8-sig')
    import numpy as np
    from sentence_transformers import SentenceTransformer

    qa_path = root/'data/evaluation/qa_gold.json'
    corpus_path = root/'data/processed/legal_articles.json'
    meta_path = root/'data/indexes/dense_e5/metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
    if hashlib.sha256(corpus_path.read_bytes()).hexdigest() != meta['manifest']['corpus_sha256']:
        raise ValueError('Corpus/index mismatch; stop and check the existing index.')
    questions = [q for q in json.loads(qa_path.read_text(encoding='utf-8-sig'))
                 if q.get('split') == 'development' and q.get('answerable') is True
                 and q['language'] == 'en']
    chunks = [c for c in meta['chunks'] if c['language'] == 'en']
    if not questions or not chunks:
        raise ValueError('Missing English development questions or passages.')
    keys = [(c['document_id'], int(c['article_number'])) for c in chunks]
    for q in questions:
        gold = {(r['document_id'], int(r['article_number'])) for r in q['gold_articles']}
        if not gold or not gold.issubset(set(keys)):
            raise ValueError(f"Missing English reference: {q['id']}")
    texts = ['passage: '+c['chunk_text'] for c in chunks]
    queries = ['query: '+q['question'].strip() for q in questions]
    rows, summaries, model_metadata = [], [], []
    print(f'English development questions: {len(questions)}; shared passages: {len(chunks)}', flush=True)
    for name in MODELS:
        print(f'Loading {name} on CPU; first use downloads weights.', flush=True)
        model = SentenceTransformer(name, device='cpu')
        model.max_seq_length = 512
        lengths = [len(model.tokenizer(t, truncation=False)['input_ids']) for t in texts+queries]
        if max(lengths) > 512:
            raise ValueError('Shared passages exceed this tokenizer limit. No silent truncation; revise common boundaries.')
        passages = model.encode(texts, batch_size=8, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True)
        vectors = model.encode(queries, batch_size=8, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True)
        if not np.isfinite(passages).all() or not np.isfinite(vectors).all():
            raise ValueError('Non-finite embeddings.')
        similarities = vectors @ passages.T
        current = []
        for q, values in zip(questions, similarities):
            ranked, seen = [], set()
            for position in np.argsort(-values, kind='stable'):
                key = keys[int(position)]
                if key not in seen:
                    seen.add(key)
                    ranked.append(key)
                if len(ranked) == 10:
                    break
            gold = {(r['document_id'], int(r['article_number'])) for r in q['gold_articles']}
            current.append(dict(model=name, question_id=q['id'], language='en',
                                retrieved_top10=';'.join(f'{d}:{n}' for d,n in ranked),
                                **metrics(ranked, gold)))
        rows.extend(current)
        summary = dict(model=name, language='en', n=len(current), **{
            k:sum(r[k] for r in current)/len(current)
            for k in ('recall_at5','recall_at10','mrr_at5')})
        summaries.append(summary)
        model_metadata.append(dict(model=name, max_input_tokens=max(lengths),
                                   dimension=int(passages.shape[1]),
                                   revision=getattr(model[0].auto_model.config, '_commit_hash', None)))
        print(f"{name} | n={len(current)} | Recall@5={summary['recall_at5']:.3f} | Recall@10={summary['recall_at10']:.3f} | MRR@5={summary['mrr_at5']:.3f}", flush=True)
        del model, passages, vectors, similarities
        gc.collect()
    output = root/'reports'/('e3_english_models_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'details.csv', rows)
    write_csv(output/'summary.csv', summaries)
    manifest = dict(experiment='E3_English_model_comparison', models=model_metadata,
                    selected_ids=[q['id'] for q in questions], shared_passages=len(chunks),
                    normalized=True, pooling='maximum passage similarity per article',
                    hashes={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (qa_path, corpus_path, meta_path, Path(__file__))},
                    limitations=['English same-language arm only; no Arabic monolingual model evaluated.',
                                 'Different pretrained models/tokenizers; differences are not attributable solely to multilingual training.',
                                 'Development only; no answer generation or test questions.'])
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved: {output}')
    print('No Gemini calls. Existing FAISS files are unchanged.')


if __name__ == '__main__':
    main()
