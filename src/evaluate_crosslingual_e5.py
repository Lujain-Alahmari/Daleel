"""E3 development pilot: matched-query same/cross-language E5 retrieval.

Target pools are restricted to one language. Gold identity is document+article,
with an available translation required in both pools. This does not replace the
required monolingual-vs-multilingual model comparison or answer evaluation.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

def scores(ranked,gold):
    first=next((i for i,k in enumerate(ranked[:5],1) if k in gold),None)
    return {'recall_at5':len(set(ranked[:5])&gold)/len(gold),
            'recall_at10':len(set(ranked[:10])&gold)/len(gold),
            'mrr_at5':1/first if first else 0.0}

def prepare(questions,articles):
    selected=[q for q in questions if q.get('split')=='development' and q.get('answerable') is True]
    if not selected or len({q['id'] for q in selected})!=len(selected):
        raise ValueError('Missing or duplicate development questions.')
    available={(a['document_id'],int(a['article_number']),a['language']) for a in articles}
    for q in selected:
        if q['language'] not in {'ar','en'} or not q['gold_articles']:
            raise ValueError(f"Invalid language/references: {q['id']}")
        for ref in q['gold_articles']:
            for target in ('ar','en'):
                if (ref['document_id'],int(ref['article_number']),target) not in available:
                    raise ValueError(f"Missing gold translation for {q['id']}: {ref}, {target}")
    return selected

def evaluate(selected,retriever):
    rows=[]
    for number,q in enumerate(selected,1):
        gold={(a['document_id'],int(a['article_number'])) for a in q['gold_articles']}
        opposite='en' if q['language']=='ar' else 'ar'
        for mode,target in [('same_language',q['language']),('cross_language',opposite)]:
            print(f"Evaluating {number}/{len(selected)}: {q['id']} | {q['language']} -> {target}",flush=True)
            results=retriever.search(q['question'],k=10,language=target)
            if any(r['article']['language']!=target for r in results):
                raise ValueError('Retriever returned the wrong target language.')
            ranked=[(r['article']['document_id'],int(r['article']['article_number'])) for r in results]
            if len(ranked)!=len(set(ranked)):
                raise ValueError('Duplicate articles in top-k.')
            rows.append({'question_id':q['id'],'query_language':q['language'],
                'target_language':target,'mode':mode,'split':q['split'],
                'review_status':q.get('review_status'),
                'retrieved_top10':';'.join(f'{d}:{n}' for d,n in ranked),
                'gold_articles':';'.join(f'{d}:{n}' for d,n in sorted(gold)),**scores(ranked,gold)})
    return rows

def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-parent',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    from dotenv import load_dotenv
    load_dotenv(root/'.env',override=False,encoding='utf-8-sig')
    # Import only after cache configuration has been loaded.
    from search_dense import DenseRetriever
    qa_path=root/'data/evaluation/qa_gold.json'
    corpus_path=root/'data/processed/legal_articles.json'
    questions=json.loads(qa_path.read_text(encoding='utf-8-sig'))
    articles=json.loads(corpus_path.read_text(encoding='utf-8-sig'))
    selected=prepare(questions,articles)
    retriever=DenseRetriever()
    rows=evaluate(selected,retriever)
    grouped=defaultdict(list)
    for row in rows:grouped[(row['query_language'],row['target_language'],row['mode'])].append(row)
    summary=[]
    for (query,target,mode),group in sorted(grouped.items()):
        row={'query_language':query,'target_language':target,'mode':mode,'n':len(group),**{
            k:sum(r[k] for r in group)/len(group) for k in ('recall_at5','recall_at10','mrr_at5')}}
        summary.append(row)
        print(f"{query} -> {target} | {mode} | n={len(group)} | Recall@5={row['recall_at5']:.3f} | Recall@10={row['recall_at10']:.3f} | MRR@5={row['mrr_at5']:.3f}")
    output=(args.output_parent or root/'reports')/('e3_crosslingual_e5_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
    output.mkdir(parents=True,exist_ok=False)
    save_csv(output/'details.csv',rows);save_csv(output/'summary.csv',summary)
    manifest={'experiment':'E3_E5_cross_language_pilot','selected_ids':[q['id'] for q in selected],
        'qa_sha256':hashlib.sha256(qa_path.read_bytes()).hexdigest(),
        'corpus_sha256':hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        'dense_manifest':retriever.manifest,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'target_article_counts':{lang:sum(a['language']==lang for a in articles) for lang in ('ar','en')},
        'limitations':['E5 only; monolingual embedding comparison remains to be done.',
            'Dense retrieval only; no generation, reranking or answer grading.',
            'Same question used for paired same/cross-language conditions; article-level scoring.']}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved: {output}')
    print('Development only. No Gemini/API calls; existing FAISS index is read-only.')

if __name__=='__main__':main()
