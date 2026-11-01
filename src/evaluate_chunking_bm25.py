"""E2 development pilot: article units vs fixed-size windows, BM25 held fixed.

This measures retrieval, not generated-answer quality. Fixed windows stay inside
their source article so citation metadata remains valid. Scores are aggregated
by maximum window score per article; top-k counts unique articles in both arms.
No test questions, Gemini requests, model downloads or existing index changes.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from rank_bm25 import BM25Okapi

def tokenize(text):
    text = re.sub(r'[\u0610-\u061a\u064b-\u065f\u0670ـ]', '', text.lower())
    return re.findall(r'\w+', re.sub('[أإآٱ]', 'ا', text))

def windows(tokens, size=220, overlap=40):
    if size < 1 or not 0 <= overlap < size:
        raise ValueError('Require size > 0 and 0 <= overlap < size.')
    for start in range(0, len(tokens), size-overlap):
        end = min(start+size, len(tokens))
        yield tokens[start:end], start, end
        if end == len(tokens):
            break

def make_index(articles, mode, size, overlap):
    units, metadata = [], []
    for article in articles:
        tokens = tokenize(article['text'])
        if not tokens:
            raise ValueError(f"Empty token sequence: {article['id']}")
        spans = [(tokens, 0, len(tokens))] if mode == 'article' else list(windows(tokens,size,overlap))
        for words,start,end in spans:
            units.append(words)
            metadata.append({'document_id':article['document_id'],
                'article_number':int(article['article_number']),
                'language':article['language'],'start_token':start,'end_token':end})
    return BM25Okapi(units),metadata

def retrieve(index, metadata, question):
    scores = index.get_scores(tokenize(question))
    order = sorted(range(len(metadata)), key=lambda i:(-float(scores[i]),i))
    seen, results = set(),[]
    for i in order:
        if scores[i] <= 0:
            continue
        item = metadata[i]
        key = (item['document_id'],item['article_number'])
        if key not in seen:
            results.append(key)
            seen.add(key)
        if len(results) == 10:
            break
    return results

def metrics(ranked,gold):
    first=next((r for r,key in enumerate(ranked[:5],1) if key in gold),None)
    return {'recall_at5':len(set(ranked[:5])&gold)/len(gold),
            'recall_at10':len(set(ranked[:10])&gold)/len(gold),
            'mrr_at5':1/first if first else 0.0}

def csv_save(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output-parent',type=Path)
    parser.add_argument('--chunk-size',type=int,default=220)
    parser.add_argument('--overlap',type=int,default=40)
    args=parser.parse_args()
    # Validate even if all articles would fit one window.
    list(windows(['check'],args.chunk_size,args.overlap))
    corpus_path=args.project_root/'data/processed/legal_articles.json'
    qa_path=args.project_root/'data/evaluation/qa_gold.json'
    articles=json.loads(corpus_path.read_text(encoding='utf-8-sig'))
    questions=json.loads(qa_path.read_text(encoding='utf-8-sig'))
    development=[q for q in questions if q.get('split')=='development']
    selected=[q for q in development if q.get('answerable') is True]
    if not selected or len({q['id'] for q in selected})!=len(selected):
        raise ValueError('No selected development questions, or duplicate IDs.')
    available={(a['document_id'],int(a['article_number']),a['language']) for a in articles}
    for q in selected:
        gold={(a['document_id'],int(a['article_number']),q['language']) for a in q['gold_articles']}
        if not gold or not gold<=available:
            raise ValueError(f"Invalid gold references for {q['id']}")
    indexes={}
    sizes={}
    for language in ('ar','en'):
        records=[a for a in articles if a['language']==language]
        for mode in ('article','fixed'):
            indexes[(language,mode)]=make_index(records,mode,args.chunk_size,args.overlap)
            sizes[f'{language}/{mode}']=len(indexes[(language,mode)][1])
    details=[]
    for q in selected:
        gold={(a['document_id'],int(a['article_number'])) for a in q['gold_articles']}
        for mode in ('article','fixed'):
            index,metadata=indexes[(q['language'],mode)]
            ranked=retrieve(index,metadata,q['question'])
            details.append({'question_id':q['id'],'language':q['language'],'mode':mode,
                'split':q['split'],'review_status':q.get('review_status'),
                'retrieved_top10':';'.join(f'{d}:{n}' for d,n in ranked),
                'gold_articles':';'.join(f'{d}:{n}' for d,n in sorted(gold)),**metrics(ranked,gold)})
    grouped=defaultdict(list)
    for row in details:grouped[(row['language'],row['mode'])].append(row)
    summary=[]
    for (language,mode),rows in sorted(grouped.items()):
        row={'language':language,'mode':mode,'n':len(rows),**{
            key:sum(x[key] for x in rows)/len(rows) for key in ('recall_at5','recall_at10','mrr_at5')}}
        summary.append(row)
        print(f"{language} | {mode} | n={len(rows)} | Recall@5={row['recall_at5']:.3f} | Recall@10={row['recall_at10']:.3f} | MRR@5={row['mrr_at5']:.3f}")
    output=(args.output_parent or args.project_root/'reports')/('e2_chunking_bm25_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
    output.mkdir(parents=True,exist_ok=False)
    csv_save(output/'details.csv',details);csv_save(output/'summary.csv',summary)
    manifest={'experiment':'E2_BM25_retrieval_pilot','selected_ids':[q['id'] for q in selected],
        'split':'development','unit_counts':sizes,'size_words':args.chunk_size,'overlap_words':args.overlap,
        'article_aggregation':'maximum unit score; unique-article top-k',
        'corpus_sha256':hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        'qa_sha256':hashlib.sha256(qa_path.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['BM25 retrieval only; not full end-to-end E2 answer evaluation.',
            'Fixed windows do not cross article boundaries; citation metadata retained.',
            'Window unit count and BM25 length/IDF statistics change with chunking.',
            'Source translation differences remain relevant to later answer grading.']}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved: {output}')
    print(f'Excluded {len(development)-len(selected)} unanswerable/pending development questions. Test questions excluded.')
    print('Retrieval-only pilot. No Gemini calls. Existing reports and indexes preserved.')

if __name__=='__main__':main()
