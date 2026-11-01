"""Paired development-only retrieval test; no Gemini or generation calls.

Compare hybrid top-20 candidates with their union with top-10 char TF-IDF
candidates. Both arms use the same cross-encoder scores for shared passages.
Default is dry run. --run executes; rerunning resumes saved question results.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

RUN='retrieval_character_candidates_v1'


def key(article):
    return article['document_id'],int(article['article_number']),article['language']


def normalize(text):
    return re.sub('[أإآٱ]','ا',re.sub(r'[\u0610-\u061a\u064b-\u065f\u0670ـ]','',text.lower()))


class CharacterCandidates:
    def __init__(self,articles):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.indexes={}
        for language in ('ar','en'):
            records=[a for a in articles if a['language']==language]
            vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5))
            matrix=vectorizer.fit_transform([normalize(a['text']) for a in records])
            self.indexes[language]=records,vectorizer,matrix

    def search(self,question,language,k=10):
        records,vectorizer,matrix=self.indexes[language]
        scores=(matrix@vectorizer.transform([normalize(question)]).T).toarray().ravel()
        order=sorted(range(len(records)),key=lambda i:(-float(scores[i]),i))
        return [{'article':records[i],'score':float(scores[i])} for i in order[:k] if scores[i]>0]


def union_candidates(baseline,character_results):
    candidates=list(baseline)
    seen={key(r['article']) for r in baseline}
    for r in character_results:
        k=key(r['article'])
        if k not in seen:
            candidates.append(r);seen.add(k)
    return candidates


def paired_rankings(scored,baseline):
    baseline_keys={key(r['article']) for r in baseline}
    # The original candidates occur first in the union, preserving baseline tie order.
    before=[r for r in scored if key(r['article']) in baseline_keys]
    assert {key(r['article']) for r in before}==baseline_keys
    return before,scored


def metrics(ranked,gold):
    first=next((i for i,k in enumerate(ranked[:5],1) if k in gold),None)
    return {'recall_at5':len(set(ranked[:5])&gold)/len(gold),
            'recall_at10':len(set(ranked[:10])&gold)/len(gold),
            'mrr_at5':1/first if first else 0.0}


def save_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def export(output,state):
    rows=[]
    for result in state['results'].values():rows.extend(result['rows'])
    if not rows:return
    def write(name,items):
        path=output/name;temp=path.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
        temp.replace(path)
    write('details.csv',rows)
    summary=[]
    for lang in ('ar','en'):
        for method in ('hybrid_reranked','character_expanded_reranked'):
            group=[r for r in rows if r['language']==lang and r['method']==method]
            if not group:continue
            row={'language':lang,'method':method,'n':len(group)}
            row.update({k:sum(r[k] for r in group)/len(group)
                        for k in ('recall_at5','recall_at10','mrr_at5','candidate_recall','candidate_count')})
            summary.append(row)
    write('summary.csv',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--limit',type=int,default=25)
    args=parser.parse_args()
    if args.limit<1:parser.error('--limit must be positive')
    root=Path(__file__).resolve().parent.parent
    qa_path=root/'data/evaluation/qa_gold.json';corpus_path=root/'data/processed/legal_articles.json'
    questions=[q for q in json.loads(qa_path.read_text(encoding='utf-8-sig'))
               if q.get('split')=='development' and q.get('answerable') is True]
    if not questions or len({q['id'] for q in questions})!=len(questions):
        raise ValueError('Missing/duplicate development questions.')
    articles=json.loads(corpus_path.read_text(encoding='utf-8-sig'))
    available={key(a) for a in articles}
    for q in questions:
        if q['language'] not in ('ar','en') or not q['gold_articles']:
            raise ValueError('Invalid question language/references.')
        if not all((a['document_id'],int(a['article_number']),q['language']) in available for a in q['gold_articles']):
            raise ValueError('Gold source missing in target language.')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'protocol':RUN,'corpus_sha256':digest(corpus_path),
      'development_sha256':hashlib.sha256(json.dumps(questions,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
      'selected_ids':[q['id'] for q in questions],
      'files':{p.name:digest(p) for p in [Path(__file__).resolve(),root/'src/search_reranked.py',
               root/'src/search_hybrid.py',root/'src/search_dense.py',root/'data/indexes/dense_e5/metadata.json']},
      'baseline_candidates':20,'character_top_k':10,'character_ngrams':[3,5],
      'evaluation_k':[5,10],'reranker':'BAAI/bge-reranker-v2-m3',
      'limitations':['Development answerable questions only; no abstention or answer-quality conclusions.',
                     'Candidate recall is diagnostic: union is a superset, so increased candidate recall alone is not success.',
                     'Both ranking arms share one cross-encoder scoring pass; top-k article recall and rank can regress.']}
    output=root/'reports'/RUN;checkpoint=output/'checkpoint.json'
    state=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {'manifest':manifest,'results':{}}
    if state['manifest']!=manifest:
        raise ValueError('Inputs/code changed; preserve the old run and create a new experiment version.')
    pending=[q for q in questions if q['id'] not in state['results']]
    print(f'Development answerable questions: {len(questions)}; saved: {len(state["results"])}; unfinished: {len(pending)}')
    print('Test and unanswerable questions excluded. No Gemini requests. Existing FAISS index unchanged; model files may be checked via Hugging Face.')
    if not args.run:
        print('Dry run: add --run --limit 25 to evaluate/resume. CPU reranking may take several minutes.')
        return
    if not pending:
        export(output,state);print(f'All questions complete. Reports: {output}');return
    from dotenv import load_dotenv
    load_dotenv(root/'.env',override=False,encoding='utf-8-sig')
    from search_reranked import RerankedRetriever
    retriever=RerankedRetriever();character=CharacterCandidates(articles)
    output.mkdir(parents=True,exist_ok=True)
    save_json(checkpoint,state)
    try:
        for position,q in enumerate(pending[:args.limit],1):
            print(f'Evaluating {position}/{min(args.limit,len(pending))}: {q["id"]}',flush=True)
            baseline=retriever.hybrid.search(q['question'],k=20,language=q['language'],candidate_k=20)
            candidates=union_candidates(baseline,character.search(q['question'],q['language'],k=10))
            if any(r['article']['language']!=q['language'] for r in candidates):raise ValueError('Wrong-language candidate.')
            scored=retriever.rerank(q['question'],candidates,k=len(candidates))
            before,after=paired_rankings(scored,baseline)
            gold={(a['document_id'],int(a['article_number'])) for a in q['gold_articles']}
            rows=[]
            for method,ranking,pool in [('hybrid_reranked',before,baseline),('character_expanded_reranked',after,candidates)]:
                ranked=[(r['article']['document_id'],int(r['article']['article_number'])) for r in ranking]
                pool_keys={(r['article']['document_id'],int(r['article']['article_number'])) for r in pool}
                rows.append({'question_id':q['id'],'language':q['language'],'method':method,
                    'candidate_count':len(pool),'candidate_recall':len(gold&pool_keys)/len(gold),
                    'retrieved_top10':';'.join(f'{d}:{n}' for d,n in ranked[:10]),**metrics(ranked,gold)})
            state['results'][q['id']]={'rows':rows,'added_ids':[r['article']['id'] for r in candidates[len(baseline):]]}
            save_json(checkpoint,state);summary=export(output,state)
            print(f'Saved: {len(state["results"])}/{len(questions)}',flush=True)
    except KeyboardInterrupt:
        print('Stopped; completed questions retained. Rerun the same command to resume.')
        return
    for r in summary:
        print(f"{r['language']} | {r['method']} | n={r['n']} | Recall@5={r['recall_at5']:.3f} | Recall@10={r['recall_at10']:.3f} | MRR@5={r['mrr_at5']:.3f}")
    print(f'Reports: {output}')


if __name__=='__main__':main()
