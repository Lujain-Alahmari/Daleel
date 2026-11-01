"""Development candidate-coverage diagnostic; no reranker or Gemini.

Fixed bilingual domain vocabulary creates up to three supplementary queries.
Gold references are used only for scoring after candidates are retrieved.
This does not measure final top-k ranking or generated answer quality.
"""
import argparse,csv,hashlib,json,re
from pathlib import Path
from evaluate_character_retrieval import CharacterCandidates,key,normalize

RUN='retrieval_query_expansion_v1'
MAX_POOL=30
# Query labels describe legal topics, not conditions or known article answers.
LEXICON=[
 ('disclosure',r'مشارك|افصاح|\bshar(?:e|ing)\b|\bdisclos',
  'شروط الإفصاح عن البيانات الشخصية لجهة أخرى','conditions for disclosure of personal data to another party'),
 ('recipient_role',r'وكال|طرف ثالث|جهة خارجي|مشارك|مزود|معالج|\bagenc|\bprovider|\bprocessor|\bvendor|\bthird.part',
  'جهة التحكم وجهة المعالجة وضمانات اختيار جهة المعالجة واتفاق المعالجة',
  'controller and processor roles processor selection guarantees processing agreement'),
 ('marketing',r'تسويق|\bmarketing',
  'شروط معالجة البيانات الشخصية لأغراض تسويقية','conditions for processing personal data for marketing purposes'),
 ('access',r'وصول|اطلاع|\baccess\b',
  'حق الوصول إلى البيانات الشخصية ومهلة الطلب وقيوده','personal data access right request period and restrictions'),
 ('destruction',r'حذف|اتلاف|\bdelet|\bdestruct|\berasure',
  'طلب إتلاف البيانات الشخصية وحدود الاحتفاظ','personal data destruction requests and retention limits'),
 ('consent',r'موافق|\bconsent',
  'شروط الموافقة والعدول عنها وآثار العدول','consent conditions withdrawal and effects of withdrawal'),
 ('breach',r'تسرب|\bbreach',
  'الإشعار بحادثة تسرب البيانات الشخصية وشروطه ومحتواه','personal data breach notification conditions timing and content'),
 ('transfer',r'خارج المملك|\boutside the kingdom\b|\bcross.border|\boverseas',
  'نقل البيانات الشخصية إلى خارج المملكة وشروطه وضماناته','personal data transfer outside the Kingdom conditions and safeguards'),
 ('assessment',r'تقويم.*اثر|تقييم.*اثر|\bimpact assessment|\bdpia\b',
  'تقويم أثر معالجة البيانات الشخصية وحالاته وعناصره','personal data processing impact assessment triggers and elements'),
 ('correction',r'تصحيح|تحديث|\bcorrect|\bupdat',
  'تصحيح البيانات الشخصية والإشعار والتوثيق','personal data correction notification and documentation'),
 ('records',r'سجل|سجلات|\brecord',
  'سجلات أنشطة معالجة البيانات الشخصية ومدد الاحتفاظ','records of personal data processing activities and retention periods'),
 ('officer',r'مسؤول حماية|\bofficer\b|\bdpo\b',
  'حالات تعيين مسؤول حماية البيانات الشخصية','personal data protection officer appointment conditions'),
]

def expanded_queries(question,language):
    text=normalize(question)
    return [{'topic':topic,'query':ar if language=='ar' else en}
            for topic,pattern,ar,en in LEXICON if re.search(pattern,text)][:3]

def extend_pool(baseline,lists,limit=MAX_POOL):
    if len(baseline)>limit:raise ValueError('Baseline exceeds pool budget')
    result=list(baseline);seen={key(r['article']) for r in result}
    for position in range(max((len(v) for v in lists),default=0)):
        for values in lists:
            if position>=len(values):continue
            candidate=values[position];k=key(candidate['article'])
            if k not in seen:
                result.append(candidate);seen.add(k)
                if len(result)==limit:return result
    return result

def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def export(output,state):
    rows=[row for result in state['results'].values() for row in result['rows']]
    if not rows:return []
    summary=[]
    for language in ('ar','en'):
        for method in ('hybrid_pool20','hybrid_character_pool30','expanded_query_pool30'):
            group=[r for r in rows if r['language']==language and r['method']==method]
            if group:summary.append({'language':language,'method':method,'n':len(group),
               'candidate_recall':sum(r['candidate_recall'] for r in group)/len(group),
               'mean_pool_size':sum(r['pool_size'] for r in group)/len(group)})
    for name,items in [('details.csv',rows),('summary.csv',summary)]:
        path=output/name;temp=path.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
        temp.replace(path)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--limit',type=int,default=25)
    args=p.parse_args()
    if args.limit<1:p.error('Use a positive limit')
    root=Path(__file__).resolve().parent.parent
    questions=[q for q in json.loads((root/'data/evaluation/qa_gold.json').read_text(encoding='utf-8-sig'))
               if q.get('split')=='development' and q.get('answerable') is True]
    if not questions or len({q['id'] for q in questions})!=len(questions):raise ValueError('Invalid development IDs')
    articles=json.loads((root/'data/processed/legal_articles.json').read_text(encoding='utf-8-sig'))
    available={key(a) for a in articles}
    for q in questions:
        if not q['gold_articles'] or not all((g['document_id'],int(g['article_number']),q['language']) in available for g in q['gold_articles']):
            raise ValueError('Missing gold article in target language')
    files=[Path(__file__).resolve(),root/'src/evaluate_character_retrieval.py',root/'src/search_hybrid.py',
           root/'src/search_dense.py',root/'data/processed/legal_articles.json',root/'data/indexes/dense_e5/metadata.json']
    manifest={'protocol':RUN,'max_pool':MAX_POOL,'max_added_queries':3,
      'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
      'development_sha256':hashlib.sha256(json.dumps(questions,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
      'selected_ids':[q['id'] for q in questions],
      'limitations':['Candidate coverage only. Superset coverage cannot establish top-k ranking or answer improvement.',
                     'A fixed domain lexicon may miss expressions or introduce irrelevant topics; evaluate unseen cases later.',
                     'No test or unanswerable questions used. No article numbers or gold text used to construct queries.']}
    output=root/'reports'/RUN;path=output/'checkpoint.json'
    state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'manifest':manifest,'results':{}}
    if state['manifest']!=manifest:raise ValueError('Inputs/code changed; preserve run and use a new experiment version.')
    pending=[q for q in questions if q['id'] not in state['results']]
    print(f'Development questions: {len(questions)}; saved: {len(state["results"])}; pending: {len(pending)}')
    print('Measures candidate coverage BEFORE reranking. No Gemini or cross-encoder calls. Existing files unchanged.')
    if not args.run:
        print('Dry run. Add --run to execute; E5 model files may be checked/downloaded via Hugging Face.');return
    if not pending:export(output,state);print(f'Complete: {output}');return
    from dotenv import load_dotenv
    load_dotenv(root/'.env',override=False,encoding='utf-8-sig')
    from search_hybrid import HybridRetriever
    hybrid=HybridRetriever();characters=CharacterCandidates(articles)
    output.mkdir(parents=True,exist_ok=True);save(path,state)
    for number,q in enumerate(pending[:args.limit],1):
        print(f'Evaluating {number}/{min(args.limit,len(pending))}: {q["id"]}',flush=True)
        baseline=hybrid.search(q['question'],k=20,language=q['language'],candidate_k=20)
        char=characters.search(q['question'],q['language'],k=10)
        facets=expanded_queries(q['question'],q['language'])
        lists=[char]+[hybrid.search(f['query'],k=20,language=q['language'],candidate_k=20) for f in facets]
        pools=[('hybrid_pool20',baseline),('hybrid_character_pool30',extend_pool(baseline,[char])),
               ('expanded_query_pool30',extend_pool(baseline,lists))]
        gold={(g['document_id'],int(g['article_number'])) for g in q['gold_articles']}
        rows=[]
        for method,pool in pools:
            if len(pool)>MAX_POOL or any(r['article']['language']!=q['language'] for r in pool):raise ValueError('Invalid candidate pool')
            identities={(r['article']['document_id'],int(r['article']['article_number'])) for r in pool}
            if len(identities)!=len(pool):raise ValueError('Duplicate candidate')
            rows.append({'question_id':q['id'],'language':q['language'],'method':method,'pool_size':len(pool),
              'candidate_recall':len(gold&identities)/len(gold),
              'candidate_ids':';'.join(f'{d}:{n}' for d,n in sorted(identities)),
              'missing_gold_ids':';'.join(f'{d}:{n}' for d,n in sorted(gold-identities))})
        state['results'][q['id']]={'rows':rows,'supplementary_queries':facets}
        save(path,state);summary=export(output,state)
        print(f'Saved: {len(state["results"])}/{len(questions)}',flush=True)
    for r in summary:print(f"{r['language']} | {r['method']} | n={r['n']} | Candidate Recall={r['candidate_recall']:.3f} | Mean pool={r['mean_pool_size']:.1f}")
    print(f'Reports: {output}')

if __name__=='__main__':main()
