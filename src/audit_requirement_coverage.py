"""Link a saved RAG draft to source-backed requirements without an API call.

Evidence links do not establish semantic coverage. Optional source-comparison
decisions are kept separate from the original draft and generation metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid


def norm(text):
    text=re.sub(r'[\u0610-\u061a\u064b-\u065f\u0670ـ]','',text.lower())
    return re.sub('[أإآٱ]','ا',text)


def topics_for(key):
    if key.startswith('processor_') or key=='subprocessing':return {'recipient_role'}
    if key.startswith('disclosure_'):return {'disclosure'}
    if key.startswith('direct_marketing_') or key=='marketing_conditions':return {'marketing'}
    if key=='consent_conditions':return {'consent','marketing'}
    if key=='explicit_consent':return {'consent'}
    if key.startswith('withdrawal_'):return {'withdrawal'}
    if key.startswith('breach_'):return {'breach'}
    if key.startswith('impact_assessment_'):return {'assessment'}
    if key.startswith('transfer_'):return {'transfer'}
    if key=='privacy_policy':return {'privacy_policy'}
    return set()


def overlap(start,end,other_start,other_end):
    return max(start,other_start)<min(end,other_end)


def target_hash(audit):
    view=audit.get('source_snapshot',audit)
    target={k:view[k] for k in ('question','language','parts','sources','segments')}
    target.update(answer=audit['answer'],actors=audit.get('actors',[]))
    return hashlib.sha256(json.dumps(target,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def build_report(audit,register,corpus,decisions=None):
    view=audit.get('source_snapshot',audit)
    language=view['language'];question=view['question'];parts=view['parts']
    sources=view['sources'];segments=view['segments']
    answer=audit['answer'];actors=audit.get('actors',[])
    corpus_by_id={f"{a['document_id']}:{a['article_number']}:{a['language']}":a for a in corpus}
    topics={p['topic'] for p in parts}
    text=norm(question)
    if re.search(r'عدول|رجوع|سحب.*موافق|\bwithdraw|\brevok',text):topics.add('withdrawal')
    if re.search(r'سياس.*خصوص|\bprivacy\s+policy\b',text):topics.add('privacy_policy')
    actor_roles={a['legal_role'] for a in actors}
    records=[]
    for requirement in register['requirements']:
        if requirement['language']!=language:continue
        key=requirement['requirement_id'].rsplit(':',1)[0]
        definition=requirement['kind']=='definition'
        if definition:
            if requirement['responsible_role'] not in actor_roles:continue
        elif not topics_for(key)&topics:continue
        sid=requirement['source_id'];article=corpus_by_id[sid]
        lo,hi=requirement['source_start'],requirement['source_end']
        if article['text'][lo:hi]!=requirement['source_text']:
            raise ValueError('Requirements register no longer matches corpus text.')
        if hashlib.sha256(article['text'].encode()).hexdigest()!=requirement['article_sha256']:
            raise ValueError('Requirements article hash changed.')
        citing=[];overlapping=[]
        for number,claim in enumerate(answer['claims'],1):
            evidence=[e for e in claim['evidence'] if e['source_id']==sid]
            if evidence:citing.append(number)
            if any(overlap(lo,hi,e['start'],e['end']) for e in evidence):overlapping.append(number)
        linked_actors=[]
        for actor in actors:
            evidence=[segments[e] for e in actor['evidence_segment_ids']]
            if any(e['source_id']==sid and overlap(lo,hi,e['start'],e['end']) for e in evidence):
                linked_actors.append(actor['actor_id'])
        records.append(dict(requirement,source_in_context=sid in sources,
            article_citing_claims=citing,paragraph_linked_claims=overlapping,linked_actors=linked_actors,
            coverage='unassessed',reason='',review_claim_numbers=[]))
    if len({r['requirement_id'] for r in records})!=len(records):raise ValueError('Duplicate requirement ID.')
    by_id={r['requirement_id']:r for r in records}
    if decisions:
        if decisions['question']!=question:raise ValueError('Review belongs to another question.')
        if decisions['target_sha256']!=target_hash(audit):raise ValueError('Review belongs to a different answer, context or actor map.')
        rh=hashlib.sha256(json.dumps(register,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if decisions['register_sha256']!=rh:raise ValueError('Requirements changed after review.')
        seen=set()
        for item in decisions['decisions']:
            key=item['requirement_id']
            if key not in by_id or key in seen:raise ValueError('Invalid or duplicate review requirement.')
            seen.add(key)
            status=item['coverage'];reason=item['reason'];numbers=item['claim_numbers']
            if status not in ('covered','partial','incorrect','not_addressed','not_applicable'):
                raise ValueError('Invalid review status.')
            if not isinstance(reason,str) or not reason.strip():raise ValueError('Review reason is required.')
            if not isinstance(numbers,list) or any(not isinstance(n,int) or isinstance(n,bool) or not 1<=n<=len(answer['claims']) for n in numbers):
                raise ValueError('Invalid claim number in review.')
            record=by_id[key]
            if status=='covered':
                if not record['source_in_context']:raise ValueError('A requirement with absent source cannot be marked covered.')
                if not (set(numbers)&set(record['paragraph_linked_claims'])) and not record['linked_actors']:
                    raise ValueError('Covered status needs a claim or actor linked to its source paragraph.')
            record.update(coverage=status,reason=reason,review_claim_numbers=numbers)
    for sid,source in sources.items():
        if sid not in corpus_by_id or source['text']!=corpus_by_id[sid]['text']:
            raise ValueError('Saved source changed from corpus.')
    for segment in segments.values():
        if sources[segment['source_id']]['text'][segment['start']:segment['end']]!=segment['text']:
            raise ValueError('Saved evidence offsets changed.')
    unresolved=[r['requirement_id'] for r in records if r['coverage']=='unassessed']
    gaps=[r['requirement_id'] for r in records if r['coverage'] in ('partial','incorrect','not_addressed')]
    return {'protocol':'source-requirements-review-v1','question':question,'language':language,
        'status':'review_incomplete' if unresolved or not records else ('gaps_found' if gaps else 'selected_requirements_reviewed'),
        'records':records,'unassessed_requirements':unresolved,'gaps':gaps,
        'claims':answer['claims'],'actors':actors,
        'limitations':['Topic-based selection produces candidate requirements, not guaranteed applicability.',
         'A citation/paragraph link is not a semantic correctness or completeness score.',
         'The register is not exhaustive; selected_requirements_reviewed is not full-law compliance.',
         'Original generations, reference answers and model metrics are unchanged. No API call.']}


def markdown(report):
    roles={'controller':'جهة التحكم','processor':'جهة المعالجة','data_subject':'صاحب البيانات',
        'controller_and_processor':'جهة التحكم وجهة المعالجة؛ يفصل النص دور كل منهما'}
    statuses={'unassessed':'لم يحسم بعد','covered':'مغطى وفق المراجعة','partial':'ناقص',
              'incorrect':'يحتاج تصحيحاً','not_addressed':'لم تتناوله الإجابة','not_applicable':'غير منطبق مع بيان السبب'}
    lines=['# مراجعة الإجابة مقابل متطلبات المصادر','',report['question'],'',
           'هذا تقرير ربط ومراجعة دون API. الاستشهاد بالمادة أو بمقطع منها لا يثبت وحده أن شروطها وردت في الجواب. البنود المختارة مرشحة بحسب الموضوع؛ يحدد انطباقها لكل سؤال، ولا يطالب الجواب تلقائياً بكل أحكام المادة.','',
           f"عدد البنود: {len(report['records'])}. غير المحسوم: {len(report['unassessed_requirements'])}. البنود التي سجل فيها نقص أو خطأ: {len(report['gaps'])}.",'']
    for record in report['records']:
        reference=record['document_name']+'، حسب المادة رقم ('+str(record['article_number'])+')'
        if record['paragraph_number'] is not None:reference+='، الفقرة '+str(record['paragraph_number'])
        lines += ['## '+record['topic_title'],'',
            'الجهة المسؤولة أو محل التعريف: '+roles.get(record['responsible_role'],record['responsible_role']),
            'المرجع: '+reference,
            'الحالة: '+statuses[record['coverage']],
            'الادعاءات التي تستشهد بالمادة: '+('، '.join(map(str,record['article_citing_claims'])) or 'لا يوجد'),
            'الادعاءات المرتبطة بمقطع هذه الفقرة: '+('، '.join(map(str,record['paragraph_linked_claims'])) or 'لا يوجد'),
            'هل دخل نص المادة في سياق الإجابة؟ '+('نعم' if record['source_in_context'] else 'لا'),'']
        if record['reason']:lines += ['ملاحظة المراجعة: '+record['reason'],'']
        lines += ['النص الداعم بشروطه واستثناءاته:','',record['source_text'],'']
    lines += ['## ادعاءات الإجابة المحفوظة','']
    for number,claim in enumerate(report['claims'],1):lines += [f"{number}. {claim['text']}",'']
    lines += ['## حدود النتيجة','',
        'لا يغير هذا التقرير الجواب الأصلي ولا يصححه آلياً. المجموعة الحالية لا تغطي جميع أحكام PDPL. المراجعة الكاملة للاختبار والاتفاق بين المقيمين تبقى عملاً مستقلاً.','']
    return '\n'.join(lines)


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit',type=Path)
    parser.add_argument('--review',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    corpus_path=root/'data/processed/legal_articles.json'
    register_path=root/'data/checklists/legal_requirements_v1.json'
    register=json.loads(register_path.read_text(encoding='utf-8-sig'))
    if hashlib.sha256(corpus_path.read_bytes()).hexdigest()!=register['base_corpus_sha256']:
        raise ValueError('Corpus changed since requirements register was prepared.')
    audit=json.loads(args.audit.read_text(encoding='utf-8-sig'))
    decisions=json.loads(args.review.read_text(encoding='utf-8-sig')) if args.review else None
    report=build_report(audit,register,json.loads(corpus_path.read_text(encoding='utf-8-sig')),decisions)
    report['source_audit_sha256']=hashlib.sha256(args.audit.read_bytes()).hexdigest()
    report['requirements_sha256']=hashlib.sha256(register_path.read_bytes()).hexdigest()
    folder=root/'reports/requirements_review_v1'/uuid.uuid4().hex
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'coverage.md').write_text(markdown(report),encoding='utf-8')
    print('Requirements:',len(report['records']),'| Unassessed:',len(report['unassessed_requirements']),'| Gaps:',len(report['gaps']))
    print('Report:',folder/'coverage.md')
    print('No API calls. Original answer and experiments retained.')


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as error:
        print('Setup/review error:',error)
        raise SystemExit(1)
