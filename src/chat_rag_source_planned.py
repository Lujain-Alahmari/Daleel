"""Experimental RAG with per-topic sources, cross-references and explicit actors.

One Gemini call after retrieval. Structural checks do not establish legal
entailment; this candidate must be evaluated on development before adoption.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
import uuid
import chat_rag_gemini as base
import chat_rag_gemini_v2 as v2
from legal_source_plan import plan_parts,assemble_context,identity,norm

ROOT=Path(__file__).resolve().parent.parent
MODEL=v2.MODEL
VERSION='pdpl-source-plan-actors-v1'
DISPLAY_VERSION='grouped-article-number-citations-rtl-v3'
INSTRUCTION=v2.SYSTEM_INSTRUCTION+'''

The input includes a required source_plan. Return exactly one question_parts
entry per supplied part_id, without merging or dropping parts. Topic queries
are retrieval aids, not evidence or extra facts about the scenario. Preserve
all material requests from the original question within their relevant parts.
Answer only from the supplied segments. If a requested condition or role has
no adequate source support, mark the affected part partial or not_covered.
Related marketing rules alone do not establish an answer about all sharing
or recipient-role requirements. Missing sources are not proof that no rule exists.

Return actors identifying the scenario's parties. For each specify actor_id,
scenario_name, legal_role, role_status, condition and evidence_segment_ids.
Use a named party from the question when established. Do not assign a vendor
or agency a role merely from its commercial name. Explain supported possible
roles conditionally if the scenario does not establish the role. Use unresolved
when support is unavailable. A source-supported or conditional legal role
requires evidence, and a conditional role requires its condition to be stated.
Do not classify supervisory authorities as courts or officers as controllers
merely because they carry out an assigned function.

Each claim must identify the responsible or relevant actor_ids. Its text must
state who performs the action and who receives a request or notice, as applicable.
Distinguish the data subject's consent from another actor's approval, and
distinguish who sends a notice from who receives it. Do not shift one party's
duty to another. Select enough evidence segments to support the entire claim,
including its actor, conditions and exceptions. Where roles are conditional,
keep the corresponding claims conditional. Do not add role facts from memory.
Use the requested language for scenario names, conditions, claims and coverage.
'''
ROLES=('controller','processor','data_subject','competent_authority','court','dpo','recipient','unresolved')


def schema_for(parts,segments):
    schema=v2.draft_schema(segments)
    claim=schema['$defs']['DraftClaim']
    claim['properties']['actor_ids']={'type':'array','items':{'type':'string'},'minItems':1}
    claim['required'].append('actor_ids')
    part=schema['$defs']['Part']
    part['properties']['part_id']={'type':'string','enum':[p['id'] for p in parts]}
    part['required'].append('part_id')
    role={'type':'object','properties':{
        'actor_id':{'type':'string'},'scenario_name':{'type':'string'},
        'legal_role':{'type':'string','enum':list(ROLES)},
        'role_status':{'type':'string','enum':['source_supported','conditional','unresolved']},
        'condition':{'type':'string'},
        'evidence_segment_ids':{'type':'array','items':{'type':'string','enum':list(segments)}}},
        'required':['actor_id','scenario_name','legal_role','role_status','condition','evidence_segment_ids'],
        'additionalProperties':False}
    schema['properties']['actors']={'type':'array','items':role}
    schema['required'].append('actors')
    return schema


def validate_response(raw,parts,sources,segments):
    draft=json.loads(raw)
    if set(draft)!={'actors','claims','question_parts'}:raise ValueError('Unexpected response fields.')
    actors={}
    role_keys={'actor_id','scenario_name','legal_role','role_status','condition','evidence_segment_ids'}
    for actor in draft['actors']:
        if set(actor)!=role_keys:raise ValueError('Unexpected actor fields.')
        aid=actor['actor_id']
        if not isinstance(aid,str) or not aid.strip() or aid in actors:raise ValueError('Missing/duplicate actor ID.')
        if not isinstance(actor['scenario_name'],str) or not actor['scenario_name'].strip():raise ValueError('Missing actor name.')
        if actor['legal_role'] not in ROLES or actor['role_status'] not in ('source_supported','conditional','unresolved'):
            raise ValueError('Invalid actor role/status.')
        if not isinstance(actor['condition'],str):raise ValueError('Invalid role condition.')
        evidence=actor['evidence_segment_ids']
        if not isinstance(evidence,list) or any(not isinstance(s,str) or s not in segments for s in evidence):
            raise ValueError('Actor cites an unavailable evidence segment.')
        if actor['role_status']!='unresolved' and not evidence:raise ValueError('Asserted role lacks evidence.')
        if actor['role_status']=='conditional' and not actor['condition'].strip():raise ValueError('Conditional role lacks condition.')
        if actor['legal_role']=='unresolved' and actor['role_status']!='unresolved':raise ValueError('Inconsistent unresolved role.')
        actors[aid]=actor
    expected={p['id'] for p in parts}
    received=[p.get('part_id') for p in draft['question_parts']]
    if len(received)!=len(expected) or set(received)!=expected:raise ValueError('A planned question part is missing or duplicated.')
    clean_claims=[]
    for claim in draft['claims']:
        if set(claim)!={'text','evidence_segment_ids','actor_ids'}:raise ValueError('Unexpected claim fields.')
        ids=claim['actor_ids']
        if not isinstance(ids,list) or not ids or any(not isinstance(a,str) or a not in actors for a in ids):
            raise ValueError('Claim lacks a known actor.')
        clean_claims.append({k:v for k,v in claim.items() if k!='actor_ids'})
    clean_parts=[];downgrades=[]
    for part in draft['question_parts']:
        if set(part)!={'part_id','part','coverage','claim_numbers','missing_information'}:raise ValueError('Unexpected part fields.')
        clean={k:v for k,v in part.items() if k!='part_id'}
        unresolved=[]
        for number in part['claim_numbers']:
            if not isinstance(number,int) or isinstance(number,bool) or not 1<=number<=len(draft['claims']):
                raise ValueError('Question part references an invalid claim.')
            for aid in draft['claims'][number-1]['actor_ids']:
                if actors[aid]['role_status']=='unresolved':unresolved.append(actors[aid]['scenario_name'])
        if unresolved and part['coverage']=='covered':
            clean['coverage']='partial'
            prefix='دور الطرف غير محسوم: ' if any('\u0600'<=c<='\u06ff' for name in unresolved for c in name) else 'Unresolved actor role: '
            clean['missing_information']=(prefix+', '.join(sorted(set(unresolved))))
            downgrades.append({'part_id':part['part_id'],'reason':'unresolved_actor_role'})
        clean_parts.append(clean)
    grounded=v2.ground_response(json.dumps({'claims':clean_claims,'question_parts':clean_parts},ensure_ascii=False),sources,segments)
    return grounded,list(actors.values()),downgrades


def build_live_context(question,language):
    from search_reranked import RerankedRetriever
    from evaluate_character_retrieval import CharacterCandidates
    from evaluate_query_expansion import extend_pool
    articles=json.loads((ROOT/'data/processed/legal_articles.json').read_text(encoding='utf-8-sig'))
    lookup={identity(a):a for a in articles}
    retriever=RerankedRetriever();characters=CharacterCandidates(articles)
    parts=plan_parts(question,language)
    original=retriever.hybrid.search(question,k=20,language=language,candidate_k=20)
    supplementary=[retriever.hybrid.search(p['query'],k=20,language=language,candidate_k=20) for p in parts]
    pool=extend_pool(original,[characters.search(question,language,k=10),*supplementary],limit=30)
    ranked=retriever.rerank(question,pool,k=len(pool))
    aspect_rankings=[retriever.rerank(p['query'],pool,k=len(pool)) for p in parts]
    context=assemble_context([r['article'] for r in ranked],
        [[r['article'] for r in ranking] for ranking in aspect_rankings],lookup,[p['id'] for p in parts])
    return parts,context


def asks_for_roles(question):
    text=norm(question)
    arabic=(r'(?:ما\s*(?:هي\s*)?|وضح\s*|اشرح\s*|حدد\s*|بين\s*).*?(?:ادوار|دور\s+(?:كل|الجه|الشرك|التطبيق|الوكال))',
            r'^\s*(?:من|اي\s*جهة).*?(?:جهة\s*(?:التحكم|المعالجة)|مسؤول\s*حماية)',
            r'(?:الفرق|فرق|تمييز|ميز).*?جهة\s*(?:التحكم|المعالجة)')
    english=(r'\b(?:explain|describe|identify|clarify|list)\b.*\broles?\b',
             r'\bwhat\b.*\broles?\b',r'\broles?\s+of\b',
             r'^\s*(?:who|which\s+(?:party|entity|organisation|organization))\b.*\b(?:controller|processor|officer)\b',
             r'\b(?:difference|distinguish)\b.*\b(?:controller|processor)\b')
    return any(re.search(pattern,text) for pattern in (*arabic,*english))


def document_title(document_id,language):
    names={
        'ar':{'pdpl':'نظام حماية البيانات الشخصية',
              'implementing_regulation':'اللائحة التنفيذية لنظام حماية البيانات الشخصية',
              'transfer_regulation':'لائحة نقل البيانات الشخصية إلى خارج المملكة'},
        'en':{'pdpl':'Personal Data Protection Law',
              'implementing_regulation':'Implementing Regulation of the Personal Data Protection Law',
              'transfer_regulation':'Regulation on Personal Data Transfer outside the Kingdom'}}
    name=names[language].get(document_id)
    if name is None:raise ValueError('Missing display name for source document.')
    return name


def citation_groups(source_ids,sources,language):
    groups={}
    for sid in source_ids:
        source=sources[sid]
        groups.setdefault(source['document_id'],set()).add(int(source['article_number']))
    labels=[]
    for document,numbers in groups.items():
        if language=='ar':
            values='، '.join(map(str,sorted(numbers)))
            labels.append(document_title(document,language)+'، حسب المادة رقم '+ '\u2066('+values+')\u2069')
        else:
            suffix=', Article ' if len(numbers)==1 else ', Articles '
            values=', '.join(map(str,sorted(numbers)))
            labels.append(document_title(document,language)+suffix+values)
    return labels


def citation_label(source_id,sources,language):
    return citation_groups([source_id],sources,language)[0]


def inline_citations(source_ids,sources,language):
    label=('؛ ' if language=='ar' else '; ').join(citation_groups(source_ids,sources,language))
    if not label:return ''
    # Arabic numbers already have parentheses in an LTR isolate; avoid nesting.
    return ('\u2067'+label+'\u2069') if language=='ar' else '('+label+')'


def render(answer,actors,sources,segments,language,show_roles=False):
    labels_ar={'controller':'جهة التحكم','processor':'جهة المعالجة','data_subject':'صاحب البيانات',
        'competent_authority':'الجهة المختصة','court':'المحكمة','dpo':'مسؤول حماية البيانات','recipient':'متلقي البيانات','unresolved':'دور غير محسوم'}
    lines=['هذه أداة معلومات ولا تعد استشارة قانونية.' if language=='ar' else 'This is an information tool, not legal advice.']
    if show_roles:lines+=['','الأطراف وأدوارهم:' if language=='ar' else 'Parties and roles:']
    for actor in (actors if show_roles else []):
        label=labels_ar[actor['legal_role']] if language=='ar' else actor['legal_role'].replace('_',' ')
        condition=actor['condition'].strip()
        state=actor['role_status']
        if state=='conditional':label=('دور مشروط: ' if language=='ar' else 'Conditional role: ')+label
        if state=='unresolved':label=labels_ar['unresolved'] if language=='ar' else 'Role unresolved'
        refs=sorted({segments[sid]['source_id'] for sid in actor['evidence_segment_ids']})
        lines.append('- '+actor['scenario_name']+' — '+label+('؛ '+condition if condition else '')+
                     (' '+inline_citations(refs,sources,language) if refs else ''))
    lines.append('')
    used=[]
    if answer.status=='not_covered':
        lines.append('المواد المسترجعة لا تتضمن معلومات كافية للإجابة عن هذا السؤال.' if language=='ar'
                     else 'The retrieved sources do not contain enough information to answer this question.')
    else:
        if answer.status=='partial':
            lines.append('إجابة جزئية بحسب المواد المسترجعة:' if language=='ar' else 'Partial answer from the retrieved sources:')
        for claim in answer.claims:
            refs=list(dict.fromkeys(e.source_id for e in claim.evidence))
            lines.append('- '+claim.text.strip()+' '+inline_citations(refs,sources,language))
            used.extend(sid for sid in refs if sid not in used)
        if answer.missing_information:
            lines+=['','ما لم تغطه الإجابة:' if language=='ar' else 'Not covered by the answer:']
            lines.extend('- '+item for item in answer.missing_information)
    role_refs=sorted({segments[e]['source_id'] for actor in actors for e in actor['evidence_segment_ids']})
    if show_roles:used.extend(sid for sid in role_refs if sid not in used)
    if used:
        lines+=['','المراجع:' if language=='ar' else 'References:']
        lines.extend('- '+label for label in citation_groups(used,sources,language))
    return '\n'.join(lines)


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question')
    parser.add_argument('--saved-context',type=Path)
    parser.add_argument('--language',choices=['ar','en'])
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--render-audit',type=Path,help='Redisplay a saved answer without retrieval or API calls.')
    display=parser.add_mutually_exclusive_group()
    display.add_argument('--show-roles',action='store_true')
    display.add_argument('--hide-roles',action='store_true')
    args=parser.parse_args()
    if args.render_audit:
        if args.question or args.saved_context or args.plan_only:
            parser.error('--render-audit cannot be combined with generation/plan inputs.')
        audit=json.loads(args.render_audit.read_text(encoding='utf-8-sig'))
        source_view=audit.get('source_snapshot',audit)
        answer=v2.GroundedAnswer.model_validate(audit['answer'])
        show=args.show_roles or (not args.hide_roles and asks_for_roles(source_view['question']))
        print(render(answer,audit['actors'],source_view['sources'],source_view['segments'],source_view['language'],show))
        return
    if bool(args.question)==bool(args.saved_context):
        parser.error('Use exactly one of --question or --saved-context.')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False,encoding='utf-8-sig')
    if args.saved_context:
        frozen=json.loads(args.saved_context.read_text(encoding='utf-8-sig'))
        question=frozen['question'];language=frozen['language']
        if args.language and args.language!=language:raise ValueError('Saved context language mismatch.')
        parts=frozen['parts'];sources=frozen['sources'];context=frozen['context_diagnostics']
        if parts!=plan_parts(question,language):raise ValueError('Saved question plan changed.')
        articles=json.loads((ROOT/'data/processed/legal_articles.json').read_text(encoding='utf-8-sig'))
        corpus={f"{a['document_id']}:{a['article_number']}:{a['language']}":a for a in articles}
        if any(sid not in corpus or s['text']!=corpus[sid]['text'] or s['language']!=language for sid,s in sources.items()):
            raise ValueError('Saved source differs from the local same-language corpus.')
        if len(sources)>24 or sum(len(s['text']) for s in sources.values())>60000:
            raise ValueError('Saved context exceeds source budget.')
        print('Using verified saved source plan; no retrieval/model loading.')
    else:
        question=args.question;language=args.language or base.detect_language(question)
        parts,context=build_live_context(question,language)
        sources={f"{a['document_id']}:{a['article_number']}:{a['language']}":a for a in context['articles']}
    if not sources:raise ValueError('No usable sources.')
    segments=v2.source_segments(sources)
    code_files=['chat_rag_source_planned.py','legal_source_plan.py','evaluate_query_expansion.py',
                'evaluate_character_retrieval.py','chat_rag_gemini.py','chat_rag_gemini_v2.py']
    audit={'version':VERSION,'display_version':DISPLAY_VERSION,'model':MODEL,'question':question,'language':language,'parts':parts,
           'code_sha256':{name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in code_files},
           'corpus_sha256':hashlib.sha256((ROOT/'data/processed/legal_articles.json').read_bytes()).hexdigest(),
           'generation_config':{'max_output_tokens':4096,'thinking_level':'low','fixed_seed':None},
           'sources':sources,'segments':segments,'context_diagnostics':{k:v for k,v in context.items() if k!='articles'}}
    folder=ROOT/'reports/rag_source_planned_v1';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(uuid.uuid4().hex+'.json')
    try:
        if args.plan_only:
            print(json.dumps({'parts':parts,'sources':list(sources),'gaps':context['gaps']},ensure_ascii=False,indent=2))
            return
        source_input=[{'source_id':sid,'document_id':s['document_id'],'article_number':s['article_number'],
                       'language':s['language'],'segments':[{'segment_id':seg['segment_id'],'text':seg['text']}
                          for seg in segments.values() if seg['source_id']==sid]} for sid,s in sources.items()]
        payload={'model':MODEL,'system_instruction':INSTRUCTION,
            'input':json.dumps({'question':question,'language':language,
                'source_plan':[dict(p,relevant_source_ids=context.get('part_source_ids',{}).get(p['id'],[])) for p in parts],
                'sources':source_input,'source_gaps':context['gaps']},ensure_ascii=False),
            'response_format':{'type':'text','mime_type':'application/json','schema':schema_for(parts,segments)},
            'generation_config':{'max_output_tokens':4096,'thinking_level':'low'},'store':False}
        response=base.generate_with_deadline(payload,timeout_seconds=180)
        audit['raw_response']=response
        if response.get('status')!='completed':raise ValueError('Incomplete API response.')
        answer,actors,downgrades=validate_response(response['text'],parts,sources,segments)
        audit.update(answer=answer.model_dump(),actors=actors,downgrades=downgrades)
        show=args.show_roles or (not args.hide_roles and asks_for_roles(question))
        text=render(answer,actors,sources,segments,language,show)
        audit['display_roles']=show
        audit['answer_text']=text
        print(text)
    except Exception as error:
        audit['error']=str(error)
        raise
    finally:
        path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Audit:',path)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,base.ResponseValidationError) as error:
        print('Request/setup error:',error)
        raise SystemExit(1)
