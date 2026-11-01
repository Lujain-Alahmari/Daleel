"""Source planning by topic and explicit one-hop legal cross-references.

No gold answers/article IDs are used to select sources. Topic detection is a
fixed development lexicon, not a guarantee of complete question decomposition.
"""
import re
import unicodedata
from evaluate_query_expansion import expanded_queries


def norm(text):
    text=unicodedata.normalize('NFKC',text).lower()
    text=re.sub(r'[\u0610-\u061a\u064b-\u065f\u0670ـ]','',text)
    return re.sub('[أإآٱ]','ا',text)


def identity(article):
    return article['document_id'],int(article['article_number']),article['language']


def plan_parts(question,language):
    facets=expanded_queries(question,language)
    if not facets:
        facets=[{'topic':'question','query':question}]
    return [{'id':f'P{i}','topic':f['topic'],'query':f['query']}
            for i,f in enumerate(facets,1)]


def ordinal(value):
    value=re.sub(r'\s+','',norm(value))
    try:return int(value)
    except ValueError:pass
    ones=['الاولى','الثانية','الثالثة','الرابعة','الخامسة','السادسة','السابعة','الثامنة','التاسعة']
    for i,word in enumerate(ones,1):
        if value==word:return i
    if value=='العاشرة':return 10
    teens=['الحادية','الثانية','الثالثة','الرابعة','الخامسة','السادسة','السابعة','الثامنة','التاسعة']
    for i,word in enumerate(teens,11):
        if value==word+'عشرة':return i
    for tens,forms in [(20,['العشرون','العشرين']),(30,['الثلاثون','الثلاثين']),(40,['الاربعون','الاربعين'])]:
        for form in forms:
            if value==form:return tens
            for i,word in enumerate(['الحادية',*ones[1:]],1):
                if value==word+'و'+form:return tens+i
    return None


def referenced_articles(article):
    """Resolve explicit references to this corpus only; no external-law guessing."""
    text=re.sub(r'\s+',' ',norm(article['text']));doc=article['document_id'];language=article['language']
    if language=='ar':
        pattern=r'(?:المادة|للمادة|المادتين|للمادتين|المادتان|المواد|للمواد)\s*(?P<body>\([^)]{1,65}\)(?:\s*(?:و|،|,)\s*\([^)]{1,65}\))*)\s*(?=(?P<tail>[^.\n]{0,100}))'
    else:
        pattern=r'\barticles?\s*(?P<body>\(?\d{1,2}\)?(?:\s*(?:and|,|&)\s*\(?\d{1,2}\)?)*)\s*(?=(?P<tail>[^.\n]{0,100}))'
    found=[];seen=set()
    for match in re.finditer(pattern,text):
        tail=match['tail'];target=None
        if language=='ar':
            if re.match(r'(?:من\s*)?(?:هذه\s*)?اللائحة\s*التنفيذية',tail):target='implementing_regulation'
            elif re.match(r'(?:من\s*)?(?:هذا\s*)?(?:النظام|نظام\s*حماية\s*البيانات)',tail):target='pdpl'
            elif re.match(r'(?:من\s*)?(?:هذه\s*)?اللائحة',tail):target=doc if doc!='pdpl' else None
            values=re.findall(r'\(([^)]+)\)',match['body'])
            numbers=[ordinal(v) for v in values]
        else:
            if re.match(r'of\s+(?:the\s+)?implementing\s+regulations?',tail):target='implementing_regulation'
            elif re.match(r'of\s+(?:(?:the|this)\s+law|(?:the\s+)?personal\s+data\s+protection\s+law)|herein',tail):target='pdpl'
            elif re.match(r'of\s+this\s+regulation',tail):target=doc if doc!='pdpl' else None
            numbers=[int(v) for v in re.findall(r'\d+',match['body'])]
        if target is None:continue
        for number in numbers:
            if number is None:continue
            key=(target,number,language)
            if key not in seen:
                found.append({'key':key,'reference':match.group(0)})
                seen.add(key)
    return found


def assemble_context(original,aspect_rankings,articles,part_ids,max_articles=24,max_chars=60000):
    """Preserve original top 10, add top 3 per aspect, then one-hop references.

    References are collected from initial sources only, avoiding recursive drift.
    Selection does not use gold relevance. Report source provenance and budget gaps.
    """
    if len(aspect_rankings)!=len(part_ids):raise ValueError('Aspect/part mismatch')
    selected=[];provenance={};gaps=[];total=0
    def add(article,reason):
        nonlocal total
        key=identity(article)
        if key in provenance:
            if reason not in provenance[key]:provenance[key].append(reason)
            return True
        if len(selected)>=max_articles or total+len(article['text'])>max_chars:
            gaps.append({'article':key,'reason':'context_budget','requested_by':reason})
            return False
        selected.append(article);provenance[key]=[reason];total+=len(article['text'])
        return True
    for article in original[:10]:
        if not add(article,'original_question'):
            raise ValueError('Budget cannot preserve the original ten-source context.')
    for rank in range(3):
        for part_id,ranking in zip(part_ids,aspect_rankings):
            if rank<len(ranking):add(ranking[rank],part_id)
    initial=list(selected)
    links=[]
    for article in initial:
        for reference in referenced_articles(article):
            key=reference['key']
            if key not in articles:
                gaps.append({'article':key,'reason':'referenced_source_unavailable','from':identity(article)})
                continue
            okay=add(articles[key],'reference_from:'+':'.join(map(str,identity(article))))
            if okay:
                for part_id in part_ids:
                    if part_id in provenance[identity(article)] and part_id not in provenance[key]:
                        provenance[key].append(part_id)
            links.append({'from':identity(article),'to':key,'included':okay,'reference':reference['reference']})
    return {'articles':selected,'provenance':{':'.join(map(str,k)):v for k,v in provenance.items()},
            'links':links,'gaps':gaps,'context_chars':total,'initial_count':len(initial),
            'part_source_ids':{p:[':'.join(map(str,k)) for k,v in provenance.items() if p in v] for p in part_ids}}
