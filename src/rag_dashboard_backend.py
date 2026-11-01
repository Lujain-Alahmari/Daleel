"""Dashboard adapter for the evaluated role-prompt-only generator.
The retrieval algorithm, generator prompt and validation are unchanged.
"""
from pathlib import Path
import hashlib,json,os,threading,time,uuid
ROOT=Path(__file__).resolve().parent.parent

def verify_core(root=ROOT):
    manifest=json.loads((root/"config/dashboard_model_v1.json").read_text(encoding="utf-8"))
    for rel,expected in manifest["files"].items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError("Selected model/source file changed: "+rel)
    return manifest

class DashboardRag:
    def __init__(self,root=ROOT,retriever=None,characters=None,generate=None):
        self.root=Path(root);verify_core(self.root)
        import chat_rag_roles_prompt_only as candidate
        self.candidate=candidate;self.engine=candidate.engine
        self.articles=json.loads((self.root/"data/processed/legal_articles.json").read_text(encoding="utf-8-sig"))
        self.register=json.loads((self.root/"data/checklists/legal_requirements_v2.json").read_text(encoding="utf-8-sig"))
        if self.register["base_corpus_sha256"]!=hashlib.sha256((self.root/"data/processed/legal_articles.json").read_bytes()).hexdigest():
            raise ValueError("Requirement corpus differs.")
        self.lookup={self.engine.identity(a):a for a in self.articles}
        if retriever is None:
            from search_reranked import RerankedRetriever
            retriever=RerankedRetriever()
        if characters is None:
            from evaluate_character_retrieval import CharacterCandidates
            characters=CharacterCandidates(self.articles)
        self.retriever=retriever;self.characters=characters
        self._generate=generate;self._lock=threading.Lock()

    def build_context(self,question,language):
        # Same sequence and limits as frozen build_live_context; resources reused.
        from evaluate_query_expansion import extend_pool
        parts=self.engine.plan_parts(question,language)
        original=self.retriever.hybrid.search(question,k=20,language=language,candidate_k=20)
        supplementary=[self.retriever.hybrid.search(p["query"],k=20,language=language,candidate_k=20) for p in parts]
        pool=extend_pool(original,[self.characters.search(question,language,k=10),*supplementary],limit=30)
        ranked=self.retriever.rerank(question,pool,k=len(pool))
        aspect_rankings=[self.retriever.rerank(p["query"],pool,k=len(pool)) for p in parts]
        context=self.engine.assemble_context([r["article"] for r in ranked],
            [[r["article"] for r in ranking] for ranking in aspect_rankings],
            self.lookup,[p["id"] for p in parts])
        return parts,context

    def answer(self,question,on_stage=None):
        question=question.strip()
        if not question:raise ValueError("السؤال فارغ.")
        if len(question)>3000:raise ValueError("اختصر السؤال إلى 3000 حرف كحد أقصى.")
        from dotenv import load_dotenv
        load_dotenv(self.root/".env",override=False,encoding="utf-8-sig")
        if self._generate is None and not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("مفتاح الاتصال غير متاح في إعدادات المشروع.")
        with self._lock:
            verify_core(self.root)
            start=time.monotonic();language=self.engine.base.detect_language(question)
            result={"question":question,"language":language,"status":"error","sources":{},"cited_source_ids":[]}
            audit={"version":"dashboard-adapter-v1","generator_version":self.candidate.VERSION,
                "model":self.engine.MODEL,"question":question,"language":language}
            try:
                if on_stage:on_stage("retrieval")
                retrieval_parts,context=self.build_context(question,language)
                sources={f"{a['document_id']}:{a['article_number']}:{a['language']}":a for a in context["articles"]}
                if not sources:raise ValueError("لم تتوفر مصادر للبحث.")
                result["sources"]=sources
                segments=self.engine.v2.source_segments(sources)
                units=self.candidate.select_units(question,language,retrieval_parts,sources,segments,self.register)
                for unit in units:unit["part_ids"]=["P1"]
                parts=[{"id":"P1","topic":"question","query":question}]
                diagnostics={k:v for k,v in context.items() if k!="articles"}
                payload=self.candidate.make_payload(question,language,parts,sources,segments,units,diagnostics)
                audit.update(parts=parts,retrieval_parts=retrieval_parts,sources=sources,segments=segments,
                    requirement_units=units,context_diagnostics=diagnostics,
                    generation_config=payload["generation_config"],
                    payload_sha256=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest())
                if on_stage:on_stage("generation")
                generate=self._generate or self.engine.base.generate_with_deadline
                raw=generate(payload,timeout_seconds=180);audit["raw_response"]=raw
                if raw.get("status")!="completed":raise ValueError("Incomplete API response.")
                answer,_,checks,gaps,_=self.candidate.validate(raw["text"],parts,sources,segments,units,language)
                text=self.engine.render(answer,[],sources,segments,language,False)
                cited=list(dict.fromkeys(e.source_id for claim in answer.claims for e in claim.evidence))
                result.update(status=answer.status,answer_text=text,cited_source_ids=cited)
                audit.update(answer=answer.model_dump(),requirement_coverage=checks,requirement_gaps=gaps)
            except Exception as error:
                audit["error"]=str(error)
                result["message"]=("تعذر إعداد إجابة موثقة لهذه المحاولة. يمكنك مراجعة النصوص المسترجعة أدناه."
                    if language=="ar" else "A verified response could not be prepared for this attempt. You can inspect the retrieved sources below.")
            result["elapsed_seconds"]=round(time.monotonic()-start,1)
            folder=self.root/"reports/dashboard_live_v1";folder.mkdir(parents=True,exist_ok=True)
            path=folder/(uuid.uuid4().hex+".json")
            audit["status"]=result["status"];audit["elapsed_seconds"]=result["elapsed_seconds"]
            path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
            return result

def evaluation_cards(summary):
    total=summary["questions"]
    full=summary["full_four_criteria_answers"]+summary["correct_abstentions"]
    return {"total":total,"complete":full,"complete_rate":full/total,
       "execution_success":total-summary["execution_errors"],
       "execution_rate":summary["execution_success_rate"],
       "abstentions":summary["correct_abstentions"],"unanswerable":summary["unanswerable"],
       "accepted_answers":summary["accepted_answerable_answers"],"errors":summary["execution_errors"]}
