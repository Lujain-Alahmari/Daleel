"""E5 stage 2: controlled explicit-abstain ablation; 15 preselected development cases; at most 30 first requests."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from prepare_e5_abstention import read_inputs,digest,save
RUN="e5_small_development_v1"
COMMON="""You answer Saudi PDPL questions using only the supplied source text.
Do not use general legal knowledge. Respond in the question's language.
Give a direct, concise answer with only necessary conditions and exceptions.
Preserve the responsible party, legal object, mandatory/optional meaning and AND/OR.
Use only the supplied evidence segment IDs to support each factual claim.
Return JSON with status and claims according to the schema. A claim has text
and evidence_segment_ids. Do not output internal checks or additional fields.
"""
EXPLICIT="""If the supplied texts do not contain sufficient information to answer
the question, return status not_covered with an empty claims list rather than
guessing. Do not confuse a related topic with evidence that answers the question.
"""

def response_schema(segments):
    return {"type":"object","properties":{
      "status":{"type":"string","enum":["answered","not_covered"]},
      "claims":{"type":"array","items":{"type":"object","properties":{
        "text":{"type":"string"},"evidence_segment_ids":{"type":"array",
          "items":{"type":"string","enum":list(segments)},"minItems":1}},
        "required":["text","evidence_segment_ids"],"additionalProperties":False}}},
      "required":["status","claims"],"additionalProperties":False}

def check_response(text,sources,segments):
    data=json.loads(text)
    if set(data)!={"status","claims"} or data["status"] not in ("answered","not_covered"):
        raise ValueError("Invalid status/fields.")
    if not isinstance(data["claims"],list):raise ValueError("Claims must be a list.")
    if data["status"]=="not_covered":
        if data["claims"]:raise ValueError("Abstention must not contain claims.")
    elif not data["claims"]:raise ValueError("Answered response lacks claims.")
    for c in data["claims"]:
        if set(c)!={"text","evidence_segment_ids"} or not isinstance(c["text"],str) or not c["text"].strip():
            raise ValueError("Invalid claim.")
        ids=c["evidence_segment_ids"]
        if not isinstance(ids,list) or not ids or any(not isinstance(e,str) or e not in segments for e in ids):
            raise ValueError("Claim cites unavailable evidence.")
        for key in ids:
            s=segments[key]
            if s["source_id"] not in sources or sources[s["source_id"]]["text"][s["start"]:s["end"]]!=s["text"]:
                raise ValueError("Evidence text differs from original source.")
    return data

def summarize(cases,attempts,scores,threshold):
    table=[]
    for prompt in ("without_explicit","with_explicit"):
        for gate in (False,True):
            row={"prompt":prompt,"threshold_enabled":gate,"questions":len(cases),
                 "pending":0,"errors":0,"correct_unanswerable_abstentions":0,
                 "false_refusals":0,"answerable_answered":0,"unanswerable_answered":0}
            for c in cases:
                if gate and scores[c["id"]]["max_score"]<threshold:
                    status="not_covered"
                else:
                    saved=attempts.get(c["id"]+"|"+prompt,[])
                    if not saved:row["pending"]+=1;continue
                    first=saved[0]
                    if not first["success"]:row["errors"]+=1;continue
                    status=first["status"]
                if c["answerable"]:
                    row["false_refusals" if status=="not_covered" else "answerable_answered"]+=1
                else:
                    row["correct_unanswerable_abstentions" if status=="not_covered" else "unanswerable_answered"]+=1
            n_a=sum(c["answerable"] for c in cases);n_u=len(cases)-n_a
            row.update(answerable_n=n_a,unanswerable_n=n_u,
                unanswerable_accuracy=row["correct_unanswerable_abstentions"]/n_u,
                false_refusal_rate=row["false_refusals"]/n_a,
                incomplete=bool(row["pending"]),
                caution="Low refusal rate is not evidence of correct answers; errors are separate.")
            table.append(row)
    return table

def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument("--run",action="store_true")
    p.add_argument("--limit",type=int,default=2,help="Maximum API requests in this batch.")
    p.add_argument("--retry-failed",action="store_true")
    args=p.parse_args()
    if args.limit<1:p.error("--limit must be positive")
    root=args.root;manifest_path,manifest,all_cases=read_inputs(root)
    subset_path=root/"config/e5_small_selection.json"
    subset=json.loads(subset_path.read_text(encoding="utf-8-sig"))
    for relative,expected in subset["files"].items():
        if digest(root/relative)!=expected:raise ValueError("Frozen E5 subset input changed: "+relative)
    selected=set(subset["question_ids"])
    cases=[c for c in all_cases if c["id"] in selected]
    if len(cases)!=15 or len(selected)!=15 or any(c["split"]!="development" for c in cases):
        raise ValueError("Invalid development subset.")
    if sum(not c["answerable"] for c in cases)!=5:
        raise ValueError("Expected all five development unanswerable cases.")
    expected={c["id"] for c in all_cases if not c["answerable"]}
    for lang in ("ar","en"):
        expected.update(c["id"] for c in sorted(
            [c for c in all_cases if c["answerable"] and c["language"]==lang],
            key=lambda c:c["id"])[:5])
    if expected!=selected:raise ValueError("Subset differs from predeclared ID-order selection.")
    folder=root/"reports"/RUN
    source_folder=root/"reports/e5_abstention_development_v1"
    selection_path=source_folder/"threshold_selection.json";scores_path=source_folder/"score_checkpoint.json"
    if not selection_path.exists() or not scores_path.exists():
        print("Stage 1 scores/threshold pending. No API requests.")
        return
    selection=json.loads(selection_path.read_text(encoding="utf-8"))
    scores_state=json.loads(scores_path.read_text(encoding="utf-8"))
    if selection["manifest_sha256"]!=digest(manifest_path) or selection["score_checkpoint_sha256"]!=digest(scores_path):
        raise ValueError("Threshold provenance differs from frozen inputs.")
    scores=scores_state["scores"];threshold=selection["selection"]["threshold"]
    checkpoint=folder/"generation_checkpoint.json"
    state=json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {
        "manifest_sha256":digest(manifest_path),"threshold_selection_sha256":digest(selection_path),"subset_selection_sha256":digest(subset_path),"attempts":{}}
    if state["manifest_sha256"]!=digest(manifest_path) or state["threshold_selection_sha256"]!=digest(selection_path) or state.get("subset_selection_sha256")!=digest(subset_path):
        raise ValueError("Generation checkpoint belongs to another experiment.")
    tasks=[]
    for i,c in enumerate(cases):
        order=("without_explicit","with_explicit") if i%2==0 else ("with_explicit","without_explicit")
        for prompt in order:
            key=c["id"]+"|"+prompt;prior=state["attempts"].get(key,[])
            if not prior or (args.retry_failed and not prior[-1]["success"]):tasks.append((c,prompt,key))
    print(f"E5 generation: {len(cases)} development cases; {len(state['attempts'])}/30 attempted; {len(tasks)} pending.")
    print("15 development cases: all 5 unanswerable + first 5 answerable per language by ID.")
    print("Threshold was calibrated on 30 development cases; this is a subset generation ablation.")
    print("2 prompt conditions, 2 threshold conditions. Threshold arms reuse the same generations.")
    print("This is a controlled E5 ablation, not a replacement for the frozen production prompt.")
    if not args.run:
        print("Dry run; no API. --run --limit 2 starts or resumes at most 2 requests.")
        return
    import chat_rag_gemini as base
    import chat_rag_gemini_v2 as v2
    from dotenv import load_dotenv
    load_dotenv(root/".env",override=False,encoding="utf-8-sig")
    audits=folder/"generation_audits";audits.mkdir(parents=True,exist_ok=True)
    for number,(case,prompt,key) in enumerate(tasks[:args.limit],1):
        print(f"Generating {case['id']} | {prompt} ({number}/{min(args.limit,len(tasks))})",flush=True)
        frozen=json.loads((root/case["context_file"]).read_text(encoding="utf-8"))
        if frozen["question"]!=case["question"] or frozen["language"]!=case["language"]:
            raise ValueError("Case/context mismatch.")
        sources=frozen["sources"];segments=v2.source_segments(sources)
        # Deliberate allowlist: no answerability, gold article IDs, reference or review notes.
        model_input={"question":case["question"],"language":case["language"],
          "sources":[{"source_id":sid,"document_id":s["document_id"],"article_number":s["article_number"],
              "segments":[{"segment_id":e,"text":v["text"]} for e,v in segments.items() if v["source_id"]==sid]}
              for sid,s in sources.items()]}
        payload={"model":manifest["generator_model"],"system_instruction":COMMON+(EXPLICIT if prompt=="with_explicit" else ""),
          "input":json.dumps(model_input,ensure_ascii=False),
          "response_format":{"type":"text","mime_type":"application/json","schema":response_schema(segments)},
          "generation_config":{"max_output_tokens":4096,"thinking_level":"low"},"store":False}
        attempt_number=len(state["attempts"].get(key,[]))+1
        path=audits/(case["id"]+"_"+prompt+"_attempt"+str(attempt_number)+".json")
        if path.exists():raise ValueError("Existing uncheckpointed audit; inspect before resuming.")
        audit={"question_id":case["id"],"prompt_condition":prompt,"question":case["question"],
          "language":case["language"],"model":manifest["generator_model"],"sources":sources,"segments":segments,
          "payload_sha256":hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
        attempt={"success":False,"status":None,"audit_file":str(path.relative_to(root))}
        try:
            response=base.generate_with_deadline(payload,timeout_seconds=180);audit["raw_response"]=response
            if response.get("status")!="completed":raise ValueError("Incomplete provider response.")
            answer=check_response(response["text"],sources,segments)
            audit["answer"]=answer;attempt.update(success=True,status=answer["status"])
        except Exception as error:
            audit["error"]=str(error);attempt["error"]=str(error)
        save(path,audit);attempt["audit_sha256"]=digest(path)
        state["attempts"].setdefault(key,[]).append(attempt)
        save(checkpoint,state)
        save(folder/"abstention_comparison.json",{"stage":"15-case development ablation; threshold fitted on full 30-case development set","question_ids":sorted(selected),
            "rows":summarize(cases,state["attempts"],scores,threshold),
            "manual_answer_grading":"required; do not infer correctness from status/citations",
            "additional_attempts":sum(max(0,len(a)-1) for a in state["attempts"].values())})
        if not attempt["success"]:
            print("Stopped; failure preserved separately from legal abstention. Default resume skips it.")
            break
        if number<min(args.limit,len(tasks)):
            print("Pausing 20 seconds...",flush=True);time.sleep(20)
    print("Reports:",folder)

if __name__=="__main__":
    try:main()
    except (ValueError,OSError,RuntimeError) as error:
        print("E5 evaluation error:",error);raise SystemExit(1)
