"""E5 stage 1: development-only context relevance scores. No Gemini calls."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
RUN="e5_abstention_development_v1"

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,data):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(path)

def metrics(cases,scores,threshold):
    answerable=[c for c in cases if c["answerable"]]
    unanswerable=[c for c in cases if not c["answerable"]]
    rejected=lambda c:scores[c["id"]]["max_score"]<threshold
    false=sum(rejected(c) for c in answerable)
    correct=sum(rejected(c) for c in unanswerable)
    fr=false/len(answerable) if answerable else None
    ar=correct/len(unanswerable) if unanswerable else None
    return {"threshold":threshold,"answerable_n":len(answerable),"unanswerable_n":len(unanswerable),
        "false_refusals":false,"correct_abstentions":correct,
        "false_refusal_rate":fr,"unanswerable_abstention_rate":ar,
        "balanced_accuracy":((1-fr)+ar)/2 if fr is not None and ar is not None else None}

def choose_threshold(cases,scores):
    values=sorted({scores[c["id"]]["max_score"] for c in cases})
    if not values or not all(math.isfinite(v) for v in values):raise ValueError("Invalid scores.")
    candidates=[values[0]]+[(a+b)/2 for a,b in zip(values,values[1:])]+[math.nextafter(values[-1],math.inf)]
    curve=[metrics(cases,scores,t) for t in candidates]
    if any(r["balanced_accuracy"] is None for r in curve):raise ValueError("Both label classes required.")
    # Tune only on development: maximize balanced accuracy, fewer false refusals
    # on ties, then lower threshold. These are fitted development metrics.
    best=max(curve,key=lambda r:(r["balanced_accuracy"],-r["false_refusals"],-r["threshold"]))
    return best,curve

def read_inputs(root):
    manifest_path=root/"data/evaluation/e5_development_v1/manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for relative,expected in manifest["files"].items():
        if digest(root/relative)!=expected:raise ValueError("Frozen E5 file changed: "+relative)
    cases=json.loads((root/manifest["cases_file"]).read_text(encoding="utf-8"))
    if len(cases)!=30 or len({c["id"] for c in cases})!=30:raise ValueError("Expected 30 unique development cases.")
    if any(c["split"]!="development" for c in cases):raise ValueError("Held-out test cases are forbidden here.")
    return manifest_path,manifest,cases

def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument("--run",action="store_true")
    p.add_argument("--limit",type=int,default=5)
    args=p.parse_args()
    if args.limit<1:p.error("--limit must be positive")
    root=args.root;manifest_path,manifest,cases=read_inputs(root)
    folder=root/"reports"/RUN;checkpoint=folder/"score_checkpoint.json"
    state=json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {
        "manifest_sha256":digest(manifest_path),"scores":{}}
    if state["manifest_sha256"]!=digest(manifest_path):raise ValueError("Checkpoint manifest differs.")
    pending=[c for c in cases if c["id"] not in state["scores"]]
    print(f"Development cases: {len(cases)} | scored: {len(state['scores'])} | pending: {len(pending)}")
    print("E5 stage 1: score retrieved cached sources on CPU. No Gemini/API requests.")
    print("Score: maximum cross-encoder relevance over each cached context; not a confidence percentage.")
    if not args.run:
        print("Dry run passed. Use --run --limit 5 to score or resume locally.")
        return
    folder.mkdir(parents=True,exist_ok=True)
    if pending:
        # Rerank existing sources only: no dense model, FAISS or new retrieval.
        from search_reranked import RerankedRetriever
        reranker=RerankedRetriever(hybrid=object())
        for i,case in enumerate(pending[:args.limit],1):
            print(f"Scoring {case['id']} ({i}/{min(args.limit,len(pending))})",flush=True)
            context=json.loads((root/case["context_file"]).read_text(encoding="utf-8"))
            if context["question"]!=case["question"] or context["language"]!=case["language"]:
                raise ValueError("Context question or language mismatch.")
            candidates=[{"article":a,"score":0.0} for a in context["sources"].values()]
            if not candidates:raise ValueError("Empty cached context.")
            start=time.monotonic()
            ranked=reranker.rerank(case["question"],candidates,k=len(candidates))
            values=[float(r["score"]) for r in ranked]
            if not values or not all(math.isfinite(v) for v in values):raise ValueError("Invalid reranker result.")
            state["scores"][case["id"]]={"max_score":max(values),"source_count":len(candidates),
                "seconds":time.monotonic()-start,
                "article_scores":[{"document_id":r["article"]["document_id"],
                  "article_number":r["article"]["article_number"],"score":float(r["score"])} for r in ranked]}
            save(checkpoint,state)
            print("Saved scores:",len(state["scores"]),"/",len(cases),flush=True)
    if len(state["scores"])==len(cases):
        best,curve=choose_threshold(cases,state["scores"])
        selected={"experiment":RUN,"score_kind":manifest["score_kind"],"selection":best,
            "selection_rule":"max development balanced accuracy; ties prefer fewer false refusals then lower threshold",
            "fit_scope":"30 development cases; fitted performance, not independent test performance",
            "manifest_sha256":digest(manifest_path),"score_checkpoint_sha256":digest(checkpoint),
            "generation_comparison_complete":False,
            "limitations":["Only 5 unanswerable development cases.","Not a calibrated answer-confidence score.",
              "Threshold must not be chosen again on held-out test labels.",
              "Stage 2 with/without explicit abstain generation is still required."]}
        save(folder/"threshold_selection.json",selected);save(folder/"threshold_curve.json",curve)
        print("Development-selected threshold:",best["threshold"])
        print("Fitted false refusals:",best["false_refusals"],"/",best["answerable_n"])
        print("Fitted unanswerable rejections:",best["correct_abstentions"],"/",best["unanswerable_n"])
        print("Stage 1 complete. E5 generation ablation still pending; production candidate unchanged.")
    print("Reports:",folder)

if __name__=="__main__":
    try:main()
    except (ValueError,OSError,RuntimeError) as error:
        print("E5 setup/scoring error:",error)
        raise SystemExit(1)
