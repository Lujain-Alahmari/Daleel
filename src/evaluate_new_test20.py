"""Frozen 20-question test: live retrieval + role-prompt-only generator.
No API, retrieval or model loading unless --run is explicit.
First attempts only. Failed/interrupted cases remain failures; no best-of retries.
"""
import argparse,hashlib,json,subprocess,sys,time,shutil,os,importlib.util
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    t=p.with_suffix(".tmp");t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8");t.replace(p)

def read_protocol(root):
    path=root/"config/rag_new_test20_v1.json"
    protocol=json.loads(path.read_text(encoding="utf-8-sig"))
    for rel,expected in protocol["files"].items():
        if digest(root/rel)!=expected:raise ValueError("Frozen test input/code changed: "+rel)
    data=json.loads((root/"data/evaluation/new_test20_v1/model_questions.json").read_text(encoding="utf-8-sig"))
    # Projection only: no answerability, reference answer, gold articles or notes
    # are retained for retrieval/generation.
    cases=[{k:q[k] for k in ("id","question","language","split")} for q in data if q["split"]=="test"]
    if len(cases)!=20 or len({c["id"] for c in cases})!=20:raise ValueError("Expected 20 unique new test cases.")
    if sorted(c["id"] for c in cases)!=protocol["question_ids"]:raise ValueError("New-test IDs differ.")
    if any(c["language"] not in ("ar","en") for c in cases):raise ValueError("Unsupported language.")
    return path,protocol,sorted(cases,key=lambda c:c["id"])

def summary(state,total):
    rows=list(state["attempts"].values())
    return {"total_questions":total,"attempted":len(rows),
      "successful_first_attempts":sum(r.get("success",False) for r in rows),
      "failed_or_interrupted_first_attempts":sum(not r.get("success",False) for r in rows),
      "pending":total-len(rows),
      "status_counts":{k:sum(r.get("status")==k for r in rows) for k in ("answered","partial","not_covered")},
      "semantic_accuracy":"not yet graded; execution/validation success is not answer accuracy"}

def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument("--run",action="store_true")
    p.add_argument("--limit",type=int,default=3,help="Maximum new test questions in this batch.")
    args=p.parse_args()
    if args.limit<1:p.error("--limit must be positive")
    root=args.root;protocol_path,protocol,cases=read_protocol(root)
    folder=root/"reports/rag_new_test20_v1";checkpoint=folder/"checkpoint.json"
    state=json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {
       "protocol_sha256":digest(protocol_path),"attempts":{}}
    if state["protocol_sha256"]!=digest(protocol_path):raise ValueError("Test protocol changed; preserve existing results.")
    pending=[c for c in cases if c["id"] not in state["attempts"]]
    print(f"New test set: {len(cases)} questions | attempted: {len(state['attempts'])} | pending: {len(pending)}")
    print("Live retrieval + frozen selected generator. No E5 threshold. One generation request per question.")
    print("No reference answers or gold articles enter retrieval/generation. Errors are not legal abstentions.")
    if not args.run:
        print("Dry run passed: no API, retrieval or model loading. --run --limit 3 starts/resumes.")
        return
    if not pending:
        print("All first attempts recorded. Do not rerun cases to replace errors.");return
    import chat_rag_source_planned as engine
    from dotenv import load_dotenv
    load_dotenv(root/".env",override=False,encoding="utf-8-sig")
    if not os.environ.get("GEMINI_API_KEY"):raise ValueError("Missing GEMINI_API_KEY; no new-test attempts started.")
    try:sdk=importlib.util.find_spec("google.genai")
    except ModuleNotFoundError:sdk=None
    if sdk is None:raise ValueError("Missing google-genai; activate pdpl-copilot.")
    # Selected modules use their own project root; require real project location.
    if engine.ROOT.resolve()!=root.resolve():raise ValueError("Imported generator belongs to another project.")
    folder.mkdir(parents=True,exist_ok=True)
    contexts=folder/"contexts";contexts.mkdir(exist_ok=True)
    audits=folder/"audits";audits.mkdir(exist_ok=True)
    lock=folder/"running.lock"
    try:handle=lock.open("x",encoding="utf-8")
    except FileExistsError:raise ValueError("Another test run or stale running.lock exists; inspect before resuming.")
    try:
        handle.write("New-question evaluation in progress. Do not launch another process.\n");handle.close()
        for i,case in enumerate(pending[:args.limit],1):
            qid=case["id"]
            row={"success":False,"status":None,"phase":"retrieval",
                 "error":"Interrupted or failed before completion","language":case["language"]}
            state["attempts"][qid]=row;save(checkpoint,state)
            started=time.monotonic()
            print(f"Testing {qid} ({i}/{min(args.limit,len(pending))})",flush=True)
            try:
                # Calls the already frozen live-context implementation unchanged.
                parts,context=engine.build_live_context(case["question"],case["language"])
                sources={f"{a['document_id']}:{a['article_number']}:{a['language']}":a for a in context["articles"]}
                frozen={"question":case["question"],"language":case["language"],"parts":parts,"sources":sources,
                    "context_diagnostics":{k:v for k,v in context.items() if k!="articles"}}
                context_path=contexts/(qid+".json");save(context_path,frozen)
                row.update(context_file=str(context_path.relative_to(root)),context_sha256=digest(context_path),
                    retrieval_seconds=time.monotonic()-started,phase="generation")
                save(checkpoint,state)
                raw_folder=root/"reports/rag_roles_prompt_only_v1"
                before=set(raw_folder.glob("*.json")) if raw_folder.exists() else set()
                cmd=[sys.executable,"-u",str(root/"src/chat_rag_roles_prompt_only.py"),
                     "--saved-context",str(context_path),"--run"]
                child=subprocess.Popen(cmd,cwd=str(root),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,encoding="utf-8",errors="replace",bufsize=1)
                try:
                    for line in child.stdout:print(line,end="",flush=True)
                    exit_code=child.wait()
                except KeyboardInterrupt:
                    child.terminate();child.wait(timeout=10);raise
                after=set(raw_folder.glob("*.json")) if raw_folder.exists() else set()
                created=after-before
                if len(created)!=1:raise ValueError("Expected one generation audit; check concurrent activity or setup error.")
                original=created.pop()
                audit=json.loads(original.read_text(encoding="utf-8-sig"))
                if audit.get("question")!=case["question"] or audit.get("language")!=case["language"]:
                    raise ValueError("Generated audit does not match new-test question.")
                path=audits/(qid+".json")
                if path.exists():raise ValueError("Existing new-test audit would be overwritten.")
                shutil.copy2(original,path)
                row.update(audit_file=str(path.relative_to(root)),audit_sha256=digest(path),
                   original_audit_file=str(original.relative_to(root)),exit_code=exit_code,
                   success=exit_code==0 and bool(audit.get("answer")) and not audit.get("error"),
                   status=audit.get("answer",{}).get("status"),error=audit.get("error"),phase="completed")
            except KeyboardInterrupt:
                row.update(error="Interrupted first attempt; retained as error.",phase="interrupted")
                raise
            except Exception as error:
                row.update(error=str(error),phase="failed")
            finally:
                row["total_seconds"]=time.monotonic()-started
                save(checkpoint,state);save(folder/"execution_summary.json",summary(state,len(cases)))
            if not row["success"]:
                print("Stopped after failed first attempt. Default resume continues remaining questions; failure stays recorded.")
                break
            if i<min(args.limit,len(pending)):
                print("Pausing 20 seconds...",flush=True);time.sleep(20)
    finally:
        if not handle.closed:handle.close()
        lock.unlink(missing_ok=True)
    print("Test reports:",folder)

if __name__=="__main__":
    try:main()
    except (ValueError,OSError,RuntimeError) as error:
        print("New test setup error:",error);raise SystemExit(1)
