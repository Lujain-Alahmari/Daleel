# 🛡️ Daleel | دليل

A bilingual Saudi personal-data protection information assistant with evidence-backed legal answers, privacy-policy clause classification and policy disclosure-gap analysis.

**Information tool, not legal advice. A covered policy statement does not establish actual legal compliance.**

## 🖥️ Interfaces
- `app.py`: legal questions, one shared policy input with classification and gap-analysis tabs, and article browsing.
- `evaluation_dashboard.py`: independent saved-results dashboard for RAG, OPP-115 and policy analysis. No API requests are made by this dashboard.
- Arabic and English interface controls, RTL/LTR content direction and a light theme with explicit navigation contrast. Original source text and stored findings retain their original language.

## 🚀 Setup
Use Python 3.11 in a virtual or Conda environment.

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```
Set `GEMINI_API_KEY` privately in `.env`. Never place a real key in `.env.example` or commit `.env`.

```powershell
python -m streamlit run app.py
python -m streamlit run evaluation_dashboard.py --server.port 8502
```
Run the commands in separate terminals. The question route requires a Gemini key and may download multilingual E5 and BGE reranker weights on first use. The saved-results dashboard can be used without the classifier weight. Restart Streamlit after changing theme settings.

## 🧠 Classifier model
The 255 MiB classifier weight is intentionally **not tracked by ordinary Git**. Obtain the team's existing model artefact separately and restore it with:

```powershell
python scripts/restore_classifier.py --from-folder "PATH_TO_EXISTING_MODEL_FOLDER"
```
The source folder must contain the saved `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json` and `results.json`. SHA-256 hashes are checked before copying. Model files stay in the ignored `models/` directory. No hosted model-download URL is claimed. Without these assets, policy classification and the application's policy route cannot run.

## 📊 Saved evaluation
| Component | Result | Scope |
|---|---|---|
| Legal QA | 17/20 full-rubric answers or appropriate abstentions; 18/20 execution success | Frozen 20-question sample, not universal accuracy |
| OPP-115 classifier | Micro F1 0.7682; Macro F1 0.6710 | 671 English test segments; balanced TF-IDF has higher Macro F1 |
| Policy gap experiment | 76/125 status agreement (60.8%); Macro F1 0.5757 | Separate batched saved-output experiment; 32 validation errors among 180 outputs |

Metrics belong to different tasks and must not be combined. The 55 unresolved external/conditional policy-reference judgments are excluded from the primary text score. Team confirmation and independent reviewer agreement are not established. A whitespace-only evidence validation correction reused the same model responses and unchanged reference; it is not a new independent held-out test.

The latest live smoke answered a legal question, abstained on an outside-scope question and classified an example policy. The live 18-request policy-analysis route failed on its first request. It is **not** verified for a complete live policy demonstration. Browser visual verification was unavailable; simulated rendering-path and contrast checks passed.

## ✅ Verification
```powershell
python -m unittest discover -s tests
python scripts/check_setup.py
```
The setup command checks imports and protected-file hashes, not API service availability. Local hashes do not automatically detect new legal amendments.

## 📚 Data and reproducibility
The repository includes the article-level legal corpus, evaluated FAISS snapshot, question references, selected metrics, training scripts and the technical report. Raw OPP-115 data, full third-party local-policy snapshots, model weights, runtime drafts and private request logs are excluded. Obtain original datasets under their owners' terms; this repository does not grant redistribution rights to third-party content. Dataset source records and policy URLs/hashes are included. Full internal evaluation evidence remains in the submission bundle.

See [the technical report](docs/Daleel_Technical_Report_Final.pdf), its [editable Markdown source](docs/DALEEL_TECHNICAL_REPORT.md), and [model setup](docs/MODEL_SETUP.md).
