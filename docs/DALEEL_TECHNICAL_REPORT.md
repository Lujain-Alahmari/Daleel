# Daleel Technical Report
Integrated legal assistant and privacy policy analysis | 1 November 2026
## Summary

Daleel combines bilingual question answering over the Saudi personal data protection corpus with privacy-policy clause classification and disclosure-gap review. The interface and saved-results dashboard are separate Streamlit entry points. Users can choose Arabic or English; original source passages retain their original language.

| Component | Measured result | Interpretation |
| --- | --- | --- |
| Legal question answering | 17/20 full-rubric answers or appropriate abstentions; 18/20 execution success | Frozen 20-question evaluation, not universal legal accuracy |
| OPP-115 classifier | Micro F1 0.7682; Macro F1 0.6710 | English multi-label test on 671 segments |
| Policy gap analysis | 76/125 text-status agreement (60.8%); Macro F1 0.5757 | Separate batched experiment; 32 validation errors among 180 outputs |

These results measure different tasks and cannot be combined into a single project accuracy. The gap-analyser result is weaker than the other components and requires review before use. Independent reviewer agreement has not been measured. A covered policy statement does not establish operational compliance.

## Delivered application

app.py provides legal questions, policy analysis and legal article browsing. Policy analysis uses one text input and two result tabs: paragraph classification and gap analysis. evaluation_dashboard.py displays saved RAG, classifier and policy-analysis results without making API requests.

## Evaluation boundaries

The historical RAG evaluation remains tied to its frozen configuration. Later interface changes do not constitute a new benchmark. The policy-score correction reuses existing model responses and changes only whitespace-sensitive evidence matching; it is not a new independent held-out test.

# Knowledge base and legal retrieval
The article-level knowledge base contains 180 records: 43 PDPL articles, 38 Implementing Regulation articles and 9 Transfer Regulation articles in each of Arabic and English. The multilingual dense index contains 210 embedding windows. Article identity and full source text are retained for final citations.

Metadata identifies the document, language, article number and recorded source. Provenance sidecars and file hashes support traceability. The frozen corpus SHA-256 is 242b23fa1eba69ce577812d939404e6827213caca9ae25a08412b034ee106b86. Hash checks detect changes in the local snapshot; they do not automatically detect newly published legal amendments.

BM25 supplies exact-term retrieval. intfloat/multilingual-e5-base supplies multilingual dense embeddings. Reciprocal rank fusion combines rankings with k=60, and BAAI/bge-reranker-v2-m3 reranks article candidates. The selected live route supplements candidate retrieval with character and query-part searches. Retrieved source texts, rather than model memory, are supplied as evidence.

## Development retrieval comparison

| Language | Method | n | Recall@5 | Recall@10 | MRR@5 |
| --- | --- | --- | --- | --- | --- |
| ar | bm25 | 13 | 0.625 | 0.636 | 0.846 |
| ar | dense | 13 | 0.793 | 0.908 | 0.846 |
| ar | hybrid | 13 | 0.690 | 0.842 | 0.814 |
| ar | hybrid_reranked | 13 | 0.755 | 0.881 | 0.923 |
| en | bm25 | 12 | 0.708 | 0.778 | 0.840 |
| en | dense | 12 | 0.819 | 0.889 | 1.000 |
| en | hybrid | 12 | 0.778 | 0.917 | 0.896 |
| en | hybrid_reranked | 12 | 0.882 | 0.965 | 1.000 |

The comparison uses 25 answerable development questions (13 Arabic, 12 English). Five unanswerable or pending cases and test questions were excluded from retrieval scoring. Dense retrieval is strongest at Arabic Recall@5 in this comparison; reranking improves English recall and Arabic first-result ranking. Hybrid retrieval is not claimed to dominate every metric.

# Legal question answering evaluation
The generator is Gemini gemini-3.5-flash-lite in the selected role-prompt-only configuration. Roles are handled in the prompt rather than requiring a separate actor-output gate. Source identity, evidence and requirement-scope checks remain. An answer must rely on the supplied legal articles and can abstain when they do not support an answer.

| Frozen test outcome | Count |
| --- | --- |
| Questions | 20 (10 Arabic, 10 English) |
| Answerable / outside source coverage | 16 / 4 |
| Accepted answerable answers | 14 |
| Full four-criteria answers | 13 |
| Appropriate abstentions | 4 |
| Execution errors | 2 |
| Full rubric answers + appropriate abstentions | 17/20 (85%) |
| Execution success | 18/20 (90%) |

| Accepted-answer quality metric | Mean out of 2 |
| --- | --- |
| Correctness | 2.0 |
| Relevance | 2.0 |
| Citation Support | 1.929 |
| Necessary Completeness | 2.0 |

The quality means cover only the 14 accepted answers. They exclude errors and abstentions and therefore must not be presented as performance over all questions. One failed test request returned HTTP 400; another failed a clause/scope evidence check. The sample contains only four unanswerable questions, so abstention estimates are limited.

## Development evidence

E1 compares LLM-only, naive dense and hybrid-reranked generation on development questions. E2 compares article-level and fixed chunks. E3 covers cross-language retrieval and English embedding models. E5 studies abstention. Their saved evidence archives are retained separately under reports/evidence. Development findings are not substituted for the final frozen test score.

Citations are checked programmatically for identity and evidence presence. These checks do not prove that each interpretation follows from its quotation. Semantic comparison against the local sources was recorded; independent two-reviewer agreement is not available.

# Privacy policy clause classification
The selected classifier is a fine-tuned distilbert-base-uncased model with ten OPP-115 labels. Training uses a multi-label sigmoid formulation and per-class thresholds. The saved experiment records best epoch 3, a maximum input length of 256 tokens and batch size 8 at inference.

The audited data contain 92 training policies with 3,089 segments and 23 test policies with 679 segments. Eight normalized-text duplicates were removed from the test scoring set, leaving 671 test segments. No policy-level overlap or remaining normalized-text overlap was found in the audit. The training scripts use policy-grouped splits and validation-based threshold selection.

| Model | Micro F1 | Macro F1 |
| --- | --- | --- |
| TF-IDF baseline | 0.6460 | 0.3984 |
| TF-IDF tuned | 0.7334 | 0.6625 |
| TF-IDF balanced | 0.7420 | 0.6951 |
| DistilBERT | 0.7682 | 0.6710 |

DistilBERT has the highest micro F1 among these saved runs, while the balanced TF-IDF baseline has the higher macro F1 (0.6951 versus 0.6710). The transformer is not uniformly superior. Its exact label-set match is 46.8%; a correct subset of labels is insufficient for this strict metric.

| Category | F1 | Support |
| --- | --- | --- |
| Data Retention | 0.300 | 23 |
| Data Security | 0.734 | 59 |
| Do Not Track | 0.444 | 7 |
| First Party Collection/Use | 0.823 | 260 |
| International and Specific Audiences | 0.870 | 56 |
| Other | 0.763 | 301 |
| Policy Change | 0.738 | 37 |
| Third Party Sharing/Collection | 0.845 | 239 |
| User Access, Edit and Deletion | 0.535 | 48 |
| User Choice/Control | 0.657 | 140 |

Data Retention, Do Not Track and User Access/Edit/Deletion are weaker categories. The classifier was evaluated on English OPP-115, mainly US website policies; these numbers do not establish Arabic classification quality or Saudi legal compliance. Long paragraphs can lose content beyond the 256-token inference limit.

# Mapping and local policy evaluation
The written checklist has 18 requirements and 30 subchecks. It covers policy and collection notices, data categories and purposes, rights, recipients, retention and conditional disclosures. OPP-115 labels map explicitly to candidate checklist evidence; the categories do not define PDPL requirements. Do Not Track and Other have no direct mandatory requirement mapping. Availability and DPO-contact requirements are not established by a category match alone.

The application policy route supplies policy text, checklist scope, relevant legal passages and optional classifier candidates. It assesses requirements separately and can send up to 18 requests. A separate batched experiment used one request per policy. Only the latter has the ten-policy comparison reported below; its score must not be attributed to the 18-request application route.

Ten policies were collected from official sites on 1 November 2026: eXtra, stc, Salla, Nahdi, Saudi Awwal Bank, Nahdha, Jarir, Almosafer, Saudi Post SPL and Saudi Pro League. Eight are English and two Arabic. The convenience sample was selected for accessible text, not by random sampling. URLs, hashes and exact source spans identify each reference judgment.

| Reference or result | Value |
| --- | --- |
| Reference judgments | 180 (10 policies x 18 requirements) |
| Covered / partial / missing | 84 / 29 / 12 |
| Unresolved external or conditional | 55; excluded from primary text denominator |
| Primary text-status agreement | 76/125 (60.8%) |
| Macro F1 over supported text classes | 0.5757 |
| Validated / rejected requirement outputs | 148 / 32 |
| Secondary all-status agreement | 90/180 (50.0%) |

The original strict exact-string validation yielded 34.4% text-status agreement. Collapsing whitespace runs for matching accepted 53 quotations that differed only in spaces or line breaks. The verifier returns the original source span and offsets. No words, punctuation, labels or model responses were changed, no new generation requests were made, and errors remain counted.

The reference judgments are document-based and have not been confirmed as team annotations. Publication timing, role appointment and conditional applicability remain unresolved where public evidence is insufficient. No independent reviewer agreement is claimed. The remaining quotation and coverage errors limit reliability.

# Interface and release verification
The assistant and evaluation dashboard support Arabic and English interface labels. Arabic uses right-to-left layout; English uses left-to-right layout. Navigation remains on the right of the assistant. The light theme explicitly defines text, button, tab and input colors to prevent dark-theme text from disappearing on light surfaces. Primary palette contrast ratios range from 6.59 to 10.64.

The policy input is shared by classification and gap-analysis tabs. The evaluation dashboard is a separate entry point with three large navigation buttons and no model loading or API generation. Saved source passages and previously generated findings retain their original language; switching interface language is not translation of stored evidence.

| Release check | Outcome |
| --- | --- |
| Simulated Streamlit rendering paths | 12 passed: 3 assistant pages and 3 dashboard views in both languages |
| Contrast calculations | 4 key text/background pairs passed ratio 4.5 |
| Evidence/runtime unit checks | 9 passed |
| Protected local files | Manifest hashes checked after UI-only updates |
| Browser screenshot verification | Not completed: local browser automation unavailable |
| Live smoke tests | See reports/release_verification/live_smoke_summary.json |

The rendering-path tests use simulated Streamlit widgets and read real local report files. They verify page execution, navigation labels, shared input and metric loading; they are not browser screenshots. Live smoke tests, when completed, verify the service path separately and do not replace benchmark evaluation. Failed live requests are recorded rather than reported as successful abstentions.

| Live smoke case | Observed status |
| --- | --- |
| legal | answered |
| outside | not_covered |
| classifier | classification_completed |
| policy_analysis | error (0/18 requirements saved) |

## Responsible use

A visible notice states that the tool provides information, not legal advice. A covered clause can still be incorrect or not followed in practice. The interface lets users inspect source passages. API keys remain in a local .env excluded from distribution. Uploaded policy drafts and analysis checkpoints are written locally; users should avoid submitting confidential content and should manage local retention. This application is designed for local use, not an authenticated multi-user service.

Current official legal amendments are not monitored automatically. An update process must fetch the official text, compare it, review changes, update provenance and rebuild affected indexes. Local hash checks alone do not establish legal currency.

# Reproduction and evidence register
## Run the application

From the extracted Daleel folder, activate the existing pdpl-copilot environment, place the private .env beside app.py and run: python -m streamlit run app.py. Run the independent dashboard with: python -m streamlit run evaluation_dashboard.py --server.port 8502. Restart Streamlit after changing the theme configuration.

The package includes the local classifier model and retrieval index. The classifier weight exceeds the normal GitHub single-file limit; use a separate artefact distribution or Git LFS for public hosting. Do not commit .env, runtime drafts or API request logs. Public source availability does not imply permission to redistribute entire third-party policies.

| Evidence | Project location |
| --- | --- |
| Frozen RAG summary | reports/final_test20/review_summary.json |
| Retrieval development summary | reports/retrieval_comparison/reranker_comparison_summary.csv |
| Classifier model and metrics | models/opp115_distilbert_other_f2_experiment/results.json |
| Training/baseline evidence | opp115_training/scripts and data/processed |
| Written checklist | config/PDPL_Policy_Gap_Checklist_FINAL_EDITABLE.json |
| Category mapping | src/pdpl_integration.py and docs/POLICY_MAPPING.md |
| Local-policy reference | data/evaluation/local_policies/annotations_reviewed.csv |
| Batched corrected result | reports/local_policy_evaluation/evaluation.json |
| Release checks | reports/release_verification |
| Technical report source | docs/DALEEL_TECHNICAL_REPORT.md |

## Conclusion

The frozen legal-answering evaluation and English clause-classifier results support a bounded prototype. The policy-gap comparison is weaker and cannot be presented as automated compliance certification. The interface has been separated from evaluation reporting and packaged with reproducible source evidence. Remaining limits are the unresolved policy conditions, lack of independent reviewer agreement, incomplete browser visual verification and the latency and quota dependence of the API-backed routes.
