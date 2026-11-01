## Local privacy-policy reference set

Ten publicly accessible privacy policies from organisations operating in Saudi Arabia were collected on 1 November 2026: eXtra, stc, Salla, Nahdi, Saudi Awwal Bank, Nahdha, Jarir, Almosafer, Saudi Post SPL and the Saudi Pro League. The set contains eight English policies and two Arabic policies. Selection was based on accessibility and sector diversity, rather than random sampling. Source URLs, retrieval dates and SHA-256 hashes identify the frozen policy texts.

Each policy was reviewed against 18 checklist requirements, producing 180 requirement-level reference judgments. The reviewed labels are 84 covered, 29 partially covered, 12 missing and 55 needs_review. These counts describe policy-text coverage; they are not model-performance scores or legal-compliance determinations. The 55 unresolved cases concern publication timing, appointment or applicability evidence, indirect-collection notification, changed-purpose notification, enhanced transparency and capacity-specific notices. Every unresolved case has a documented reason. A public-page desk review did not establish the operational facts needed to resolve these cases. An explicit DPO contact found in a separate official stc notice was retained as supplementary evidence, without altering the frozen policy input or its reference labels.

A comparison should report exact status agreement on the 125 assessable text judgments as its primary result, retain the 55 unresolved cases as a separate stratum, and include failed or missing model outputs rather than treating them as lawful abstentions. Any all-status agreement should be labelled secondary because needs_review matches can inflate it. The model has not yet been run on these ten policies, so no gap-analyser accuracy or F1 is reported here. Independent reviewer agreement has not been measured, and the reference set is not yet team-confirmed.


## Ten-policy gap-analyser evaluation

A separate batched experiment (`gap-policy-batched-v1`) assessed ten frozen policies with Gemini `gemini-3.5-flash-lite`, using one generation request per policy (10 requests in total). Each request included the full policy, 30 checklist checks grouped into 18 requirements, official legal passages and OPP-115 classifier candidate hints. Reference judgments and review notes were excluded from generation. This experiment differs from the application's 18-request policy route and must not be described as its evaluation.

The reference contains 180 requirement judgments. The primary comparison uses 125 text-assessable judgments; 55 unresolved external/conditional judgments are reported separately. Exact status accuracy was **34.4% (43/125)**, and macro F1 over the three supported text classes was **0.4543**. Across all 180 judgments, 110 outputs passed structural/quotation validation and 70 were errors. Failed outputs remained in the denominator. Passing quotation validation does not establish semantic correctness.

All-status agreement, including the unresolved stratum, was 31.1% (56/180); this is a secondary diagnostic. The result indicates that this batched gap-analyser configuration is not reliable enough to claim accurate policy assessment. The reference is document-based and has not been confirmed by independent team reviewers; no reviewer-agreement statistic is available. These are neither RAG question-answering scores nor legal-compliance scores.

Validation error categories: {"Quote is not in policy": 89, "Covered check has no quote": 1}. Exact quote mismatches may include formatting differences or altered quotations; they were not silently accepted or repaired. First attempts were retained. No generation retry was used to replace these results.


## Whitespace-only evidence validation correction

The same ten-policy model responses were revalidated with a documented correction that collapses whitespace runs solely for evidence matching. The verifier returns the exact original source span and its offsets. It does not change words, case, punctuation, spelling or reference labels, and does not join disconnected passages. This accepted 53 whitespace-only check quotations; 38 previously rejected requirement outputs became valid. No new API calls or model generations were used. The original experiment and scores remain archived.

Exact text-status accuracy increased from 34.4% (43/125) to **60.8% (76/125)**. Macro F1 over the three supported text classes increased from 0.4543 to **0.5757**. Validated requirement outputs increased from 110/180 to **148/180**, with **32 errors retained**. Secondary all-status agreement was 50.0% (90/180). These are status-comparison results for the separate batched policy experiment, not legal-compliance accuracy, semantic evidence entailment, or RAG question-answering performance.

The reference's 55 unresolved external/conditional judgments remain excluded from the primary text denominator. The reference was unchanged and is document-based, without independently confirmed team annotation or reviewer-agreement measurement. This correction addresses formatting-related false rejections; remaining model-quotation and coverage errors remain unresolved. It is a post-run validation correction on the same responses, not an independent new held-out evaluation.
