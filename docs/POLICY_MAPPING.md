# OPP-115 to PDPL candidate mapping

The mapping in src/pdpl_integration.py routes categories to possible checklist items. All ten OPP categories have explicit entries. Do Not Track and Other have no direct checklist mapping. N01 publication availability and N14 DPO contact applicability require evidence outside this category vocabulary. Classifier labels never establish legal coverage. All policy text and legal source passages are assessed even when a classifier returns no candidate.

The current classifier was evaluated on English OPP-115 policies. Arabic classifier quality is not established. Policy paragraphs are truncated to 256 tokens for classification; the gap assessor receives the full policy text separately. Security and international-audience labels are broad candidate hints only. The checklist contains disclosure checks and external checks; it is not an exhaustive operational compliance audit.

No ten-policy local evaluation is claimed. Complete data/evaluation/local_policies templates using actual public Saudi-market policies and team annotations before reporting gap accuracy.
