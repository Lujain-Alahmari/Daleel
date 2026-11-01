# التجارب
| التجربة | السكربت والدليل |
|---|---|
| مقارنة الاسترجاع | evaluate_retrieval_development_v2.py وreports/retrieval_comparison |
| E1 | evaluate_e1_rag.py وreports/evidence/e1_evidence.zip |
| E2 | evaluate_chunking_bm25.py وreports/evidence/e2_evidence.zip |
| E3 | evaluate_crosslingual_e5.py وevaluate_e3_english_models.py وreports/evidence/e3_evidence.zip |
| E4 | المصنف؛ يدمج من عمل الفريق |
| E5 | prepare_e5_abstention.py وevaluate_e5_abstention_small.py وreports/evidence/e5_evidence.zip |
| الاختبار النهائي | evaluate_new_test20.py وreports/final_test20 وreports/evidence/final_test20_evidence.zip |

السكربتات داخل src. كل أرشيف أدلة يحتفظ بالمسارات الأصلية. عند الحاجة لاستخراجه من جذر المشروع:
~~~powershell
Expand-Archive -LiteralPath '.\reports\evidence\e5_evidence.zip' -DestinationPath '.' -Force
~~~
السجلات التفصيلية لا تلزم لتشغيل الواجهة؛ استخراجها يعيد نقاط التحقق والسياقات.
قابلية فتح سجل لا تعني جواز استكماله بعد تغيير الكود أو الأسئلة؛ راجع البصمات.
E2 مقارنة استرجاع BM25 وتقسيم ثابت/مادة، وE3 تجربة عبر اللغات ومقارنة النموذج الإنجليزي.
E5 معايرة العتبة على 30 تطويرا ومقارنة توليد على 15 تطويرا؛ ليست نتيجته اختبارا محجوزا.
تساوت حالات E5 في هذه العينة ولم تضف العتبة للنسخة النهائية.
الواجهة تعرض اختبار النسخة النهائي الجديد فقط، ولا تستخدم تقييم إصدار آخر.
لا تعد التجارب المكتملة لمجرد تجهيز المستودع. التشغيل مع --run قد يستهلك API في سكربتات التوليد.
