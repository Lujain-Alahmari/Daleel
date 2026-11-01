# تنظيم المشروع
~~~text
PDPL-Copilot/
  app.py                     واجهة العرض
  src/                       كود التشغيل وسكربتات التجارب والبيانات
  scripts/check_setup.py     فحص محلي دون API
  config/                    تعريف النسخة وبصمات اختبارها
  data/
    raw/                     PDF قانونية أصلية
    processed/               corpus والجداول ومعلومات المصادر
    checklists/              المتطلبات المستخدمة في التحقق
    indexes/dense_e5/         الفهرس المثبت وبياناته
    evaluation/              الأسئلة الأساسية والاختبار النهائي المنفصل
  reports/
    final_test20/            التقرير النهائي والمراجعة
    retrieval_comparison/    جداول مقارنة الاسترجاع
    evidence/                أدلة التجارب مضغوطة
    dashboard_live_v1/       سجلات محلية لا ترفع
  docs/                      التشغيل والاختبار والتنظيم
  requirements.txt
  .env.example               مثال بلا مفتاح
~~~
بعض الملفات المسماة evaluate أو v2 تستخدمها النسخة الحالية مباشرة؛ لا تحذفها بسبب الاسم.
chat_rag_gemini.py وchat_rag_gemini_v2.py وchat_rag_source_planned.py اعتماديات للمولد الحالي.
evaluate_character_retrieval.py وevaluate_query_expansion.py يتضمنان وظائف البحث الفعلي.
المولد المستخدم src/chat_rag_roles_prompt_only.py، والواجهة app.py.
النسخ غير اللازمة بقيت في أرشيف محلي خارج نسخة التسليم.
أدلة التجارب مضغوطة مع مساراتها الأصلية لتقليل الملفات الظاهرة دون حذفها.
