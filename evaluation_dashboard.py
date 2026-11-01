"""Standalone evaluation dashboard. Reads saved reports only; no API or model loading."""
from pathlib import Path
import sys, json, html
import streamlit as st
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from rag_dashboard_backend import evaluation_cards
from ui_i18n import translate_ui
st.set_page_config(page_title='Daleel — Evaluation dashboard', page_icon='📊', layout='wide')
if 'dashboard_language' not in st.session_state:
    st.session_state['dashboard_language'] = 'ar'
a, b = st.columns(2)
if a.button('العربية', key='dashboard_ar'):
    st.session_state['dashboard_language'] = 'ar'
    st.rerun()
if b.button('English', key='dashboard_en'):
    st.session_state['dashboard_language'] = 'en'
    st.rerun()
ui_language = st.session_state['dashboard_language']
from ui_theme import apply_ui_theme
apply_ui_theme(ROOT, ui_language, sidebar_right=False)

def ui(value):
    return translate_ui(value, ui_language)
direction = 'rtl' if ui_language == 'ar' else 'ltr'
alignment = 'right' if ui_language == 'ar' else 'left'
st.title('Evaluation dashboard' if ui_language == 'en' else 'داشبورد نتائج التقييم')
st.caption('Saved results only. The three components use different tasks, datasets and metrics; no combined accuracy is reported.' if ui_language == 'en' else 'نتائج محفوظة فقط. الأجزاء الثلاثة تختلف في المهام والبيانات والمقاييس، ولا تعرض دقة موحدة للمشروع.')

def heading(title, pill=''):
    st.subheader(ui(title))
    if pill:
        st.caption(ui(pill))
if 'evaluation_view' not in st.session_state:st.session_state['evaluation_view']='rag'
views=st.columns(3)
labels={'rag':('الأسئلة القانونية','Legal questions'),'classifier':('مصنف السياسات','Policy classifier'),'policies':('تحليل السياسات','Policy analysis')}
for col,key in zip(views,labels):
    if col.button(labels[key][1 if ui_language=='en' else 0],key='view_'+key,use_container_width=True,type='primary' if st.session_state['evaluation_view']==key else 'secondary'):
        st.session_state['evaluation_view']=key
        st.rerun()
evaluation_view=st.session_state['evaluation_view']

if evaluation_view == 'rag':
    path = ROOT / 'reports/final_test20/review_summary.json'
    if not path.exists():
        st.warning(ui('ملخص النتائج غير متاح.'))
    else:
        summary = json.loads(path.read_text(encoding='utf-8'))
        cards = evaluation_cards(summary)
        st.caption(ui('عينة من ٢٠ سؤالًا: ١٠ عربية و١٠ إنجليزية — نتائج المحاولة الأولى'))
        a, b, c = st.columns(3)
        a.metric(ui('اجتياز كامل لمعايير المراجعة'), f"{cards['complete_rate']:.0%}", help=ui('الإجابات المستوفية للمعايير الأربعة والامتناعات المناسبة، من جميع الأسئلة.'))
        a.caption(ui(f"{cards['complete']} من {cards['total']} حالة"))
        b.metric(ui('نجاح التنفيذ'), f"{cards['execution_rate']:.0%}", help=ui('يشمل الإجابات والامتناعات التي اجتازت التحقق؛ ليس نسبة صحة دلالية عامة.'))
        b.caption(ui(f"{cards['execution_success']} من {cards['total']} محاولة"))
        c.metric(ui('امتناع مناسب'), f"{cards['abstentions']} / {cards['unanswerable']}")
        c.caption(ui('للأسئلة خارج نطاق المصادر'))
        total = cards['total']
        st.markdown(ui(f"""<div class="outcome-panel"><div class="outcome-bar" role="img" aria-label="توزيع نتائج الاختبار"><div class="outcome-piece answers" style="width:{cards['accepted_answers'] / total * 100}%"></div><div class="outcome-piece abstentions" style="width:{cards['abstentions'] / total * 100}%"></div><div class="outcome-piece errors" style="width:{cards['errors'] / total * 100}%"></div></div><div class="outcome-legend"><div class="legend-item"><span class="legend-dot"></span>{cards['accepted_answers']} إجابة مقبولة</div><div class="legend-item"><span class="legend-dot abstentions"></span>{cards['abstentions']} امتناعات مناسبة</div><div class="legend-item"><span class="legend-dot errors"></span>{cards['errors']} أخطاء</div></div></div>"""), unsafe_allow_html=True)
        heading('جودة الإجابات المقبولة', 'من درجتين')
        means = summary['accepted_answerable_mean_scores_out_of_2']
        labels = {'correctness': 'الصحة', 'relevance': 'الارتباط بالسؤال', 'citation_support': 'دعم المراجع', 'necessary_completeness': 'الاكتمال الضروري'}
        tiles = ''
        for key, label in labels.items():
            value = means[key]
            tiles += f'<div class="score-tile"><div class="score-label">{label}</div><div class="score-number">{value:.2f} <span>/ 2</span></div><div class="meter"><div class="meter-fill" style="width:{min(100, max(0, value / 2 * 100))}%"></div></div></div>'
        st.markdown('<div class="score-grid">' + tiles + '</div>', unsafe_allow_html=True)
        st.caption(ui('هذه المتوسطات تخص الإجابات الـ14 المقبولة فقط، ولا تشمل الأخطاء أو الامتناعات.'))
        with st.expander(ui('كيف تقرأ هذه النتائج؟')):
            st.write(ui('العينة صغيرة وتضم أربعة أسئلة غير مغطاة فقط. نجاح التنفيذ أو تطابق الاقتباس لا يثبت صحة كل تفسير.'))
            st.write(ui('المراجعة الدلالية مقابل النصوص المحلية؛ تقييم عضوين مستقلين واتفاقهما لم يسجلا بعد.'))
        report = ROOT / 'reports/final_test20/final_test_results.md'
        if report.exists():
            st.download_button(ui('تنزيل تقرير التقييم'), report.read_bytes(), 'final_test_results.md', 'text/markdown')
elif evaluation_view == 'classifier':
    st.subheader('OPP-115 classifier' if ui_language == 'en' else 'تقييم مصنف OPP-115')
    classifier_path = ROOT / 'reports/classifier_evaluation/results.json'
    if not classifier_path.exists():
        st.warning('Classifier metrics are unavailable.' if ui_language == 'en' else 'مقاييس المصنف غير متاحة.')
    else:
        metrics = json.loads(classifier_path.read_text(encoding='utf-8-sig'))
        st.caption('DistilBERT • English OPP-115 • 671 test segments • policy-level split.' if ui_language == 'en' else 'DistilBERT — بيانات OPP-115 الإنجليزية — 671 مقطع اختبار — التقسيم حسب السياسة.')
        c1, c2, c3 = st.columns(3)
        c1.metric('Micro F1', f"{metrics['micro_f1']:.1%}")
        c2.metric('Macro F1', f"{metrics['macro_f1']:.1%}")
        c3.metric('Exact label-set match' if ui_language == 'en' else 'تطابق مجموعة الفئات', f"{metrics['exact_match_accuracy']:.1%}")
        st.caption('Multi-label classification metrics; they do not measure PDPL compliance or Arabic classification quality.' if ui_language == 'en' else 'مقاييس تصنيف متعدد الفئات؛ لا تقيس الامتثال للنظام أو جودة التصنيف بالعربية.')
        comparison = []
        for label, rel in [('TF-IDF baseline', 'opp115_training/data/processed/baseline_results.json'), ('TF-IDF tuned', 'opp115_training/data/processed/tuned_results.json'), ('TF-IDF balanced', 'opp115_training/data/processed/final_balanced_results.json'), ('DistilBERT', 'reports/classifier_evaluation/results.json')]:
            candidate = ROOT / rel
            if candidate.exists():
                values = json.loads(candidate.read_text(encoding='utf-8-sig'))
                if 'micro_f1' in values and 'macro_f1' in values:
                    comparison.append({'Model': label, 'Micro F1': round(values['micro_f1'], 4), 'Macro F1': round(values['macro_f1'], 4)})
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        per_class = [{'Category': key, 'Precision': round(v['precision'], 4), 'Recall': round(v['recall'], 4), 'F1': round(v['f1-score'], 4), 'Support': int(v['support'])} for key, v in metrics['classification_report'].items() if key in metrics['thresholds']]
        with st.expander('Per-class results' if ui_language == 'en' else 'النتائج لكل فئة'):
            st.dataframe(per_class, hide_index=True, use_container_width=True)
        st.download_button('Download classifier metrics' if ui_language == 'en' else 'تنزيل مقاييس المصنف', classifier_path.read_bytes(), 'opp115_metrics.json', 'application/json')
else:
    st.subheader('Policy analysis evaluation' if ui_language == 'en' else 'تقييم تحليل السياسات')
    policy_metrics = ROOT / 'reports/local_policy_evaluation/evaluation.json'
    if not policy_metrics.exists():
        st.warning('Policy-analysis metrics are unavailable.' if ui_language == 'en' else 'مقاييس تحليل السياسات غير متاحة.')
    else:
        metrics = json.loads(policy_metrics.read_text(encoding='utf-8-sig'))
        st.caption('Separate batched experiment on 10 policies. Saved responses revalidated using whitespace-only quote matching.' if ui_language == 'en' else 'تجربة مجمعة مستقلة على 10 سياسات، مع إعادة التحقق من الردود المحفوظة بتوحيد المسافات والأسطر فقط.')
        c1, c2, c3 = st.columns(3)
        c1.metric('Text-status agreement' if ui_language == 'en' else 'مطابقة الأحكام النصية', f"{metrics['text_status_accuracy_primary']:.1%}", help='76 / 125 text-assessable reference judgments; errors remain in the denominator.' if ui_language == 'en' else '76 من 125 حكماً نصياً؛ الأخطاء باقية في المقام.')
        c2.metric('Macro F1', f"{metrics['text_macro_f1_supported_classes']:.3f}")
        c3.metric('Validation errors' if ui_language == 'en' else 'أخطاء التحقق', f"{metrics['invalid_or_missing_outputs']} / {metrics['reference_judgments']}")
        st.info('55 external or conditional reference judgments are unresolved and excluded from the primary text score. This is not a legal-compliance score or a RAG question-answering score.' if ui_language == 'en' else '55 حكماً مرجعياً خارجياً أو مشروطاً غير محسوم، مستبعدة من المقياس النصي الأساسي. هذه ليست نسبة امتثال قانوني أو نتيجة RAG للأسئلة.')
        st.caption('Reference labels and model responses were unchanged. No new API calls for revalidation. Independent team confirmation and reviewer agreement have not been established.' if ui_language == 'en' else 'لم تتغير الأحكام المرجعية أو ردود النموذج، ولم تستخدم طلبات API جديدة لإعادة التحقق. اعتماد الفريق المستقل واتفاق المراجعين غير مثبتين.')
        st.dataframe([{'Status': key, **{k: round(v, 4) if isinstance(v, float) else v for k, v in values.items()}} for key, values in metrics['text_per_class'].items()], hide_index=True, use_container_width=True)
        st.download_button('Download policy-analysis metrics' if ui_language == 'en' else 'تنزيل مقاييس تحليل السياسات', policy_metrics.read_bytes(), 'policy_analysis_metrics.json', 'application/json')
