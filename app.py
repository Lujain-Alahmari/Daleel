"""Local Arabic-first dashboard. Generator and evidence checks are unchanged."""
from pathlib import Path
import hashlib, html, json, sys
import streamlit as st
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from rag_dashboard_backend import DashboardRag, evaluation_cards, verify_core
st.set_page_config(page_title='دليل | Daleel', page_icon='⚖️', layout='wide', initial_sidebar_state='expanded')
SHIELD = '<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M16 3 27 7v8c0 7-6 12-11 15C11 27 5 22 5 15V7L16 3Z"/><path d="m10 16 4 4 8-9"/></svg>'
BOOK = '<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M16 8C12 5 7 5 3 6v20c4-1 9-1 13 2 4-3 9-3 13-2V6c-4-1-9-1-13 2Zm0 0v20M7 11l5 1m-5 4 5 1m8-5 5-1m-5 6 5-1"/></svg>'
from ui_i18n import translate_ui
if 'ui_language' not in st.session_state:
    st.session_state['ui_language'] = 'ar'
language_ar, language_en = st.sidebar.columns(2)
if language_ar.button('العربية', key='language_ar', use_container_width=True, type='primary' if st.session_state['ui_language'] == 'ar' else 'secondary'):
    st.session_state['ui_language'] = 'ar'
    st.session_state['corpus_language'] = 'ar'
    st.rerun()
if language_en.button('English', key='language_en', use_container_width=True, type='primary' if st.session_state['ui_language'] == 'en' else 'secondary'):
    st.session_state['ui_language'] = 'en'
    st.session_state['corpus_language'] = 'en'
    st.rerun()
ui_language = st.session_state['ui_language']
from ui_theme import apply_ui_theme
apply_ui_theme(ROOT, ui_language, sidebar_right=True)

def ui(value):
    return translate_ui(value, ui_language)
direction = 'rtl' if ui_language == 'ar' else 'ltr'
alignment = 'right' if ui_language == 'ar' else 'left'
st.sidebar.markdown(ui('<div class="sidebar-brand"><div class="brand-line"><div class="brand-emblem">' + SHIELD + '</div><div><div class="brand-name" dir="ltr">دليل<br>DALEEL</div><div class="brand-caption">مساعد حماية البيانات الشخصية</div></div></div></div><div class="sidebar-label">مساحة العمل</div>'), unsafe_allow_html=True)
if 'active_page' not in st.session_state:
    st.session_state['active_page'] = 'السؤال والمراجع'
if st.session_state.get('active_page') == 'نتائج التقييم':
    st.session_state['active_page'] = 'السؤال والمراجع'
if st.session_state.get('active_page') == 'تصنيف سياسة الخصوصية':
    st.session_state['active_page'] = 'تحليل فجوات PDPL'
for nav_page in ['السؤال والمراجع', 'تحليل فجوات PDPL', 'المواد القانونية']:
    if st.sidebar.button(ui('تحليل السياسات' if nav_page == 'تحليل فجوات PDPL' else nav_page), key='nav_' + nav_page, use_container_width=True, type='primary' if st.session_state['active_page'] == nav_page else 'secondary'):
        st.session_state['active_page'] = nav_page
        st.rerun()
page = st.session_state['active_page']
st.sidebar.markdown(ui('<div class="sidebar-note"><strong>المصدر أمامك.</strong><br>تظهر الإجابة مع مرجعها، ويمكنك فتح النص القانوني وقراءته بنفسك.<br><span class="sidebar-pill">العربية والإنجليزية</span></div>'), unsafe_allow_html=True)
PAGES = {'السؤال والمراجع': ('اسأل • افهم • ارجع للمصدر', 'المعلومة القانونية، ومعها دليلها.', 'اكتب سؤالك عن حماية البيانات الشخصية، واطّلع على الإجابة والمواد التي تستند إليها.'), 'تصنيف سياسة الخصوصية': ('تحليل نص السياسة', 'تصنيف فقرات سياسة الخصوصية', 'تصنيف الفقرات باستخدام نموذج OPP-115.'), 'تحليل فجوات PDPL': ('تحليل الامتثال', 'فجوات سياسة الخصوصية', 'راجع المتطلبات والأدلة والمراجع القانونية، مع الفصل بين التصنيف والامتثال.'), 'المواد القانونية': ('النصوص بين يديك', 'استكشف المرجع، مادةً بمادة.', 'تصفّح النصوص العربية والإنجليزية للنظام ولائحتيه، واختر الوثيقة والمادة التي تريد قراءتها.')}
eyebrow, title, description = (ui(value) for value in PAGES[page])
if page == 'تحليل فجوات PDPL':
    title = ui('تحليل السياسات')
ART = '<svg viewBox="0 0 180 180" fill="none"><circle cx="90" cy="90" r="73" stroke="#d3ba7c" stroke-opacity=".3"/><circle cx="90" cy="90" r="58" stroke="#d3ba7c" stroke-opacity=".2"/><path d="M90 37l34 13v35c0 22-19 42-34 52-16-10-34-30-34-52V50l34-13Z" stroke="#d7c18b" stroke-width="1.6" fill="#ffffff06"/><path d="m73 85 13 13 24-28" stroke="#d7c18b" stroke-width="2.3"/><path d="M21 47h14m-7-7v14m114 74h14m-7-7v14" stroke="#d7c18b" stroke-opacity=".55"/></svg>'
st.markdown(ui('<div class="hero"><div class="hero-art" aria-hidden="true">' + ART + '</div><div class="hero-main"><div class="eyebrow">' + eyebrow + '</div><h1>' + title + '</h1><p>' + description + '</p><div class="hero-tags"><span class="hero-tag">نظام حماية البيانات الشخصية</span><span class="hero-tag">اللائحة التنفيذية</span><span class="hero-tag">لائحة النقل إلى الخارج</span></div></div></div>'), unsafe_allow_html=True)
st.markdown(ui('<div class="notice" role="note"><span class="notice-icon">ⓘ</span><span>هذه أداة معلومات ولا تعد استشارة قانونية. وجود بند في سياسة لا يثبت الامتثال الفعلي.</span></div>'), unsafe_allow_html=True)

def display_text(text, language='ar', source=False):
    css = 'source-box' if source else 'answer-box'
    direction = 'rtl' if language == 'ar' else 'ltr'
    st.markdown(f'<div class="{css}" dir="{direction}">{html.escape(text)}</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def backend(model_fingerprint):
    return DashboardRag()

@st.cache_data(show_spinner=False)
def corpus(expected_hash):
    raw = (ROOT / 'data/processed/legal_articles.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('Source snapshot changed.')
    return json.loads(raw.decode('utf-8-sig'))

def heading(title, pill=''):
    badge = '<span class="pill">' + html.escape(ui(pill)) + '</span>' if pill else ''
    st.markdown('<div class="section-heading"><h3>' + html.escape(ui(title)) + '</h3>' + badge + '</div>', unsafe_allow_html=True)

def show_sources(sources, language, cited):
    if not sources:
        return
    heading('راجع النص القانوني', str(len(sources)) + ' مادة مسترجعة')
    cited_set = set(cited)
    choices = ['المراجع المستخدمة', 'جميع المواد المسترجعة'] if cited else ['جميع المواد المسترجعة']
    mode = st.radio(ui('النصوص'), choices, format_func=ui, horizontal=True, key='source_filter', label_visibility='collapsed')
    for sid, s in sources.items():
        if mode == 'المراجع المستخدمة' and sid not in cited_set:
            continue
        name = s.get('document_name') or s['document_id']
        label = ui(name) + ' — ' + ('المادة ' if s['language'] == 'ar' else 'Article ') + str(s['article_number'])
        with st.expander(label):
            display_text(s['text'], s['language'], source=True)
            url = s.get('official_source_url', '')
            if isinstance(url, str) and url.startswith('https://'):
                st.link_button(ui('فتح المصدر الرسمي المسجل ↗'), url)
if page == 'السؤال والمراجع':
    st.markdown(ui('<div class="stat-strip"><div class="mini-stat"><strong>٣</strong> وثائق قانونية</div><div class="mini-stat"><strong>١٨٠</strong> نص مادة باللغتين</div><div class="mini-stat"><strong>AR / EN</strong> سؤال بلغتك</div></div>'), unsafe_allow_html=True)
    with st.form('ask_form'):
        question = st.text_area(ui('ما الذي تريد معرفته؟'), height=145, max_chars=3000, placeholder=ui('صف الحالة واكتب سؤالك بالعربية أو الإنجليزية…'))
        st.caption(ui('كل سؤال مستقل. حدّد الحالة وما تريد معرفته لتحصل على جواب أقرب إلى طلبك.'))
        submitted = st.form_submit_button(ui('اسأل المساعد'), type='primary', use_container_width=True)
    if submitted:
        st.session_state.pop('rag_result', None)
        if not question.strip():
            st.warning(ui('اكتب السؤال أولًا.'))
        else:
            phase = st.empty()
            try:
                phase.info(ui('جار تجهيز المساعد…'))
                model = verify_core(ROOT)
                signature = hashlib.sha256(json.dumps(model['files'], sort_keys=True).encode()).hexdigest()
                service = backend(signature)

                def stage(name):
                    phase.info(ui('جار البحث في النصوص القانونية…' if name == 'retrieval' else 'جار إعداد الإجابة من المصادر…'))
                with st.spinner(ui('نبحث عن النصوص المرتبطة بسؤالك…')):
                    st.session_state['rag_result'] = service.answer(question, on_stage=stage)
            except Exception:
                st.error(ui('تعذر بدء الخدمة. تحقق من إعدادات المشروع في الطرفية.'))
            finally:
                phase.empty()
    result = st.session_state.get('rag_result')
    if result:
        heading('الإجابة', 'من النصوص المسترجعة')
        st.markdown('<div class="question-context" dir="auto">' + html.escape(result['question']) + '</div>', unsafe_allow_html=True)
        if result['status'] == 'error':
            st.error(result['message'])
        else:
            text = result['answer_text']
            text = text.split('\n', 1)[1].lstrip() if '\n' in text else text
            display_text(text, result['language'])
            st.download_button(ui('حفظ الإجابة'), result['answer_text'], 'pdpl_answer.txt', 'text/plain')
        st.caption(ui('زمن البحث وإعداد الإجابة: ' + str(result['elapsed_seconds']) + ' ثانية'))
        show_sources(result['sources'], result['language'], result['cited_source_ids'])
    else:
        st.markdown(ui('<div class="empty-panel"><div class="empty-icon">' + BOOK + '</div><div><h4>ابدأ بسؤال، ثم اقرأ الدليل.</h4><p>الإجابة مختصرة، والمراجع قابلة للفتح. إذا لم تكفِ النصوص المتاحة، يوضح المساعد ذلك بدل اختلاق معلومة.</p></div></div>'), unsafe_allow_html=True)
elif page == 'تحليل فجوات PDPL':

    def show_gap_evidence(evidence, legal=False):
        """Display saved evidence without raw JSON."""
        if isinstance(evidence, str):
            st.text(evidence)
            return
        if isinstance(evidence, dict):
            evidence = [evidence]
        if not isinstance(evidence, list):
            st.write(evidence)
            return
        for index, entry in enumerate(evidence, 1):
            with st.container(border=True):
                if isinstance(entry, str):
                    st.text(entry)
                    continue
                if not isinstance(entry, dict):
                    st.write(entry)
                    continue
                if legal:
                    reference = entry.get('reference', 'مرجع قانوني')
                    st.markdown(ui('**المرجع:** ' + str(reference)))
                    st.caption(ui('المادة: ' + str(entry.get('article_id', '—'))))
                    if entry.get('text'):
                        st.text(entry['text'])
                    if entry.get('subparagraph_verified') is False:
                        st.warning(ui('الإحالة الفرعية غير مثبتة بشكل مستقل.'))
                else:
                    number = entry.get('paragraph_number', index)
                    st.markdown(ui(f'**الدليل من الفقرة {number}**'))
                    st.text(str(entry.get('text') or entry.get('quote') or entry.get('evidence_quote') or ui('لا يوجد اقتباس نصي.')))
    import os
    from gap_analyzer import MODEL as GAP_MODEL
    st.subheader(ui('تحليل السياسات'))
    st.caption(ui('الفحص يراجع الإفصاحات الواردة في السياسة، ولا يثبت الامتثال التشغيلي أو يمثل استشارة قانونية.'))
    draft_file = ROOT / 'reports' / 'gap_draft.txt'
    if 'gap_policy_text' not in st.session_state:
        st.session_state['gap_policy_text'] = ''

    def save_gap_draft():
        draft_file.parent.mkdir(parents=True, exist_ok=True)
        draft_file.write_text(st.session_state.get('gap_policy_text', ''), encoding='utf-8')
    gap_text = st.text_area(ui('نص سياسة الخصوصية'), height=240, key='gap_policy_text', on_change=save_gap_draft, placeholder=ui('الصق نص السياسة هنا مع فصل الفقرات بأسطر فارغة.'))
    if gap_text.strip():
        policy_hash = hashlib.sha256(gap_text.encode('utf-8')).hexdigest()
        checklist_path = ROOT / 'config' / 'PDPL_Policy_Gap_Checklist_FINAL_EDITABLE.json'
        checklist_hash = hashlib.sha256(checklist_path.read_bytes()).hexdigest()
        run_dir = ROOT / 'reports' / 'gap_runs' / (policy_hash[:16] + '_grounded_v1')
        checkpoint = run_dir / 'checkpoint.json'
        report = run_dir / 'gap_report.json'
        previous = ROOT / 'reports' / 'gap_smoke_test_grounded_v1' / 'checkpoint.json'
        if previous.exists():
            try:
                saved_previous = json.loads(previous.read_text(encoding='utf-8'))
                if saved_previous.get('policy_sha256') == policy_hash and saved_previous.get('checklist_sha256') == checklist_hash and (saved_previous.get('model') == GAP_MODEL):
                    checkpoint = previous
                    report = previous.parent / 'gap_report.json'
            except (ValueError, OSError):
                pass
        saved = {}
        if checkpoint.exists():
            try:
                data = json.loads(checkpoint.read_text(encoding='utf-8'))
                if data.get('policy_sha256') == policy_hash and data.get('checklist_sha256') == checklist_hash and (data.get('model') == GAP_MODEL) and isinstance(data.get('completed'), dict):
                    from gap_analyzer import validate_saved, LEGAL
                    current_items = json.loads(checklist_path.read_text(encoding='utf-8-sig'))['items']
                    item_lookup = {x['id']: x for x in current_items}
                    if data.get('pipeline_version') == 'daleel-grounded-gap-v1' and data.get('legal_sha256') == hashlib.sha256(LEGAL.read_bytes()).hexdigest() and (data.get('code_sha256') == hashlib.sha256((ROOT / 'src/gap_analyzer.py').read_bytes()).hexdigest()) and all((k in item_lookup and validate_saved(item_lookup[k], v, gap_text) for k, v in data['completed'].items())):
                        saved = data['completed']
                    else:
                        st.warning(ui('السجل السابق لا يطابق مسار التحقق الحالي؛ ابدأ تشغيلًا جديدًا.'))
                else:
                    st.warning(ui('ملف التقدم لا يطابق النص أو القائمة أو النموذج الحالي.'))
            except (ValueError, OSError) as exc:
                st.error(ui(f'تعذر قراءة ملف التقدم: {exc}'))
        classification_tab, gaps_tab = st.tabs(['Paragraph classification', 'Gap analysis'] if ui_language == 'en' else ['تصنيف الفقرات', 'تحليل الفجوات'])
        with classification_tab:
            if st.button(ui('تصنيف الفقرات باستخدام OPP-115'), type='primary', use_container_width=True, key='gap_opp_button'):
                try:
                    from privacy_policy_pipeline import analyze_privacy_policy
                    with st.spinner(ui('جار تصنيف الفقرات...')):
                        result = analyze_privacy_policy(gap_text)
                    st.session_state['gap_opp_result'] = {'hash': policy_hash, 'data': result}
                except Exception as exc:
                    st.error(ui('تعذر تصنيف الفقرات.'))
                    st.exception(exc)
        with gaps_tab:
            st.caption(ui('تشغيل التحليل الكامل قد يرسل حتى 18 طلب Gemini ويستهلك رصيد API؛ النتائج المحفوظة تُستكمل.'))
            run_gap = st.button(ui('تشغيل / استكمال تحليل Gemini'), use_container_width=True, key='gap_run_button', disabled=len(saved) >= 18)
            if run_gap:
                try:
                    from gap_analyzer import analyze_policy_text
                    report.parent.mkdir(parents=True, exist_ok=True)
                    with st.spinner(ui('جار تحليل المتطلبات المتبقية باستخدام Gemini...')):
                        analyze_policy_text(gap_text, report_path=report, checkpoint_path=checkpoint)
                    st.success(ui('اكتمل تحليل الفجوات.'))
                    st.rerun()
                except Exception as exc:
                    st.error(ui('توقف التحليل. النتائج المكتملة محفوظة ويمكن استكمالها.'))
                    st.exception(exc)
        with classification_tab:
            opp_state = st.session_state.get('gap_opp_result')
            if opp_state and opp_state.get('hash') == policy_hash:
                opp = opp_state['data']
                st.subheader(ui('نتائج OPP-115'))
                st.metric(ui('الفقرات المصنفة'), opp.get('paragraph_count', 0))
                for paragraph in opp.get('classifications', []):
                    with st.expander(ui(f"الفقرة {paragraph.get('paragraph_number', '-')}")):
                        st.write(paragraph.get('text', ''))
                        st.write(ui('التصنيفات:'), paragraph.get('predicted_categories', []))
                st.caption(ui('تصنيفات OPP-115 تساعد في فهم محتوى السياسة ولا تثبت تغطية المتطلبات القانونية.'))
                st.subheader(ui('متطلبات PDPL المرشحة للمراجعة'))
                from pdpl_integration import suggest_requirements
                checklist_data = json.loads(checklist_path.read_text(encoding='utf-8-sig'))
                suggestions = suggest_requirements(opp, checklist_data)
                if suggestions:
                    for suggestion in suggestions:
                        requirement_id = suggestion['requirement_id']
                        with st.expander(ui(f'{requirement_id} — مرشح للمراجعة')):
                            for match in suggestion['matches']:
                                st.write(ui('**الفقرة:**'), match.get('paragraph_number'))
                                st.write(ui('**تصنيف OPP-115:**'), match.get('category'))
                                st.write(ui('**النص المرشح:**'), match.get('paragraph_text', ''))
                            st.caption(ui('ترشيح آلي أولي، وليس حكمًا بتغطية المتطلب أو الامتثال القانوني.'))
                else:
                    st.info(ui('لم تُكتشف متطلبات مرشحة من هذه التصنيفات.'))
        with classification_tab:
            if not opp_state or opp_state.get('hash') != policy_hash:
                st.info('Run paragraph classification to view category results here.' if ui_language == 'en' else 'اضغط تصنيف الفقرات لعرض نتائج الفئات هنا.')
        with gaps_tab:
            st.metric(ui('المتطلبات المحفوظة'), f'{len(saved)} / 18')
            if saved:
                if ui_language == 'en':
                    st.caption('Saved findings and quotations retain the language used when the analysis was generated. Interface labels are shown in English.')
                st.subheader(ui('نتائج Gap Analysis المحفوظة'))
                status_names = {'covered': ui('مغطى'), 'partially_covered': ui('مغطى جزئيًا'), 'missing': ui('مفقود'), 'needs_review': ui('يحتاج مراجعة'), 'not_assessed': ui('لم يُقيّم')}
                for requirement_id, item in sorted(saved.items()):
                    if not isinstance(item, dict):
                        continue
                    status = item.get('status', 'unknown')
                    with st.expander(f'{requirement_id} — {status_names.get(status, status)}'):
                        st.write(ui('**الحالة:**'), status_names.get(status, status))
                        st.write(ui('**سبب التقييم:**'), ui(item.get('assessment_reason', '')))
                        if item.get('assessment_evidence'):
                            st.write(ui('**أدلة التقييم:**'))
                            show_gap_evidence(item['assessment_evidence'])
                        for check in item.get('checks', []):
                            if isinstance(check, dict):
                                st.write(ui('**الفحص:**'), check.get('check_id', check.get('id', '')))
                                st.write(ui('الحالة:'), status_names.get(check.get('status'), check.get('status', '')))
                                st.write(ui('السبب:'), check.get('reason', ''))
                        st.markdown(ui('**المراجع القانونية:**'))
                        references = item.get('legal_references', [])
                        if references:
                            for reference in references:
                                st.markdown('- `' + str(reference) + '`')
                        else:
                            st.caption(ui('لا توجد مراجع مرتبطة.'))
                        if item.get('legal_evidence'):
                            with st.expander(ui('النصوص القانونية المرتبطة')):
                                show_gap_evidence(item['legal_evidence'], legal=True)
                st.download_button(ui('تنزيل النتائج المحفوظة JSON'), data=json.dumps(list(saved.values()), ensure_ascii=False, indent=2), file_name='pdpl_gap_saved_results.json', mime='application/json', key='gap_download')
    else:
        st.info(ui('أدخل نص سياسة الخصوصية لعرض النتائج المحفوظة أو البدء بتحليل جديد.'))
else:
    try:
        manifest = verify_core(ROOT)
        records = corpus(manifest['files']['data/processed/legal_articles.json'])
        f1, f2, f3 = st.columns([1, 2, 1])
        with f1:
            language = st.selectbox(ui('لغة النص'), ['ar', 'en'], format_func=lambda v: 'العربية' if v == 'ar' else 'English', key='corpus_language')
        titles = {a['document_id']: a.get('document_name') or {'pdpl': 'نظام حماية البيانات الشخصية', 'implementing_regulation': 'اللائحة التنفيذية لنظام حماية البيانات الشخصية', 'transfer_regulation': 'لائحة نقل البيانات الشخصية إلى خارج المملكة'}.get(a['document_id'], a['document_id']) for a in records if a['language'] == language}
        with f2:
            document = st.selectbox(ui('الوثيقة'), list(titles), format_func=lambda v: ui(titles[v]), key='corpus_document_' + language)
        selected = sorted([a for a in records if a['language'] == language and a['document_id'] == document], key=lambda a: int(a['article_number']))
        with f3:
            article = st.selectbox(ui('رقم المادة'), [a['article_number'] for a in selected], key='corpus_article_' + language + document)
        source = next((a for a in selected if a['article_number'] == article))
        name = titles[document]
        st.markdown(ui('<div class="document-banner"><div class="document-title">' + html.escape(ui(name)) + '</div><div class="document-subtitle">' + ('المادة ' if language == 'ar' else 'Article ') + str(article) + '</div></div>'), unsafe_allow_html=True)
        display_text(source['text'], language, source=True)
        url = source.get('official_source_url', '')
        if isinstance(url, str) and url.startswith('https://'):
            st.link_button(ui('فتح المصدر الرسمي المسجل ↗'), url)
        st.caption(ui('هذا النص المحلي المحفوظ؛ لا تؤكد الأداة تلقائيًا عدم وجود تعديل رسمي أحدث.'))
    except Exception:
        st.error(ui('تعذر تحميل نسخة المصادر المحددة.'))
