"""Evaluator demo using the existing evaluated legal QA adapter."""
from pathlib import Path
import os
import sys
import threading
import time

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
st.set_page_config(page_title="Daleel | دليل", page_icon="⚖️", layout="wide")


@st.cache_resource
def demo_resources():
    # Limits are shared by sessions on this process and reset after a restart.
    return {"lock": threading.Lock(), "count": 0, "last": 0.0}


@st.cache_resource(show_spinner=False)
def get_service():
    import torch
    torch.set_num_threads(2)
    from rag_dashboard_backend import DashboardRag
    return DashboardRag()


language = st.selectbox("Language / اللغة", ["العربية", "English"])
ar = language == "العربية"
def tr(a, e):
    return a if ar else e

st.markdown("<style>.stTextArea textarea {direction:" + ("rtl" if ar else "ltr") +
            ";text-align:" + ("right" if ar else "left") + ";}</style>", unsafe_allow_html=True)
st.title(tr("دليل — تجربة الأسئلة القانونية", "Daleel — Legal question demo"))
st.info(tr("هذه أداة معلومات ولا تعد استشارة قانونية.", "Information tool, not legal advice."))
st.caption(tr("لا تدخل بيانات شخصية أو سرية. تُرسل الأسئلة والنصوص المسترجعة إلى Gemini؛ يحفظ الخادم سجل الطلب محلياً. قد يستغرق تحميل النماذج والبحث عدة دقائق.",
              "Do not enter personal or confidential data. Questions and retrieved text are sent to Gemini; the server saves a local request audit. Model loading and retrieval may take several minutes."))

try:
    key = st.secrets.get("GEMINI_API_KEY", "")
    if key:
        os.environ["GEMINI_API_KEY"] = str(key)
    access_code = str(st.secrets.get("DEMO_ACCESS_CODE", ""))
    max_requests = int(st.secrets.get("DEMO_MAX_REQUESTS", 20))
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    access_code = os.environ.get("DEMO_ACCESS_CODE", "")
    max_requests = int(os.environ.get("DEMO_MAX_REQUESTS", "20"))

if access_code:
    import hmac
    supplied = st.text_input(tr("رمز تجربة المقيمين", "Evaluator access code"), type="password")
    if not hmac.compare_digest(supplied, access_code):
        st.caption(tr("أدخل رمز التجربة للمتابعة.", "Enter the demo code to continue."))
        st.stop()

with st.form("question_form"):
    question = st.text_area(tr("سؤالك بالعربية أو الإنجليزية", "Your question in Arabic or English"), max_chars=1500,
                            placeholder=tr("ما متطلبات سياسة الخصوصية؟", "What must a privacy policy include?"))
    submit = st.form_submit_button(tr("اسأل دليل", "Ask Daleel"), type="primary")

if submit:
    if not question.strip():
        st.warning(tr("اكتب السؤال أولاً.", "Enter a question first."))
    elif not os.environ.get("GEMINI_API_KEY"):
        st.error(tr("مفتاح الخدمة غير مضبوط لدى مسؤول الديمو.", "The demo administrator must configure the API key."))
    else:
        state = demo_resources()
        if not state["lock"].acquire(blocking=False):
            st.warning(tr("هناك طلب قيد التنفيذ؛ حاول بعد انتهائه.", "Another request is running. Try again when it finishes."))
        else:
            try:
                if state["count"] >= max_requests:
                    st.warning(tr("انتهى حد هذه الجلسة من تشغيل الخادم.", "This server run has reached its request limit."))
                elif time.monotonic() - state["last"] < 30:
                    st.warning(tr("انتظر 30 ثانية بين المحاولات.", "Allow 30 seconds between attempts."))
                else:
                    state["count"] += 1
                    state["last"] = time.monotonic()
                    with st.spinner(tr("جار تحميل المساعد والبحث وإعداد الإجابة…", "Loading the assistant, retrieving and generating…")):
                        service = get_service()
                        st.session_state["demo_result"] = service.answer(question)
            except Exception:
                st.error(tr("تعذر تشغيل المساعد. راجع مسؤول الديمو وسجل الاستضافة.", "The assistant could not start. Ask the demo administrator to check hosting logs."))
            finally:
                state["lock"].release()

result = st.session_state.get("demo_result")
if result:
    st.subheader(tr("الإجابة", "Answer"))
    st.write(result.get("answer_text") or result.get("message"))
    st.caption(tr("الزمن: ", "Elapsed: ") + str(result["elapsed_seconds"]) + " s")
    for sid, source in result.get("sources", {}).items():
        with st.expander(sid):
            st.text(source["text"])

st.caption(tr("حد الطلبات مؤقت لكل عملية خادم ويُعاد عند إعادة التشغيل؛ ليس حد فوترة مضموناً.",
              "The request cap is per server process and resets on restart; it is not a billing guarantee."))
