"""Shared interface palette and language direction; no model configuration."""
from pathlib import Path
import streamlit as st

def apply_ui_theme(root: Path, language: str, sidebar_right=False):
    st.markdown('<style>'+ (root/'assets/dashboard.css').read_text(encoding='utf-8')+'</style>', unsafe_allow_html=True)
    direction='rtl' if language=='ar' else 'ltr'
    align='right' if language=='ar' else 'left'
    art='left:20px;right:auto;' if language=='ar' else 'right:20px;left:auto;'
    sidebar='[data-testid="stAppViewContainer"]{flex-direction:row-reverse;} [data-testid="stSidebarCollapsedControl"]{left:auto!important;right:0.5rem!important;}' if sidebar_right else ''
    st.markdown(f'''<style>
html,body,[data-testid="stApp"],[data-testid="stMain"],[data-testid="stSidebar"],
[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"],
[data-testid="stRadio"],[data-testid="stRadio"] label,[data-testid="stSelectbox"],
[data-baseweb="select"],[data-testid="stHorizontalBlock"],[data-testid="stMetricLabel"],
[data-testid="stExpander"] summary,[data-testid="stAlert"] {{direction:{direction};text-align:{align};}}
h1,h2,h3,.score-number,[data-testid="stMetricValue"],[data-testid="stSidebar"] button {{text-align:{align};}}
[data-testid="stTextArea"] textarea {{direction:{direction};text-align:{align};unicode-bidi:plaintext;}}
.hero-art {{{art}}}
.answer-box[dir="rtl"],.source-box[dir="rtl"] {{direction:rtl;text-align:right;unicode-bidi:plaintext;}}
.answer-box[dir="ltr"],.source-box[dir="ltr"] {{direction:ltr;text-align:left;unicode-bidi:plaintext;}}
.question-context {{unicode-bidi:plaintext;}}
{sidebar}
</style>''',unsafe_allow_html=True)
