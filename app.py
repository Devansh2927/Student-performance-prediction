import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.inspection import permutation_importance

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Syne', sans-serif !important;
    background: #080c14 !important;
    color: #e2e8f0 !important;
}

#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid rgba(99,179,237,0.12) !important;
}
[data-testid="stSidebar"] label {
    color: #64748b !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #1a2236 !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #64748b !important; }
[data-testid="stSidebar"] [data-baseweb="menu"] {
    background: #1a2236 !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
}
[data-testid="stSidebar"] [data-baseweb="option"] { background: #1a2236 !important; color: #e2e8f0 !important; }
[data-testid="stSidebar"] [data-baseweb="option"]:hover { background: #243047 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: linear-gradient(90deg, #38bdf8, #6366f1) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="thumb"] {
    background: #38bdf8 !important;
    box-shadow: 0 0 10px #38bdf8 !important;
    width: 18px !important;
    height: 18px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(99,179,237,0.12) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 12px 22px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 28px !important;
    background: transparent !important;
}

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 24px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 32px rgba(99,102,241,0.6) !important;
    transform: translateY(-1px) !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }

@keyframes pulse {
    0%,100%{ opacity:1; transform:scale(1); }
    50%{ opacity:0.4; transform:scale(1.4); }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = ['Age','Gender','Ethnicity','ParentalEducation','StudyTimeWeekly',
            'Absences','Tutoring','ParentalSupport','Extracurricular','Sports','Music','Volunteering']

GRADE_LABELS  = {0:'F', 1:'D', 2:'C', 3:'B', 4:'A'}
GRADE_COLORS  = {0:'#f43f5e', 1:'#f59e0b', 2:'#818cf8', 3:'#38bdf8', 4:'#10b981'}
GRADE_MEANING = {
    0:'Failing — critical intervention needed',
    1:'Below average — needs improvement',
    2:'Average — meeting baseline',
    3:'Good — above average',
    4:'Excellent — top-tier achievement',
}
GRADE_GPA = {0:'< 1.0', 1:'1.0–1.9', 2:'2.0–2.9', 3:'3.0–3.9', 4:'≥ 4.0'}

PBGL  = 'rgba(0,0,0,0)'
GRIDC = 'rgba(99,179,237,0.07)'
FONT  = 'Syne, sans-serif'
CHART_CFG = {'displayModeBar': False}

# ── Data & Model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('Student_performance_data.csv')

@st.cache_resource
def train_models(_df):
    X = _df[FEATURES]
    y_cls = _df['GradeClass'].astype(int)
    y_reg = _df['GPA']
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42)
    clf = Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
    clf.fit(X_tr, yc_tr)
    reg = Pipeline([('sc', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=5))])
    reg.fit(X_tr, yr_tr)
    yc_pred = clf.predict(X_te)
    yr_pred = reg.predict(X_te)
    acc = accuracy_score(yc_te, yc_pred)
    mse = mean_squared_error(yr_te, yr_pred)
    r2  = r2_score(yr_te, yr_pred)
    pi  = permutation_importance(clf, X_te, yc_te, n_repeats=10, random_state=42)
    fi  = dict(zip(FEATURES, pi.importances_mean.tolist()))
    cm  = confusion_matrix(yc_te, yc_pred)
    return clf, reg, X_tr, yc_tr, yr_tr, acc, mse, r2, fi, cm

# ── HTML helpers ──────────────────────────────────────────────────────────────
def stat_block(label, value, sub, color):
    return f"""
    <div style="background:#111827;border:1px solid rgba(99,179,237,0.1);border-radius:14px;
                padding:20px 22px;position:relative;overflow:hidden;height:100%;">
        <div style="position:absolute;top:0;left:0;width:3px;height:100%;
                    background:{color};border-radius:3px 0 0 3px;"></div>
        <div style="font-size:10px;font-weight:700;color:#475569;letter-spacing:0.12em;
                    text-transform:uppercase;margin-bottom:8px;">{label}</div>
        <div style="font-size:26px;font-weight:800;color:{color};line-height:1;
                    font-family:'JetBrains Mono',monospace;">{value}</div>
        <div style="font-size:11px;color:#334155;margin-top:5px;">{sub}</div>
    </div>"""

def dark_layout(**kw):
    base = dict(
        paper_bgcolor=PBGL, plot_bgcolor=PBGL,
        font=dict(family=FONT, color='#94a3b8', size=12),
        margin=dict(l=10, r=10, t=44, b=10),
        xaxis=dict(showgrid=True, gridcolor=GRIDC, zeroline=False,
                   linecolor=GRIDC, tickfont=dict(size=11, color='#64748b')),
        yaxis=dict(showgrid=True, gridcolor=GRIDC, zeroline=False,
                   linecolor=GRIDC, tickfont=dict(size=11, color='#64748b')),
    )
    base.update(kw)
    return base

def chart_card(content_fn, *args, **kwargs):
    st.markdown('<div style="background:#111827;border:1px solid rgba(99,179,237,0.1);'
                'border-radius:16px;padding:20px;margin-bottom:16px;">', unsafe_allow_html=True)
    content_fn(*args, **kwargs)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:24px 20px 20px;border-bottom:1px solid rgba(99,179,237,0.1);">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <div style="width:34px;height:34px;background:linear-gradient(135deg,#0ea5e9,#6366f1);
                            border-radius:9px;display:flex;align-items:center;justify-content:center;
                            font-size:17px;flex-shrink:0;">⚡</div>
                <div>
                    <div style="font-size:15px;font-weight:800;color:#f1f5f9;">EduPredict AI</div>
                    <div style="font-size:10px;color:#38bdf8;letter-spacing:0.1em;font-weight:600;">KNN · 2392 Students</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div style="padding:18px 4px 6px;">
            <div style="font-size:10px;font-weight:700;color:#38bdf8;letter-spacing:0.14em;
                        text-transform:uppercase;margin-bottom:12px;padding-left:4px;">Demographics</div>
        </div>""", unsafe_allow_html=True)

        age    = st.selectbox("Age", [15,16,17,18], index=2)
        gender = st.selectbox("Gender", ["Female","Male"])
        eth    = st.selectbox("Ethnicity", ["Group A","Group B","Group C","Group D"])
        ped    = st.selectbox("Parental Education",
                            ["None","High school","Some college","Bachelor's","Higher"], index=2)

        st.markdown("""<div style="padding:12px 4px 4px;">
            <div style="font-size:10px;font-weight:700;color:#6366f1;letter-spacing:0.14em;
                        text-transform:uppercase;padding-left:4px;">Academic</div>
        </div>""", unsafe_allow_html=True)

        study    = st.slider("Study time (hrs / week)", 0.0, 20.0, 10.0, 0.5)
        absences = st.slider("Absences (days)", 0, 29, 8)
        tutoring = st.selectbox("Tutoring", ["No","Yes"])
        psupport = st.selectbox("Parental Support",
                                ["None","Low","Moderate","High","Very high"], index=2)

        st.markdown("""<div style="padding:12px 4px 4px;">
            <div style="font-size:10px;font-weight:700;color:#10b981;letter-spacing:0.14em;
                        text-transform:uppercase;padding-left:4px;">Activities</div>
        </div>""", unsafe_allow_html=True)

        extra    = st.selectbox("Extracurricular", ["No","Yes"])
        sports   = st.selectbox("Sports", ["No","Yes"])
        music    = st.selectbox("Music", ["No","Yes"])
        volunteer= st.selectbox("Volunteering", ["No","Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button("⚡  Run Prediction")
        st.markdown("<br>", unsafe_allow_html=True)

    inputs = {
        'Age':              age,
        'Gender':           1 if gender=="Male" else 0,
        'Ethnicity':        ["Group A","Group B","Group C","Group D"].index(eth),
        'ParentalEducation':["None","High school","Some college","Bachelor's","Higher"].index(ped),
        'StudyTimeWeekly':  study,
        'Absences':         absences,
        'Tutoring':         1 if tutoring=="Yes" else 0,
        'ParentalSupport':  ["None","Low","Moderate","High","Very high"].index(psupport),
        'Extracurricular':  1 if extra=="Yes" else 0,
        'Sports':           1 if sports=="Yes" else 0,
        'Music':            1 if music=="Yes" else 0,
        'Volunteering':     1 if volunteer=="Yes" else 0,
    }
    return inputs, btn

# ── Predict Tab ───────────────────────────────────────────────────────────────
def render_predict_tab(inputs, btn, clf, reg, X_tr, yc_tr, yr_tr, acc, mse, r2):
    c1, c2, c3, c4 = st.columns(4)
    data = [
        ("Dataset size","2,392","training samples","#38bdf8"),
        ("Grade accuracy",f"{acc*100:.1f}%","classification","#10b981"),
        ("GPA R² score",f"{r2:.3f}","regression fit","#6366f1"),
        ("Neighbors (k)","5","used per query","#f59e0b"),
    ]
    for col,(lbl,val,sub,clr) in zip([c1,c2,c3,c4], data):
        with col:
            st.markdown(stat_block(lbl,val,sub,clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if btn or st.session_state.get('pred'):
        if btn:
            inp_df = pd.DataFrame([inputs])[FEATURES]
            grade  = int(clf.predict(inp_df)[0])
            gpa    = float(np.clip(reg.predict(inp_df)[0], 0, 4))
            scaler = clf.named_steps['sc']
            dists  = np.sqrt(((scaler.transform(X_tr) - scaler.transform(inp_df))**2).sum(axis=1))
            top5   = np.argsort(dists)[:5]
            nb     = [(int(i), float(dists[i]), int(yc_tr.iloc[i]), float(yr_tr.iloc[i])) for i in top5]
            st.session_state['pred'] = dict(grade=grade, gpa=gpa, neighbors=nb, inputs=inputs)

        p = st.session_state['pred']
        grade, gpa = p['grade'], p['gpa']
        gc = GRADE_COLORS[grade]
        gl = GRADE_LABELS[grade]
        pct = (gpa/4.0)*100

        left, right = st.columns([3, 2])

        with left:
            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=gpa,
                number=dict(font=dict(family='JetBrains Mono',size=52,color=gc),
                            valueformat=".2f"),
                delta=dict(reference=2.0, valueformat=".2f",
                        increasing=dict(color='#10b981'),
                        decreasing=dict(color='#f43f5e')),
                gauge=dict(
                    axis=dict(range=[0,4],tickvals=[0,1,2,3,4],
                            ticktext=['0','1','2','3','4'],
                            tickfont=dict(size=10,color='#475569'),
                            tickcolor='#1e293b'),
                    bar=dict(color=gc, thickness=0.25),
                    bgcolor='rgba(0,0,0,0)',
                    borderwidth=0,
                    steps=[
                        dict(range=[0,1], color='rgba(244,63,94,0.07)'),
                        dict(range=[1,2], color='rgba(245,158,11,0.07)'),
                        dict(range=[2,3], color='rgba(129,140,248,0.07)'),
                        dict(range=[3,4], color='rgba(16,185,129,0.07)'),
                    ],
                    threshold=dict(line=dict(color=gc,width=3),thickness=0.85,value=gpa),
                ),
            ))
            fig_g.update_layout(
                height=270, margin=dict(l=30,r=30,t=20,b=10),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(family=FONT),
            )
            st.markdown(f"""
            <div style="background:#111827;border:1px solid {gc}30;border-radius:18px;
                        padding:26px;box-shadow:0 0 40px {gc}12;margin-bottom:16px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
                    <div style="font-size:10px;font-weight:700;color:#475569;
                                letter-spacing:0.14em;text-transform:uppercase;">Prediction Result</div>
                    <div style="background:{gc}18;border:1px solid {gc}40;color:{gc};
                                padding:4px 14px;border-radius:20px;font-size:12px;font-weight:700;
                                font-family:'JetBrains Mono';">
                        Grade {gl}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig_g, width='stretch', config=CHART_CFG)
            st.markdown(f"""
                <div style="text-align:center;margin:-8px 0 18px;">
                    <div style="font-size:12px;color:{gc};font-weight:600;margin-bottom:3px;">
                        GPA range: {GRADE_GPA[grade]}
                    </div>
                    <div style="font-size:12px;color:#475569;">{GRADE_MEANING[grade]}</div>
                </div>
                <div style="height:6px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;margin-bottom:5px;">
                    <div style="width:{pct:.1f}%;height:100%;
                                background:linear-gradient(90deg,{gc}80,{gc});
                                border-radius:3px;transition:width 0.8s ease;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:9px;
                            color:#334155;font-family:'JetBrains Mono';margin-bottom:4px;">
                    <span>0.0</span><span>1.0</span><span>2.0</span><span>3.0</span><span>4.0</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Factor pills
            abs_v   = p['inputs']['Absences']
            study_v = p['inputs']['StudyTimeWeekly']
            tut_v   = p['inputs']['Tutoring']
            ps_v    = p['inputs']['ParentalSupport']
            factors = []
            if abs_v > 20:
                factors.append(("⚠ High absences", f"{abs_v} days", "#f43f5e"))
            elif abs_v < 5:
                factors.append(("✓ Low absences", f"{abs_v} days", "#10b981"))
            if study_v > 14:
                factors.append(("✓ Intensive study", f"{study_v:.1f} h/wk", "#10b981"))
            elif study_v < 5:
                factors.append(("⚠ Low study time", f"{study_v:.1f} h/wk", "#f43f5e"))
            if tut_v:
                factors.append(("✓ Tutoring", "Active", "#38bdf8"))
            if ps_v >= 3:
                factors.append(("✓ Parent support", ["–","Low","Mod","High","V.High"][ps_v], "#6366f1"))

            if factors:
                cols = st.columns(min(3, len(factors)))
                for col, (lbl, val, clr) in zip(cols, factors[:3]):
                    with col:
                        st.markdown(f"""
                        <div style="background:{clr}12;border:1px solid {clr}35;border-radius:10px;
                                    padding:10px 12px;text-align:center;">
                            <div style="font-size:11px;font-weight:700;color:{clr};">{lbl}</div>
                            <div style="font-size:11px;color:#475569;margin-top:2px;">{val}</div>
                        </div>""", unsafe_allow_html=True)

        with right:
            # Neighbors
            st.markdown(f"""
            <div style="background:#111827;border:1px solid rgba(99,179,237,0.1);border-radius:18px;
                        padding:20px;margin-bottom:16px;">
                <div style="font-size:10px;font-weight:700;color:#475569;letter-spacing:0.14em;
                            text-transform:uppercase;margin-bottom:14px;">5 Nearest Neighbors</div>
            """, unsafe_allow_html=True)
            for rank, (idx, dist, nb_g, nb_gpa) in enumerate(p['neighbors'], 1):
                gl_nb = GRADE_LABELS[nb_g]
                gc_nb = GRADE_COLORS[nb_g]
                sim   = max(0, int((1 - dist/3.5)*100))
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:9px 0;
                            border-bottom:1px solid rgba(255,255,255,0.04);">
                    <div style="width:22px;height:22px;border-radius:6px;
                                background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.15);
                                display:flex;align-items:center;justify-content:center;
                                font-size:10px;font-weight:700;color:#475569;flex-shrink:0;">{rank}</div>
                    <div style="flex:1;min-width:0;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                            <span style="font-size:12px;font-weight:600;color:#cbd5e1;
                                        font-family:'JetBrains Mono';">#{idx+1}</span>
                            <span style="background:{gc_nb}20;color:{gc_nb};padding:2px 8px;
                                        border-radius:10px;font-size:10px;font-weight:700;">{gl_nb}</span>
                        </div>
                        <div style="height:3px;background:rgba(255,255,255,0.04);border-radius:2px;overflow:hidden;">
                            <div style="width:{sim}%;height:100%;
                                        background:linear-gradient(90deg,{gc_nb}50,{gc_nb});"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-top:3px;">
                            <span style="font-size:10px;color:#334155;">d={dist:.3f}</span>
                            <span style="font-size:10px;color:#334155;">GPA {nb_gpa:.2f}</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Radar
            r_vals = [
                (inputs['Age']-15)/3,
                inputs['StudyTimeWeekly']/20,
                1 - inputs['Absences']/29,
                inputs['Tutoring'],
                inputs['ParentalSupport']/4,
                (inputs['Extracurricular'] + inputs['Sports'] + inputs['Music'])/3,
            ]
            r_labs = ['Age','Study','Attendance','Tutoring','Parent','Activities']
            _h = gc.lstrip('#'); _r,_g,_b = int(_h[0:2],16),int(_h[2:4],16),int(_h[4:6],16)
            _fc = f'rgba({_r},{_g},{_b},0.08)'
            fig_r = go.Figure(go.Scatterpolar(
                r=r_vals + [r_vals[0]],
                theta=r_labs + [r_labs[0]],
                fill='toself',
                fillcolor=_fc,
                line=dict(color=gc, width=2),
                marker=dict(color=gc, size=5),
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0,1],
                                showticklabels=False, gridcolor=GRIDC, linecolor=GRIDC),
                    angularaxis=dict(tickfont=dict(size=10,color='#64748b'),
                                    linecolor=GRIDC, gridcolor=GRIDC),
                ),
                showlegend=False, height=230,
                margin=dict(l=30,r=30,t=14,b=14),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(family=FONT),
            )
            st.markdown(f"""
            <div style="background:#111827;border:1px solid rgba(99,179,237,0.1);
                        border-radius:18px;padding:18px;">
                <div style="font-size:10px;font-weight:700;color:#475569;letter-spacing:0.14em;
                            text-transform:uppercase;margin-bottom:4px;">Profile Radar</div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig_r, width='stretch', config=CHART_CFG)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#111827;border:2px dashed rgba(56,189,248,0.15);border-radius:18px;
                    padding:60px;text-align:center;margin-top:8px;">
            <div style="font-size:44px;margin-bottom:16px;opacity:0.3;">⚡</div>
            <div style="font-size:20px;font-weight:800;color:#e2e8f0;margin-bottom:8px;">
                Ready to predict
            </div>
            <div style="font-size:13px;color:#475569;max-width:280px;margin:0 auto;line-height:1.6;">
                Configure the student profile in the sidebar, then click
                <span style="color:#38bdf8;font-weight:700;">Run Prediction</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Insights Tab ──────────────────────────────────────────────────────────────
def render_insights_tab(df, fi):
    c1, c2 = st.columns(2)
    grade_cts = df['GradeClass'].value_counts().sort_index()

    with c1:
        lbls = [GRADE_LABELS[int(g)] for g in grade_cts.index]
        clrs = [GRADE_COLORS[int(g)] for g in grade_cts.index]
        fig = go.Figure(go.Bar(
            x=lbls, y=grade_cts.values,
            marker=dict(color=clrs, opacity=0.8,
                        line=dict(color=clrs, width=1)),
            text=grade_cts.values, textposition='outside',
            textfont=dict(size=11, color='#64748b'),
        ))
        fig.update_layout(
            title=dict(text='Grade Distribution', font=dict(size=14,color='#e2e8f0',family=FONT)),
            height=300, showlegend=False,
            **dark_layout()
        )
        def _grade_chart():
            st.plotly_chart(fig, width='stretch', config=CHART_CFG)
        chart_card(_grade_chart)

    with c2:
        fig2 = go.Figure(go.Histogram(
            x=df['GPA'], nbinsx=32,
            marker=dict(color='#6366f1', opacity=0.7,
                        line=dict(color='#818cf8', width=0.5)),
        ))
        fig2.update_layout(
            title=dict(text='GPA Distribution', font=dict(size=14,color='#e2e8f0',family=FONT)),
            height=300, showlegend=False,
            **dark_layout(xaxis=dict(title='GPA', showgrid=True, gridcolor=GRIDC,
                                    zeroline=False, tickfont=dict(size=11,color='#64748b')),
                          yaxis=dict(title='Count', showgrid=True, gridcolor=GRIDC,
                                    zeroline=False, tickfont=dict(size=11,color='#64748b')))
        )
        def _gpa_chart():
            st.plotly_chart(fig2, width='stretch', config=CHART_CFG)
        chart_card(_gpa_chart)

    # Feature importance
    fi_sorted = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
    fi_names  = [x[0] for x in fi_sorted]
    fi_vals   = [x[1]*100 for x in fi_sorted]
    clrs_fi   = ['#10b981' if v >= 0 else '#f43f5e' for v in fi_vals]

    fig3 = go.Figure(go.Bar(
        x=fi_vals, y=fi_names, orientation='h',
        marker=dict(color=clrs_fi, opacity=0.8,
                    line=dict(color=clrs_fi, width=0.8)),
        text=[f"{v:+.2f}%" for v in fi_vals], textposition='outside',
        textfont=dict(size=10, color='#64748b'),
    ))
    fig3.update_layout(
        title=dict(text='Feature Importance — Permutation Method', font=dict(size=14,color='#e2e8f0',family=FONT)),
        height=360, showlegend=False,
        yaxis=dict(autorange='reversed', showgrid=False, zeroline=False,
                   tickfont=dict(size=11, color='#94a3b8')),
        xaxis=dict(showgrid=True, gridcolor=GRIDC, zeroline=True,
                   zerolinecolor='rgba(99,179,237,0.18)', tickfont=dict(size=11, color='#64748b')),
        margin=dict(l=10, r=60, t=44, b=10),
        paper_bgcolor=PBGL, plot_bgcolor=PBGL,
        font=dict(family=FONT),
    )
    def _fi_chart():
        st.plotly_chart(fig3, width='stretch', config=CHART_CFG)
        st.markdown("""
        <div style="background:rgba(56,189,248,0.06);border-left:3px solid #38bdf8;
                    padding:11px 14px;border-radius:0 8px 8px 0;font-size:12px;color:#7dd3fc;margin-top:4px;">
            <b>Absences</b> is the strongest predictor (+22%). Green = positive contribution · Red = negative/noise.
        </div>""", unsafe_allow_html=True)
    chart_card(_fi_chart)

    c3, c4 = st.columns(2)
    sample = df.sample(600, random_state=42)

    with c3:
        fig4 = px.scatter(
            sample, x='Absences', y='GPA', color='GradeClass', opacity=0.6,
            color_continuous_scale=['#f43f5e','#f59e0b','#818cf8','#38bdf8','#10b981'],
            title='Absences vs GPA',
        )
        fig4.update_traces(marker=dict(size=5))
        fig4.update_layout(height=300, coloraxis_colorbar=dict(
            title='Grade', tickfont=dict(size=10, color='#64748b'), outlinewidth=0),
            **dark_layout(title=dict(text='Absences vs GPA', font=dict(size=14,color='#e2e8f0'))))
        def _sc1():
            st.plotly_chart(fig4, width='stretch', config=CHART_CFG)
        chart_card(_sc1)

    with c4:
        fig5 = px.scatter(
            sample, x='StudyTimeWeekly', y='GPA', color='GradeClass', opacity=0.6,
            color_continuous_scale=['#f43f5e','#f59e0b','#818cf8','#38bdf8','#10b981'],
        )
        fig5.update_traces(marker=dict(size=5))
        fig5.update_layout(height=300, coloraxis_colorbar=dict(
            title='Grade', tickfont=dict(size=10, color='#64748b'), outlinewidth=0),
            **dark_layout(title=dict(text='Study Time vs GPA', font=dict(size=14,color='#e2e8f0'))))
        def _sc2():
            st.plotly_chart(fig5, width='stretch', config=CHART_CFG)
        chart_card(_sc2)

# ── Model Info Tab ────────────────────────────────────────────────────────────
def render_model_tab(acc, mse, r2, cm):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(stat_block("Algorithm","KNN","K-Nearest Neighbors","#38bdf8"), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_block("Train / test","80 / 20","1913 / 479 samples","#6366f1"), unsafe_allow_html=True)
    with c3:
        st.markdown(stat_block("Features","12","input dimensions","#10b981"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([3, 2])

    with cl:
        labels = ['F','D','C','B','A']
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=[f'Pred {l}' for l in labels],
            y=[f'Actual {l}' for l in labels],
            colorscale=[[0,'#080c14'],[0.25,'#0c2a4a'],[0.6,'#1d4ed8'],[1,'#38bdf8']],
            text=cm, texttemplate='<b>%{text}</b>',
            textfont=dict(size=13, color='white'),
            showscale=True,
            colorbar=dict(tickfont=dict(color='#64748b',size=10), outlinewidth=0,
                        tickcolor='#334155'),
        ))
        fig_cm.update_layout(
            title=dict(text='Confusion Matrix', font=dict(size=14,color='#e2e8f0',family=FONT)),
            height=380, margin=dict(l=10,r=10,t=44,b=10),
            paper_bgcolor=PBGL, plot_bgcolor=PBGL, font=dict(family=FONT),
            yaxis=dict(autorange='reversed', tickfont=dict(color='#94a3b8',size=11), showgrid=False),
            xaxis=dict(tickfont=dict(color='#94a3b8',size=11), showgrid=False),
        )
        def _cm_chart():
            st.plotly_chart(fig_cm, width='stretch', config=CHART_CFG)
        chart_card(_cm_chart)

    with cr:
        st.markdown("""
        <div style="background:#111827;border:1px solid rgba(99,179,237,0.1);border-radius:16px;padding:22px;">
            <div style="font-size:10px;font-weight:700;color:#475569;letter-spacing:0.14em;
                        text-transform:uppercase;margin-bottom:18px;">Pipeline Config</div>
        """, unsafe_allow_html=True)
        params = [
            ("n_neighbors","5","#38bdf8"),
            ("weights","uniform","#94a3b8"),
            ("algorithm","auto","#94a3b8"),
            ("metric","minkowski p=2","#94a3b8"),
            ("scaler","StandardScaler","#6366f1"),
            ("features","12","#10b981"),
            ("targets","GPA + GradeClass","#f59e0b"),
        ]
        for k, v, clr in params:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:9px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="font-size:12px;color:#64748b;">{k}</span>
                <span style="font-family:'JetBrains Mono';font-size:12px;font-weight:500;color:{clr};">{v}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:20px;">
            <div style="font-size:10px;font-weight:700;color:#475569;letter-spacing:0.14em;
                        text-transform:uppercase;margin-bottom:14px;">Performance Metrics</div>
        """, unsafe_allow_html=True)

        perf = [
            ("Classification accuracy", f"{acc*100:.1f}%", "#10b981", int(acc*100)),
            ("GPA MSE", f"{mse:.4f}", "#f59e0b", max(5, int((1-mse)*100))),
            ("GPA R² score", f"{r2:.4f}", "#38bdf8", int(r2*100)),
        ]
        for k, v, clr, bw in perf:
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:12px;color:#64748b;">{k}</span>
                    <span style="font-size:13px;font-weight:700;color:{clr};font-family:'JetBrains Mono';">{v}</span>
                </div>
                <div style="height:4px;background:rgba(255,255,255,0.04);border-radius:2px;overflow:hidden;">
                    <div style="width:{bw}%;height:100%;background:{clr};border-radius:2px;opacity:0.75;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if 'pred' not in st.session_state:
        st.session_state['pred'] = None

    df  = load_data()
    clf, reg, X_tr, yc_tr, yr_tr, acc, mse, r2, fi, cm = train_models(df)
    inputs, btn = render_sidebar()

    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:8px 0 24px;border-bottom:1px solid rgba(99,179,237,0.08);margin-bottom:28px;">
        <div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                <div style="width:8px;height:8px;background:#10b981;border-radius:50%;
                            box-shadow:0 0 8px #10b981;animation:pulse 2s infinite;"></div>
                <span style="font-size:10px;color:#10b981;font-weight:700;letter-spacing:0.14em;
                            text-transform:uppercase;">Model Active</span>
            </div>
            <h1 style="font-size:28px;font-weight:800;color:#f1f5f9;margin:0;line-height:1.1;">
                EduPredict <span style="color:#38bdf8;">AI</span>
            </h1>
            <p style="font-size:13px;color:#334155;margin:4px 0 0;font-weight:500;">
                K-Nearest Neighbors · Student Performance Intelligence
            </p>
        </div>
        <div style="display:flex;gap:10px;">
            <div style="background:rgba(56,189,248,0.07);border:1px solid rgba(56,189,248,0.18);
                        border-radius:10px;padding:10px 16px;text-align:center;">
                <div style="font-size:18px;font-weight:800;color:#38bdf8;
                            font-family:'JetBrains Mono';">{acc*100:.1f}%</div>
                <div style="font-size:9px;color:#334155;letter-spacing:0.1em;text-transform:uppercase;">accuracy</div>
            </div>
            <div style="background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.18);
                        border-radius:10px;padding:10px 16px;text-align:center;">
                <div style="font-size:18px;font-weight:800;color:#10b981;
                            font-family:'JetBrains Mono';">{r2:.3f}</div>
                <div style="font-size:9px;color:#334155;letter-spacing:0.1em;text-transform:uppercase;">R² score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⚡  Predict", "📊  Insights", "🧠  Model Info"])
    with tab1:
        render_predict_tab(inputs, btn, clf, reg, X_tr, yc_tr, yr_tr, acc, mse, r2)
    with tab2:
        render_insights_tab(df, fi)
    with tab3:
        render_model_tab(acc, mse, r2, cm)

if __name__ == "__main__":
    main()






# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
# 
# ---------- UTIL ----------
# def hex_to_rgba(hex_color, alpha=0.3):
    # hex_color = hex_color.lstrip('#')
    # r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # return f'rgba({r},{g},{b},{alpha})'
# 
# ---------- PAGE ----------
# st.set_page_config(page_title="EduPredict AI", layout="wide")
# 
# ---------- ELITE CSS ----------
# st.markdown("""
# <style>
# body {
    # background: linear-gradient(135deg,#0f172a,#020617);
# }
# .block-container {
    # padding-top: 2rem;
# }
# .card {
    # background: rgba(255,255,255,0.04);
    # border: 1px solid rgba(255,255,255,0.08);
    # backdrop-filter: blur(12px);
    # border-radius: 16px;
    # padding: 20px;
# }
# .title {
    # font-size: 28px;
    # font-weight: 800;
    # color: #e2e8f0;
# }
# .metric {
    # font-size: 26px;
    # font-weight: 700;
# }
# </style>
# """, unsafe_allow_html=True)
# 
# ---------- DATA ----------
# FEATURES = ['Age','Gender','Ethnicity','ParentalEducation','StudyTimeWeekly',
            # 'Absences','Tutoring','ParentalSupport','Extracurricular','Sports','Music','Volunteering']
# 
# GRADE_LABELS  = {0:'F',1:'D',2:'C',3:'B',4:'A'}
# GRADE_COLORS  = {0:'#f43f5e',1:'#f59e0b',2:'#818cf8',3:'#38bdf8',4:'#10b981'}
# 
# @st.cache_data
# def load_data():
    # return pd.read_csv('Student_performance_data.csv')
# 
# @st.cache_resource
# def train(df):
    # X = df[FEATURES]
    # y1 = df['GradeClass']
    # y2 = df['GPA']
# 
    # X_tr, X_te, y1_tr, y1_te, y2_tr, y2_te = train_test_split(X,y1,y2,test_size=0.2)
# 
    # clf = Pipeline([('sc',StandardScaler()),('knn',KNeighborsClassifier(5))])
    # reg = Pipeline([('sc',StandardScaler()),('knn',KNeighborsRegressor(5))])
# 
    # clf.fit(X_tr,y1_tr)
    # reg.fit(X_tr,y2_tr)
# 
    # acc = accuracy_score(y1_te, clf.predict(X_te))
    # r2  = r2_score(y2_te, reg.predict(X_te))
    # cm  = confusion_matrix(y1_te, clf.predict(X_te))
# 
    # return clf, reg, acc, r2, cm
# 
# df = load_data()
# clf, reg, acc, r2, cm = train(df)
# 
# ---------- HEADER ----------
# st.markdown('<div class="title">⚡ EduPredict AI Dashboard</div>', unsafe_allow_html=True)
# 
# c1,c2,c3 = st.columns(3)
# c1.markdown(f'<div class="card"><div>Accuracy</div><div class="metric" style="color:#10b981">{acc*100:.1f}%</div></div>', unsafe_allow_html=True)
# c2.markdown(f'<div class="card"><div>R² Score</div><div class="metric" style="color:#38bdf8">{r2:.3f}</div></div>', unsafe_allow_html=True)
# c3.markdown(f'<div class="card"><div>Model</div><div class="metric">KNN</div></div>', unsafe_allow_html=True)
# 
# st.markdown("---")
# 
# ---------- INPUT ----------
# col1,col2,col3 = st.columns(3)
# 
# age = col1.slider("Age",15,18,17)
# study = col2.slider("Study Hours",0.0,20.0,10.0)
# absences = col3.slider("Absences",0,30,5)
# 
# btn = st.button("🚀 Predict")
# 
# if btn:
    # inp = pd.DataFrame([{
        # 'Age':age,'Gender':0,'Ethnicity':1,'ParentalEducation':2,
        # 'StudyTimeWeekly':study,'Absences':absences,'Tutoring':0,
        # 'ParentalSupport':2,'Extracurricular':0,'Sports':0,'Music':0,'Volunteering':0
    # }])
# 
    # grade = int(clf.predict(inp)[0])
    # gpa = float(reg.predict(inp)[0])
    # gc = GRADE_COLORS[grade]
# 
    # st.markdown(f"""
    # <div class="card">
        # <h3 style="color:{gc}">Prediction Result</h3>
        # <h1 style="color:{gc}">Grade {GRADE_LABELS[grade]}</h1>
        # <h2>GPA: {gpa:.2f}</h2>
    # </div>
    # """, unsafe_allow_html=True)
# 
    # ---------- RADAR ----------
    # fig = go.Figure(go.Scatterpolar(
        # r=[0.6,0.8,0.7,0.5,0.6,0.7],
        # theta=['Age','Study','Attendance','Tutoring','Parent','Activities'],
        # fill='toself',
        # fillcolor=hex_to_rgba(gc,0.1),
        # line=dict(color=gc,width=3)
    # ))
    # fig.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)'), showlegend=False)
    # st.plotly_chart(fig, width='stretch')
# 
# ---------- DISTRIBUTION ----------
# grade_cts = df['GradeClass'].value_counts().sort_index()
# lbls = [GRADE_LABELS[int(g)] for g in grade_cts.index]
# clrs = [GRADE_COLORS[int(g)] for g in grade_cts.index]
# 
# fig2 = go.Figure(go.Bar(
    # x=lbls,
    # y=grade_cts.values,
    # marker=dict(
        # color=clrs,
        # line=dict(color=[hex_to_rgba(c,0.3) for c in clrs])
    # )
# ))
# fig2.update_layout(title="Grade Distribution", plot_bgcolor='rgba(0,0,0,0)')
# st.plotly_chart(fig2, width='stretch')
# 
# ---------- CONFUSION MATRIX ----------
# fig_cm = go.Figure(go.Heatmap(
    # z=cm,
    # colorscale="Blues"
# ))
# fig_cm.update_layout(title="Confusion Matrix")
# st.plotly_chart(fig_cm, width='stretch')











































































