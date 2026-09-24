#!/usr/bin/env python3
"""
AI Physio Triage Tool — Vertex AI Edition (v10)
================================================
Same app as before, but swapped from local Ollama to Google
Vertex AI (Gemini) for instant responses + HK data residency.

SETUP:
    conda activate physio
    pip install llama-index-llms-vertex llama-index-embeddings-vertex google-cloud-aiplatform

BEFORE RUNNING:
    Make sure you've completed:
      gcloud auth application-default login
      gcloud config set project YOUR_PROJECT_ID
      gcloud services enable aiplatform.googleapis.com

    Update GCP_PROJECT_ID below to your actual project ID.

RUN:
    streamlit run app_vertex.py
"""

import os
import re
import json
import glob
import streamlit as st

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.vertex import Vertex
from llama_index.embeddings.vertex import VertexTextEmbedding
from google.oauth2 import service_account as _sa

def _load_vertex_credentials():
    """Load credentials from local file OR Streamlit Cloud secrets."""
    local_key = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vertex-key.json")
    if os.path.exists(local_key):
        return _sa.Credentials.from_service_account_file(local_key)
    try:
        import streamlit as _st
        info = dict(_st.secrets["gcp_service_account"])
        return _sa.Credentials.from_service_account_info(info)
    except Exception as e:
        raise RuntimeError(
            "No Vertex credentials found. Add vertex-key.json locally, "
            "or configure [gcp_service_account] in Streamlit secrets."
        ) from e

_VERTEX_CREDS = _load_vertex_credentials()
st.set_page_config(page_title="Pain Assessment", page_icon="🩺", layout="centered")

# ============================================================
# VERTEX AI CONFIG — fill in your project ID
# ============================================================
GCP_PROJECT_ID = "physio-triage-tool"   # <-- your actual project ID
GCP_REGION = "asia-east2"               # Hong Kong region for data residency

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #FFFFFF; }
    h1, h2, h3 { color: #111827 !important; font-family: 'Inter', sans-serif; }
    .ai-bubble {
        background-color: #F9FAFB; border-radius: 16px;
        padding: 16px 20px; margin-bottom: 16px;
        font-size: 18px; color: #111827; line-height: 1.4;
    }
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        background-color: #2563EB; color: white; border: none;
        border-radius: 8px; padding: 12px 24px; font-weight: 600; height: 50px;
    }
    div[data-testid="stCheckbox"] label {
        background-color: white; border: 1px solid #E5E7EB;
        border-radius: 12px; padding: 12px 16px; margin-bottom: 8px; width: 100%;
    }
    div[data-testid="stCheckbox"] label:hover {
        border-color: #2563EB; background-color: #EFF6FF;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF; border: 1px solid #E5E7EB !important;
        border-radius: 16px; padding: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

DIFFICULTY_BADGE = {
    "beginner": "🟢 Beginner",
    "intermediate": "🟡 Intermediate",
    "advanced": "🔴 Advanced",
}


# ============================================================
# ENGINE — now powered by Vertex AI
# ============================================================
@st.cache_resource
def setup_engine():
    # LLM: Gemini 1.5 Flash — fast, cheap, great for this use case
    Settings.llm = Vertex(
        model="gemini-1.5-flash-002",
        project=GCP_PROJECT_ID,
        location=GCP_REGION,
        temperature=0.2,
        max_tokens=1024,
        credentials=_VERTEX_CREDS,
    )
    # Embeddings: Vertex's text embedding model
    Settings.embed_model = VertexTextEmbedding(
        model_name="text-embedding-004",
        project=GCP_PROJECT_ID,
        location=GCP_REGION,
        credentials=_VERTEX_CREDS,
    )

    conditions_data, docs = {}, []
    for fname in sorted(glob.glob("MSK-*.json")):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        cond_name = data.get("condition_name_en")
        if not cond_name:
            continue
        conditions_data[cond_name] = data

        parts = [f"Condition: {cond_name}",
                 f"Body area: {', '.join(data.get('body_area', []))}"]
        aliases = data.get('patient_language_aliases', [])
        if aliases:
            parts.append("Also described as: " + ", ".join(str(a) for a in aliases))
        symptoms = data.get('symptoms', [])
        if symptoms:
            parts.append("Symptoms: " + "; ".join(
                s.get('description_en', '') if isinstance(s, dict) else str(s)
                for s in symptoms))
        causes = data.get('possible_causes', [])
        if causes:
            parts.append("Possible causes: " + "; ".join(
                c.get('cause_en', '') if isinstance(c, dict) else str(c)
                for c in causes))
        docs.append(Document(text="\n".join(parts),
                             metadata={"condition_name": cond_name}))

    index = VectorStoreIndex.from_documents(docs)
    return index.as_retriever(similarity_top_k=1), conditions_data, len(docs)


def match_condition(retriever, body_area, pain_quality, pain_timing):
    query = f"{body_area} pain, feels {pain_quality}, occurs {pain_timing}"
    nodes = retriever.retrieve(query)
    if not nodes:
        return None, "low"
    best = nodes[0]
    score = best.score if hasattr(best, "score") and best.score else 0
    conf = "high" if score > 0.5 else ("medium" if score > 0.3 else "low")
    return best.metadata.get("condition_name"), conf


def get_recovery(conditions_data, condition, severity):
    d = conditions_data.get(condition, {})
    return d.get("recovery_timeline", {}).get(f"{severity.lower()}_estimate", "Varies")


def parse_plan_days(condition_data, severity="Mild"):
    tl = condition_data.get("recovery_timeline", {})
    text = tl.get(f"{severity.lower()}_estimate", "") or ""
    wk = re.findall(r"(\d+)\s*[-–—]\s*(\d+)\s*week", text)
    if wk:
        return min(max(int(wk[0][0]), 1) * 7, 42), text
    wk1 = re.findall(r"(\d+)\s*week", text)
    if wk1:
        return min(int(wk1[0]) * 7, 42), text
    dy = re.findall(r"(\d+)\s*[-–—]\s*(\d+)\s*day", text)
    if dy:
        return min(int(dy[0][1]), 42), text
    return 14, text


# ============================================================
# STATE
# ============================================================
def init_state():
    defaults = {
        "step": 0, "answers": {}, "mode": "assessment",
        "current_day": 1, "week_offset": 0,
        "completed": {}, "checkins": {}, "factors": {},
        "view_exercise": None, "show_checkin": False,
        "rc_screen": "factors",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def next_step(): st.session_state.step += 1
def prev_step():
    if st.session_state.step > 0: st.session_state.step -= 1

def restart():
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    init_state()

def ai_bubble(text):
    st.markdown(f"<div class='ai-bubble'>{text}</div>", unsafe_allow_html=True)


# ============================================================
# RECOVERY COACH SCREENS
# ============================================================
STATUS_ICON = {"done": "✅", "today": "📍", "today_done": "✅",
               "missed": "⚪", "upcoming": "⚪"}

def day_status(day, current, completed, n_ex):
    if day > current: return "upcoming"
    done = len(completed.get(day, set()))
    if day == current: return "today_done" if done >= n_ex else "today"
    return "done" if done >= n_ex else "missed"


def rc_factors():
    st.subheader("A few quick questions")
    st.caption("This helps your physiotherapist understand your context. "
               "Your plan is not automatically adjusted — discuss any "
               "modifications with a professional.")
    with st.form("factors_form"):
        age = st.selectbox("Age range:",
                           ["18–29", "30–39", "40–49", "50–59", "60–69", "70+"])
        act = st.selectbox("Current activity level:",
                           ["Very active (5+ days/week)",
                            "Moderately active (2–4 days/week)",
                            "Lightly active (1–2 days/week)", "Mostly sedentary"])
        other = st.multiselect("Do you have any of these? (optional)",
                               ["Diabetes", "Heart condition", "Osteoporosis",
                                "Arthritis", "Pregnancy", "Recent surgery",
                                "High blood pressure"])
        prior = st.radio("Seen a physiotherapist for this before?",
                         ["No", "Yes, currently", "Yes, in the past"])
        if st.form_submit_button("Start My Plan →", type="primary",
                                 use_container_width=True):
            st.session_state.factors = {"age_range": age, "activity": act,
                                        "other_conditions": other, "prior_physio": prior}
            st.session_state.rc_screen = "overview"
            st.rerun()


def rc_overview(cd, plan_days, timeline_text):
    name = cd.get("condition_name_en", "your condition")
    name_zh = cd.get("condition_name_zh", "")
    exercises = cd.get("self_care_exercises", [])
    weeks = (plan_days + 6) // 7

    st.subheader("Your Recovery Plan")
    st.markdown(f"### {name}")
    if name_zh: st.caption(name_zh)

    with st.container(border=True):
        st.markdown(f"**📅 Plan length:** {plan_days} days ({weeks} weeks)")
        st.markdown(f"**🏃 Exercises:** {len(exercises)} per day")
        if timeline_text:
            st.markdown(f"**🕒 Expected recovery:** {timeline_text}")

    f = st.session_state.factors
    if f:
        with st.expander("Your context (shared with your physiotherapist)"):
            st.markdown(f"- Age range: {f.get('age_range','—')}")
            st.markdown(f"- Activity: {f.get('activity','—')}")
            oc = f.get("other_conditions") or []
            st.markdown(f"- Other conditions: {', '.join(oc) if oc else 'None reported'}")
            st.markdown(f"- Prior physio: {f.get('prior_physio','—')}")

    st.write("")
    st.markdown("**Exercises in this plan:**")
    for ex in exercises:
        d = DIFFICULTY_BADGE.get(str(ex.get("difficulty_level", "")).lower(), "")
        st.markdown(f"- {ex.get('exercise_name_en','Exercise')} — {ex.get('frequency','')} {d}")

    st.write("")
    st.info("💡 Consistency matters more than intensity. Stop any exercise that "
            "causes sharp pain.")

    if f.get("other_conditions") or f.get("age_range") in ["60–69", "70+"]:
        st.warning("⚠️ Because of the health factors you shared, we recommend "
                   "discussing this plan with a physiotherapist before starting.")

    st.write("")
    if st.button("View My Calendar →", type="primary", use_container_width=True):
        st.session_state.rc_screen = "calendar"
        st.rerun()


def rc_calendar(cd, plan_days):
    exercises = cd.get("self_care_exercises", [])
    n_ex = len(exercises)
    cur = st.session_state.current_day
    completed = st.session_state.completed
    off = st.session_state.week_offset
    total_weeks = (plan_days + 6) // 7
    week_num = off + 1
    start_day = off * 7 + 1
    end_day = min(start_day + 6, plan_days)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("← Prev", disabled=(off == 0), use_container_width=True):
            st.session_state.week_offset -= 1; st.rerun()
    with c2:
        st.markdown(f"<div style='text-align:center;font-weight:600;padding-top:6px;'>"
                    f"Week {week_num} of {total_weeks}</div>", unsafe_allow_html=True)
    with c3:
        if st.button("Next →", disabled=(week_num >= total_weeks), use_container_width=True):
            st.session_state.week_offset += 1; st.rerun()

    st.write("")
    days = list(range(start_day, end_day + 1))
    cols = st.columns(7)
    for i, col in enumerate(cols):
        with col:
            if i < len(days):
                d = days[i]
                icon = STATUS_ICON[day_status(d, cur, completed, n_ex)]
                label = "Today" if d == cur else ""
                st.markdown(f"<div style='text-align:center;'>"
                            f"<div style='font-size:12px;color:#6B7280;'>Day {d}</div>"
                            f"<div style='font-size:28px;'>{icon}</div>"
                            f"<div style='font-size:11px;color:#2563EB;'>{label}</div></div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### Day {cur} — Today's exercises")
    done_set = completed.get(cur, set())

    if not exercises:
        st.warning("No exercises found for this condition.")
        return

    for idx, ex in enumerate(exercises):
        is_done = idx in done_set
        with st.container(border=True):
            a, b = st.columns([4, 1])
            with a:
                st.markdown(f"**{'✅ ' if is_done else ''}{ex.get('exercise_name_en','Exercise')}**")
                st.caption(f"{ex.get('frequency','')} · "
                           f"{DIFFICULTY_BADGE.get(str(ex.get('difficulty_level','')).lower(),'')}")
            with b:
                if not is_done:
                    if st.button("Do it", key=f"do_{cur}_{idx}", use_container_width=True):
                        st.session_state.view_exercise = idx; st.rerun()
            if is_done:
                st.caption("Completed")

    st.write("")
    st.progress(len(done_set) / n_ex if n_ex else 0)
    st.caption(f"{len(done_set)} of {n_ex} exercises done today")

    if len(done_set) >= n_ex and cur not in st.session_state.checkins:
        st.success("All exercises done! Ready for your daily check-in.")
        if st.button("Daily Check-In →", type="primary", use_container_width=True):
            st.session_state.show_checkin = True; st.rerun()


def rc_exercise_detail(ex, idx, total, day):
    st.caption(f"Day {day} · Exercise {idx+1} of {total}")
    st.markdown(f"### {ex.get('exercise_name_en','Exercise')}")
    if ex.get("exercise_name_zh"): st.caption(ex["exercise_name_zh"])

    a, b = st.columns(2)
    with a:
        st.markdown(f"**Difficulty**  \n"
                    f"{DIFFICULTY_BADGE.get(str(ex.get('difficulty_level','')).lower(),'—')}")
    with b:
        st.markdown(f"**Frequency**  \n{ex.get('frequency','—')}")

    st.divider()
    with st.container(border=True):
        st.markdown("<div style='text-align:center;padding:30px;color:#6B7280;'>"
                    "▶️<br>Exercise video<br><small>(coming soon)</small></div>",
                    unsafe_allow_html=True)
    st.divider()
    st.markdown("**How to do it:**")
    st.write(ex.get("instructions_en", "Instructions not available."))
    if ex.get("instructions_zh"):
        with st.expander("中文說明"):
            st.write(ex["instructions_zh"])
    if ex.get("contraindications"):
        st.warning(f"⚠️ **Important:** {ex['contraindications']}")

    st.write("")
    a, b = st.columns(2)
    with a:
        if st.button("← Back to plan", use_container_width=True):
            st.session_state.view_exercise = None; st.rerun()
    with b:
        if st.button("Mark as Complete ✓", type="primary", use_container_width=True):
            d = st.session_state.current_day
            st.session_state.completed.setdefault(d, set()).add(idx)
            st.session_state.view_exercise = None; st.rerun()


def rc_checkin():
    day = st.session_state.current_day
    st.subheader(f"Day {day} Check-In 🎉")
    st.write("Great work completing today's exercises!")
    with st.form("checkin"):
        pain = st.radio("**How's your pain right now?**",
                        ["🟢 Better", "😐 About the same", "🔴 Worse"])
        notes = st.text_area("Any notes? (optional)")
        if st.form_submit_button("Submit & Continue", type="primary",
                                 use_container_width=True):
            st.session_state.checkins[day] = pain
            st.session_state.current_day += 1
            st.session_state.week_offset = (st.session_state.current_day - 1) // 7
            st.session_state.show_checkin = False
            st.rerun()


def rc_progress(plan_days):
    cur = st.session_state.current_day
    checkins = st.session_state.checkins
    completed = st.session_state.completed

    st.subheader("Your Progress")
    with st.container(border=True):
        pct = min(int(((cur - 1) / plan_days) * 100), 100)
        st.markdown(f"**Day {cur} of {plan_days}**")
        st.progress(min((cur - 1) / plan_days, 1.0))
        st.caption(f"{pct}% complete")

    streak = len([d for d in completed if completed[d]])
    if streak >= 2:
        st.markdown(f"🔥 **{streak} days completed**")

    st.divider()
    st.markdown("**Your pain trend**")
    if not checkins:
        st.caption("No check-ins yet — complete a day to start tracking.")
    else:
        with st.container(border=True):
            for d in sorted(checkins):
                st.markdown(f"Day {d} — {checkins[d]}")

    st.divider()
    st.info("💡 If pain worsens or doesn't improve, see a physiotherapist.")


# ============================================================
# MAIN
# ============================================================
try:
    retriever, conditions_data, n_conditions = setup_engine()
except Exception as e:
    st.error(f"⚠️ Could not connect to Vertex AI: {e}")
    st.info("Check that: (1) GCP_PROJECT_ID is correct, (2) you've run "
            "`gcloud auth application-default login`, (3) billing is enabled, "
            "(4) the Vertex AI API is enabled.")
    st.stop()

A = st.session_state.answers

# ---------- RECOVERY COACH MODE ----------
if st.session_state.mode == "recovery":
    cond_name = A.get("matched_condition")
    cd = conditions_data.get(cond_name, {})
    plan_days, timeline_text = parse_plan_days(cd, A.get("severity", "Mild"))

    with st.sidebar:
        st.markdown("### Recovery Coach")
        st.caption(cond_name or "—")
        nav = st.radio("Go to:", ["Plan Overview", "Calendar", "Progress"],
                       index={"factors": 0, "overview": 0, "calendar": 1,
                              "progress": 2}.get(st.session_state.rc_screen, 0))
        if nav == "Plan Overview" and st.session_state.rc_screen not in ("factors", "overview"):
            st.session_state.rc_screen = "overview"; st.rerun()
        if nav == "Calendar" and st.session_state.rc_screen != "calendar":
            st.session_state.rc_screen = "calendar"; st.rerun()
        if nav == "Progress" and st.session_state.rc_screen != "progress":
            st.session_state.rc_screen = "progress"; st.rerun()
        st.divider()
        if st.button("← Back to Summary", use_container_width=True):
            st.session_state.mode = "assessment"; st.rerun()
        if st.button("Start Over", use_container_width=True):
            restart(); st.rerun()

    if st.session_state.view_exercise is not None:
        ex_list = cd.get("self_care_exercises", [])
        i = st.session_state.view_exercise
        if 0 <= i < len(ex_list):
            rc_exercise_detail(ex_list[i], i, len(ex_list), st.session_state.current_day)
    elif st.session_state.show_checkin:
        rc_checkin()
    elif st.session_state.rc_screen == "factors":
        rc_factors()
    elif st.session_state.rc_screen == "overview":
        rc_overview(cd, plan_days, timeline_text)
    elif st.session_state.rc_screen == "calendar":
        rc_calendar(cd, plan_days)
    elif st.session_state.rc_screen == "progress":
        rc_progress(plan_days)

# ---------- CLINIC MATCH MODE (placeholder) ----------
elif st.session_state.mode == "clinic":
    st.subheader("Find a Physiotherapist")
    st.info("Clinic matching — coming next. (Supabase integration pending)")
    if st.button("← Back to Summary"):
        st.session_state.mode = "assessment"; st.rerun()

# ---------- ASSESSMENT MODE ----------
else:
    total_steps = 8
    if 0 < st.session_state.step <= total_steps:
        st.progress(st.session_state.step / total_steps)

    if st.session_state.step == 0:
        st.markdown("<h1 style='text-align:center;'>🩺</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>Understand your pain,<br>"
                    "find the right care.</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#6B7280;'>了解痛症<br>"
                    "找到最合適的護理方式</p>", unsafe_allow_html=True)
        st.write("")
        st.info("ℹ️ This tool provides information only and does not diagnose.")
        st.caption(f"Knowledge base: {n_conditions} conditions loaded")
        st.write("")
        st.button("Let's Get Started →", on_click=next_step, type="primary")

    elif st.session_state.step == 1:
        ai_bubble("Where does it hurt? Look at the body map and select the area.")
        try:
            st.image("body_map.png", use_container_width=True)
        except Exception:
            pass
        st.button("← Back", on_click=prev_step)
        with st.form("body_form"):
            body = st.selectbox("Select the area:",
                                ["Lower back", "Upper back", "Neck", "Left shoulder",
                                 "Right shoulder", "Left elbow", "Right elbow",
                                 "Left wrist", "Right wrist", "Left hip", "Right hip",
                                 "Left knee", "Right knee", "Left ankle", "Right ankle",
                                 "Left foot", "Right foot"])
            custom = st.text_input("Or describe in your own words (optional):")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                A["body_area"] = custom.strip() if custom.strip() else body
                next_step(); st.rerun()

    elif st.session_state.step == 2:
        ai_bubble(f"Got it — {A.get('body_area','')} pain. "
                  f"Can you describe what the pain feels like?")
        st.button("← Back", on_click=prev_step)
        with st.form("quality_form"):
            st.caption("Select all that apply:")
            opts = {"🔥 Burning": "Burning", "⚡ Sharp": "Sharp",
                    "😣 Dull ache": "Dull ache", "📌 Pins & needles": "Pins & needles",
                    "💪 Stiffness": "Stiffness", "🔋 Weakness": "Weakness",
                    "🔨 Throbbing": "Throbbing", "🧊 Clicking or locking": "Clicking or locking"}
            chosen = [v for l, v in opts.items() if st.checkbox(l)]
            custom = st.text_input("Or describe in your own words...")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                if custom.strip(): chosen.append(custom.strip())
                A["pain_quality"] = ", ".join(chosen) if chosen else "unspecified"
                next_step(); st.rerun()

    elif st.session_state.step == 3:
        ai_bubble("When does the pain usually happen?")
        st.button("← Back", on_click=prev_step)
        with st.form("timing_form"):
            st.caption("Select all that apply:")
            opts = ["During activity", "In the morning", "At night", "When sitting",
                    "When walking", "After exercise", "Constant"]
            chosen = [o for o in opts if st.checkbox(o)]
            custom = st.text_input("Or describe in your own words...")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                if custom.strip(): chosen.append(custom.strip())
                A["pain_timing"] = ", ".join(chosen) if chosen else "unspecified"
                next_step(); st.rerun()

    elif st.session_state.step == 4:
        ai_bubble("How intense is your pain right now?")
        st.button("← Back", on_click=prev_step)
        with st.form("sev_form"):
            sev = st.radio("Select one:", ["🟢 Mild", "🟡 Moderate", "🔴 Severe"])
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                A["severity"] = sev.split(" ", 1)[1]
                next_step(); st.rerun()

    elif st.session_state.step == 5:
        ai_bubble(f"Got it — {A.get('severity','')} pain. When did this start?")
        st.button("← Back", on_click=prev_step)
        with st.form("dur_form"):
            dur = st.radio("Select one:",
                           ["Just today", "Few days", "1-4 weeks", "Over a month"])
            custom = st.text_input("Or describe when it started...")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                A["duration"] = custom.strip() if custom.strip() else dur
                next_step(); st.rerun()

    elif st.session_state.step == 6:
        st.markdown("<h3 style='text-align:center;'>⚠️ Safety Check</h3>",
                    unsafe_allow_html=True)
        st.caption("Please tell us if you're experiencing any of these. "
                   "You can choose more than one.")
        st.button("← Back", on_click=prev_step)
        with st.form("flags_form"):
            opts = ["Numbness in both legs", "Loss of bladder or bowel control",
                    "Fever with pain", "Recent significant injury or fall",
                    "Unexplained weight loss", "History of cancer",
                    "Sudden severe weakness"]
            chosen = [o for o in opts if st.checkbox(o)]
            st.divider()
            none_sel = st.checkbox("**None of the above**")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                A["red_flags"] = [] if none_sel else chosen
                next_step(); st.rerun()

    elif st.session_state.step == 7:
        ai_bubble("Last question — what activities do you do regularly?")
        st.button("← Back", on_click=prev_step)
        with st.form("act_form"):
            st.caption("Select all that apply:")
            opts = {"🏃 Running": "Running", "🏋️ Weightlifting": "Weightlifting",
                    "🚴 Cycling": "Cycling", "🧘 Yoga/Pilates": "Yoga/Pilates",
                    "🎾 Racquet sports": "Racquet sports", "🏀 Team sports": "Team sports",
                    "💻 Desk work": "Desk work", "💺 Mostly sedentary": "Mostly sedentary"}
            chosen = [v for l, v in opts.items() if st.checkbox(l)]
            custom = st.text_input("Or describe your activity...")
            if st.form_submit_button("Get My Assessment →", type="primary",
                                     use_container_width=True):
                if custom.strip(): chosen.append(custom.strip())
                A["activity"] = ", ".join(chosen) if chosen else "unspecified"
                next_step(); st.rerun()

    elif st.session_state.step == 8:
        real_flags = [f for f in A.get("red_flags", []) if f]
        if real_flags:
            st.markdown("<h1 style='text-align:center;'>🚨</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center;color:#EF4444;'>"
                        "Get medical care right away</h2>", unsafe_allow_html=True)
            st.write("Based on what you shared, your symptoms may need urgent attention.")
            st.info("**Recommended actions:**\n\n- Go to your nearest A&E\n"
                    "- Call 999 for severe symptoms\n- Don't wait — get checked")
            st.button("Find Emergency Care", type="primary")
            st.button("Start Over", on_click=restart)
        else:
            if "matched_condition" not in A:
                with st.spinner("Analyzing your symptoms..."):
                    cond, conf = match_condition(retriever, A["body_area"],
                                                 A["pain_quality"], A["pain_timing"])
                    A["matched_condition"] = cond
                    A["confidence"] = conf
            cond = A["matched_condition"]
            conf = A["confidence"]
            recovery = get_recovery(conditions_data, cond, A["severity"])

            mild = A["severity"].lower() == "mild"
            next_text = "Try guided self-care" if mild else "See a physiotherapist"

            st.subheader("Your Pain Summary")
            st.caption("Generated by AI Physio Triage Tool")
            se = {"mild": "🟢", "moderate": "🟡", "severe": "🔴"}.get(
                A["severity"].lower(), "🟡")

            with st.container(border=True):
                st.markdown(f"**📍 Location**  \n{A['body_area']}")
                st.markdown(f"**⚡ Type**  \n{A['pain_quality']}")
                st.markdown(f"**📅 When**  \n{A['pain_timing']}")
                st.markdown(f"**⏱️ Duration**  \n{A['duration']}")
                st.markdown(f"**🎯 Severity**  \n{se} {A['severity']}")
                st.markdown(f"**🏃 Activity**  \n{A['activity']}")
                st.divider()
                st.markdown(f"**📋 What this might be**  \n{cond} *({conf} confidence)*")
                st.markdown(f"**🕒 Recovery estimate**  \n{recovery}")
                st.divider()
                st.markdown(f"**🩺 Suggested next step**  \n{next_text}")

            st.caption("ℹ️ Not a diagnosis. Reviewed by HK physiotherapists.")
            st.write("")

            c1, c2 = st.columns(2)
            with c1:
                st.button("Share Summary", type="primary", use_container_width=True)
            with c2:
                if mild:
                    if st.button("Start Recovery Plan", use_container_width=True):
                        st.session_state.mode = "recovery"
                        st.session_state.rc_screen = "factors"
                        st.rerun()
                else:
                    if st.button("Find a Physio", use_container_width=True):
                        st.session_state.mode = "clinic"; st.rerun()

            if not mild:
                st.write("")
                if st.button("Or start a self-care plan", use_container_width=True):
                    st.session_state.mode = "recovery"
                    st.session_state.rc_screen = "factors"
                    st.rerun()

            st.write("")
            st.button("Start Over", on_click=restart)
