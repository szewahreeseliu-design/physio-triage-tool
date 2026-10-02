#!/usr/bin/env python3
"""
AI Physio Triage Tool - Beta v0.2
=================================
Fixes from Beta0.1 Review Log (Oct 1):
  1. Severity levels now have clear descriptions
  2. Find a Physio shows real clinics by area (Supabase)
  3. Calendar days are clickable + "Skip today" option
  4. Progress page has a "Talk to a physiotherapist" button
  5. "Back to Summary" renamed to "Back to Pain Summary"
  6. Removed "Knowledge base: 20 conditions loaded" from front page
  7. "Find a Physio" offered for mild cases too
  8. Low-confidence / mismatched results show "no confident match"
  9. Habit questions + routine-based daily schedule

LOCAL RUN:
    conda activate physio
    python -m streamlit run app_vertex.py

Secrets needed (local: .streamlit/secrets.toml, cloud: App settings > Secrets):
    [gcp_service_account]   ... (already set up)
    [supabase]
    url = "https://hnhpyeimpllrdaxtfevn.supabase.co"
    key = "YOUR_NEW_SERVICE_ROLE_KEY"
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

st.set_page_config(page_title="Pain Assessment", page_icon="🩺", layout="centered")

# ============================================================
# CONFIG
# ============================================================
GCP_PROJECT_ID = "physio-triage-tool"
GCP_REGION = "asia-east2"

# Matching safety: below this similarity score, show "no confident match".
# Calibrate by opening the app with ?debug=1 in the URL to see scores.
MIN_MATCH_SCORE = 0.55
HIGH_MATCH_SCORE = 0.70

DEBUG = st.query_params.get("debug") == "1"


def _load_vertex_credentials():
    local_key = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vertex-key.json")
    if os.path.exists(local_key):
        return _sa.Credentials.from_service_account_file(local_key)
    info = dict(st.secrets["gcp_service_account"])
    return _sa.Credentials.from_service_account_info(info)


_VERTEX_CREDS = _load_vertex_credentials()

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
        border-radius: 8px; font-weight: 600;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
    }
    .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

DIFFICULTY_BADGE = {
    "beginner": "🟢 Beginner",
    "intermediate": "🟡 Intermediate",
    "advanced": "🔴 Advanced",
}

# Body area -> keywords expected in a condition's body_area list
BODY_KEYWORDS = {
    "lower back": ["back", "lumbar", "spine", "buttock"],
    "upper back": ["back", "thoracic", "spine", "shoulder blade"],
    "neck": ["neck", "cervical"],
    "shoulder": ["shoulder"],
    "elbow": ["elbow", "forearm"],
    "wrist": ["wrist", "hand", "finger"],
    "hip": ["hip", "buttock", "thigh", "groin"],
    "knee": ["knee", "thigh"],
    "ankle": ["ankle", "foot", "heel", "calf", "leg"],
    "foot": ["foot", "heel", "ankle", "calf", "arch"],
}

REGIONS = {
    "Hong Kong Island": "HK_Island",
    "Kowloon": "Kowloon",
    "New Territories": "NT",
    "Outlying Islands": "Outlying_Islands",
}


# ============================================================
# ENGINE
# ============================================================
@st.cache_resource
def setup_engine():
    Settings.llm = Vertex(model="gemini-1.5-flash-002", project=GCP_PROJECT_ID,
                          location=GCP_REGION, temperature=0.2, max_tokens=1024,
                          credentials=_VERTEX_CREDS)
    Settings.embed_model = VertexTextEmbedding(model_name="text-embedding-004",
                                               project=GCP_PROJECT_ID, location=GCP_REGION,
                                               credentials=_VERTEX_CREDS)
    conditions_data, docs = {}, []
    for fname in sorted(glob.glob("MSK-*.json")):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        name = data.get("condition_name_en")
        if not name:
            continue
        conditions_data[name] = data
        parts = [f"Condition: {name}",
                 f"Body area: {', '.join(data.get('body_area', []))}"]
        aliases = data.get("patient_language_aliases", [])
        if aliases:
            parts.append("Also described as: " + ", ".join(str(a) for a in aliases))
        symptoms = data.get("symptoms", [])
        if symptoms:
            parts.append("Symptoms: " + "; ".join(
                s.get("description_en", "") if isinstance(s, dict) else str(s) for s in symptoms))
        causes = data.get("possible_causes", [])
        if causes:
            parts.append("Possible causes: " + "; ".join(
                c.get("cause_en", "") if isinstance(c, dict) else str(c) for c in causes))
        docs.append(Document(text="\n".join(parts), metadata={"condition_name": name}))
    index = VectorStoreIndex.from_documents(docs)
    return index.as_retriever(similarity_top_k=3), conditions_data


def body_consistent(body_area, condition_data):
    """True if the selected body area plausibly matches the condition."""
    sel = body_area.lower().replace("left ", "").replace("right ", "").strip()
    keywords = BODY_KEYWORDS.get(sel)
    if not keywords:          # free-text answer: can't check, allow it
        return True
    cond_area = " ".join(condition_data.get("body_area", [])).lower()
    return any(k in cond_area for k in keywords)


def match_condition(retriever, conditions_data, A):
    """Returns (condition_name or None, confidence, score)."""
    query = (f"{A['body_area']} pain, feels {A['pain_quality']}, "
             f"occurs {A['pain_timing']}, activity: {A.get('activity', '')}")
    nodes = retriever.retrieve(query)
    for node in nodes:
        name = node.metadata.get("condition_name")
        score = node.score or 0
        if not body_consistent(A["body_area"], conditions_data.get(name, {})):
            continue
        if score < MIN_MATCH_SCORE:
            return None, "low", score
        conf = "high" if score >= HIGH_MATCH_SCORE else "medium"
        return name, conf, score
    best = nodes[0].score if nodes else 0
    return None, "low", best


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
# SUPABASE (clinic list)
# ============================================================
@st.cache_resource
def get_supabase():
    from supabase import create_client
    cfg = st.secrets["supabase"]
    return create_client(cfg["url"], cfg["key"])


@st.cache_data(ttl=3600)
def fetch_clinics(region_code):
    sb = get_supabase()
    res = (sb.table("clinics")
             .select("id, clinic_name_en, clinic_name_zh, address_en, phone, district")
             .eq("region", region_code).limit(60).execute())
    clinics = res.data or []
    ids = [c["id"] for c in clinics]
    insurers = {}
    if ids:
        links = (sb.table("clinic_insurer_panels")
                   .select("clinic_id, insurers(insurer_name)")
                   .in_("clinic_id", ids).execute())
        for l in links.data or []:
            ins = (l.get("insurers") or {}).get("insurer_name")
            if ins:
                insurers.setdefault(l["clinic_id"], set()).add(ins)
    for c in clinics:
        c["insurers"] = sorted(insurers.get(c["id"], []))
    return clinics


# ============================================================
# STATE + NAV
# ============================================================
def init_state():
    defaults = {
        "step": 0, "answers": {}, "mode": "assessment", "prev_mode": "assessment",
        "current_day": 1, "selected_day": 1, "week_offset": 0,
        "completed": {}, "skipped": set(), "checkins": {}, "factors": {},
        "view_exercise": None, "view_day": None, "show_checkin": False,
        "rc_screen": "factors",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def next_step():
    st.session_state.step += 1


def prev_step():
    if st.session_state.step > 0:
        st.session_state.step -= 1


def restart():
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    init_state()


def go_clinic():
    st.session_state.prev_mode = st.session_state.mode
    st.session_state.mode = "clinic"


def ai_bubble(text):
    st.markdown(f"<div class='ai-bubble'>{text}</div>", unsafe_allow_html=True)


# ============================================================
# DAILY SCHEDULE (habit-based timing only, not dosage)
# ============================================================
SLOTS = {
    "Spread through the day (short breaks)":
        ["☕ Mid-morning break", "🍱 Lunch break", "🕒 Mid-afternoon break", "🌙 Evening"],
    "Morning, before work or school": ["🌅 Morning"],
    "Lunch break": ["🍱 Lunch break"],
    "Evening, after work": ["🌙 Evening"],
}


def build_schedule(exercises, factors):
    slots = SLOTS.get(factors.get("preferred_time"), ["🌙 Evening"])
    schedule = {s: [] for s in slots}
    for i, ex in enumerate(exercises):
        schedule[slots[i % len(slots)]].append(ex)
    return {s: exs for s, exs in schedule.items() if exs}


def schedule_reason(factors):
    work = factors.get("work_type", "")
    sitting = factors.get("sitting_hours", "")
    if work == "Desk / office work" or sitting in ["6–8 hours", "More than 8 hours"]:
        return ("Because you sit for long periods, we've placed your exercises "
                "into breaks so you move regularly through the day.")
    if work == "Standing / on my feet":
        return "We've placed your exercises at times that fit around a standing job."
    if work == "Physical / manual work":
        return "We've placed your exercises outside your working hours."
    return "We've arranged your exercises around the time you said suits you best."


# ============================================================
# RECOVERY COACH SCREENS
# ============================================================
def day_status(day):
    s = st.session_state
    if day in s.skipped:
        return "⏭️"
    if day > s.current_day:
        return "⚪"
    if day in s.checkins:
        return "✅"
    if day == s.current_day:
        return "📍"
    return "⚪"


def rc_factors():
    st.subheader("A few quick questions")
    st.caption("This helps us fit your plan into your routine and gives your "
               "physiotherapist useful context. Exercises and repetitions are "
               "not changed — discuss modifications with a professional.")
    with st.form("factors_form"):
        age = st.selectbox("Age range:", ["18–29", "30–39", "40–49", "50–59", "60–69", "70+"])
        work = st.selectbox("What best describes your work or daily routine?",
                            ["Desk / office work", "Standing / on my feet",
                             "Physical / manual work", "Student",
                             "Retired / at home", "Mixed / varies"])
        sitting = st.selectbox("How many hours do you sit on a typical day?",
                               ["Less than 4 hours", "4–6 hours", "6–8 hours",
                                "More than 8 hours"])
        pref = st.radio("When is it easiest for you to do exercises?",
                        list(SLOTS.keys()))
        act = st.selectbox("Current activity level:",
                           ["Very active (5+ days/week)", "Moderately active (2–4 days/week)",
                            "Lightly active (1–2 days/week)", "Mostly sedentary"])
        other = st.multiselect("Do you have any of these? (optional)",
                               ["Diabetes", "Heart condition", "Osteoporosis", "Arthritis",
                                "Pregnancy", "Recent surgery", "High blood pressure"])
        prior = st.radio("Seen a physiotherapist for this before?",
                         ["No", "Yes, currently", "Yes, in the past"])
        if st.form_submit_button("Create My Plan →", type="primary", use_container_width=True):
            st.session_state.factors = {
                "age_range": age, "work_type": work, "sitting_hours": sitting,
                "preferred_time": pref, "activity": act,
                "other_conditions": other, "prior_physio": prior}
            st.session_state.rc_screen = "overview"
            st.rerun()


def rc_overview(cd, plan_days, timeline_text):
    exercises = cd.get("self_care_exercises", [])
    f = st.session_state.factors
    st.subheader("Your Recovery Plan")
    st.markdown(f"### {cd.get('condition_name_en', '')}")
    if cd.get("condition_name_zh"):
        st.caption(cd["condition_name_zh"])

    with st.container(border=True):
        st.markdown(f"**📅 Plan length:** {plan_days} days ({(plan_days + 6) // 7} weeks)")
        st.markdown(f"**🏃 Exercises:** {len(exercises)} per day")
        if timeline_text:
            st.markdown(f"**🕒 Expected recovery:** {timeline_text}")

    st.markdown("#### Your daily schedule")
    st.caption(schedule_reason(f))
    for slot, exs in build_schedule(exercises, f).items():
        with st.container(border=True):
            st.markdown(f"**{slot}**")
            for ex in exs:
                st.markdown(f"- {ex.get('exercise_name_en', 'Exercise')} — {ex.get('frequency', '')}")
    if f.get("work_type") == "Desk / office work" or f.get("sitting_hours") in ["6–8 hours", "More than 8 hours"]:
        st.info("💡 Tip: standing up and moving for a minute or two between long "
                "periods of sitting can help. Ask your physiotherapist how often "
                "is right for you.")

    with st.expander("Your context (shared with your physiotherapist)"):
        for label, key in [("Age range", "age_range"), ("Routine", "work_type"),
                           ("Sitting per day", "sitting_hours"), ("Activity", "activity"),
                           ("Prior physio", "prior_physio")]:
            st.markdown(f"- {label}: {f.get(key, '—')}")
        oc = f.get("other_conditions") or []
        st.markdown(f"- Other conditions: {', '.join(oc) if oc else 'None reported'}")

    if f.get("other_conditions") or f.get("age_range") in ["60–69", "70+"]:
        st.warning("⚠️ Because of the health factors you shared, we recommend "
                   "discussing this plan with a physiotherapist before starting.")
        st.button("Find a Physiotherapist", on_click=go_clinic, use_container_width=True)

    st.info("💡 Stop any exercise that causes sharp pain.")
    if st.button("View My Calendar →", type="primary", use_container_width=True):
        st.session_state.rc_screen = "calendar"
        st.rerun()


def rc_calendar(cd, plan_days):
    s = st.session_state
    exercises = cd.get("self_care_exercises", [])
    n_ex = len(exercises)
    total_weeks = (plan_days + 6) // 7
    off = s.week_offset
    start_day = off * 7 + 1
    end_day = min(start_day + 6, plan_days)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("← Prev", disabled=(off == 0), use_container_width=True):
            s.week_offset -= 1
            st.rerun()
    with c2:
        st.markdown(f"<div style='text-align:center;font-weight:600;padding-top:6px;'>"
                    f"Week {off + 1} of {total_weeks}</div>", unsafe_allow_html=True)
    with c3:
        if st.button("Next →", disabled=(off + 1 >= total_weeks), use_container_width=True):
            s.week_offset += 1
            st.rerun()

    st.caption("Tap a day to see its exercises.")
    cols = st.columns(7)
    for i, d in enumerate(range(start_day, end_day + 1)):
        with cols[i]:
            label = f"{day_status(d)}\n\nDay {d}"
            if st.button(label, key=f"day_{d}", use_container_width=True,
                         type="primary" if d == s.selected_day else "secondary"):
                s.selected_day = d
                st.rerun()

    st.divider()
    day = s.selected_day
    is_today = day == s.current_day
    is_future = day > s.current_day
    title = "Today" if is_today else ("Upcoming" if is_future else "Past day")
    st.markdown(f"### Day {day} — {title}")

    if day in s.skipped:
        st.caption("You skipped this day. You can still do the exercises below if you'd like.")
    if is_future:
        st.caption("Preview only — you can do these exercises when you reach this day.")

    done_set = s.completed.get(day, set())
    for idx, ex in enumerate(exercises):
        is_done = idx in done_set
        with st.container(border=True):
            a, b = st.columns([4, 1])
            with a:
                st.markdown(f"**{'✅ ' if is_done else ''}{ex.get('exercise_name_en', 'Exercise')}**")
                st.caption(f"{ex.get('frequency', '')} · "
                           f"{DIFFICULTY_BADGE.get(str(ex.get('difficulty_level', '')).lower(), '')}")
            with b:
                btn = "View" if (is_done or is_future) else "Do it"
                if st.button(btn, key=f"do_{day}_{idx}", use_container_width=True):
                    s.view_exercise, s.view_day = idx, day
                    st.rerun()

    if not is_future and n_ex:
        st.progress(len(done_set) / n_ex)
        st.caption(f"{len(done_set)} of {n_ex} exercises done")

    if is_today:
        if len(done_set) >= n_ex and day not in s.checkins:
            st.success("All exercises done! Ready for your daily check-in.")
            if st.button("Daily Check-In →", type="primary", use_container_width=True):
                s.show_checkin = True
                st.rerun()
        st.write("")
        if st.button("⏭️ Skip today and move to the next day", use_container_width=True):
            s.skipped.add(day)
            s.current_day = min(day + 1, plan_days)
            s.selected_day = s.current_day
            s.week_offset = (s.current_day - 1) // 7
            st.rerun()


def rc_exercise_detail(ex, idx, total, day):
    s = st.session_state
    st.caption(f"Day {day} · Exercise {idx + 1} of {total}")
    st.markdown(f"### {ex.get('exercise_name_en', 'Exercise')}")
    if ex.get("exercise_name_zh"):
        st.caption(ex["exercise_name_zh"])
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Difficulty**  \n"
                    f"{DIFFICULTY_BADGE.get(str(ex.get('difficulty_level', '')).lower(), '—')}")
    with b:
        st.markdown(f"**Frequency**  \n{ex.get('frequency', '—')}")
    st.divider()
    st.markdown("**How to do it:**")
    st.write(ex.get("instructions_en", "Instructions not available."))
    if ex.get("instructions_zh"):
        with st.expander("中文說明"):
            st.write(ex["instructions_zh"])
    if ex.get("contraindications"):
        st.warning(f"⚠️ **Important:** {ex['contraindications']}")

    can_complete = day <= s.current_day and idx not in s.completed.get(day, set())
    a, b = st.columns(2)
    with a:
        if st.button("← Back to calendar", use_container_width=True):
            s.view_exercise = None
            st.rerun()
    with b:
        if can_complete and st.button("Mark as Complete ✓", type="primary", use_container_width=True):
            s.completed.setdefault(day, set()).add(idx)
            s.view_exercise = None
            st.rerun()


def rc_checkin():
    s = st.session_state
    day = s.current_day
    st.subheader(f"Day {day} Check-In 🎉")
    with st.form("checkin"):
        pain = st.radio("**How's your pain right now?**",
                        ["🟢 Better", "😐 About the same", "🔴 Worse"])
        st.text_area("Any notes? (optional)")
        if st.form_submit_button("Submit & Continue", type="primary", use_container_width=True):
            s.checkins[day] = pain
            s.current_day += 1
            s.selected_day = s.current_day
            s.week_offset = (s.current_day - 1) // 7
            s.show_checkin = False
            st.rerun()


def rc_progress(plan_days):
    s = st.session_state
    st.subheader("Your Progress")
    with st.container(border=True):
        st.markdown(f"**Day {s.current_day} of {plan_days}**")
        st.progress(min((s.current_day - 1) / plan_days, 1.0))
    if s.skipped:
        st.caption(f"Days skipped: {len(s.skipped)}")

    st.markdown("**Your pain trend**")
    if not s.checkins:
        st.caption("No check-ins yet — complete a day to start tracking.")
    else:
        with st.container(border=True):
            for d in sorted(s.checkins):
                st.markdown(f"Day {d} — {s.checkins[d]}")

    worse = sum(1 for v in s.checkins.values() if "Worse" in v)
    if worse:
        st.warning("⚠️ You've reported worse pain. We recommend seeing a "
                   "physiotherapist for an assessment.")
    else:
        st.info("💡 If pain worsens or doesn't improve, see a physiotherapist.")
    st.button("🩺 Talk to a physiotherapist", on_click=go_clinic,
              type="primary" if worse else "secondary", use_container_width=True)


# ============================================================
# CLINIC SCREEN
# ============================================================
def clinic_screen():
    s = st.session_state
    back_label = "← Back to Recovery Plan" if s.prev_mode == "recovery" else "← Back to Pain Summary"
    if st.button(back_label):
        s.mode = s.prev_mode
        st.rerun()

    st.subheader("Find a Physiotherapist")
    st.caption("Choose your area to see physiotherapy clinics near you.")
    with st.form("clinic_form"):
        area = st.selectbox("📍 Area:", list(REGIONS.keys()))
        insurer = st.selectbox("🏥 Your insurance (optional):",
                               ["Not sure / no insurance", "FWD", "AIA", "Bupa",
                                "Blue Cross", "Cigna"])
        go = st.form_submit_button("Show Clinics →", type="primary", use_container_width=True)
    if go:
        s.answers["clinic_area"], s.answers["clinic_insurer"] = area, insurer

    if "clinic_area" not in s.answers:
        return
    area, insurer = s.answers["clinic_area"], s.answers["clinic_insurer"]

    try:
        clinics = fetch_clinics(REGIONS[area])
    except Exception as e:
        st.error("Clinic search is temporarily unavailable. Please try again later.")
        if DEBUG:
            st.exception(e)
        return

    if not clinics:
        st.info(f"No clinics found in {area} yet. Try another area.")
        return

    if insurer != "Not sure / no insurance":
        clinics = sorted(clinics, key=lambda c: insurer not in c["insurers"])
    st.success(f"{len(clinics)} clinics in {area}")
    st.caption("⚠️ Insurance information is still being verified and may be "
               "incomplete. Please confirm coverage with the clinic before booking.")

    for c in clinics:
        name = c.get("clinic_name_en") or c.get("clinic_name_zh") or "Clinic"
        with st.container(border=True):
            st.markdown(f"**{name}**")
            if c.get("clinic_name_zh") and c.get("clinic_name_en"):
                st.caption(c["clinic_name_zh"])
            if c.get("district"):
                st.markdown(f"📍 {c['district']}")
            if c.get("address_en"):
                st.caption(c["address_en"])
            if c.get("phone"):
                st.markdown(f"📞 {c['phone']}")
            if c["insurers"]:
                tag = " ".join(f"`{i}`" for i in c["insurers"])
                st.markdown(f"🏥 Listed with: {tag}")


# ============================================================
# MAIN
# ============================================================
try:
    retriever, conditions_data = setup_engine()
except Exception as e:
    st.error("⚠️ The assessment service is temporarily unavailable.")
    if DEBUG:
        st.exception(e)
    st.stop()

A = st.session_state.answers
S = st.session_state

# ---------- RECOVERY MODE ----------
if S.mode == "recovery":
    cd = conditions_data.get(A.get("matched_condition"), {})
    plan_days, timeline_text = parse_plan_days(cd, A.get("severity", "Mild"))

    with st.sidebar:
        st.markdown("### Recovery Coach")
        st.caption(A.get("matched_condition") or "—")
        idx = {"factors": 0, "overview": 0, "calendar": 1, "progress": 2}.get(S.rc_screen, 0)
        nav = st.radio("Go to:", ["Plan Overview", "Calendar", "Progress"], index=idx)
        target = {"Plan Overview": "overview", "Calendar": "calendar", "Progress": "progress"}[nav]
        if S.factors and target != S.rc_screen and not (target == "overview" and S.rc_screen == "factors"):
            S.rc_screen, S.view_exercise, S.show_checkin = target, None, False
            st.rerun()
        st.divider()
        st.button("🩺 Find a Physiotherapist", on_click=go_clinic, use_container_width=True)
        if st.button("← Back to Pain Summary", use_container_width=True):
            S.mode = "assessment"
            st.rerun()
        if st.button("Start Over", use_container_width=True):
            restart()
            st.rerun()

    if S.view_exercise is not None:
        exs = cd.get("self_care_exercises", [])
        if 0 <= S.view_exercise < len(exs):
            rc_exercise_detail(exs[S.view_exercise], S.view_exercise, len(exs), S.view_day)
    elif S.show_checkin:
        rc_checkin()
    elif S.rc_screen == "factors" or not S.factors:
        rc_factors()
    elif S.rc_screen == "overview":
        rc_overview(cd, plan_days, timeline_text)
    elif S.rc_screen == "calendar":
        rc_calendar(cd, plan_days)
    elif S.rc_screen == "progress":
        rc_progress(plan_days)

# ---------- CLINIC MODE ----------
elif S.mode == "clinic":
    clinic_screen()

# ---------- ASSESSMENT MODE ----------
else:
    if 0 < S.step <= 8:
        st.progress(S.step / 8)

    if S.step == 0:
        st.markdown("<h1 style='text-align:center;'>🩺</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>Understand your pain,<br>find the right care.</h2>",
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#6B7280;'>了解痛症<br>找到最合適的護理方式</p>",
                    unsafe_allow_html=True)
        st.info("ℹ️ This tool provides information only and does not diagnose.")
        st.button("Let's Get Started →", on_click=next_step, type="primary")

    elif S.step == 1:
        ai_bubble("Where does it hurt? Look at the body map and select the area.")
        try:
            st.image("body_map.png", use_container_width=True)
        except Exception:
            pass
        st.button("← Back", on_click=prev_step)
        with st.form("body_form"):
            body = st.selectbox("Select the area:",
                                ["Lower back", "Upper back", "Neck", "Left shoulder", "Right shoulder",
                                 "Left elbow", "Right elbow", "Left wrist", "Right wrist",
                                 "Left hip", "Right hip", "Left knee", "Right knee",
                                 "Left ankle", "Right ankle", "Left foot", "Right foot"])
            custom = st.text_input("Or describe in your own words (optional):")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                A["body_area"] = custom.strip() or body
                A.pop("matched_condition", None)
                next_step()
                st.rerun()

    elif S.step == 2:
        ai_bubble(f"Got it — {A.get('body_area', '')} pain. Can you describe what the pain feels like?")
        st.button("← Back", on_click=prev_step)
        with st.form("quality_form"):
            st.caption("Select all that apply:")
            opts = {"🔥 Burning": "Burning", "⚡ Sharp": "Sharp", "😣 Dull ache": "Dull ache",
                    "📌 Pins & needles": "Pins & needles", "💪 Stiffness": "Stiffness",
                    "🔋 Weakness": "Weakness", "🔨 Throbbing": "Throbbing",
                    "🧊 Clicking or locking": "Clicking or locking"}
            chosen = [v for l, v in opts.items() if st.checkbox(l)]
            custom = st.text_input("Or describe in your own words...")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                if custom.strip():
                    chosen.append(custom.strip())
                A["pain_quality"] = ", ".join(chosen) or "unspecified"
                A.pop("matched_condition", None)
                next_step()
                st.rerun()

    elif S.step == 3:
        ai_bubble("When does the pain usually happen?")
        st.button("← Back", on_click=prev_step)
        with st.form("timing_form"):
            st.caption("Select all that apply:")
            opts = ["During activity", "In the morning", "At night", "When sitting",
                    "When walking", "After exercise", "Constant"]
            chosen = [o for o in opts if st.checkbox(o)]
            custom = st.text_input("Or describe in your own words...")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                if custom.strip():
                    chosen.append(custom.strip())
                A["pain_timing"] = ", ".join(chosen) or "unspecified"
                A.pop("matched_condition", None)
                next_step()
                st.rerun()

    elif S.step == 4:
        ai_bubble("How much is the pain affecting you right now?")
        st.button("← Back", on_click=prev_step)
        with st.form("sev_form"):
            sev = st.radio(
                "Choose the description that fits best:",
                ["🟢 Mild", "🟡 Moderate", "🔴 Severe"],
                captions=[
                    "Noticeable, but you can carry on with work and daily activities. "
                    "It doesn't wake you at night. (Roughly 1–3 out of 10)",
                    "It limits some activities, like sport, sitting for long, or stairs. "
                    "You may move differently or need painkillers. (Roughly 4–6 out of 10)",
                    "It stops you doing normal daily activities, makes it hard to sleep "
                    "or focus, or hurts even at rest. (Roughly 7–10 out of 10)",
                ])
            st.caption("Not sure between two levels? Choose the higher one.")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                A["severity"] = sev.split(" ", 1)[1]
                next_step()
                st.rerun()

    elif S.step == 5:
        ai_bubble(f"Got it — {A.get('severity', '').lower()} pain. When did this start?")
        st.button("← Back", on_click=prev_step)
        with st.form("dur_form"):
            dur = st.radio("Select one:", ["Just today", "Few days", "1-4 weeks", "Over a month"])
            custom = st.text_input("Or describe when it started...")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                A["duration"] = custom.strip() or dur
                next_step()
                st.rerun()

    elif S.step == 6:
        st.markdown("<h3 style='text-align:center;'>⚠️ Safety Check</h3>", unsafe_allow_html=True)
        st.caption("Please tell us if you're experiencing any of these. You can choose more than one.")
        st.button("← Back", on_click=prev_step)
        with st.form("flags_form"):
            opts = ["Numbness in both legs", "Loss of bladder or bowel control",
                    "Fever with pain", "Recent significant injury or fall",
                    "Unexplained weight loss", "History of cancer", "Sudden severe weakness"]
            chosen = [o for o in opts if st.checkbox(o)]
            st.divider()
            none_sel = st.checkbox("**None of the above**")
            if st.form_submit_button("Continue →", type="primary", use_container_width=True):
                A["red_flags"] = [] if none_sel else chosen
                next_step()
                st.rerun()

    elif S.step == 7:
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
            if st.form_submit_button("Get My Assessment →", type="primary", use_container_width=True):
                if custom.strip():
                    chosen.append(custom.strip())
                A["activity"] = ", ".join(chosen) or "unspecified"
                A.pop("matched_condition", None)
                next_step()
                st.rerun()

    elif S.step == 8:
        if A.get("red_flags"):
            st.markdown("<h1 style='text-align:center;'>🚨</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center;color:#EF4444;'>Get medical care right away</h2>",
                        unsafe_allow_html=True)
            st.write("Based on what you shared, your symptoms may need urgent attention.")
            st.error("**Recommended actions:**\n\n- Go to your nearest A&E\n"
                     "- Call 999 for severe symptoms\n- Don't wait — get checked")
            st.button("Start Over", on_click=restart)
        else:
            if "matched_condition" not in A:
                with st.spinner("Analysing your symptoms..."):
                    cond, conf, score = match_condition(retriever, conditions_data, A)
                A["matched_condition"], A["confidence"], A["score"] = cond, conf, score
            cond, conf = A["matched_condition"], A["confidence"]
            mild = A["severity"].lower() == "mild"
            se = {"mild": "🟢", "moderate": "🟡", "severe": "🔴"}.get(A["severity"].lower(), "🟡")

            st.subheader("Your Pain Summary")
            with st.container(border=True):
                st.markdown(f"**📍 Location**  \n{A['body_area']}")
                st.markdown(f"**⚡ Type**  \n{A['pain_quality']}")
                st.markdown(f"**📅 When**  \n{A['pain_timing']}")
                st.markdown(f"**⏱️ Duration**  \n{A['duration']}")
                st.markdown(f"**🎯 Severity**  \n{se} {A['severity']}")
                st.markdown(f"**🏃 Activity**  \n{A['activity']}")
                st.divider()
                if cond:
                    st.markdown(f"**📋 What this might be**  \n{cond} *({conf} confidence)*")
                    st.markdown(f"**🕒 Recovery estimate**  \n"
                                f"{get_recovery(conditions_data, cond, A['severity'])}")
                    nxt = "Try guided self-care" if mild else "See a physiotherapist"
                else:
                    st.markdown("**📋 What this might be**  \n"
                                "We couldn't confidently match your symptoms to a condition.")
                    nxt = "See a physiotherapist for an assessment"
                st.divider()
                st.markdown(f"**🩺 Suggested next step**  \n{nxt}")
            st.caption("ℹ️ Not a diagnosis. Clinical content pending physiotherapist review.")
            if DEBUG:
                st.caption(f"[debug] match score: {A.get('score', 0):.3f}")

            if not cond:
                st.button("Find a Physiotherapist", on_click=go_clinic,
                          type="primary", use_container_width=True)
            elif mild:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Start Recovery Plan", type="primary", use_container_width=True):
                        S.mode, S.rc_screen = "recovery", "factors"
                        st.rerun()
                with c2:
                    st.button("Find a Physiotherapist", on_click=go_clinic, use_container_width=True)
            else:
                st.button("Find a Physiotherapist", on_click=go_clinic,
                          type="primary", use_container_width=True)
                if st.button("Or start a self-care plan", use_container_width=True):
                    S.mode, S.rc_screen = "recovery", "factors"
                    st.rerun()

            st.button("Share Summary", use_container_width=True, disabled=True,
                      help="Coming soon")
            st.button("Start Over", on_click=restart)
