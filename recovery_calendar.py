#!/usr/bin/env python3
"""
Recovery Coach — Condition-Based Calendar (v2)
===============================================
- Plan length derived from each condition's recovery_timeline
- Weekly calendar view with navigation
- Factor collection (age, other conditions) shown as context only
  (NO automatic clinical adjustment — flagged for physio review)
- Daily exercise checklist with real progress tracking

Standalone test:
    streamlit run recovery_calendar.py
"""

import json
import glob
import re
import streamlit as st
from datetime import datetime, timedelta


# ============================================================
# LOAD CONDITIONS
# ============================================================
@st.cache_data
def load_all_conditions():
    conditions = {}
    for fname in sorted(glob.glob("MSK-*.json")):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("condition_name_en")
            if name:
                conditions[name] = data
        except Exception:
            continue
    return conditions


# ============================================================
# PLAN LENGTH — derived from recovery_timeline
# ============================================================
def parse_plan_days(condition_data, severity="Mild"):
    """
    Extract plan length in days from recovery_timeline text.
    e.g. "1-3 weeks" -> 21 days ; "6-12 weeks" -> 42 days (cap)
    Falls back to 14 days if unparseable.
    """
    tl = condition_data.get("recovery_timeline", {})
    text = tl.get(f"{severity.lower()}_estimate", "") or ""

    # Find week ranges like "1-3 weeks" or "6–12 weeks"
    wk = re.findall(r"(\d+)\s*[-–—]\s*(\d+)\s*week", text)
    if wk:
        low, high = int(wk[0][0]), int(wk[0][1])
        # Use the LOWER bound as the initial plan, then reassess
        weeks = max(low, 1)
        return min(weeks * 7, 42), text   # cap at 6 weeks

    # Single week value like "2 weeks"
    wk_single = re.findall(r"(\d+)\s*week", text)
    if wk_single:
        return min(int(wk_single[0]) * 7, 42), text

    # Day ranges
    dy = re.findall(r"(\d+)\s*[-–—]\s*(\d+)\s*day", text)
    if dy:
        return min(int(dy[0][1]), 42), text

    return 14, text   # default


DIFFICULTY_BADGE = {
    "beginner": "🟢 Beginner",
    "intermediate": "🟡 Intermediate",
    "advanced": "🔴 Advanced",
}


# ============================================================
# SESSION STATE
# ============================================================
def init_state(plan_days):
    if "current_day" not in st.session_state:
        st.session_state.current_day = 1
    if "week_offset" not in st.session_state:
        st.session_state.week_offset = 0
    if "completed" not in st.session_state:
        # {day: set of exercise indices completed}
        st.session_state.completed = {}
    if "checkins" not in st.session_state:
        # {day: "Better"/"Same"/"Worse"}
        st.session_state.checkins = {}
    if "factors" not in st.session_state:
        st.session_state.factors = {}


def day_status(day, current_day, completed, exercises_count):
    if day > current_day:
        return "upcoming"
    done = len(completed.get(day, set()))
    if day == current_day:
        return "today_done" if done >= exercises_count else "today"
    return "done" if done >= exercises_count else "missed"


STATUS_ICON = {
    "done": "✅",
    "today": "📍",
    "today_done": "✅",
    "missed": "⚪",
    "upcoming": "⚪",
}


# ============================================================
# SCREEN: Factor Collection
# ============================================================
def factors_screen():
    st.subheader("A few quick questions")
    st.caption("This helps your physiotherapist understand your context. "
               "Your plan is not automatically adjusted — discuss any "
               "modifications with a professional.")

    with st.form("factors_form"):
        age_range = st.selectbox("Age range:",
                                 ["18–29", "30–39", "40–49", "50–59", "60–69", "70+"])
        activity = st.selectbox("Current activity level:",
                                ["Very active (5+ days/week)",
                                 "Moderately active (2–4 days/week)",
                                 "Lightly active (1–2 days/week)",
                                 "Mostly sedentary"])
        other_conditions = st.multiselect(
            "Do you have any of these? (optional)",
            ["Diabetes", "Heart condition", "Osteoporosis", "Arthritis",
             "Pregnancy", "Recent surgery", "High blood pressure"])
        prior_physio = st.radio("Have you seen a physiotherapist for this before?",
                                ["No", "Yes, currently", "Yes, in the past"])

        submitted = st.form_submit_button("Start My Plan →", type="primary",
                                          use_container_width=True)

    if submitted:
        st.session_state.factors = {
            "age_range": age_range,
            "activity": activity,
            "other_conditions": other_conditions,
            "prior_physio": prior_physio,
        }
        return True
    return False


# ============================================================
# SCREEN: Plan Overview
# ============================================================
def plan_overview(condition_data, plan_days, timeline_text, severity):
    name = condition_data.get("condition_name_en", "your condition")
    name_zh = condition_data.get("condition_name_zh", "")
    exercises = condition_data.get("self_care_exercises", [])
    weeks = (plan_days + 6) // 7

    st.subheader("Your Recovery Plan")
    st.markdown(f"### {name}")
    if name_zh:
        st.caption(name_zh)

    with st.container(border=True):
        st.markdown(f"**📅 Plan length:** {plan_days} days ({weeks} weeks)")
        st.markdown(f"**🏃 Exercises:** {len(exercises)} per day")
        if timeline_text:
            st.markdown(f"**🕒 Expected recovery:** {timeline_text}")

    # Factors as context (no auto-adjustment)
    f = st.session_state.get("factors", {})
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
        diff = DIFFICULTY_BADGE.get(str(ex.get("difficulty_level", "")).lower(), "")
        st.markdown(f"- {ex.get('exercise_name_en','Exercise')} — "
                    f"{ex.get('frequency','')} {diff}")

    st.write("")
    st.info("💡 Consistency matters more than intensity. Stop any exercise that "
            "causes sharp pain. If symptoms worsen or don't improve, see a physiotherapist.")

    # Safety note for relevant factors
    f = st.session_state.get("factors", {})
    if f.get("other_conditions") or f.get("age_range") in ["60–69", "70+"]:
        st.warning("⚠️ Because of the health factors you shared, we recommend "
                   "discussing this plan with a physiotherapist before starting.")


# ============================================================
# SCREEN: Weekly Calendar
# ============================================================
def calendar_view(condition_data, plan_days):
    exercises = condition_data.get("self_care_exercises", [])
    n_ex = len(exercises)
    current_day = st.session_state.current_day
    completed = st.session_state.completed
    offset = st.session_state.week_offset

    total_weeks = (plan_days + 6) // 7
    week_num = offset + 1
    start_day = offset * 7 + 1
    end_day = min(start_day + 6, plan_days)

    # Week navigation
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("← Prev", disabled=(offset == 0), use_container_width=True):
            st.session_state.week_offset -= 1
            st.rerun()
    with c2:
        st.markdown(f"<div style='text-align:center; font-weight:600; padding-top:6px;'>"
                    f"Week {week_num} of {total_weeks}</div>", unsafe_allow_html=True)
    with c3:
        if st.button("Next →", disabled=(week_num >= total_weeks), use_container_width=True):
            st.session_state.week_offset += 1
            st.rerun()

    st.write("")

    # Calendar row
    days = list(range(start_day, end_day + 1))
    cols = st.columns(7)
    for i, col in enumerate(cols):
        with col:
            if i < len(days):
                d = days[i]
                status = day_status(d, current_day, completed, n_ex)
                icon = STATUS_ICON[status]
                label = "Today" if d == current_day else ""
                st.markdown(f"<div style='text-align:center;'>"
                            f"<div style='font-size:12px; color:#6B7280;'>Day {d}</div>"
                            f"<div style='font-size:28px;'>{icon}</div>"
                            f"<div style='font-size:11px; color:#2563EB;'>{label}</div>"
                            f"</div>", unsafe_allow_html=True)
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)

    st.divider()

    # Today's exercises
    st.markdown(f"### Day {current_day} — Today's exercises")
    done_set = completed.get(current_day, set())

    if not exercises:
        st.warning("No exercises found for this condition.")
        return

    for idx, ex in enumerate(exercises):
        is_done = idx in done_set
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{'✅ ' if is_done else ''}{ex.get('exercise_name_en','Exercise')}**")
                st.caption(f"{ex.get('frequency','')} · "
                           f"{DIFFICULTY_BADGE.get(str(ex.get('difficulty_level','')).lower(),'')}")
            with c2:
                if not is_done:
                    if st.button("Do it", key=f"do_{current_day}_{idx}",
                                 use_container_width=True):
                        st.session_state.view_exercise = idx
                        st.rerun()

            if is_done:
                st.caption("Completed")

    # Progress for today
    st.write("")
    st.progress(len(done_set) / n_ex if n_ex else 0)
    st.caption(f"{len(done_set)} of {n_ex} exercises done today")

    # Complete day / check-in
    if len(done_set) >= n_ex and current_day not in st.session_state.checkins:
        st.success("All exercises done! Ready for your daily check-in.")
        if st.button("Daily Check-In →", type="primary", use_container_width=True):
            st.session_state.show_checkin = True
            st.rerun()


# ============================================================
# SCREEN: Exercise Detail
# ============================================================
def exercise_detail(ex, idx, total, day):
    st.caption(f"Day {day} · Exercise {idx + 1} of {total}")
    st.markdown(f"### {ex.get('exercise_name_en','Exercise')}")
    if ex.get("exercise_name_zh"):
        st.caption(ex["exercise_name_zh"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Difficulty**  \n"
                    f"{DIFFICULTY_BADGE.get(str(ex.get('difficulty_level','')).lower(),'—')}")
    with c2:
        st.markdown(f"**Frequency**  \n{ex.get('frequency','—')}")

    st.divider()
    with st.container(border=True):
        st.markdown("<div style='text-align:center; padding:30px; color:#6B7280;'>"
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
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to plan", use_container_width=True):
            st.session_state.view_exercise = None
            st.rerun()
    with c2:
        if st.button("Mark as Complete ✓", type="primary", use_container_width=True):
            d = st.session_state.current_day
            st.session_state.completed.setdefault(d, set()).add(idx)
            st.session_state.view_exercise = None
            st.rerun()


# ============================================================
# SCREEN: Daily Check-In
# ============================================================
def checkin_screen():
    day = st.session_state.current_day
    st.subheader(f"Day {day} Check-In 🎉")
    st.write("Great work completing today's exercises!")

    with st.form("checkin"):
        pain = st.radio("**How's your pain right now?**",
                        ["🟢 Better", "😐 About the same", "🔴 Worse"])
        notes = st.text_area("Any notes? (optional)",
                             placeholder="e.g. Felt tightness during the second exercise")
        if st.form_submit_button("Submit & Continue", type="primary",
                                 use_container_width=True):
            st.session_state.checkins[day] = pain
            st.session_state.current_day += 1
            st.session_state.show_checkin = False
            # auto-advance week view if needed
            st.session_state.week_offset = (st.session_state.current_day - 1) // 7
            st.rerun()

    if "🔴 Worse" in str(st.session_state.checkins.values()):
        st.warning("⚠️ If your pain is getting worse, consider seeing a physiotherapist.")


# ============================================================
# SCREEN: Progress
# ============================================================
def progress_view(plan_days):
    current = st.session_state.current_day
    checkins = st.session_state.checkins
    completed = st.session_state.completed

    st.subheader("Your Progress")

    with st.container(border=True):
        pct = min(int(((current - 1) / plan_days) * 100), 100)
        st.markdown(f"**Day {current} of {plan_days}**")
        st.progress(min((current - 1) / plan_days, 1.0))
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
    st.info("💡 Keep going — consistency matters more than intensity. "
            "If pain worsens or doesn't improve, see a physiotherapist.")


# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(page_title="Recovery Coach", page_icon="💪", layout="centered")

    conditions = load_all_conditions()
    if not conditions:
        st.error("No MSK-*.json files found in this folder.")
        return

    st.sidebar.title("Recovery Coach")
    chosen = st.sidebar.selectbox("Condition:", sorted(conditions.keys()))
    severity = st.sidebar.selectbox("Severity:", ["Mild", "Moderate", "Severe"])
    condition_data = conditions[chosen]

    plan_days, timeline_text = parse_plan_days(condition_data, severity)
    init_state(plan_days)

    st.sidebar.caption(f"Plan: {plan_days} days")
    st.sidebar.caption(f"{len(conditions)} conditions loaded")
    if st.sidebar.button("Reset plan"):
        for k in ["current_day", "week_offset", "completed", "checkins",
                  "factors", "view_exercise", "show_checkin"]:
            st.session_state.pop(k, None)
        st.rerun()

    view = st.sidebar.radio("Screen:",
                            ["Factors", "Plan Overview", "Calendar", "Progress"])

    # Exercise detail overrides
    if st.session_state.get("view_exercise") is not None and view == "Calendar":
        exercises = condition_data.get("self_care_exercises", [])
        idx = st.session_state.view_exercise
        if 0 <= idx < len(exercises):
            exercise_detail(exercises[idx], idx, len(exercises),
                            st.session_state.current_day)
            return

    if st.session_state.get("show_checkin") and view == "Calendar":
        checkin_screen()
        return

    if view == "Factors":
        if factors_screen():
            st.success("Saved! Switch to 'Plan Overview' to see your plan.")
    elif view == "Plan Overview":
        plan_overview(condition_data, plan_days, timeline_text, severity)
    elif view == "Calendar":
        calendar_view(condition_data, plan_days)
    elif view == "Progress":
        progress_view(plan_days)


if __name__ == "__main__":
    main()
