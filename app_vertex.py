#!/usr/bin/env python3
"""
AI Physio Triage Tool - Beta v0.3 (English + Traditional Chinese)
=================================================================
New in v0.3:
  - Full English / Traditional Chinese interface
  - Language auto-detected from the browser, with an EN | 中文 toggle on every screen
  - Clinical content (conditions, exercises) read from the *_zh fields in the JSONs,
    falling back to English where a Chinese field is missing
  - Draft banner in Chinese mode: Chinese clinical content pending physio review
  - Multilingual embeddings so Chinese free-text input can be matched
  - Answers are stored in English internally, so routing logic is unchanged

LOCAL RUN:
    conda activate physio
    python -m streamlit run app_vertex.py
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

st.set_page_config(page_title="Pain Assessment 痛症評估", page_icon="🩺", layout="centered")

# ============================================================
# CONFIG
# ============================================================
GCP_PROJECT_ID = "physio-triage-tool"
GCP_REGION = "asia-east2"
MIN_MATCH_SCORE = 0.55     # calibrate with ?debug=1
HIGH_MATCH_SCORE = 0.70
DEBUG = st.query_params.get("debug") == "1"


def _load_vertex_credentials():
    local_key = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vertex-key.json")
    if os.path.exists(local_key):
        return _sa.Credentials.from_service_account_file(local_key)
    return _sa.Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]))


_VERTEX_CREDS = _load_vertex_credentials()

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #FFFFFF; }
    h1, h2, h3 { color: #111827 !important; }
    .ai-bubble {
        background-color: #F9FAFB; border-radius: 16px;
        padding: 16px 20px; margin-bottom: 16px;
        font-size: 18px; color: #111827; line-height: 1.5;
    }
    .stButton > button[kind="primary"], .stFormSubmitButton > button {
        background-color: #2563EB; color: white; border: none;
        border-radius: 8px; font-weight: 600;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
    .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LANGUAGE
# ============================================================
def detect_browser_lang():
    try:
        loc = st.context.locale
        if loc and str(loc).lower().startswith("zh"):
            return "zh"
    except Exception:
        pass
    return "en"


if "lang" not in st.session_state:
    st.session_state.lang = detect_browser_lang()


def zh():
    return st.session_state.lang == "zh"


STR = {
    "en": {
        "title": "Understand your pain,<br>find the right care.",
        "disclaimer": "ℹ️ This tool provides information only and does not diagnose.",
        "start": "Let's Get Started →",
        "back": "← Back",
        "continue": "Continue →",
        "q_body": "Where does it hurt? Look at the body map and select the area.",
        "select_area": "Select the area:",
        "own_words_opt": "Or describe in your own words (optional):",
        "own_words": "Or describe in your own words...",
        "q_quality": "Got it — {body} pain. Can you describe what the pain feels like?",
        "select_all": "Select all that apply:",
        "q_timing": "When does the pain usually happen?",
        "q_sev": "How much is the pain affecting you right now?",
        "sev_choose": "Choose the description that fits best:",
        "sev_caps": [
            "Noticeable, but you can carry on with work and daily activities. It doesn't wake you at night. (Roughly 1–3 out of 10)",
            "It limits some activities, like sport, sitting for long, or stairs. You may move differently or need painkillers. (Roughly 4–6 out of 10)",
            "It stops you doing normal daily activities, makes it hard to sleep or focus, or hurts even at rest. (Roughly 7–10 out of 10)",
        ],
        "sev_unsure": "Not sure between two levels? Choose the higher one.",
        "q_dur": "Got it — {sev} pain. When did this start?",
        "select_one": "Select one:",
        "dur_own": "Or describe when it started...",
        "safety": "⚠️ Safety Check",
        "safety_cap": "Please tell us if you're experiencing any of these. You can choose more than one.",
        "none_above": "**None of the above**",
        "q_activity": "Last question — what activities do you do regularly?",
        "activity_own": "Or describe your activity...",
        "get_assessment": "Get My Assessment →",
        "urgent_title": "Get medical care right away",
        "urgent_text": "Based on what you shared, your symptoms may need urgent attention.",
        "urgent_actions": "**Recommended actions:**\n\n- Go to your nearest A&E\n- Call 999 for severe symptoms\n- Don't wait — get checked",
        "start_over": "Start Over",
        "analysing": "Analysing your symptoms...",
        "summary": "Your Pain Summary",
        "f_location": "📍 Location", "f_type": "⚡ Type", "f_when": "📅 When",
        "f_duration": "⏱️ Duration", "f_severity": "🎯 Severity", "f_activity": "🏃 Activity",
        "f_might": "📋 What this might be", "f_recovery": "🕒 Recovery estimate",
        "f_next": "🩺 Suggested next step",
        "conf": {"high": "high confidence", "medium": "medium confidence"},
        "no_match": "We couldn't confidently match your symptoms to a condition.",
        "next_self": "Try guided self-care",
        "next_physio": "See a physiotherapist",
        "next_assess": "See a physiotherapist for an assessment",
        "not_dx": "ℹ️ Not a diagnosis. Clinical content pending physiotherapist review.",
        "start_plan": "Start Recovery Plan",
        "find_physio": "Find a Physiotherapist",
        "or_self": "Or start a self-care plan",
        "share": "Share Summary",
        "soon": "Coming soon",
        "factors_title": "A few quick questions",
        "factors_cap": "This helps us fit your plan into your routine and gives your physiotherapist useful context. Exercises and repetitions are not changed — discuss modifications with a professional.",
        "age": "Age range:",
        "work": "What best describes your work or daily routine?",
        "sitting": "How many hours do you sit on a typical day?",
        "pref": "When is it easiest for you to do exercises?",
        "act": "Current activity level:",
        "other": "Do you have any of these? (optional)",
        "prior": "Seen a physiotherapist for this before?",
        "create_plan": "Create My Plan →",
        "plan_title": "Your Recovery Plan",
        "plan_len": "📅 Plan length:", "days_weeks": "{d} days ({w} weeks)",
        "ex_per_day": "🏃 Exercises:", "per_day": "{n} per day",
        "expected": "🕒 Expected recovery:",
        "sched": "Your daily schedule",
        "reason_desk": "Because you sit for long periods, we've placed your exercises into breaks so you move regularly through the day.",
        "reason_stand": "We've placed your exercises at times that fit around a standing job.",
        "reason_manual": "We've placed your exercises outside your working hours.",
        "reason_default": "We've arranged your exercises around the time you said suits you best.",
        "tip_desk": "💡 Tip: standing up and moving for a minute or two between long periods of sitting can help. Ask your physiotherapist how often is right for you.",
        "context": "Your context (shared with your physiotherapist)",
        "ctx": {"age_range": "Age range", "work_type": "Routine", "sitting_hours": "Sitting per day",
                "activity": "Activity", "prior_physio": "Prior physio", "other": "Other conditions"},
        "none_reported": "None reported",
        "warn_factors": "⚠️ Because of the health factors you shared, we recommend discussing this plan with a physiotherapist before starting.",
        "stop_sharp": "💡 Stop any exercise that causes sharp pain.",
        "view_cal": "View My Calendar →",
        "prev": "← Prev", "next": "Next →", "week": "Week {a} of {b}",
        "tap_day": "Tap a day to see its exercises.",
        "day": "Day {d}", "today": "Today", "upcoming": "Upcoming", "past": "Past day",
        "skipped_note": "You skipped this day. You can still do the exercises below if you'd like.",
        "preview_note": "Preview only — you can do these exercises when you reach this day.",
        "view": "View", "do_it": "Do it",
        "done_of": "{a} of {b} exercises done",
        "all_done": "All exercises done! Ready for your daily check-in.",
        "checkin_btn": "Daily Check-In →",
        "skip_today": "⏭️ Skip today and move to the next day",
        "ex_of": "Day {d} · Exercise {a} of {b}",
        "difficulty": "Difficulty", "frequency": "Frequency",
        "how_to": "How to do it:", "other_lang": "English instructions",
        "important": "⚠️ **Important:**",
        "back_cal": "← Back to calendar", "mark_done": "Mark as Complete ✓",
        "checkin_title": "Day {d} Check-In 🎉",
        "pain_now": "**How's your pain right now?**",
        "notes": "Any notes? (optional)", "submit": "Submit & Continue",
        "progress": "Your Progress", "day_of": "Day {a} of {b}",
        "skipped_days": "Days skipped: {n}", "trend": "**Your pain trend**",
        "no_checkins": "No check-ins yet — complete a day to start tracking.",
        "worse_warn": "⚠️ You've reported worse pain. We recommend seeing a physiotherapist for an assessment.",
        "see_info": "💡 If pain worsens or doesn't improve, see a physiotherapist.",
        "talk_physio": "🩺 Talk to a physiotherapist",
        "rc": "### Recovery Coach", "go_to": "Go to:",
        "nav": ["Plan Overview", "Calendar", "Progress"],
        "back_summary": "← Back to Pain Summary", "back_plan": "← Back to Recovery Plan",
        "find_title": "Find a Physiotherapist",
        "choose_area": "Choose your area to see physiotherapy clinics near you.",
        "area": "📍 Area:", "insurance": "🏥 Your insurance (optional):",
        "show_clinics": "Show Clinics →",
        "unavailable": "Clinic search is temporarily unavailable. Please try again later.",
        "none_found": "No clinics found in {a} yet. Try another area.",
        "n_clinics": "{n} clinics in {a}",
        "ins_note": "⚠️ Insurance information is still being verified and may be incomplete. Please confirm coverage with the clinic before booking.",
        "listed_with": "🏥 Listed with:",
        "service_down": "⚠️ The assessment service is temporarily unavailable.",
        "draft": "",
    },
    "zh": {
        "title": "了解你的痛症，<br>找到合適的護理。",
        "disclaimer": "ℹ️ 本工具只提供資訊，並不作診斷。",
        "start": "開始評估 →",
        "back": "← 返回",
        "continue": "繼續 →",
        "q_body": "哪裡痛？請參考人體圖並選擇位置。",
        "select_area": "選擇位置：",
        "own_words_opt": "或用你自己的話描述（選填）：",
        "own_words": "或用你自己的話描述……",
        "q_quality": "明白，{body}痛。可以描述一下痛的感覺嗎？",
        "select_all": "可選多項：",
        "q_timing": "痛楚通常在甚麼時候出現？",
        "q_sev": "痛楚現時對你有多大影響？",
        "sev_choose": "請選擇最貼切的描述：",
        "sev_caps": [
            "感覺得到，但仍可如常工作和進行日常活動，晚上不會痛醒。（約 10 分中的 1–3 分）",
            "影響部分活動，例如運動、久坐或上落樓梯；你可能要改變動作或需要服止痛藥。（約 10 分中的 4–6 分）",
            "令你無法進行日常活動、難以入睡或集中精神，或休息時也會痛。（約 10 分中的 7–10 分）",
        ],
        "sev_unsure": "在兩個程度之間猶豫？請選較高的一個。",
        "q_dur": "明白，{sev}程度的痛楚。痛楚是何時開始的？",
        "select_one": "請選一項：",
        "dur_own": "或描述痛楚何時開始……",
        "safety": "⚠️ 安全檢查",
        "safety_cap": "請告訴我們你是否有以下情況，可選多項。",
        "none_above": "**以上皆沒有**",
        "q_activity": "最後一條問題：你平時經常做甚麼活動？",
        "activity_own": "或描述你的活動……",
        "get_assessment": "查看評估結果 →",
        "urgent_title": "請立即求醫",
        "urgent_text": "根據你提供的資料，你的症狀可能需要緊急醫療處理。",
        "urgent_actions": "**建議行動：**\n\n- 前往最近的急症室\n- 如症狀嚴重，請致電 999\n- 不要拖延，盡快求診",
        "start_over": "重新開始",
        "analysing": "正在分析你的症狀……",
        "summary": "你的痛症摘要",
        "f_location": "📍 位置", "f_type": "⚡ 類型", "f_when": "📅 何時出現",
        "f_duration": "⏱️ 持續時間", "f_severity": "🎯 嚴重程度", "f_activity": "🏃 活動",
        "f_might": "📋 可能的情況", "f_recovery": "🕒 預計康復時間",
        "f_next": "🩺 建議下一步",
        "conf": {"high": "高信心", "medium": "中等信心"},
        "no_match": "我們未能有把握地將你的症狀配對到某個情況。",
        "next_self": "嘗試自我護理計劃",
        "next_physio": "求診物理治療師",
        "next_assess": "請求診物理治療師作評估",
        "not_dx": "ℹ️ 這不是診斷。臨床內容尚待物理治療師審核。",
        "start_plan": "開始康復計劃",
        "find_physio": "尋找物理治療師",
        "or_self": "或開始自我護理計劃",
        "share": "分享摘要",
        "soon": "即將推出",
        "factors_title": "幾條簡單問題",
        "factors_cap": "這些資料幫助我們把計劃配合你的日常作息，並讓物理治療師了解你的情況。運動內容和次數不會因此改變，如需調整請諮詢專業人士。",
        "age": "年齡組別：",
        "work": "以下哪項最能描述你的工作或日常作息？",
        "sitting": "你平日大約坐多少小時？",
        "pref": "你最方便在甚麼時候做運動？",
        "act": "現時活動量：",
        "other": "你有以下任何情況嗎？（選填）",
        "prior": "以前曾因這個問題看過物理治療師嗎？",
        "create_plan": "建立我的計劃 →",
        "plan_title": "你的康復計劃",
        "plan_len": "📅 計劃長度：", "days_weeks": "{d} 天（{w} 星期）",
        "ex_per_day": "🏃 運動：", "per_day": "每天 {n} 項",
        "expected": "🕒 預計康復時間：",
        "sched": "你的每日時間表",
        "reason_desk": "由於你需要長時間坐著，我們把運動分散到休息時段，讓你全日都定時活動。",
        "reason_stand": "我們把運動安排在配合站立工作的時間。",
        "reason_manual": "我們把運動安排在你的工作時間以外。",
        "reason_default": "我們按你最方便的時間安排了運動。",
        "tip_desk": "💡 提示：長時間坐著之間，站起來活動一兩分鐘會有幫助。可向物理治療師查詢適合你的頻率。",
        "context": "你的背景資料（會與物理治療師分享）",
        "ctx": {"age_range": "年齡組別", "work_type": "日常作息", "sitting_hours": "每日坐著時間",
                "activity": "活動量", "prior_physio": "曾否看物理治療", "other": "其他健康狀況"},
        "none_reported": "沒有",
        "warn_factors": "⚠️ 根據你提供的健康資料，建議你在開始前先與物理治療師商討這個計劃。",
        "stop_sharp": "💡 如任何運動引起劇痛，請立即停止。",
        "view_cal": "查看我的日曆 →",
        "prev": "← 上週", "next": "下週 →", "week": "第 {a} 週，共 {b} 週",
        "tap_day": "點選日子以查看當天的運動。",
        "day": "第{d}天", "today": "今天", "upcoming": "未來", "past": "過去",
        "skipped_note": "你跳過了這一天。如你願意，仍可完成以下運動。",
        "preview_note": "只供預覽，到了這一天便可以進行這些運動。",
        "view": "查看", "do_it": "開始",
        "done_of": "已完成 {a} / {b} 項運動",
        "all_done": "全部運動已完成！可以進行每日記錄。",
        "checkin_btn": "每日記錄 →",
        "skip_today": "⏭️ 跳過今天，進入下一天",
        "ex_of": "第{d}天 · 第 {a} 項運動，共 {b} 項",
        "difficulty": "難度", "frequency": "次數",
        "how_to": "做法：", "other_lang": "英文說明 English",
        "important": "⚠️ **注意：**",
        "back_cal": "← 返回日曆", "mark_done": "標示為已完成 ✓",
        "checkin_title": "第{d}天記錄 🎉",
        "pain_now": "**你現在的痛楚如何？**",
        "notes": "備註（選填）", "submit": "提交並繼續",
        "progress": "你的進度", "day_of": "第 {a} 天，共 {b} 天",
        "skipped_days": "已跳過日數：{n}", "trend": "**你的痛楚趨勢**",
        "no_checkins": "未有記錄，完成一天後便會開始追蹤。",
        "worse_warn": "⚠️ 你表示痛楚加劇，建議你求診物理治療師作評估。",
        "see_info": "💡 如痛楚加劇或沒有改善，請求診物理治療師。",
        "talk_physio": "🩺 聯絡物理治療師",
        "rc": "### 康復教練", "go_to": "前往：",
        "nav": ["計劃概覽", "日曆", "進度"],
        "back_summary": "← 返回痛症摘要", "back_plan": "← 返回康復計劃",
        "find_title": "尋找物理治療師",
        "choose_area": "選擇地區以查看附近的物理治療診所。",
        "area": "📍 地區：", "insurance": "🏥 你的保險（選填）：",
        "show_clinics": "顯示診所 →",
        "unavailable": "診所搜尋暫時未能使用，請稍後再試。",
        "none_found": "{a}暫時未有診所資料，請嘗試其他地區。",
        "n_clinics": "{a}共 {n}間診所",
        "ins_note": "⚠️ 保險資料仍在核實中，可能不完整。預約前請向診所確認是否受保。",
        "listed_with": "🏥 列於：",
        "service_down": "⚠️ 評估服務暫時未能使用。",
        "draft": "⚠️ 中文內容為初稿，尚待物理治療師審核。Chinese content is a draft pending physiotherapist review.",
    },
}

# English option value -> Traditional Chinese label (values stay English internally)
OPT_ZH = {
    # body areas
    "Lower back": "下背", "Upper back": "上背", "Neck": "頸", "Left shoulder": "左肩",
    "Right shoulder": "右肩", "Left elbow": "左手肘", "Right elbow": "右手肘",
    "Left wrist": "左手腕", "Right wrist": "右手腕", "Left hip": "左髖", "Right hip": "右髖",
    "Left knee": "左膝", "Right knee": "右膝", "Left ankle": "左腳踝", "Right ankle": "右腳踝",
    "Left foot": "左腳", "Right foot": "右腳",
    # pain quality
    "Burning": "灼熱", "Sharp": "刺痛", "Dull ache": "隱隱作痛", "Pins & needles": "針刺麻痺",
    "Stiffness": "僵硬", "Weakness": "無力", "Throbbing": "抽痛", "Clicking or locking": "有聲響或卡住",
    # timing
    "During activity": "活動時", "In the morning": "早上", "At night": "晚上",
    "When sitting": "坐著時", "When walking": "走路時", "After exercise": "運動後", "Constant": "持續",
    # severity
    "Mild": "輕微", "Moderate": "中等", "Severe": "嚴重",
    # duration
    "Just today": "今天才開始", "Few days": "幾天", "1-4 weeks": "1–4 星期", "Over a month": "超過一個月",
    # red flags
    "Numbness in both legs": "雙腳麻痺", "Loss of bladder or bowel control": "大小便失禁",
    "Fever with pain": "痛楚伴隨發燒", "Recent significant injury or fall": "最近有嚴重受傷或跌倒",
    "Unexplained weight loss": "原因不明的體重下降", "History of cancer": "曾患癌症",
    "Sudden severe weakness": "突然嚴重無力",
    # activities
    "Running": "跑步", "Weightlifting": "舉重", "Cycling": "踏單車", "Yoga/Pilates": "瑜伽／普拉提",
    "Racquet sports": "球拍類運動", "Team sports": "團隊球類運動", "Desk work": "辦公室工作",
    "Mostly sedentary": "大部分時間坐著",
    # factors
    "Desk / office work": "辦公室工作", "Standing / on my feet": "需要長時間站立",
    "Physical / manual work": "體力勞動工作", "Student": "學生", "Retired / at home": "退休／在家",
    "Mixed / varies": "混合／不固定",
    "Less than 4 hours": "少於 4 小時", "4–6 hours": "4–6 小時", "6–8 hours": "6–8 小時",
    "More than 8 hours": "多於 8 小時",
    "Spread through the day (short breaks)": "分散在一天內（短暫休息時）",
    "Morning, before work or school": "早上，上班或上學前",
    "Lunch break": "午飯時間", "Evening, after work": "晚上，下班後",
    "Very active (5+ days/week)": "非常活躍（每週 5 天或以上）",
    "Moderately active (2–4 days/week)": "中度活躍（每週 2–4 天）",
    "Lightly active (1–2 days/week)": "輕度活躍（每週 1–2 天）",
    "Diabetes": "糖尿病", "Heart condition": "心臟病", "Osteoporosis": "骨質疏鬆",
    "Arthritis": "關節炎", "Pregnancy": "懷孕", "Recent surgery": "最近做過手術",
    "High blood pressure": "高血壓",
    "No": "沒有", "Yes, currently": "有，現正接受治療", "Yes, in the past": "有，以前曾經",
    # schedule slots
    "☕ Mid-morning break": "☕ 上午休息", "🍱 Lunch break": "🍱 午飯時間",
    "🕒 Mid-afternoon break": "🕒 下午休息", "🌙 Evening": "🌙 晚上", "🌅 Morning": "🌅 早上",
    # check-in
    "Better": "好轉", "About the same": "差不多", "Worse": "轉差",
    # regions / insurance
    "Hong Kong Island": "港島", "Kowloon": "九龍", "New Territories": "新界",
    "Outlying Islands": "離島", "Not sure / no insurance": "不確定／沒有保險",
    # difficulty
    "beginner": "初級", "intermediate": "中級", "advanced": "高級",
}


def T(key, **kw):
    s = STR[st.session_state.lang].get(key, STR["en"].get(key, key))
    return s.format(**kw) if kw and isinstance(s, str) else s


def tr(value):
    """Translate an English option value for display."""
    if zh():
        return OPT_ZH.get(value, value)
    return value


def tr_list(text):
    """Translate a comma-joined answer (custom free text is shown as typed)."""
    if not text or text == "unspecified":
        return "—"
    return "、".join(tr(p.strip()) for p in text.split(",")) if zh() else text


def loc(d, base):
    """Read a localised field from condition/exercise JSON with English fallback."""
    if zh() and d.get(f"{base}_zh"):
        return d[f"{base}_zh"]
    return d.get(f"{base}_en") or d.get(base) or ""


DIFF_ICON = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}


def diff_badge(level):
    level = str(level or "").lower()
    if level not in DIFF_ICON:
        return ""
    return f"{DIFF_ICON[level]} {tr(level) if zh() else level.capitalize()}"


def lang_toggle():
    _, c = st.columns([4, 1])
    with c:
        st.radio("Language", ["en", "zh"], key="lang", horizontal=True,
                 format_func=lambda x: "EN" if x == "en" else "中文",
                 label_visibility="collapsed")
    if zh():
        st.warning(T("draft"))


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
REGIONS = {"Hong Kong Island": "HK_Island", "Kowloon": "Kowloon",
           "New Territories": "NT", "Outlying Islands": "Outlying_Islands"}
BODY_OPTS = ["Lower back", "Upper back", "Neck", "Left shoulder", "Right shoulder",
             "Left elbow", "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip",
             "Left knee", "Right knee", "Left ankle", "Right ankle", "Left foot", "Right foot"]


# ============================================================
# ENGINE
# ============================================================
def _embed_model():
    """Prefer a multilingual model so Chinese input can be matched."""
    for name in ["text-multilingual-embedding-002", "text-embedding-004"]:
        try:
            m = VertexTextEmbedding(model_name=name, project=GCP_PROJECT_ID,
                                    location=GCP_REGION, credentials=_VERTEX_CREDS)
            m.get_text_embedding("test")
            return m, name
        except Exception:
            continue
    raise RuntimeError("No Vertex embedding model available")


@st.cache_resource
def setup_engine():
    Settings.llm = Vertex(model="gemini-1.5-flash-002", project=GCP_PROJECT_ID,
                          location=GCP_REGION, temperature=0.2, max_tokens=1024,
                          credentials=_VERTEX_CREDS)
    Settings.embed_model, embed_name = _embed_model()

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
        parts = [f"Condition: {name} ({data.get('condition_name_zh', '')})",
                 f"Body area: {', '.join(data.get('body_area', []))}"]
        for key in ["patient_language_aliases", "patient_language_aliases_zh"]:
            aliases = data.get(key, [])
            if aliases:
                parts.append("Also described as: " + ", ".join(str(a) for a in aliases))
        sym_en, sym_zh = [], []
        for s in data.get("symptoms", []):
            if isinstance(s, dict):
                sym_en.append(s.get("description_en", ""))
                sym_zh.append(s.get("description_zh", ""))
            else:
                sym_en.append(str(s))
        parts.append("Symptoms: " + "; ".join(x for x in sym_en if x))
        if any(sym_zh):
            parts.append("症狀：" + "；".join(x for x in sym_zh if x))
        causes = data.get("possible_causes", [])
        if causes:
            parts.append("Possible causes: " + "; ".join(
                c.get("cause_en", "") if isinstance(c, dict) else str(c) for c in causes))
        docs.append(Document(text="\n".join(parts), metadata={"condition_name": name}))
    index = VectorStoreIndex.from_documents(docs)
    return index.as_retriever(similarity_top_k=3), conditions_data, embed_name


def body_consistent(body_area, condition_data):
    sel = body_area.lower().replace("left ", "").replace("right ", "").strip()
    keywords = BODY_KEYWORDS.get(sel)
    if not keywords:
        return True
    cond_area = " ".join(condition_data.get("body_area", [])).lower()
    return any(k in cond_area for k in keywords)


def match_condition(retriever, conditions_data, A):
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
        return name, ("high" if score >= HIGH_MATCH_SCORE else "medium"), score
    return None, "low", (nodes[0].score if nodes else 0)


def get_recovery(cd, severity):
    tl = cd.get("recovery_timeline", {})
    key = f"{severity.lower()}_estimate"
    if zh() and tl.get(f"{key}_zh"):
        return tl[f"{key}_zh"]
    return tl.get(key, "—")


def parse_plan_days(cd, severity="Mild"):
    text = cd.get("recovery_timeline", {}).get(f"{severity.lower()}_estimate", "") or ""
    wk = re.findall(r"(\d+)\s*[-–—]\s*(\d+)\s*week", text)
    if wk:
        return min(max(int(wk[0][0]), 1) * 7, 42)
    wk1 = re.findall(r"(\d+)\s*week", text)
    if wk1:
        return min(int(wk1[0]) * 7, 42)
    dy = re.findall(r"(\d+)\s*[-–—]\s*(\d+)\s*day", text)
    if dy:
        return min(int(dy[0][1]), 42)
    return 14


# ============================================================
# SUPABASE
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
             .select("id, clinic_name_en, clinic_name_zh, address_en, address_zh, phone, district")
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
    lang = st.session_state.lang
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    st.session_state.lang = lang
    init_state()


def go_clinic():
    st.session_state.prev_mode = st.session_state.mode
    st.session_state.mode = "clinic"


def ai_bubble(text):
    st.markdown(f"<div class='ai-bubble'>{text}</div>", unsafe_allow_html=True)


# ============================================================
# DAILY SCHEDULE (timing only, not dosage)
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


def is_sitter(f):
    return f.get("work_type") == "Desk / office work" or f.get("sitting_hours") in ["6–8 hours", "More than 8 hours"]


def schedule_reason(f):
    if is_sitter(f):
        return T("reason_desk")
    if f.get("work_type") == "Standing / on my feet":
        return T("reason_stand")
    if f.get("work_type") == "Physical / manual work":
        return T("reason_manual")
    return T("reason_default")


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
    st.subheader(T("factors_title"))
    st.caption(T("factors_cap"))
    with st.form("factors_form"):
        age = st.selectbox(T("age"), ["18–29", "30–39", "40–49", "50–59", "60–69", "70+"])
        work = st.selectbox(T("work"), ["Desk / office work", "Standing / on my feet",
                                        "Physical / manual work", "Student",
                                        "Retired / at home", "Mixed / varies"], format_func=tr)
        sitting = st.selectbox(T("sitting"), ["Less than 4 hours", "4–6 hours", "6–8 hours",
                                              "More than 8 hours"], format_func=tr)
        pref = st.radio(T("pref"), list(SLOTS.keys()), format_func=tr)
        act = st.selectbox(T("act"), ["Very active (5+ days/week)", "Moderately active (2–4 days/week)",
                                      "Lightly active (1–2 days/week)", "Mostly sedentary"],
                           format_func=tr)
        other = st.multiselect(T("other"), ["Diabetes", "Heart condition", "Osteoporosis", "Arthritis",
                                            "Pregnancy", "Recent surgery", "High blood pressure"],
                               format_func=tr)
        prior = st.radio(T("prior"), ["No", "Yes, currently", "Yes, in the past"], format_func=tr)
        if st.form_submit_button(T("create_plan"), type="primary", use_container_width=True):
            st.session_state.factors = {
                "age_range": age, "work_type": work, "sitting_hours": sitting,
                "preferred_time": pref, "activity": act,
                "other_conditions": other, "prior_physio": prior}
            st.session_state.rc_screen = "overview"
            st.rerun()


def rc_overview(cd, plan_days, severity):
    exercises = cd.get("self_care_exercises", [])
    f = st.session_state.factors
    st.subheader(T("plan_title"))
    st.markdown(f"### {loc(cd, 'condition_name')}")
    with st.container(border=True):
        st.markdown(f"**{T('plan_len')}** {T('days_weeks', d=plan_days, w=(plan_days + 6) // 7)}")
        st.markdown(f"**{T('ex_per_day')}** {T('per_day', n=len(exercises))}")
        st.markdown(f"**{T('expected')}** {get_recovery(cd, severity)}")

    st.markdown(f"#### {T('sched')}")
    st.caption(schedule_reason(f))
    for slot, exs in build_schedule(exercises, f).items():
        with st.container(border=True):
            st.markdown(f"**{tr(slot)}**")
            for ex in exs:
                st.markdown(f"- {loc(ex, 'exercise_name')} — {loc(ex, 'frequency')}")
    if is_sitter(f):
        st.info(T("tip_desk"))

    with st.expander(T("context")):
        ctx = T("ctx")
        for key in ["age_range", "work_type", "sitting_hours", "activity", "prior_physio"]:
            st.markdown(f"- {ctx[key]}: {tr(f.get(key, '—'))}")
        oc = f.get("other_conditions") or []
        st.markdown(f"- {ctx['other']}: {'、'.join(tr(o) for o in oc) if oc else T('none_reported')}")

    if f.get("other_conditions") or f.get("age_range") in ["60–69", "70+"]:
        st.warning(T("warn_factors"))
        st.button(T("find_physio"), on_click=go_clinic, use_container_width=True)

    st.info(T("stop_sharp"))
    if st.button(T("view_cal"), type="primary", use_container_width=True):
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
        if st.button(T("prev"), disabled=(off == 0), use_container_width=True):
            s.week_offset -= 1
            st.rerun()
    with c2:
        st.markdown(f"<div style='text-align:center;font-weight:600;padding-top:6px;'>"
                    f"{T('week', a=off + 1, b=total_weeks)}</div>", unsafe_allow_html=True)
    with c3:
        if st.button(T("next"), disabled=(off + 1 >= total_weeks), use_container_width=True):
            s.week_offset += 1
            st.rerun()

    st.caption(T("tap_day"))
    cols = st.columns(7)
    for i, d in enumerate(range(start_day, end_day + 1)):
        with cols[i]:
            if st.button(f"{day_status(d)}\n\n{T('day', d=d)}", key=f"day_{d}",
                         use_container_width=True,
                         type="primary" if d == s.selected_day else "secondary"):
                s.selected_day = d
                st.rerun()

    st.divider()
    day = s.selected_day
    is_today, is_future = day == s.current_day, day > s.current_day
    label = T("today") if is_today else (T("upcoming") if is_future else T("past"))
    st.markdown(f"### {T('day', d=day)} — {label}")
    if day in s.skipped:
        st.caption(T("skipped_note"))
    if is_future:
        st.caption(T("preview_note"))

    done_set = s.completed.get(day, set())
    for idx, ex in enumerate(exercises):
        is_done = idx in done_set
        with st.container(border=True):
            a, b = st.columns([4, 1])
            with a:
                st.markdown(f"**{'✅ ' if is_done else ''}{loc(ex, 'exercise_name')}**")
                st.caption(f"{loc(ex, 'frequency')} · {diff_badge(ex.get('difficulty_level'))}")
            with b:
                if st.button(T("view") if (is_done or is_future) else T("do_it"),
                             key=f"do_{day}_{idx}", use_container_width=True):
                    s.view_exercise, s.view_day = idx, day
                    st.rerun()

    if not is_future and n_ex:
        st.progress(len(done_set) / n_ex)
        st.caption(T("done_of", a=len(done_set), b=n_ex))

    if is_today:
        if len(done_set) >= n_ex and day not in s.checkins:
            st.success(T("all_done"))
            if st.button(T("checkin_btn"), type="primary", use_container_width=True):
                s.show_checkin = True
                st.rerun()
        st.write("")
        if st.button(T("skip_today"), use_container_width=True):
            s.skipped.add(day)
            s.current_day = min(day + 1, plan_days)
            s.selected_day = s.current_day
            s.week_offset = (s.current_day - 1) // 7
            st.rerun()


def rc_exercise_detail(ex, idx, total, day):
    s = st.session_state
    st.caption(T("ex_of", d=day, a=idx + 1, b=total))
    st.markdown(f"### {loc(ex, 'exercise_name')}")
    other_name = ex.get("exercise_name_en") if zh() else ex.get("exercise_name_zh")
    if other_name:
        st.caption(other_name)
    a, b = st.columns(2)
    with a:
        st.markdown(f"**{T('difficulty')}**  \n{diff_badge(ex.get('difficulty_level')) or '—'}")
    with b:
        st.markdown(f"**{T('frequency')}**  \n{loc(ex, 'frequency') or '—'}")
    st.divider()
    st.markdown(f"**{T('how_to')}**")
    st.write(loc(ex, "instructions"))
    other = ex.get("instructions_en") if zh() else ex.get("instructions_zh")
    if other:
        with st.expander(T("other_lang") if zh() else "中文說明"):
            st.write(other)
    contra = loc(ex, "contraindications")
    if contra:
        st.warning(f"{T('important')} {contra}")

    can_complete = day <= s.current_day and idx not in s.completed.get(day, set())
    a, b = st.columns(2)
    with a:
        if st.button(T("back_cal"), use_container_width=True):
            s.view_exercise = None
            st.rerun()
    with b:
        if can_complete and st.button(T("mark_done"), type="primary", use_container_width=True):
            s.completed.setdefault(day, set()).add(idx)
            s.view_exercise = None
            st.rerun()


CHECKIN_ICONS = {"Better": "🟢", "About the same": "😐", "Worse": "🔴"}


def rc_checkin():
    s = st.session_state
    day = s.current_day
    st.subheader(T("checkin_title", d=day))
    with st.form("checkin"):
        pain = st.radio(T("pain_now"), list(CHECKIN_ICONS.keys()),
                        format_func=lambda v: f"{CHECKIN_ICONS[v]} {tr(v)}")
        st.text_area(T("notes"))
        if st.form_submit_button(T("submit"), type="primary", use_container_width=True):
            s.checkins[day] = pain
            s.current_day += 1
            s.selected_day = s.current_day
            s.week_offset = (s.current_day - 1) // 7
            s.show_checkin = False
            st.rerun()


def rc_progress(plan_days):
    s = st.session_state
    st.subheader(T("progress"))
    with st.container(border=True):
        st.markdown(f"**{T('day_of', a=s.current_day, b=plan_days)}**")
        st.progress(min((s.current_day - 1) / plan_days, 1.0))
    if s.skipped:
        st.caption(T("skipped_days", n=len(s.skipped)))
    st.markdown(T("trend"))
    if not s.checkins:
        st.caption(T("no_checkins"))
    else:
        with st.container(border=True):
            for d in sorted(s.checkins):
                v = s.checkins[d]
                st.markdown(f"{T('day', d=d)} — {CHECKIN_ICONS.get(v, '')} {tr(v)}")
    worse = any(v == "Worse" for v in s.checkins.values())
    st.warning(T("worse_warn")) if worse else st.info(T("see_info"))
    st.button(T("talk_physio"), on_click=go_clinic,
              type="primary" if worse else "secondary", use_container_width=True)


# ============================================================
# CLINIC SCREEN
# ============================================================
def clinic_screen():
    s = st.session_state
    if st.button(T("back_plan") if s.prev_mode == "recovery" else T("back_summary")):
        s.mode = s.prev_mode
        st.rerun()
    st.subheader(T("find_title"))
    st.caption(T("choose_area"))
    with st.form("clinic_form"):
        area = st.selectbox(T("area"), list(REGIONS.keys()), format_func=tr)
        insurer = st.selectbox(T("insurance"), ["Not sure / no insurance", "FWD", "AIA", "Bupa",
                                                "Blue Cross", "Cigna"], format_func=tr)
        go = st.form_submit_button(T("show_clinics"), type="primary", use_container_width=True)
    if go:
        s.answers["clinic_area"], s.answers["clinic_insurer"] = area, insurer
    if "clinic_area" not in s.answers:
        return
    area, insurer = s.answers["clinic_area"], s.answers["clinic_insurer"]
    try:
        clinics = fetch_clinics(REGIONS[area])
    except Exception as e:
        st.error(T("unavailable"))
        if DEBUG:
            st.exception(e)
        return
    if not clinics:
        st.info(T("none_found", a=tr(area)))
        return
    if insurer != "Not sure / no insurance":
        clinics = sorted(clinics, key=lambda c: insurer not in c["insurers"])
    st.success(T("n_clinics", n=len(clinics), a=tr(area)))
    st.caption(T("ins_note"))
    for c in clinics:
        primary = (c.get("clinic_name_zh") if zh() else c.get("clinic_name_en")) \
            or c.get("clinic_name_en") or c.get("clinic_name_zh") or "Clinic"
        secondary = c.get("clinic_name_en") if zh() else c.get("clinic_name_zh")
        address = (c.get("address_zh") if zh() else None) or c.get("address_en")
        with st.container(border=True):
            st.markdown(f"**{primary}**")
            if secondary and secondary != primary:
                st.caption(secondary)
            if c.get("district"):
                st.markdown(f"📍 {c['district']}")
            if address:
                st.caption(address)
            if c.get("phone"):
                st.markdown(f"📞 {c['phone']}")
            if c["insurers"]:
                st.markdown(f"{T('listed_with')} " + " ".join(f"`{i}`" for i in c["insurers"]))


# ============================================================
# MAIN
# ============================================================
lang_toggle()

try:
    retriever, conditions_data, embed_name = setup_engine()
except Exception as e:
    st.error(T("service_down"))
    if DEBUG:
        st.exception(e)
    st.stop()

A = st.session_state.answers
S = st.session_state

# ---------- RECOVERY ----------
if S.mode == "recovery":
    cd = conditions_data.get(A.get("matched_condition"), {})
    sev = A.get("severity", "Mild")
    plan_days = parse_plan_days(cd, sev)
    with st.sidebar:
        st.markdown(T("rc"))
        st.caption(loc(cd, "condition_name") or "—")
        keys = ["overview", "calendar", "progress"]
        idx = {"factors": 0, "overview": 0, "calendar": 1, "progress": 2}.get(S.rc_screen, 0)
        nav = st.radio(T("go_to"), keys, index=idx,
                       format_func=lambda k: T("nav")[keys.index(k)])
        if S.factors and nav != S.rc_screen and not (nav == "overview" and S.rc_screen == "factors"):
            S.rc_screen, S.view_exercise, S.show_checkin = nav, None, False
            st.rerun()
        st.divider()
        st.button("🩺 " + T("find_physio"), on_click=go_clinic, use_container_width=True)
        if st.button(T("back_summary"), use_container_width=True):
            S.mode = "assessment"
            st.rerun()
        if st.button(T("start_over"), use_container_width=True):
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
        rc_overview(cd, plan_days, sev)
    elif S.rc_screen == "calendar":
        rc_calendar(cd, plan_days)
    elif S.rc_screen == "progress":
        rc_progress(plan_days)

# ---------- CLINIC ----------
elif S.mode == "clinic":
    clinic_screen()

# ---------- ASSESSMENT ----------
else:
    if 0 < S.step <= 8:
        st.progress(S.step / 8)

    if S.step == 0:
        st.markdown("<h1 style='text-align:center;'>🩺</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align:center;'>{T('title')}</h2>", unsafe_allow_html=True)
        st.info(T("disclaimer"))
        st.button(T("start"), on_click=next_step, type="primary")

    elif S.step == 1:
        ai_bubble(T("q_body"))
        try:
            st.image("body_map.png", use_container_width=True)
        except Exception:
            pass
        st.button(T("back"), on_click=prev_step)
        with st.form("body_form"):
            body = st.selectbox(T("select_area"), BODY_OPTS, format_func=tr)
            custom = st.text_input(T("own_words_opt"))
            if st.form_submit_button(T("continue"), type="primary", use_container_width=True):
                A["body_area"] = custom.strip() or body
                A.pop("matched_condition", None)
                next_step()
                st.rerun()

    elif S.step == 2:
        ai_bubble(T("q_quality", body=tr(A.get("body_area", ""))))
        st.button(T("back"), on_click=prev_step)
        with st.form("quality_form"):
            st.caption(T("select_all"))
            opts = {"🔥": "Burning", "⚡": "Sharp", "😣": "Dull ache", "📌": "Pins & needles",
                    "💪": "Stiffness", "🔋": "Weakness", "🔨": "Throbbing", "🧊": "Clicking or locking"}
            chosen = [v for icon, v in opts.items() if st.checkbox(f"{icon} {tr(v)}")]
            custom = st.text_input(T("own_words"))
            if st.form_submit_button(T("continue"), type="primary", use_container_width=True):
                if custom.strip():
                    chosen.append(custom.strip())
                A["pain_quality"] = ", ".join(chosen) or "unspecified"
                A.pop("matched_condition", None)
                next_step()
                st.rerun()

    elif S.step == 3:
        ai_bubble(T("q_timing"))
        st.button(T("back"), on_click=prev_step)
        with st.form("timing_form"):
            st.caption(T("select_all"))
            opts = ["During activity", "In the morning", "At night", "When sitting",
                    "When walking", "After exercise", "Constant"]
            chosen = [o for o in opts if st.checkbox(tr(o))]
            custom = st.text_input(T("own_words"))
            if st.form_submit_button(T("continue"), type="primary", use_container_width=True):
                if custom.strip():
                    chosen.append(custom.strip())
                A["pain_timing"] = ", ".join(chosen) or "unspecified"
                A.pop("matched_condition", None)
                next_step()
                st.rerun()

    elif S.step == 4:
        ai_bubble(T("q_sev"))
        st.button(T("back"), on_click=prev_step)
        icons = {"Mild": "🟢", "Moderate": "🟡", "Severe": "🔴"}
        with st.form("sev_form"):
            sev = st.radio(T("sev_choose"), list(icons.keys()),
                           format_func=lambda v: f"{icons[v]} {tr(v)}",
                           captions=T("sev_caps"))
            st.caption(T("sev_unsure"))
            if st.form_submit_button(T("continue"), type="primary", use_container_width=True):
                A["severity"] = sev
                next_step()
                st.rerun()

    elif S.step == 5:
        sev_txt = tr(A.get("severity", "")) if zh() else A.get("severity", "").lower()
        ai_bubble(T("q_dur", sev=sev_txt))
        st.button(T("back"), on_click=prev_step)
        with st.form("dur_form"):
            dur = st.radio(T("select_one"), ["Just today", "Few days", "1-4 weeks", "Over a month"],
                           format_func=tr)
            custom = st.text_input(T("dur_own"))
            if st.form_submit_button(T("continue"), type="primary", use_container_width=True):
                A["duration"] = custom.strip() or dur
                next_step()
                st.rerun()

    elif S.step == 6:
        st.markdown(f"<h3 style='text-align:center;'>{T('safety')}</h3>", unsafe_allow_html=True)
        st.caption(T("safety_cap"))
        st.button(T("back"), on_click=prev_step)
        with st.form("flags_form"):
            opts = ["Numbness in both legs", "Loss of bladder or bowel control", "Fever with pain",
                    "Recent significant injury or fall", "Unexplained weight loss",
                    "History of cancer", "Sudden severe weakness"]
            chosen = [o for o in opts if st.checkbox(tr(o))]
            st.divider()
            none_sel = st.checkbox(T("none_above"))
            if st.form_submit_button(T("continue"), type="primary", use_container_width=True):
                A["red_flags"] = [] if none_sel else chosen
                next_step()
                st.rerun()

    elif S.step == 7:
        ai_bubble(T("q_activity"))
        st.button(T("back"), on_click=prev_step)
        with st.form("act_form"):
            st.caption(T("select_all"))
            opts = {"🏃": "Running", "🏋️": "Weightlifting", "🚴": "Cycling", "🧘": "Yoga/Pilates",
                    "🎾": "Racquet sports", "🏀": "Team sports", "💻": "Desk work",
                    "💺": "Mostly sedentary"}
            chosen = [v for icon, v in opts.items() if st.checkbox(f"{icon} {tr(v)}")]
            custom = st.text_input(T("activity_own"))
            if st.form_submit_button(T("get_assessment"), type="primary", use_container_width=True):
                if custom.strip():
                    chosen.append(custom.strip())
                A["activity"] = ", ".join(chosen) or "unspecified"
                A.pop("matched_condition", None)
                next_step()
                st.rerun()

    elif S.step == 8:
        if A.get("red_flags"):
            st.markdown("<h1 style='text-align:center;'>🚨</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center;color:#EF4444;'>{T('urgent_title')}</h2>",
                        unsafe_allow_html=True)
            st.write(T("urgent_text"))
            st.error(T("urgent_actions"))
            st.button(T("start_over"), on_click=restart)
        else:
            if "matched_condition" not in A:
                with st.spinner(T("analysing")):
                    cond, conf, score = match_condition(retriever, conditions_data, A)
                A["matched_condition"], A["confidence"], A["score"] = cond, conf, score
            cond, conf = A["matched_condition"], A["confidence"]
            cd = conditions_data.get(cond, {})
            mild = A["severity"] == "Mild"
            se = {"Mild": "🟢", "Moderate": "🟡", "Severe": "🔴"}.get(A["severity"], "🟡")

            st.subheader(T("summary"))
            with st.container(border=True):
                st.markdown(f"**{T('f_location')}**  \n{tr(A['body_area'])}")
                st.markdown(f"**{T('f_type')}**  \n{tr_list(A['pain_quality'])}")
                st.markdown(f"**{T('f_when')}**  \n{tr_list(A['pain_timing'])}")
                st.markdown(f"**{T('f_duration')}**  \n{tr(A['duration'])}")
                st.markdown(f"**{T('f_severity')}**  \n{se} {tr(A['severity'])}")
                st.markdown(f"**{T('f_activity')}**  \n{tr_list(A['activity'])}")
                st.divider()
                if cond:
                    st.markdown(f"**{T('f_might')}**  \n{loc(cd, 'condition_name')} "
                                f"*（{T('conf')[conf]}）*" if zh() else
                                f"**{T('f_might')}**  \n{loc(cd, 'condition_name')} *({T('conf')[conf]})*")
                    st.markdown(f"**{T('f_recovery')}**  \n{get_recovery(cd, A['severity'])}")
                    nxt = T("next_self") if mild else T("next_physio")
                else:
                    st.markdown(f"**{T('f_might')}**  \n{T('no_match')}")
                    nxt = T("next_assess")
                st.divider()
                st.markdown(f"**{T('f_next')}**  \n{nxt}")
            st.caption(T("not_dx"))
            if DEBUG:
                st.caption(f"[debug] score {A.get('score', 0):.3f} · embeddings: {embed_name}")

            if not cond:
                st.button(T("find_physio"), on_click=go_clinic, type="primary", use_container_width=True)
            elif mild:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(T("start_plan"), type="primary", use_container_width=True):
                        S.mode, S.rc_screen = "recovery", "factors"
                        st.rerun()
                with c2:
                    st.button(T("find_physio"), on_click=go_clinic, use_container_width=True)
            else:
                st.button(T("find_physio"), on_click=go_clinic, type="primary", use_container_width=True)
                if st.button(T("or_self"), use_container_width=True):
                    S.mode, S.rc_screen = "recovery", "factors"
                    st.rerun()

            st.button(T("share"), use_container_width=True, disabled=True, help=T("soon"))
            st.button(T("start_over"), on_click=restart)
