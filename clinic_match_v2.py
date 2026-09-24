#!/usr/bin/env python3
"""
Clinic Matching Module v2 — Corrected District Mapping
=======================================================
Fixed: REGION_DISTRICTS now covers ALL districts in your data
(Chinese + English + mixed formats), so Kowloon/NT searches work.

RUN (standalone test):
    streamlit run clinic_match_v2.py
"""

import streamlit as st
from supabase import create_client

SUPABASE_URL = "https://hnhpyeimpllrdaxtfevn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhuaHB5ZWltcGxscmRheHRmZXZuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4ODY3MzMwOSwiZXhwIjoyMTA0MjQ5MzA5fQ.2sLYjcJ5pZ7auYtFbTL5ZmWAjeboNBIyukzzoVrIDiU"


@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# DISTRICT MAPPING — matches your ACTUAL data
# ============================================================
REGION_DISTRICTS = {
    "Hong Kong Island": [
        "中環", "銅鑼灣", "灣仔", "北角", "上環", "金鐘", "柴灣",
        "香港仔", "炮台山",
        "Central", "Central, Central", "Central, Central & 1 more",
        "Causeway Bay, Causeway Bay", "North Point",
    ],
    "Kowloon": [
        "尖沙咀", "旺角", "佐敦", "油麻地", "觀塘", "藍田", "紅磡",
        "牛頭角", "土瓜灣", "長沙灣", "荔枝角", "太子", "九龍塘", "深水埗",
        "Jordan, Jordan", "Mong Kok, Jordan, Mong Kok & 1 more",
        "Tsim Sha Tsui, Tsim Sha Tsui",
    ],
    "New Territories": [
        "荃灣", "沙田", "屯門", "元朗", "將軍澳", "大埔", "大圍",
        "葵芳", "西貢", "上水", "青衣",
        "Sha Tin, Tai Po, Tsuen Wan, Yuen Long, Tsim Sha Tsui, Tuen Mun, Sha Tin & 5 more",
    ],
}


def find_clinics(supabase, region, insurer, limit=10):
    districts = REGION_DISTRICTS.get(region, [])

    ins = supabase.table("insurers").select("id").eq("insurer_name", insurer).execute()
    if not ins.data:
        return []
    insurer_id = ins.data[0]["id"]

    panels = supabase.table("clinic_insurer_panels") \
        .select("clinic_id").eq("insurer_id", insurer_id).execute()
    panel_clinic_ids = [p["clinic_id"] for p in panels.data]
    if not panel_clinic_ids:
        return []

    clinics = supabase.table("clinics") \
        .select("clinic_name_en, clinic_name_zh, address_en, phone, district") \
        .in_("id", panel_clinic_ids) \
        .in_("district", districts) \
        .limit(limit).execute()
    return clinics.data


def clinic_match_screen():
    supabase = get_supabase()
    st.subheader("Find a Physiotherapist")
    st.caption("We'll match you with clinics based on your location and insurance")

    with st.form("clinic_search"):
        region = st.selectbox("📍 Your area:",
                              ["Hong Kong Island", "Kowloon", "New Territories"])
        insurer = st.selectbox("🏥 Your insurance:",
                               ["FWD", "AIA", "Bupa", "Blue Cross", "Cigna"])
        submitted = st.form_submit_button("Find Clinics →", type="primary",
                                          use_container_width=True)

    if submitted:
        with st.spinner("Searching clinics..."):
            clinics = find_clinics(supabase, region, insurer)

        if not clinics:
            st.warning(f"No {insurer} clinics found in {region}. "
                       "Try a different area or insurance.")
        else:
            st.success(f"Found {len(clinics)} clinics in {region} accepting {insurer}")
            st.write("")
            for c in clinics:
                with st.container(border=True):
                    name = c.get("clinic_name_en") or c.get("clinic_name_zh") or "Clinic"
                    st.markdown(f"### {name}")
                    if c.get("district"):
                        st.markdown(f"📍 {c['district']}")
                    if c.get("address_en"):
                        st.markdown(f"{c['address_en']}")
                    if c.get("phone"):
                        st.markdown(f"📞 {c['phone']}")
                    st.markdown(f"🏥 Accepts **{insurer}**")


if __name__ == "__main__":
    st.set_page_config(page_title="Find a Physio", page_icon="🏥", layout="centered")
    clinic_match_screen()
