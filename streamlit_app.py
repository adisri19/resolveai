"""Streamlit frontend. Talks to the FastAPI backend over HTTP only -- it has no
database access and no LLM key of its own."""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Support Decision Assistant", page_icon="🎫")


def call(method: str, path: str, **kwargs):
    """Returns (ok, payload). Auth header is attached whenever we hold a token."""
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.get("token") else {}
    try:
        resp = requests.request(method, f"{API}{path}", headers=headers, timeout=120, **kwargs)
    except requests.RequestException as exc:
        return False, f"Cannot reach the API at {API} ({exc})"
    if resp.status_code == 401:
        st.session_state.token = None
        return False, "Session expired. Please log in again."
    if not resp.ok:
        return False, resp.json().get("detail", resp.text) if resp.content else resp.reason
    return True, resp.json()


def render_decision(decision: dict):
    if not decision:
        st.warning("No decision recorded for this ticket.")
        return
    st.metric("Recommended action", decision["action"])
    st.progress(decision["confidence"], text=f"Confidence: {decision['confidence']:.0%}")
    st.write("**Reasoning**")
    st.write(decision["reason"])
    st.write("**Sources**")
    st.write(", ".join(decision["sources"]) if decision["sources"] else "_none cited_")


# ---------------------------------------------------------------- login / register
def login_view():
    st.title("Support Decision Assistant")
    tab_login, tab_register = st.tabs(["Log in", "Register"])

    with tab_login:
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                ok, data = call("POST", "/login", json={"email": email, "password": password})
                if ok:
                    st.session_state.token = data["access_token"]
                    st.rerun()
                else:
                    st.error(data)

    with tab_register:
        with st.form("register"):
            email = st.text_input("Email", key="r_email")
            password = st.text_input("Password (min 8 characters)", type="password", key="r_pw")
            if st.form_submit_button("Create account"):
                ok, data = call("POST", "/register", json={"email": email, "password": password})
                if ok:
                    st.success("Account created. Log in on the other tab.")
                else:
                    st.error(data)


# ------------------------------------------------------------------- new decision
def new_decision_view():
    st.header("New decision")
    st.caption("Leave a field on 'unknown' if you genuinely do not know it — "
               "the assistant will ask for it rather than guess.")
    with st.form("ticket"):
        message = st.text_area("What is the customer reporting?", height=120)
        col1, col2 = st.columns(2)
        value = col1.number_input("Order value (₹)", min_value=0.0, step=100.0, value=0.0)
        delivered = col1.number_input("Days since delivery (-1 = unknown)", min_value=-1, value=-1)
        dispatched = col1.number_input("Days since dispatch (-1 = unknown)", min_value=-1, value=-1)
        product = col2.selectbox("Product type", ["unknown", "food", "non_food", "mixed"])
        opened = col2.selectbox("Opened status", ["unknown", "opened", "unopened"])
        order_status = col2.selectbox("Order status", ["unknown", "processing", "dispatched", "delivered"])

        if st.form_submit_button("Get recommendation"):
            if not message.strip():
                st.error("Please describe the issue.")
                return
            payload = {
                "message": message,
                "order_value_inr": value or None,
                "days_since_delivery": None if delivered < 0 else delivered,
                "days_since_dispatch": None if dispatched < 0 else dispatched,
                "product_type": product,
                "opened_status": opened,
                "order_status": order_status,
            }
            with st.spinner("Retrieving policies and asking the model..."):
                ok, data = call("POST", "/tickets", json=payload)
            if ok:
                render_decision(data["decision"])
            else:
                st.error(data)


# ------------------------------------------------------------------------ history
def history_view():
    st.header("History")
    ok, tickets = call("GET", "/tickets")
    if not ok:
        st.error(tickets)
        return
    if not tickets:
        st.info("No tickets yet.")
        return
    for ticket in tickets:
        action = ticket["decision"]["action"] if ticket["decision"] else "—"
        with st.expander(f"#{ticket['id']} · {action} · {ticket['created_at']}"):
            st.write(ticket["message"])
            st.divider()
            # Fetch the single ticket so the per-ticket endpoint is genuinely exercised.
            ok, detail = call("GET", f"/tickets/{ticket['id']}")
            render_decision(detail["decision"] if ok else None)


# --------------------------------------------------------------------------- main
st.session_state.setdefault("token", None)

if st.session_state.token is None:
    login_view()
else:
    ok, user = call("GET", "/me")
    if not ok:
        st.rerun()
    st.sidebar.write(f"Signed in as **{user['email']}**")
    if st.sidebar.button("Log out"):
        st.session_state.token = None
        st.rerun()
    page = st.sidebar.radio("Page", ["New decision", "History"])
    (new_decision_view if page == "New decision" else history_view)()
