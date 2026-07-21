"""SPEC §6 화면 계약: Streamlit 탭 2개.

① Agentic 단독 — 시스템 선택기(현재 baseline만), 입력 + top_k 슬라이더,
   답변 + 전략 뱃지 + 출처, expander 4종(계획/hop별 판정 표/재작성 이력/중간 답·통계),
   exhausted 경고 박스
② 비교 — 입력 하나 → naive | agentic 좌우

공통: /health 사전 확인(실패 시 안내 문구), session_state 유지.
계층 규칙: frontend는 HTTP로 backend만 호출한다 (로직·DB 접근 금지).
"""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = (os.getenv("BACKEND_URL") or "").rstrip("/")

st.set_page_config(page_title="Agentic RAG — 한국어 영화 위키", layout="wide")
st.title("Agentic RAG — 한국어 위키 영화 도메인")

# ---------- 공통: /health 사전 확인 ----------
if not BACKEND_URL:
    st.error("`.env`에 BACKEND_URL이 없습니다. 예: BACKEND_URL=http://127.0.0.1:8000")
    st.stop()

try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
    assert health.get("status") == "ok"
except Exception:
    st.error(
        f"backend({BACKEND_URL})가 응답하지 않습니다. 터미널에서 서버를 먼저 실행하세요:\n\n"
        "```\n./venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000\n```"
    )
    st.stop()

st.caption(
    f"backend 정상 · 청크 {health['db_chunks']:,}개 · distance space: {health['space']}"
)


def post(path: str, question: str, top_k: int, timeout: int = 180):
    r = requests.post(
        f"{BACKEND_URL}{path}",
        json={"question": question, "top_k": top_k},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def strategy_badge(strategy: str) -> str:
    color = "#1f77b4" if strategy == "정답형" else "#9467bd"
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.85em'>{strategy}</span>"
    )


def render_sources(sources: list):
    if not sources:
        return
    st.markdown("**출처**")
    for s in sources:
        st.markdown(f"- hop {s['hop']}: {', '.join(s['titles'])}")


def render_agentic(res: dict):
    if res.get("exhausted"):
        st.warning(
            f"⚠️ 검색 한도 소진(exhausted, 사유: {res.get('exhausted_reason') or '?'}) — "
            "아래 답변은 확보된 근거까지만 반영된 제한적 답변입니다."
        )
    st.markdown(
        f"**전략** {strategy_badge(res.get('strategy') or '?')}",
        unsafe_allow_html=True,
    )
    st.markdown(res["answer"])
    render_sources(res.get("sources") or [])

    with st.expander("① 계획 (Planner)"):
        st.json(res.get("plan") or {})
    with st.expander("② hop별 판정 표 (Judge)"):
        jh = res.get("judge_history") or []
        if jh:
            st.dataframe(
                [{"hop": j["hop"], "verdict": j["verdict"], "source": j["source"],
                  "relevance": j["relevance"], "sufficiency": j["sufficiency"],
                  "reason": j["reason"]} for j in jh],
                use_container_width=True,
            )
        else:
            st.caption("판정 이력이 없습니다.")
    with st.expander("③ 재작성 이력 (Rewriter)"):
        rw = res.get("rewrite_history") or []
        if rw:
            for i, q in enumerate(rw, 1):
                st.markdown(f"{i}. {q}")
        else:
            st.caption("재작성 없이 한 번에 통과했습니다.")
    with st.expander("④ 중간 답·통계"):
        st.markdown(f"- 중간 답(intermediate): {res.get('intermediate_answers') or '없음'}")
        st.markdown(
            f"- LLM 호출 {res.get('llm_calls')}회 · 재작성 {res.get('retry_total')}회 · "
            f"도달 hop {res.get('hop_reached')} · {res.get('elapsed_sec')}초"
        )


tab_single, tab_compare = st.tabs(["① Agentic 단독", "② naive vs agentic 비교"])

# ---------- 탭 ①: Agentic 단독 ----------
with tab_single:
    col_sys, col_k = st.columns([1, 2])
    with col_sys:
        system = st.selectbox("시스템", ["baseline"], key="system_select")  # day 7 후 improved 추가
    with col_k:
        top_k = st.slider("top_k (검색 문서 수)", 1, 10, 5, key="single_topk")
    q1 = st.text_input("질문", key="single_question",
                       placeholder="예: 2012년에 개봉한 영화 러브픽션의 감독은 어떤 학교를 졸업했는가?")
    if st.button("질문하기", key="single_ask", type="primary") and q1.strip():
        with st.spinner("Agentic 그래프 실행 중… (5~20초)"):
            try:
                st.session_state["single_result"] = post("/ask", q1.strip(), top_k)
                st.session_state["single_asked"] = q1.strip()
            except Exception as e:
                st.session_state["single_result"] = None
                st.error(f"요청 실패: {e}")
    if st.session_state.get("single_result"):
        st.divider()
        st.caption(f"질문: {st.session_state.get('single_asked', '')}")
        render_agentic(st.session_state["single_result"])

# ---------- 탭 ②: 비교 ----------
with tab_compare:
    top_k2 = st.slider("top_k (양쪽 공통)", 1, 10, 5, key="compare_topk")
    q2 = st.text_input("질문 (같은 질문을 naive와 agentic에 동시에 보냅니다)",
                       key="compare_question")
    if st.button("비교 실행", key="compare_ask", type="primary") and q2.strip():
        with st.spinner("naive + agentic 실행 중…"):
            try:
                naive_res = post("/ask_naive", q2.strip(), top_k2)
                agentic_res = post("/ask", q2.strip(), top_k2)
                st.session_state["compare_results"] = (naive_res, agentic_res)
                st.session_state["compare_asked"] = q2.strip()
            except Exception as e:
                st.session_state["compare_results"] = None
                st.error(f"요청 실패: {e}")
    if st.session_state.get("compare_results"):
        naive_res, agentic_res = st.session_state["compare_results"]
        st.divider()
        st.caption(f"질문: {st.session_state.get('compare_asked', '')}")
        left, right = st.columns(2)
        with left:
            st.subheader("naive (1-pass)")
            st.markdown(naive_res["answer"])
            render_sources(naive_res.get("sources") or [])
            st.caption(
                f"LLM {naive_res.get('llm_calls')}회 · {naive_res.get('elapsed_sec')}초"
            )
        with right:
            st.subheader("agentic (baseline)")
            render_agentic(agentic_res)
