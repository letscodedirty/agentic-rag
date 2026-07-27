"""SPEC §6 화면 계약 (Day 7 5단계 확정): 단독 탭 3개, 비교 탭·시스템 선택기 없음.

① v2(기본 탭) — 기존 Agentic 구성 + 명료화 판정 expander + 정형 조회 요약 +
   되묻기 카드(choices 버튼 클릭 = question 입력창 채움 → 재제출)
② baseline — 기존 단독 탭 구성 그대로
③ naive — 입력 + 답변 + 출처 링크만

공통: /health 사전 확인, session_state 유지, 각 탭 상단 검색 코퍼스 표기,
출처 title은 https://ko.wikipedia.org/wiki/{title} 기계 조립 링크(LLM 생성 금지).
계층 규칙: frontend는 HTTP로 backend만 호출한다 (로직·DB 접근 금지).
"""
import os
from urllib.parse import quote

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
    f"backend 정상 · v1 청크 {health['db_chunks']:,}개 · distance space: {health['space']}"
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
    color = {"정답형": "#1f77b4", "탐색형": "#9467bd", "목록형": "#2ca02c"}.get(
        strategy, "#7f7f7f")
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.85em'>{strategy}</span>"
    )


def wiki_link(doc_title: str) -> str:
    """출처 링크는 기계 조립 (SPEC §6 — LLM 생성 금지, 3탭 공통)."""
    return f"[{doc_title}](https://ko.wikipedia.org/wiki/{quote(doc_title)})"


def render_sources(sources: list, cap: int = 15):
    if not sources:
        return
    st.markdown("**출처**")
    for s in sources:
        docs = list(dict.fromkeys(t.split("::")[0] for t in s.get("titles", [])))
        shown = ", ".join(wiki_link(d) for d in docs[:cap])
        more = f" 외 {len(docs) - cap}건" if len(docs) > cap else ""
        st.markdown(f"- hop {s['hop']}: {shown}{more}")


def render_agentic(res: dict):
    """baseline·v2 공통 본문 (기존 단독 탭 구성)."""
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


def _fill_v2_question(q: str):
    st.session_state["v2_question"] = q


def render_clarification_card(clar: dict):
    """되묻기 카드 (SPEC §6): category 헤더 + choices 버튼 + 자유 입력 안내.
    choices=[]면 버튼 없이 자유 입력 유도 문구만."""
    with st.container(border=True):
        st.markdown(f"### 🔍 어느 **{clar.get('category', '것')}** 말씀이신가요?")
        choices = clar.get("choices") or []
        if choices:
            st.caption("선택지를 클릭하면 입력창에 채워집니다 — 다시 '질문하기'를 눌러 주세요.")
            for i, c in enumerate(choices):
                st.button(
                    c["label"], key=f"clar_btn_{i}",
                    on_click=_fill_v2_question, args=(c["question"],),
                )
        else:
            st.markdown("질문의 대상을 특정할 수 없습니다.")
        st.info(clar.get("free_input_hint") or "질문을 구체적으로 다시 적어 주세요.")


def render_v2_extras(res: dict):
    """v2 고유 패널: 명료화 판정 + 정형 조회 요약."""
    clar = res.get("clarification")
    with st.expander("⑤ 명료화 판정 (Clarify)"):
        if clar:
            st.markdown(
                f"- 발동: **예** · 사유: {', '.join(clar.get('reason') or [])}\n"
                f"- u(x) = {clar.get('u')} / τ = {clar.get('tau')}\n"
                f"- DB 동명 매칭: {', '.join(clar.get('db_matches') or []) or '없음'}\n"
                f"- 명료화 LLM 호출: {res.get('clarify_calls', 0)}회 (상한 13)"
            )
        else:
            st.caption("발동 없음 — 명확 질문으로 판정되어 그대로 통과했습니다.")
    if not clar:
        with st.expander("⑥ 정형 조회 요약 (2·3층)"):
            sh = res.get("structured_hits") or {}
            st.markdown(
                f"- 인포박스 적중: {', '.join(sh.get('infobox') or []) or '없음'}\n"
                f"- 필모그래피 적중: {', '.join(sh.get('filmography') or []) or '없음'}"
            )
            ls = res.get("list_summary")
            if ls:
                st.markdown(
                    f"- 목록 조회: {ls.get('kind')} / 키 '{ls.get('key')}' / "
                    f"{ls.get('n_items')}항목"
                )


tab_v2, tab_base, tab_naive = st.tabs(["① v2 (Agentic+3층)", "② baseline", "③ naive"])

# ---------- 탭 ①: v2 (기본 탭) ----------
with tab_v2:
    st.caption("검색 코퍼스: 전문 3층 DB (./db_v2 — 섹션 청크·인포박스·필모 색인)")
    top_k_v2 = st.slider("top_k (검색 문서 수)", 1, 10, 10, key="v2_topk")
    q_v2 = st.text_input("질문", key="v2_question",
                         placeholder="예: 배우 유해진이 출연한 영화를 모두 알려줘")
    if st.button("질문하기", key="v2_ask", type="primary") and q_v2.strip():
        with st.spinner("v2 그래프 실행 중… (명료화 판정 포함, 5~30초)"):
            try:
                st.session_state["v2_result"] = post("/ask_v2", q_v2.strip(), top_k_v2)
                st.session_state["v2_asked"] = q_v2.strip()
            except Exception as e:
                st.session_state["v2_result"] = None
                st.error(f"요청 실패: {e}")
    if st.session_state.get("v2_result"):
        res = st.session_state["v2_result"]
        st.divider()
        st.caption(f"질문: {st.session_state.get('v2_asked', '')}")
        if res.get("clarification"):
            render_clarification_card(res["clarification"])
        else:
            render_agentic(res)
        render_v2_extras(res)

# ---------- 탭 ②: baseline (기존 단독 탭 구성 그대로) ----------
with tab_base:
    st.caption("검색 코퍼스: 서두 DB (./db — v1 서두 청크)")
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

# ---------- 탭 ③: naive (입력 + 답변 + 출처만) ----------
with tab_naive:
    st.caption("검색 코퍼스: 서두 DB (./db — v1 서두 청크)")
    q3 = st.text_input("질문", key="naive_question",
                       placeholder="예: 영화 극한직업의 장르는?")
    if st.button("질문하기", key="naive_ask", type="primary") and q3.strip():
        with st.spinner("naive 1-pass 실행 중…"):
            try:
                st.session_state["naive_result"] = post("/ask_naive", q3.strip(), 10)
                st.session_state["naive_asked"] = q3.strip()
            except Exception as e:
                st.session_state["naive_result"] = None
                st.error(f"요청 실패: {e}")
    if st.session_state.get("naive_result"):
        res = st.session_state["naive_result"]
        st.divider()
        st.caption(f"질문: {st.session_state.get('naive_asked', '')}")
        st.markdown(res["answer"])
        render_sources(res.get("sources") or [])
