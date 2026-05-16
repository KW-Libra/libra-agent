"""L4 시나리오용 지식 캐시 생성기.

libra-ingest 산출 스키마(NormalizedDocument / Event)를 미러링한다.
- positive : 반도체 업황 회복 호재 (005930.KS / 000660.KS 매핑)  -> 코어 INCREASE 기대
- mixed    : 호실적(개별) + 거시 악재(금리) 상충                  -> DEFER/HOLD 기대
semi_shock(악재)·no_signal(거시일반)은 기존 캐시 재사용.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent / "cache"


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def doc(doc_id, doc_type, source_name, stype, url, pub_org, region, title, body, pub, col):
    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "source_info": {"source_type": stype, "source_name": source_name,
                        "source_url": url, "publisher": pub_org, "region": region},
        "raw_content": {"title_raw": title, "body_raw": f"<p>{body}</p>",
                        "summary_raw": f"<p>{body[:160]}</p>"},
        "normalized_content": {"title": title, "body": body, "char_count": len(body)},
        "language_info": {"lang_final": "ko", "lang_confidence": 0.99},
        "timing_info": {"published_at": pub, "collected_at": col},
        "trace_info": {"raw_hash": _h(doc_id + title)},
    }


def stock(eid, name):
    return {"entity_type": "STOCK", "entity_id": eid, "entity_name": name, "ticker": eid}


SS, HX = stock("005930.KS", "Samsung Electronics"), stock("000660.KS", "SK hynix")


def write(name: str, documents: list, events: list) -> None:
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "normalized_documents.json").write_text(
        json.dumps({"documents": documents}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "events.json").write_text(
        json.dumps({"events": events}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {name}: {len(documents)} docs, {len(events)} events -> {out}")


# ---------------- positive: 반도체 업황 회복 호재 ----------------
positive_docs = [
    doc("news_pos_hbm_0001", "NEWS", "Yonhap Economy RSS", "RSS",
        "https://www.yna.co.kr/view/AKR20260516300000010", "Yonhap News", "KR",
        "메모리 가격 3개월 연속 반등…HBM 수요 폭증에 삼성·SK하이닉스 가동률 풀가동",
        "DRAM 고정거래가격이 3개월 연속 상승하며 전월 대비 8% 올랐다. AI 서버용 HBM은 "
        "주문이 생산능력을 초과해 삼성전자와 SK하이닉스가 라인을 풀가동 중이다. "
        "시장조사기관은 2026년 하반기 메모리 업사이클 진입을 공식화했고, 증권가는 "
        "두 회사 목표주가를 각각 18%, 22% 상향했다.",
        "2026-05-16T00:30:00Z", "2026-05-16T00:33:00Z"),
    doc("disclosure_ss_earnbeat_0002", "DISCLOSURE", "DART Disclosure", "CRAWLER",
        "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260516000120", "DART", "KR",
        "[주요사항보고서] 삼성전자 - 2026년 2분기 잠정 영업이익 컨센서스 32% 상회",
        "삼성전자가 2026년 2분기 잠정 영업이익을 11조2천억원으로 공시했다. 시장 "
        "컨센서스(8조5천억원)를 32% 상회하는 어닝서프라이즈다. 메모리 가격 반등과 "
        "HBM3E 출하 본격화로 DS부문 영업이익이 전분기 대비 2.4배로 증가했다.",
        "2026-05-16T07:10:00Z", "2026-05-16T07:12:00Z"),
    doc("news_hx_upgrade_0003", "NEWS", "Yonhap Economy RSS", "RSS",
        "https://www.yna.co.kr/view/AKR20260516300000044", "Yonhap News", "KR",
        "SK하이닉스, HBM4 양산 조기 진입…증권가 '내년 최대 실적' 컨센서스 상향",
        "SK하이닉스가 차세대 HBM4 양산을 당초 계획보다 한 분기 앞당겼다. 주요 고객사 "
        "선주문이 확정되며 증권가는 2027년 사상 최대 실적 전망과 함께 투자의견을 "
        "일제히 '매수'로 상향했다.",
        "2026-05-16T02:05:00Z", "2026-05-16T02:08:00Z"),
]
positive_events = [
    {"event_id": "evt_pos_ss_beat", "event_type": "EARNINGS",
     "event_time": "2026-05-16T07:10:00Z", "cluster_key": "005930.KS|EARNINGS|2026-05-16",
     "confidence": 0.96, "headline": "삼성전자 2Q 어닝서프라이즈 (+32% 컨센)",
     "summary": "메모리 반등·HBM3E로 DS 영업이익 2.4배.", "entities": [SS],
     "source_documents": ["disclosure_ss_earnbeat_0002", "news_pos_hbm_0001"],
     "sentiment_score": 0.79, "impact_score": 0.86, "risk_score": 0.05, "time_decay": 1.0,
     "quant_signals": {"dominant_polarity": "positive",
                       "matched_terms": {"positive": ["어닝서프라이즈", "상회", "반등"],
                                         "negative": [], "risk": []},
                       "rationale": "강한 긍정 실적."}},
    {"event_id": "evt_pos_hx_up", "event_type": "EARNINGS",
     "event_time": "2026-05-16T02:05:00Z", "cluster_key": "000660.KS|EARNINGS|2026-05-16",
     "confidence": 0.93, "headline": "SK하이닉스 HBM4 조기 양산·컨센 상향",
     "summary": "HBM4 조기 양산, 2027 최대 실적 전망, 투자의견 매수 상향.", "entities": [HX],
     "source_documents": ["news_hx_upgrade_0003", "news_pos_hbm_0001"],
     "sentiment_score": 0.72, "impact_score": 0.8, "risk_score": 0.08, "time_decay": 1.0,
     "quant_signals": {"dominant_polarity": "positive",
                       "matched_terms": {"positive": ["상향", "양산", "최대 실적"],
                                         "negative": [], "risk": []},
                       "rationale": "긍정 모멘텀."}},
]

# ---------------- mixed: 개별 호실적 vs 거시 악재 상충 ----------------
mixed_docs = [
    doc("disclosure_ss_beat_m01", "DISCLOSURE", "DART Disclosure", "CRAWLER",
        "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260516000777", "DART", "KR",
        "[주요사항보고서] 삼성전자 - 2분기 잠정 영업이익 컨센서스 19% 상회",
        "삼성전자 2026년 2분기 잠정 영업이익이 9조8천억원으로 컨센서스를 19% 상회했다. "
        "메모리 부문 견조한 실적이 견인했다.",
        "2026-05-16T07:20:00Z", "2026-05-16T07:22:00Z"),
    doc("news_macro_rate_m02", "NEWS", "Yonhap Economy RSS", "RSS",
        "https://www.yna.co.kr/view/AKR20260516300000088", "Yonhap News", "KR",
        "미 국채금리 5%대 재진입·달러 강세…신흥국 증시 자금 이탈 우려 확대",
        "미 10년물 국채금리가 다시 5%를 넘어섰고 달러 인덱스가 급등했다. 외국인의 "
        "한국 증시 순매도가 5거래일 연속 이어지며 대형주 전반에 거시 역풍이 거세다. "
        "기업 개별 실적과 무관하게 밸류에이션 압축 압력이 크다는 분석이다.",
        "2026-05-16T01:10:00Z", "2026-05-16T01:13:00Z"),
]
mixed_events = [
    {"event_id": "evt_mix_ss_beat", "event_type": "EARNINGS",
     "event_time": "2026-05-16T07:20:00Z", "cluster_key": "005930.KS|EARNINGS|2026-05-16",
     "confidence": 0.9, "headline": "삼성전자 2Q 컨센 19% 상회",
     "summary": "메모리 견조로 어닝비트.", "entities": [SS],
     "source_documents": ["disclosure_ss_beat_m01"],
     "sentiment_score": 0.55, "impact_score": 0.6, "risk_score": 0.15, "time_decay": 1.0,
     "quant_signals": {"dominant_polarity": "positive",
                       "matched_terms": {"positive": ["상회"], "negative": [], "risk": []},
                       "rationale": "개별 긍정."}},
    {"event_id": "evt_mix_macro", "event_type": "NEWS_FLOW",
     "event_time": "2026-05-16T01:10:00Z", "cluster_key": "MACRO|NEWS_FLOW|2026-05-16",
     "confidence": 0.85, "headline": "미 금리 5%대·외국인 순매도 — 거시 역풍",
     "summary": "금리 급등·달러 강세로 대형주 밸류 압축, 외국인 이탈.",
     "entities": [SS, HX],
     "source_documents": ["news_macro_rate_m02"],
     "sentiment_score": -0.6, "impact_score": 0.78, "risk_score": 0.72, "time_decay": 1.0,
     "quant_signals": {"dominant_polarity": "negative",
                       "matched_terms": {"positive": [], "negative": ["순매도", "역풍"],
                                         "risk": ["밸류에이션 압축"]},
                       "rationale": "거시 부정."}},
]

if __name__ == "__main__":
    print("building scenario knowledge caches:")
    write("positive", positive_docs, positive_events)
    write("mixed", mixed_docs, mixed_events)
    print("done. (semi_shock / no_signal reuse existing caches)")
