from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx


REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
DEMO_BASE_URL = "https://openapivts.koreainvestment.com:29443"
TOKEN_PATH = "/oauth2/tokenP"
DAILY_CHART_PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
DAILY_CHART_TR_ID = "FHKST03010100"


def _load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        raise RuntimeError(f"env file does not exist: {env_path}")
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return ""


def _kis_base_url() -> str:
    explicit = _first_env("LIBRA_KIS_BASE_URL", "KIS_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    env = _first_env("LIBRA_KIS_ENV", "KIS_ENVIRONMENT", "KIS_ENV").casefold()
    if env in {"real", "prod", "production"}:
        return REAL_BASE_URL
    return DEMO_BASE_URL


def _credentials() -> tuple[str, str]:
    app_key = _first_env(
        "LIBRA_KIS_APP_KEY",
        "LIBRA_KIS_REAL_APP_KEY",
        "LIBRA_KIS_DEMO_APP_KEY",
        "KIS_APP_KEY",
    )
    app_secret = _first_env(
        "LIBRA_KIS_APP_SECRET",
        "LIBRA_KIS_REAL_APP_SECRET",
        "LIBRA_KIS_DEMO_APP_SECRET",
        "KIS_APP_SECRET",
    )
    missing = []
    if not app_key:
        missing.append("LIBRA_KIS_APP_KEY")
    if not app_secret:
        missing.append("LIBRA_KIS_APP_SECRET")
    if missing:
        raise RuntimeError("missing KIS credentials: " + ", ".join(missing))
    return app_key, app_secret


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _yyyymmdd(day: date) -> str:
    return day.strftime("%Y%m%d")


def _date_chunks(start: date, end: date, *, days: int) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=days - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _issue_access_token(client: httpx.Client, *, base_url: str, app_key: str, app_secret: str) -> str:
    response = client.post(
        f"{base_url}{TOKEN_PATH}",
        headers={"Content-Type": "application/json"},
        json={"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret},
    )
    response.raise_for_status()
    payload = response.json()
    token = str(payload.get("access_token") or "").strip()
    if not token:
        raise RuntimeError("KIS token response did not include access_token")
    return token


def _fetch_daily_chart(
    client: httpx.Client,
    *,
    base_url: str,
    app_key: str,
    app_secret: str,
    token: str,
    ticker: str,
    market_code: str,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    response = client.get(
        f"{base_url}{DAILY_CHART_PATH}",
        headers={
            "authorization": f"Bearer {token}",
            "appkey": app_key,
            "appsecret": app_secret,
            "tr_id": DAILY_CHART_TR_ID,
            "custtype": "P",
        },
        params={
            "FID_COND_MRKT_DIV_CODE": market_code,
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": _yyyymmdd(start),
            "FID_INPUT_DATE_2": _yyyymmdd(end),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "1",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("rt_cd") or "") != "0":
        raise RuntimeError(
            f"KIS daily chart failed for {ticker}: {payload.get('msg_cd')} {payload.get('msg1')}"
        )
    rows = payload.get("output2") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _fetch_daily_chart_resilient(
    client: httpx.Client,
    *,
    base_url: str,
    app_key: str,
    app_secret: str,
    token: str,
    ticker: str,
    market_code: str,
    start: date,
    end: date,
    allow_missing: bool,
) -> list[dict[str, Any]]:
    try:
        return _fetch_daily_chart(
            client,
            base_url=base_url,
            app_key=app_key,
            app_secret=app_secret,
            token=token,
            ticker=ticker,
            market_code=market_code,
            start=start,
            end=end,
        )
    except (httpx.HTTPError, RuntimeError):
        if (end - start).days <= 10:
            if allow_missing:
                print(
                    f"warning: skipping KIS failed range {ticker} {start.isoformat()}..{end.isoformat()}",
                    flush=True,
                )
                return []
            raise
        midpoint = start + timedelta(days=(end - start).days // 2)
        return [
            *_fetch_daily_chart_resilient(
                client,
                base_url=base_url,
                app_key=app_key,
                app_secret=app_secret,
                token=token,
                ticker=ticker,
                market_code=market_code,
                start=start,
                end=midpoint,
                allow_missing=allow_missing,
            ),
            *_fetch_daily_chart_resilient(
                client,
                base_url=base_url,
                app_key=app_key,
                app_secret=app_secret,
                token=token,
                ticker=ticker,
                market_code=market_code,
                start=midpoint + timedelta(days=1),
                end=end,
                allow_missing=allow_missing,
            ),
        ]


def _normalize_chart_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for row in rows:
        raw_day = str(row.get("stck_bsop_date") or "")
        if len(raw_day) != 8:
            continue
        day = f"{raw_day[:4]}-{raw_day[4:6]}-{raw_day[6:8]}"
        try:
            normalized[day] = {
                "open": float(str(row.get("stck_oprc") or "0").replace(",", "")),
                "high": float(str(row.get("stck_hgpr") or "0").replace(",", "")),
                "low": float(str(row.get("stck_lwpr") or "0").replace(",", "")),
                "close": float(str(row.get("stck_clpr") or "0").replace(",", "")),
                "volume": float(str(row.get("acml_vol") or "0").replace(",", "")),
            }
        except ValueError:
            continue
    return normalized


def _fetch_ticker_history(
    *,
    client: httpx.Client,
    base_url: str,
    app_key: str,
    app_secret: str,
    token: str,
    ticker: str,
    market_code: str,
    start: date,
    end: date,
    chunk_days: int,
    sleep_seconds: float,
    allow_missing: bool,
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for chunk_start, chunk_end in _date_chunks(start, end, days=chunk_days):
        rows = _fetch_daily_chart_resilient(
            client,
            base_url=base_url,
            app_key=app_key,
            app_secret=app_secret,
            token=token,
            ticker=ticker,
            market_code=market_code,
            start=chunk_start,
            end=chunk_end,
            allow_missing=allow_missing,
        )
        merged.update(_normalize_chart_rows(rows))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return dict(sorted(merged.items()))


def _fetch_pykrx_history(
    *,
    ticker: str,
    start: date,
    end: date,
) -> dict[str, dict[str, float]]:
    try:
        from pykrx import stock
    except ImportError as exc:
        raise RuntimeError(
            "pykrx is required for --provider pykrx. Install with `python -m pip install pykrx`."
        ) from exc

    df = stock.get_market_ohlcv_by_date(_yyyymmdd(start), _yyyymmdd(end), ticker)
    history: dict[str, dict[str, float]] = {}
    for index, row in df.iterrows():
        day = index.strftime("%Y-%m-%d")
        history[day] = {
            "open": float(row["시가"]),
            "high": float(row["고가"]),
            "low": float(row["저가"]),
            "close": float(row["종가"]),
            "volume": float(row["거래량"]),
        }
    return dict(sorted(history.items()))


def enrich_fixture(
    fixture: Mapping[str, Any],
    *,
    history_by_ticker: Mapping[str, Mapping[str, Mapping[str, float]]],
    strict: bool,
    provider: str = "KIS inquire-daily-itemchartprice",
) -> dict[str, Any]:
    enriched = dict(fixture)
    prices = []
    missing: dict[str, list[str]] = {}
    for source_row in fixture.get("prices", []):
        if not isinstance(source_row, Mapping):
            continue
        row = dict(source_row)
        day = str(row.get("date") or "")
        volumes = dict(row.get("volumes") or {}) if isinstance(row.get("volumes"), Mapping) else {}
        for ticker, history in history_by_ticker.items():
            ticker_row = history.get(day)
            if not ticker_row:
                missing.setdefault(ticker, []).append(day)
                continue
            volume = ticker_row.get("volume")
            if volume and volume > 0:
                volumes[ticker] = volume
            else:
                missing.setdefault(ticker, []).append(day)
        if volumes:
            row["volumes"] = volumes
        prices.append(row)
    if strict and missing:
        samples = {
            ticker: days[:5]
            for ticker, days in missing.items()
            if days
        }
        if samples:
            raise RuntimeError(f"KIS volume history missing fixture dates: {samples}")
    enriched["prices"] = prices
    enriched["liquidity_history"] = {
        "provider": provider,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": sorted(history_by_ticker.keys()),
        "rows_per_ticker": {
            ticker: len(history)
            for ticker, history in sorted(history_by_ticker.items())
        },
    }
    return enriched


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach historical daily volumes to a LIBRA comparison fixture."
    )
    parser.add_argument(
        "--provider",
        default="kis",
        choices=("kis", "pykrx"),
        help="Volume source. pykrx is preferred for strict KR historical fixtures; kis uses KIS daily chart.",
    )
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--env-file", default=".env.kis.paper.local")
    parser.add_argument("--market-code", default="J")
    parser.add_argument("--ticker", action="append", help="Optional ticker override; defaults to target_weights keys.")
    parser.add_argument("--chunk-days", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    _load_env_file(args.env_file)
    fixture = _read_json(Path(args.fixture))
    tickers = args.ticker or sorted(str(ticker) for ticker in fixture.get("target_weights", {}))
    dates = [_parse_day(str(row["date"])) for row in fixture.get("prices", []) if isinstance(row, Mapping) and row.get("date")]
    if not dates:
        raise RuntimeError("fixture prices must include at least one dated row")
    start = min(dates)
    end = max(dates)
    history_by_ticker = {}
    if args.provider == "pykrx":
        for ticker in tickers:
            history_by_ticker[ticker] = _fetch_pykrx_history(
                ticker=ticker,
                start=start,
                end=end,
            )
            print(f"{ticker}: fetched {len(history_by_ticker[ticker])} pykrx daily rows", flush=True)
            if args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))
    else:
        app_key, app_secret = _credentials()
        base_url = _kis_base_url()
        with httpx.Client(timeout=max(1.0, float(args.timeout_seconds))) as client:
            token = _issue_access_token(client, base_url=base_url, app_key=app_key, app_secret=app_secret)
            for ticker in tickers:
                history_by_ticker[ticker] = _fetch_ticker_history(
                    client=client,
                    base_url=base_url,
                    app_key=app_key,
                    app_secret=app_secret,
                    token=token,
                    ticker=ticker,
                    market_code=args.market_code,
                    start=start,
                    end=end,
                    chunk_days=max(1, int(args.chunk_days)),
                    sleep_seconds=max(0.0, float(args.sleep_seconds)),
                    allow_missing=bool(args.allow_missing),
                )
                print(f"{ticker}: fetched {len(history_by_ticker[ticker])} KIS daily rows", flush=True)
    enriched = enrich_fixture(
        fixture,
        history_by_ticker=history_by_ticker,
        strict=not args.allow_missing,
        provider="pykrx.get_market_ohlcv_by_date"
        if args.provider == "pykrx"
        else "KIS inquire-daily-itemchartprice",
    )
    _write_json(Path(args.out), enriched)
    print(f"wrote enriched fixture: {args.out}", flush=True)


if __name__ == "__main__":
    main()
