from __future__ import annotations

import ast
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from ...libra_models import PortfolioHolding, PortfolioSnapshot
from ...utils import coerce_datetime

KISEnv = Literal["real", "demo"]
KIS_DEFAULT_CONFIG_PATH = Path.home() / "KIS" / "config" / "kis_devlp.yaml"
DEFAULT_USER_AGENT = "Mozilla/5.0"
REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
DEMO_BASE_URL = "https://openapivts.koreainvestment.com:29443"
DOMESTIC_BALANCE_PATH = "/uapi/domestic-stock/v1/trading/inquire-balance"
TOKEN_PATH = "/oauth2/tokenP"


class KISPortfolioBootstrapError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class KISCredentialConfig:
    env: KISEnv
    app_key: str
    app_secret: str
    account_no: str
    product_code: str
    base_url: str
    user_agent: str = DEFAULT_USER_AGENT
    timeout_seconds: float = 20.0


def build_kis_portfolio_snapshot(args: Any) -> PortfolioSnapshot:
    config = _config_from_args(args)
    client = KISDomesticPortfolioClient(config)
    return client.fetch_snapshot()


def _config_from_args(args: Any) -> KISCredentialConfig:
    env = str(getattr(args, "kis_env", "real")).strip().lower()
    if env not in {"real", "demo"}:
        raise KISPortfolioBootstrapError(f"Unsupported KIS environment: {env}")

    config_path_raw = getattr(args, "kis_config", None)
    file_values = _read_kis_config(Path(config_path_raw)) if config_path_raw else {}

    app_key = _first_non_empty(
        getattr(args, "kis_app_key", None),
        os.environ.get("LIBRA_KIS_APP_KEY"),
        file_values.get("my_app" if env == "real" else "paper_app"),
    )
    app_secret = _first_non_empty(
        getattr(args, "kis_app_secret", None),
        os.environ.get("LIBRA_KIS_APP_SECRET"),
        file_values.get("my_sec" if env == "real" else "paper_sec"),
    )
    account_no = _first_non_empty(
        getattr(args, "kis_account_no", None),
        os.environ.get("LIBRA_KIS_ACCOUNT_NO"),
        file_values.get("my_acct_stock" if env == "real" else "my_paper_stock"),
    )
    product_code = _first_non_empty(
        getattr(args, "kis_product_code", None),
        os.environ.get("LIBRA_KIS_PRODUCT_CODE"),
        file_values.get("my_prod"),
        "01",
    )
    user_agent = _first_non_empty(
        getattr(args, "kis_user_agent", None),
        os.environ.get("LIBRA_KIS_USER_AGENT"),
        file_values.get("my_agent"),
        DEFAULT_USER_AGENT,
    )
    base_url = _first_non_empty(
        file_values.get("prod" if env == "real" else "vps"),
        REAL_BASE_URL if env == "real" else DEMO_BASE_URL,
    )

    missing = [
        name
        for name, value in (
            ("app_key", app_key),
            ("app_secret", app_secret),
            ("account_no", account_no),
            ("product_code", product_code),
        )
        if not value
    ]
    if missing:
        raise KISPortfolioBootstrapError(
            "Missing KIS bootstrap values: "
            + ", ".join(missing)
            + ". Pass CLI args, LIBRA_KIS_* env vars, or --kis-config."
        )

    normalized_account = "".join(char for char in str(account_no) if char.isdigit())[:8]
    normalized_product = "".join(char for char in str(product_code) if char.isdigit())[:2]
    if len(normalized_account) != 8:
        raise KISPortfolioBootstrapError(
            "KIS account number must contain the first 8 digits of the account."
        )
    if len(normalized_product) != 2:
        raise KISPortfolioBootstrapError(
            "KIS product code must contain the final 2 digits of the account."
        )

    return KISCredentialConfig(
        env=env,
        app_key=str(app_key),
        app_secret=str(app_secret),
        account_no=normalized_account,
        product_code=normalized_product,
        base_url=str(base_url).rstrip("/"),
        user_agent=str(user_agent),
    )


class KISDomesticPortfolioClient:
    def __init__(self, config: KISCredentialConfig) -> None:
        self.config = config

    def fetch_snapshot(self, *, http_client: httpx.Client | None = None) -> PortfolioSnapshot:
        owns_client = http_client is None
        client = http_client or httpx.Client(timeout=self.config.timeout_seconds)
        try:
            token = self._issue_access_token(client)
            holdings_rows, summary_rows = self._fetch_domestic_balance(client, token=token)
            return self._to_snapshot(holdings_rows, summary_rows)
        finally:
            if owns_client:
                client.close()

    def _issue_access_token(self, client: httpx.Client) -> str:
        response = client.post(
            f"{self.config.base_url}{TOKEN_PATH}",
            headers={
                "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
            },
            json={
                "grant_type": "client_credentials",
                "appkey": self.config.app_key,
                "appsecret": self.config.app_secret,
            },
        )
        self._raise_for_http(response, "token issuance")
        payload = response.json()
        token = str(payload.get("access_token", "")).strip()
        if not token:
            raise KISPortfolioBootstrapError("KIS token response did not include access_token.")
        return token

    def _fetch_domestic_balance(
        self,
        client: httpx.Client,
        *,
        token: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        all_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        fk100 = ""
        nk100 = ""
        tr_cont = ""

        for _ in range(10):
            response = client.get(
                f"{self.config.base_url}{DOMESTIC_BALANCE_PATH}",
                headers={
                    "authorization": f"Bearer {token}",
                    "appkey": self.config.app_key,
                    "appsecret": self.config.app_secret,
                    "tr_id": "TTTC8434R" if self.config.env == "real" else "VTTC8434R",
                    "custtype": "P",
                    "tr_cont": tr_cont,
                    "User-Agent": self.config.user_agent,
                },
                params={
                    "CANO": self.config.account_no,
                    "ACNT_PRDT_CD": self.config.product_code,
                    "AFHR_FLPR_YN": "N",
                    "OFL_YN": "",
                    "INQR_DVSN": "02",
                    "UNPR_DVSN": "01",
                    "FUND_STTL_ICLD_YN": "N",
                    "FNCG_AMT_AUTO_RDPT_YN": "N",
                    "PRCS_DVSN": "01",
                    "CTX_AREA_FK100": fk100,
                    "CTX_AREA_NK100": nk100,
                },
            )
            self._raise_for_http(response, "domestic balance fetch")
            payload = response.json()
            if str(payload.get("rt_cd", "")) != "0":
                raise KISPortfolioBootstrapError(
                    f"KIS domestic balance fetch failed: {payload.get('msg_cd', '')} {payload.get('msg1', '')}".strip()
                )

            page_rows = payload.get("output1") or []
            page_summary = payload.get("output2") or []
            if isinstance(page_rows, list):
                all_rows.extend(item for item in page_rows if isinstance(item, Mapping))
            elif isinstance(page_rows, Mapping):
                all_rows.append(dict(page_rows))
            if not summary_rows:
                if isinstance(page_summary, list):
                    summary_rows = [
                        dict(item) for item in page_summary if isinstance(item, Mapping)
                    ]
                elif isinstance(page_summary, Mapping):
                    summary_rows = [dict(page_summary)]

            tr_cont = response.headers.get("tr_cont", "")
            fk100 = str(payload.get("ctx_area_fk100", "") or "")
            nk100 = str(payload.get("ctx_area_nk100", "") or "")
            if tr_cont not in {"M", "F"}:
                break
            tr_cont = "N"

        return all_rows, summary_rows

    def _to_snapshot(
        self,
        rows: list[dict[str, Any]],
        summary_rows: list[dict[str, Any]],
    ) -> PortfolioSnapshot:
        summary = summary_rows[0] if summary_rows else {}
        total_value = _as_float(summary.get("tot_evlu_amt"))
        stock_value_sum = _as_float(summary.get("evlu_amt_smtl_amt")) or sum(
            _as_float(item.get("evlu_amt")) for item in rows
        )
        cash_total = _as_float(summary.get("dnca_tot_amt"))
        if total_value <= 0:
            total_value = stock_value_sum + cash_total

        holdings: list[PortfolioHolding] = []
        for item in rows:
            ticker = str(item.get("pdno", "")).strip()
            company_name = str(item.get("prdt_name", "")).strip()
            shares = _as_float(item.get("hldg_qty"))
            if not ticker or not company_name or shares <= 0:
                continue
            eval_amount = _as_float(item.get("evlu_amt"))
            weight = eval_amount / total_value if total_value > 0 else 0.0
            aliases = _build_aliases(ticker=ticker, company_name=company_name)
            holdings.append(
                PortfolioHolding(
                    ticker=ticker,
                    company_name=company_name,
                    weight=max(0.0, min(1.0, weight)),
                    aliases=aliases,
                    shares=shares,
                    last_price=_as_float(item.get("prpr")) or None,
                )
            )

        holdings.sort(key=lambda item: item.weight, reverse=True)
        generated_at = datetime.now().astimezone()
        cash_weight = cash_total / total_value if total_value > 0 else 0.0
        return PortfolioSnapshot(
            generated_at=coerce_datetime(generated_at.isoformat()),
            holdings=tuple(holdings),
            total_value_krw=round(total_value, 2) if total_value > 0 else None,
            cash_weight=max(0.0, min(1.0, cash_weight)),
            user_preferences=(),
        )

    @staticmethod
    def _raise_for_http(response: httpx.Response, action: str) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise KISPortfolioBootstrapError(
                f"KIS {action} failed with HTTP {response.status_code}."
            ) from exc


def _read_kis_config(path: Path) -> dict[str, str]:
    if not path.exists():
        raise KISPortfolioBootstrapError(f"KIS config file does not exist: {path}")
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        parsed[normalized_key] = _parse_scalar(value.strip())
    return parsed


def _parse_scalar(value: str) -> str:
    without_comment = _strip_inline_comment(value).strip()
    if not without_comment:
        return ""
    if without_comment[0] in {"'", '"'}:
        try:
            literal = ast.literal_eval(without_comment)
        except (SyntaxError, ValueError):
            return without_comment.strip("\"'")
        return str(literal)
    return without_comment


def _strip_inline_comment(value: str) -> str:
    quoted = False
    quote_char = ""
    result: list[str] = []
    for char in value:
        if char in {"'", '"'}:
            if quoted and char == quote_char:
                quoted = False
                quote_char = ""
            elif not quoted:
                quoted = True
                quote_char = char
        if char == "#" and not quoted:
            break
        result.append(char)
    return "".join(result)


def _first_non_empty(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _as_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _build_aliases(*, ticker: str, company_name: str) -> tuple[str, ...]:
    aliases = {f"{ticker}.KS", f"KRX:{ticker}"}
    collapsed_name = company_name.replace(" ", "")
    if collapsed_name and collapsed_name != company_name:
        aliases.add(collapsed_name)
    return tuple(sorted(aliases))
