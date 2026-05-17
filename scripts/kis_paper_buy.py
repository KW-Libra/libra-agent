from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from libra_agent.libra.portfolio_sources.kis import (
    KIS_DEFAULT_CONFIG_PATH,
    KISPaperOrderClient,
    KISPortfolioBootstrapError,
    _config_from_args,
)

FIXTURE_ORDERS: tuple[tuple[str, int], ...] = (
    ("005930", 1),
    ("000660", 1),
    ("035420", 1),
    ("005380", 1),
    ("105560", 1),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Place KIS demo-only domestic market buy orders.")
    parser.add_argument(
        "--env-file",
        default=".env.kis.paper.local",
        help="Local KEY=VALUE file for LIBRA_KIS_* values. Missing files are ignored.",
    )
    parser.add_argument(
        "--kis-config",
        default=None,
        help=(
            "Optional path to kis_devlp.yaml. Defaults to LIBRA_KIS_* env vars "
            f"in --env-file; legacy default path is {KIS_DEFAULT_CONFIG_PATH}."
        ),
    )
    parser.add_argument("--kis-app-key", help="Override KIS paper app key")
    parser.add_argument("--kis-app-secret", help="Override KIS paper app secret")
    parser.add_argument("--kis-account-no", help="KIS account number, first 8 digits")
    parser.add_argument("--kis-product-code", default=None, help="KIS product code, usually 01")
    parser.add_argument("--kis-user-agent", help="Override KIS User-Agent header")
    parser.add_argument(
        "--order",
        action="append",
        default=[],
        metavar="TICKER:QTY",
        help="Paper market buy order. Repeat for multiple orders, e.g. --order 005930:1",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Buy the standard Libra test fixture: 005930, 000660, 035420, 005380, 105560, one share each.",
    )
    parser.add_argument(
        "--reserve",
        action="store_true",
        help="Place KIS reservation buy orders for the next business day instead of immediate cash orders.",
    )
    parser.add_argument(
        "--list-reservations",
        action="store_true",
        help="List KIS paper reservation orders instead of placing orders.",
    )
    parser.add_argument(
        "--reservation-start",
        default=None,
        help="Reservation query start date in YYYYMMDD. Defaults to today.",
    )
    parser.add_argument(
        "--reservation-end",
        default=None,
        help="Reservation query end date in YYYYMMDD. Defaults to --reservation-start.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_env_file(Path(args.env_file))
    orders = _parse_orders(args.order)
    if args.fixture:
        orders.extend(FIXTURE_ORDERS)
    if not orders and not args.list_reservations:
        raise SystemExit("Pass at least one --order TICKER:QTY or use --fixture.")

    config_args = argparse.Namespace(
        kis_env="demo",
        kis_config=args.kis_config,
        kis_app_key=args.kis_app_key,
        kis_app_secret=args.kis_app_secret,
        kis_account_no=args.kis_account_no,
        kis_product_code=args.kis_product_code,
        kis_user_agent=args.kis_user_agent,
    )
    try:
        config = _config_from_args(config_args)
        client = KISPaperOrderClient(config)
        if args.list_reservations:
            start_date = args.reservation_start or date.today().strftime("%Y%m%d")
            end_date = args.reservation_end or start_date
            results = client.fetch_reservation_orders(
                start_date=start_date,
                end_date=end_date,
            )
            payload = {
                "ok": True,
                "env": "demo",
                "query": {"reservation_start": start_date, "reservation_end": end_date},
                "reservation_orders": results,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.reserve:
            results = [
                client.place_market_buy_reservation(ticker=ticker, quantity=quantity)
                for ticker, quantity in orders
            ]
        else:
            results = [
                client.place_market_buy(ticker=ticker, quantity=quantity)
                for ticker, quantity in orders
            ]
    except KISPortfolioBootstrapError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "env": "demo",
                "order_mode": "reservation" if args.reserve else "cash",
                "orders": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_orders(raw_orders: list[str]) -> list[tuple[str, int]]:
    orders: list[tuple[str, int]] = []
    for raw in raw_orders:
        if ":" not in raw:
            raise SystemExit(f"Invalid order format: {raw}. Use TICKER:QTY.")
        ticker, qty_text = raw.split(":", 1)
        try:
            quantity = int(qty_text)
        except ValueError as exc:
            raise SystemExit(f"Invalid order quantity: {raw}") from exc
        if quantity <= 0:
            raise SystemExit(f"Order quantity must be positive: {raw}")
        orders.append((ticker.strip(), quantity))
    return orders


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        os.environ.setdefault(name, value)


if __name__ == "__main__":
    raise SystemExit(main())
