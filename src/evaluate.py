"""Run the decision pipeline against labelled cases and report accuracy.

    python -m src.evaluate                              # the 5 supplied sample cases
    python -m src.evaluate --dataset data/tickets.csv --limit 50
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from .database import init_db
from .decision import DecisionUnavailable, decide
from .retrieval import ingest
from .schemas import TicketIn


def _num(value, cast):
    value = (value or "").strip() if isinstance(value, str) else value
    return cast(value) if value not in (None, "", "unknown") else None


def load_cases(path: Path) -> List[Tuple[str, TicketIn, str]]:
    """Returns (case_id, ticket, expected_action). Handles the sample JSON and
    the historical tickets.csv, which label the same thing under different keys."""
    if path.suffix == ".csv":
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        id_key, label_key = "ticket_id", "resolved_action"
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        id_key, label_key = "case_id", "expected_action"

    cases = []
    for row in rows:
        ticket = TicketIn(
            message=row["message"],
            order_value_inr=_num(row.get("order_value_inr"), float),
            days_since_delivery=_num(row.get("days_since_delivery"), int),
            days_since_dispatch=_num(row.get("days_since_dispatch"), int),
            product_type=row.get("product_type") or None,
            opened_status=row.get("opened_status") or None,
            order_status=row.get("order_status") or None,
        )
        cases.append((str(row[id_key]), ticket, row[label_key]))
    return cases


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("sample_test_cases.json"))
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    args = parser.parse_args()

    init_db()
    ingest()

    cases = load_cases(args.dataset)[: args.limit]
    correct = 0
    failures = []
    errors = []
    consecutive_errors = 0

    for case_id, ticket, expected in cases:
        try:
            decision, chunks = decide(ticket)
        except DecisionUnavailable as exc:
            # A failed call is not a wrong answer. Scoring it as one turns an outage
            # or an exhausted quota into a fake accuracy number.
            errors.append((case_id, str(exc)))
            consecutive_errors += 1
            print(f"ERROR {case_id:>5}  model call failed: {str(exc)[:120]}")
            if consecutive_errors >= 3:
                print("\nAborting: three consecutive failures, the API is not answering.")
                break
            continue
        consecutive_errors = 0

        ok = decision.action == expected
        correct += ok
        print(
            f"{'PASS' if ok else 'FAIL'}  {case_id:>5}  "
            f"expected={expected:<29} got={decision.action:<29} conf={decision.confidence:.2f}"
        )
        if not ok:
            failures.append((case_id, expected, decision, [c.source for c in chunks]))

    scored = correct + len(failures)
    print(f"\n{scored} test cases")
    print(f"Correct: {correct}")
    print(f"Incorrect: {len(failures)}")
    print(f"Accuracy: {correct / scored * 100:.0f}%" if scored else "Accuracy: n/a")
    if errors:
        print(f"Errored (not scored): {len(errors)} of {len(cases)} attempted")

    if failures:
        print("\nFailures in detail:")
        for case_id, expected, decision, sources in failures:
            print(f"  {case_id}: expected {expected}, got {decision.action}")
            print(f"    reason:    {decision.reason}")
            print(f"    cited:     {decision.sources}")
            print(f"    retrieved: {sorted(set(sources))}")

    if errors:
        print(f"\nFirst error: {errors[0][1]}")

    return 0 if scored and correct == scored and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
