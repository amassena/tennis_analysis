#!/usr/bin/env python3
"""Fetch tennis-pro metadata (handedness, gender, backhand style) from Wikidata.

Per DESIGN.md principle 7: don't ask humans for facts that exist in
Wikidata. This script resolves a curated list of player names to QIDs,
queries Wikidata's SPARQL endpoint for the public attributes, and writes
the result to ``pros/wikidata_cache.json`` (separate from the curated
``pros/index.json`` so a re-fetch can't clobber human curation).

Usage:
    python scripts/fetch_pros_from_wikidata.py
    python scripts/fetch_pros_from_wikidata.py --names extra_names.txt
    python scripts/fetch_pros_from_wikidata.py --output pros/wikidata_cache.json

Names file format (one per line, blank lines / # comments allowed):
    slug<TAB>Display Name
    nadal<TAB>Rafael Nadal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "pros" / "wikidata_cache.json"

WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Curated initial roster — slug -> display name. Slugs match pros/index.json
# folder names. Mix of ATP/WTA, lefties, 1HBH players, all-court types.
DEFAULT_ROSTER: list[tuple[str, str]] = [
    # ATP — active
    ("alcaraz", "Carlos Alcaraz"),
    ("sinner", "Jannik Sinner"),
    ("djokovic", "Novak Djokovic"),
    ("medvedev", "Daniil Medvedev"),
    ("zverev", "Alexander Zverev"),
    ("tsitsipas", "Stefanos Tsitsipas"),
    ("rublev", "Andrey Rublev"),
    ("deminaur", "Alex de Minaur"),
    ("fritz", "Taylor Fritz"),
    ("hurkacz", "Hubert Hurkacz"),
    ("rune", "Holger Rune"),
    # ATP — retired greats
    ("federer", "Roger Federer"),
    ("nadal", "Rafael Nadal"),
    ("murray", "Andy Murray"),
    ("wawrinka", "Stan Wawrinka"),
    # WTA — active
    ("sabalenka", "Aryna Sabalenka"),
    ("swiatek", "Iga Swiatek"),
    ("gauff", "Coco Gauff"),
    ("rybakina", "Elena Rybakina"),
    ("pegula", "Jessica Pegula"),
    ("jabeur", "Ons Jabeur"),
    # WTA — retired greats
    ("serena", "Serena Williams"),
    ("venus", "Venus Williams"),
    ("henin", "Justine Henin"),
]

# QID-to-attribute mappings for P741 ("plays") values.
HANDEDNESS_QID = {
    "Q789447": "left",
    "Q3039938": "right",
}
BACKHAND_STYLE_QID = {
    "Q14420039": "one-handed",
    "Q14420068": "two-handed",
}
GENDER_QID = {
    "Q6581097": "male",
    "Q6581072": "female",
}

USER_AGENT = "tennis-analysis/1.0 (https://github.com/amassena/tennis_analysis)"


def http_get_json(url: str, params: dict, timeout: int = 30) -> dict:
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{url}?{qs}",
        headers={
            "Accept": "application/sparql-results+json, application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_qid(name: str) -> str | None:
    """Resolve a player name to its Wikidata QID.

    Searches Wikidata, picks the first hit whose description mentions
    "tennis". Returns None if no plausible match.
    """
    data = http_get_json(
        WIKIDATA_SEARCH,
        {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json",
            "limit": 5,
            "type": "item",
        },
    )
    for hit in data.get("search", []):
        desc = (hit.get("description") or "").lower()
        if "tennis" in desc:
            return hit["id"]
    # Fall back to first hit if nothing mentions tennis (rare but defensible
    # for players whose Wikidata description omits the word).
    hits = data.get("search", [])
    return hits[0]["id"] if hits else None


def fetch_player_attrs(qids: list[str]) -> dict[str, dict]:
    """Bulk SPARQL fetch of P741/P21/P569 for the given QIDs.

    Returns ``{qid: {handedness, backhand_style, gender, birth_year, raw_p741}}``.
    Players missing from the result map come back unchanged (caller decides
    what to do with them).
    """
    if not qids:
        return {}
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
        SELECT ?player ?prop ?value WHERE {{
          VALUES ?player {{ {values} }}
          VALUES ?prop {{ wdt:P741 wdt:P21 wdt:P569 }}
          ?player ?prop ?value .
        }}
    """
    data = http_get_json(WIKIDATA_SPARQL, {"query": query, "format": "json"})

    by_qid: dict[str, dict] = {}
    for binding in data.get("results", {}).get("bindings", []):
        qid = binding["player"]["value"].rsplit("/", 1)[-1]
        prop = binding["prop"]["value"].rsplit("/", 1)[-1]
        val = binding["value"]["value"]
        rec = by_qid.setdefault(qid, {"raw_p741": []})

        if prop == "P741":
            val_qid = val.rsplit("/", 1)[-1]
            rec["raw_p741"].append(val_qid)
            if val_qid in HANDEDNESS_QID:
                rec["handedness"] = HANDEDNESS_QID[val_qid]
            elif val_qid in BACKHAND_STYLE_QID:
                rec["backhand_style"] = BACKHAND_STYLE_QID[val_qid]
        elif prop == "P21":
            val_qid = val.rsplit("/", 1)[-1]
            if val_qid in GENDER_QID:
                rec["gender"] = GENDER_QID[val_qid]
        elif prop == "P569":
            # ISO date string, e.g. "1986-06-03T00:00:00Z"
            try:
                rec["birth_year"] = int(val[:4])
            except (TypeError, ValueError):
                pass

    # Wikidata oddity: famous right-handers (e.g. Federer) often only carry
    # the grip-style P741 values without an explicit "right-handedness" QID.
    # If we never saw left-handedness, default to right.
    for rec in by_qid.values():
        if "handedness" not in rec:
            rec["handedness"] = "right"

    return by_qid


def load_roster(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return list(DEFAULT_ROSTER)
    roster: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            slug, name = line.split("\t", 1)
        else:
            # Allow "slug Display Name" with single space tolerance
            slug, _, name = line.partition(" ")
        roster.append((slug.strip(), name.strip()))
    return roster


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--names",
        type=Path,
        default=None,
        help="Optional roster file (TSV: slug<TAB>Display Name). "
        "Falls back to the in-script DEFAULT_ROSTER.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write the JSON to stdout instead of --output.",
    )
    parser.add_argument(
        "--qid-overrides",
        type=Path,
        default=None,
        help="Optional TSV of slug<TAB>QID to skip name->QID search "
        "for ambiguous names (e.g. 'sinner' often misresolves).",
    )
    args = parser.parse_args()

    roster = load_roster(args.names)
    overrides: dict[str, str] = {}
    if args.qid_overrides and args.qid_overrides.exists():
        for line in args.qid_overrides.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            slug, _, qid = line.partition("\t")
            if slug and qid:
                overrides[slug.strip()] = qid.strip()

    print(f"Resolving {len(roster)} players to Wikidata QIDs...", file=sys.stderr)
    slug_to_qid: dict[str, str] = {}
    for slug, name in roster:
        if slug in overrides:
            slug_to_qid[slug] = overrides[slug]
            print(f"  {slug:12s} -> {overrides[slug]} (override)", file=sys.stderr)
            continue
        qid = search_qid(name)
        if qid:
            slug_to_qid[slug] = qid
            print(f"  {slug:12s} -> {qid} ({name})", file=sys.stderr)
        else:
            print(f"  {slug:12s} -> NOT FOUND ({name})", file=sys.stderr)
        time.sleep(0.1)  # be polite to Wikidata

    print(f"\nFetching P741/P21/P569 for {len(slug_to_qid)} QIDs...", file=sys.stderr)
    attrs_by_qid = fetch_player_attrs(list(slug_to_qid.values()))

    out = {
        "_meta": {
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "Wikidata SPARQL (P741, P21, P569)",
            "schema_version": 1,
        },
        "players": {},
    }
    for slug, name in roster:
        qid = slug_to_qid.get(slug)
        if not qid:
            out["players"][slug] = {"name": name, "qid": None, "error": "qid_not_found"}
            continue
        attrs = attrs_by_qid.get(qid, {})
        out["players"][slug] = {
            "name": name,
            "qid": qid,
            "handedness": attrs.get("handedness"),
            "backhand_style": attrs.get("backhand_style"),
            "gender": attrs.get("gender"),
            "birth_year": attrs.get("birth_year"),
            "raw_p741": attrs.get("raw_p741", []),
        }

    payload = json.dumps(out, indent=2, ensure_ascii=False)
    if args.stdout:
        sys.stdout.write(payload + "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"\nWrote {args.output.relative_to(PROJECT_ROOT)}", file=sys.stderr)

    missing_hand = [s for s, p in out["players"].items() if not p.get("handedness")]
    if missing_hand:
        print(
            f"\n[WARN] {len(missing_hand)} players missing handedness: {missing_hand}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
