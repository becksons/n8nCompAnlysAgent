import streamlit as st
import requests
import pandas as pd
import re
import json
from typing import Any, Dict, List, Tuple, Union

st.title("AI Agent Research Assistant Metric Finder")

user_input = st.text_area("Enter your product idea")



#Production webhook 
WEBHOOK_URL = "https://bsonbec.app.n8n.cloud/webhook/118b8ace-03f8-4a7f-8ade-138a749a02ea"




# -----------------------------------------------------------------------
#                   Parsing and counting output ranks
# -----------------------------------------------------------------------
def coerce_num(x: Any) -> int:
    """convert '7' / 7 / '7 (notes...)' -> 7"""
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+", x)
        return int(m.group(0)) if m else 0
    return 0

def rows_from_rank_schema(d: Dict[str, Any]) -> List[Tuple[int, str, int]]:
    """
    From dict like:
      {"rank1":"Accuracy","rank1_mention_#":7,"rank2":"Durability","rank2_mention_#":4,...}
    -> [(1, "Accuracy", 7), (2, "Durability", 4), ...]  (skips empty)
    """
    rows: List[Tuple[int, str, int]] = []
    # Find all rankN keys, keep numeric N for ordering
    rank_idxs = []
    for k in d.keys():
        m = re.match(r"^rank(\d+)$", str(k))
        if m:
            rank_idxs.append(int(m.group(1)))
    rank_idxs.sort()
    for idx in rank_idxs:
        name = str(d.get(f"rank{idx}", "")).strip()
        if not name:
            continue
        mentions = coerce_num(d.get(f"rank{idx}_mention_#"))
        rows.append((idx, name, mentions))
    return rows

def normalize_payload_to_rank_dicts(payload: Any) -> List[Dict[str, Any]]:
    """
    Accepts whatever the n8n webhook returns and returns a list of dicts
    each containing rankN / rankN_mention_# keys if they exist
    
    ** Data formats: 
      Direct rank object: {"rank1": "...", "rank1_mention_#": ...}
      Wrapped under {"ranks": {...}} or {"ranks": [ {...}, ... ]}
      Anything else -> ignore
    **
    
    
    
    """
    dicts: List[Dict[str, Any]] = []

    
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
          
            return dicts
 
    data_list: List[Any] = payload if isinstance(payload, list) else [payload]

    for elem in data_list:
        d = elem
        if isinstance(d, dict) and "json" in d and isinstance(d["json"], dict):
            d = d["json"]  
        
        if isinstance(d, dict):
            if any(re.match(r"^rank\d+$", k or "") for k in d.keys()):
                dicts.append(d)
                continue

            ranks = d.get("ranks")
            if isinstance(ranks, dict):
                dicts.append(ranks)
                continue
            if isinstance(ranks, list):
                for r in ranks:
                    if isinstance(r, dict) and any(re.match(r"^rank\d+$", k or "") for k in r.keys()):
                        dicts.append(r)

    return dicts

MENTION_LINE = re.compile(r"^\s*(\d+)\s*mentions?\s*[-–—:]\s*(.*)$", re.IGNORECASE)

def parse_metric_map(d: Dict[str, Any]) -> pd.DataFrame:
    """
    d = { "Accuracy": "15 mentions - text", ... }
    -> DataFrame[metric, mentions, note]
    """
    rows = []
    for metric, val in d.items():
        if not isinstance(metric, str):
            continue
        if isinstance(val, (int, float)):
            rows.append((metric.strip(), int(val), ""))
        elif isinstance(val, str):
            m = MENTION_LINE.match(val.strip())
            if m:
                rows.append((metric.strip(), int(m.group(1)), m.group(2).strip()))
            else:
                #grab first number if present
                rows.append((metric.strip(), coerce_num(val), val.strip()))
    df = pd.DataFrame(rows, columns=["metric", "Amazon Review Mentions", "note"])
    if not df.empty:
        df = df.sort_values("Amazon Review Mentions", ascending=False, kind="mergesort").reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)
    return df

def build_dataframe_from_payload(payload: Any) -> pd.DataFrame:
    # First: your rankN schema (unchanged)
    rank_dicts = normalize_payload_to_rank_dicts(payload)
    rows: List[Tuple[int, str, int]] = []
    for rd in rank_dicts:
        rows.extend(rows_from_rank_schema(rd))
    if rows:
        df = pd.DataFrame(rows, columns=["rank", "metric", "Amazon Review Mentions"]).dropna()
        return df.sort_values(["rank"], kind="mergesort").reset_index(drop=True)

    # NEW: handle metric → "N mentions - note" map (what your screenshot shows)
    if isinstance(payload, dict):
        # n8n wrap under .json
        d = payload.get("json", payload) if isinstance(payload, dict) else payload
        if isinstance(d, dict):
            # values are strings that look like “N mentions …”
            if any(isinstance(v, str) and re.search(r"\bmentions?\b", v, re.I) for v in d.values()):
                df = parse_metric_map(d)
                if not df.empty:
                    return df

    # Fallback 
    if isinstance(payload, (dict, list)):
        tallies: Dict[str, int] = {}
        items = payload if isinstance(payload, list) else [payload]
        for it in items:
            d = it.get("json", it) if isinstance(it, dict) else {}
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, (int, float)) or (isinstance(v, str) and re.search(r"\d", v)):
                        tallies[str(k).strip()] = tallies.get(str(k).strip(), 0) + coerce_num(v)
        if tallies:
            df = pd.DataFrame(
                [{"metric": k, "Amazon Review Mentions": v} for k, v in tallies.items()]
            ).sort_values("Amazon Review Mentions", ascending=False, kind="mergesort").reset_index(drop=True)
            df.insert(0, "rank", df.index + 1)
            df["note"] = ""  # no notes in this shape
            return df

    return pd.DataFrame(columns=["rank", "metric", "Amazon Review Mentions", "note"])


# -----------------------------------------------------------------------
#                               Main UI action
# -----------------------------------------------------------------------
if st.button("Submit"):
    if not user_input.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Send the product idea to n8n
                payload = {"query": user_input}
                res = requests.post(WEBHOOK_URL, json=payload, timeout=120)

                # Parsing the results
                text = res.text
                parsed: Union[dict, list, str]
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = text  # keep raw

                df = build_dataframe_from_payload(parsed)
                if df.empty:
                    st.error("No ranked metrics were found in the response.")
                else:
                    st.success(f"Found {len(df)} ranked metrics:")
                    st.dataframe(df, use_container_width=True)


                with st.expander("Raw webhook response (debug)"):
                    if isinstance(parsed, (dict, list)):
                        st.json(parsed)
                    else:
                        st.code(parsed)

            except Exception as e:
                st.error(f"❌ Error: {e}")
