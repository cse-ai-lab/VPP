
# === Standard Library ===
import os
import re
import json
import math
import time
import uuid
import ast
import copy
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

    
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Third‑Party Libraries ===
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import torch
# from transformers import BertModel, BertTokenizer


@dataclass
class Paths:
    # roots (can be overridden via env vars)
    data_root: Path = field(default_factory=lambda: Path(os.getenv("RQA_DATA_ROOT", "./data/RQA_V0")).resolve())
    eval_root: Path = field(default_factory=lambda: Path(os.getenv("RQA_EVAL_ROOT", "./evaluation")).resolve())

    # derived
    img_dir: Path = field(init=False)
    chart_json_dir: Path = field(init=False)  # structural/chart JSONs
    qa_json_dir: Path = field(init=False)     # question–answer JSONs
    filter_list: Path = field(init=False)
    tax_id_map: Path = field(init=False)
    save_dir: Path = field(init=False)

    def __post_init__(self):
        self.img_dir        = (self.data_root / "images")
        self.chart_json_dir = (self.data_root / "jsons")
        self.qa_json_dir    = (self.data_root / "qa")
        self.filter_list    = (self.data_root / "test_filenames.txt")
        self.tax_id_map     = (self.eval_root / "t_id_map.npy")
        self.save_dir       = (self.data_root / "all_premises")
P = Paths()

def _count_files(d: Path, exts: Tuple[str, ...]) -> int:
    if not d.exists(): return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)

def validate_paths(p: Paths) -> None:
    problems: List[str] = []

    checks = [
        ("img_dir", p.img_dir, True),
        ("chart_json_dir", p.chart_json_dir, True),
        ("qa_json_dir", p.qa_json_dir, True),
        ("filter_list", p.filter_list, False),
        ("tax_id_map", p.tax_id_map, False),
        ("eval_root", p.eval_root, True),
    ]
    for name, path, is_dir in checks:
        if is_dir and not path.is_dir():
            problems.append(f"[missing dir] {name}: {path}")
        if not is_dir and not path.is_file():
            problems.append(f"[missing file] {name}: {path}")

    # ensure save_dir exists (we create it if needed)
    p.save_dir.mkdir(parents=True, exist_ok=True)

    if problems:
        msg = "Path validation failed:\n" + "\n".join(" - " + x for x in problems)
        raise FileNotFoundError(msg)

def summarize_dataset(p: Paths) -> None:
    img_n   = _count_files(p.img_dir, (".png", ".jpg", ".jpeg", ".webp"))
    cjson_n = _count_files(p.chart_json_dir, (".json",))
    qa_n    = _count_files(p.qa_json_dir, (".json",))

    # filter list lines (if present)
    try:
        with open(p.filter_list, "r", encoding="utf-8") as fh:
            filter_n = sum(1 for _ in fh if _.strip())
    except FileNotFoundError:
        filter_n = 0

    print("✓ Paths OK")
    print(f"  data_root        : {p.data_root}")
    print(f"  eval_root        : {p.eval_root}")
    print(f"  images           : {img_n} files")
    print(f"  chart JSONs      : {cjson_n} files  ({p.chart_json_dir.name})")
    print(f"  QA JSONs         : {qa_n} files     ({p.qa_json_dir.name})")
    print(f"  filter list      : {filter_n} entries")
    print(f"  tax_id_map       : {'exists' if p.tax_id_map.exists() else 'missing'}")
    print(f"  save_dir         : {p.save_dir} (created if missing)")


def load_qa_jsons(input_dir: Path, filter_ids=None):
    """
    Load QA JSONs into nested dict: image_id -> list of QAs.
    filter_ids: optional set of image ids to include.
    """
    t0 = time.time()
    qa_list = []
    unused = 0

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    for file_name in tqdm(json_files, desc="Loading QA"):
        image_id = Path(file_name).stem
        if filter_ids is not None and image_id not in filter_ids:
            unused += 1
            continue

        with open(input_dir / file_name, "r") as f:
            qa_list.extend(json.load(f))

    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Images: {len(json_files)} | Used: {len(qa_list)} | Skipped: {unused}")
    return qa_list



def lookup_by_qid(qa_dict, qid):
    """Debug helper: find all occurrences of a QID across images."""
    hits = []
    for image_id, qas in qa_dict.items():
        for q in qas:
            if q.get("QID") == qid:
                hits.append((image_id, q))
    return hits



# Data Premises (DP):


def DP_val(data, role="i"):
    """
    Data-point existence premise (without legend).
    Returns one True premise and several False distractors.
    role: one of 'i', 'j', 'k' (maps to _i_, _j_, _k_)
    """
    chart_vars = data["chart"]
    y_title = data["_y_title_"]
    if y_title == '!Example Y Title' and 'y_title' in data["chart"]:
        y_title = chart_vars.get('y_title')
    
    # --- tick handling ---
    tick_true = data.get(f"_{role}_")
    if not tick_true or "X Value" in str(tick_true):  # no explicit tick in QA
        if chart_vars.get("x_ticks"):
            tick_true = random.choice(chart_vars["x_ticks"])
        else:
            tick_true = "some x-tick"

    # --- True premises ---
    true_templates = [
        f"Value in the chart plot area exists at ({tick_true}) for the axis called {y_title}",
        f"The axis of {y_title} has values at points ({tick_true})",
        f"For the axis of {y_title}, there are valid plot values corresponding to ({tick_true})",
    ]
    true_output = {
        "tag": f"DP_{role}",
        "text": random.choice(true_templates),
        "truth": True,
    }

    # --- False candidates ---
    all_txt = chart_vars.get("all_txt", [])
    x_ticks = chart_vars.get("x_ticks", [])

    # tick distractors = any text not in x_ticks or != tick_true
    tick_candidates = [t for t in all_txt if t not in x_ticks and t != tick_true]
    tick_false = random.choice(tick_candidates) if tick_candidates else f"WrongTick_{role}"

    # --- False premises ---
    false_templates = [
        f"Value in the chart plot area exists at ({tick_false}) for the axis called {y_title}",
        f"The axis of {y_title} has values at points ({tick_false})",
        f"For the axis of {y_title}, there are valid plot values corresponding to ({tick_false})",
    ]
    false_outputs = [
        {"tag": f"DP_{role}", "text": t, "truth": False} for t in false_templates
    ]

    return [true_output] + false_outputs
    

def DP_val_leg(data, role="i"):
    """
    Generate Data Premises (DP) for a given tick/legend pair.
    Returns one True premise and several False distractors.
    """
    chart_vars = data["chart"]
    y_title = data["_y_title_"]
    if y_title == '!Example Y Title' and 'y_title' in data["chart"]:
        y_title = chart_vars.get('y_title')
    
    # --- tick handling ---
    tick_true = data.get(f"_{role}_")
    if not tick_true or "X Value" in str(tick_true):  # no explicit tick in QA
        if chart_vars.get("x_ticks"):
            tick_true = random.choice(chart_vars["x_ticks"])
        else:
            tick_true = "some x-tick"
    
    legend_true = data["_legend_"]
    if legend_true == '!Example Legend Label':
        legend_true = data["_legend1_"] if role == 'i' else data["_legend2_"]
    

    # --- True premises ---
    true_templates = [
        f"Value in the chart plot area exists at ({tick_true}) for the axis called {y_title} for the data series {legend_true}",
        f"The axis of {y_title} for the data series {legend_true} has values at points ({tick_true})",
        f"For the axis of {y_title}, there are valid plot values corresponding to ({tick_true}) of the data series {legend_true}",
    ]
    true_output = {
        "tag": f"DP_leg_{role}",
        "text": random.choice(true_templates),
        "truth": True,
    }

    # --- False candidates ---
    all_txt = chart_vars.get("all_txt", [])
    x_ticks = chart_vars.get("x_ticks", [])
    
    if 'legend_labels' in chart_vars : 
        legend_labels = chart_vars["legend_labels"]
    elif 'mark_label' in chart_vars : 
        legend_labels = chart_vars["mark_label"]
    else : 
        legend_labels = []
        # print('DP_val_leg :: No Legend/Mark Label Found')
        # print('--> chart_vars')
        # for k in chart_vars : 
        #     print(k, chart_vars[k])
        # print('chart_vars <<--')
    # print(legend_labels)

    # tick distractors = any text not in x_ticks or != tick_true
    tick_candidates = [t for t in all_txt if t not in x_ticks and t != tick_true]
    # legend distractors = any text not in legend_labels or != legend_true
    try :
        legend_candidates = [l for l in all_txt if l not in legend_labels and l != legend_true]
    except :
        legend_candidates = [l for l in all_txt ]
    tick_false = random.choice(tick_candidates) if tick_candidates else f"WrongTick_{role}"
    legend_false = random.choice(legend_candidates) if legend_candidates else "WrongLegend"

    # --- False premises ---
    false_templates = [
        f"Value in the chart plot area exists at ({tick_false}) for the axis called {y_title} for the data series {legend_true}",  # wrong tick
        f"The axis of {y_title} for the data series {legend_false} has values at points ({tick_true})",  # wrong legend
        f"For the axis of {y_title}, there are valid plot values corresponding to ({tick_false}) of the data series {legend_false}",  # wrong combo
    ]
    false_outputs = [
        {"tag": f"DP_leg_{role}", "text": t, "truth": False} for t in false_templates
    ]

    return [true_output] + false_outputs

def DP_val_leg_all(data, legends=("legend1", "legend2", "legend3")):
    """
    Data Premise variant for Q72:
    Ensures that values exist across *all x-ticks* for the given legends.
    """
    y_title = data.get("_y_title_", "Y-axis")
    x_title = data.get("_x_title_", "X-axis")
    chart_vars = data["chart"]

    # Legends being checked
    legend_labels = [data.get(f"_{leg}_", f"!Example {leg}") for leg in legends]

    # X-ticks from chart
    ticks = chart_vars.get("x_ticks", [])

    # --- True templates ---
    true_texts = [
        f"For every {x_title}, valid {y_title} values exist across all x-ticks for {', '.join(legend_labels)}",
        f"Each tick of {x_title} has corresponding {y_title} values for {', '.join(legend_labels)}",
        f"The chart contains complete {y_title} data series for all {len(ticks)} {x_title} values under {', '.join(legend_labels)}"
    ]
    true_outputs = [{"tag": "DP_leg_all", "text": t, "truth": True} for t in true_texts]

    # --- False distractors ---
    false_templates = [
        f"Some ticks of {x_title} have missing {y_title} values for {random.choice(legend_labels)}",
        f"The chart only has partial {y_title} values for {random.choice(legend_labels)} across ticks",
        f"There are no valid {y_title} values at several {x_title} ticks for {random.choice(legend_labels)}"
    ]
    false_outputs = [{"tag": "DP_leg_all", "text": t, "truth": False} for t in false_templates]

    # Return 1 true + distractors
    return [random.choice(true_outputs)] + false_outputs



def DP_max_val(data):
    _y_title_ = data['_y_title_']
    _xi_, _yi_ = data['_xi_,_yi_']
    Dp_Max = [
        f'The value for {_y_title_} is maximum at ({_xi_} , {_yi_})',
        f'The maximum value of {_y_title_}, exists at ({_xi_} , {_yi_})', 
        f'Maximum {_y_title_} is at ({_xi_} , {_yi_})'
    ]
    return random.choice(Dp_Max)


def DP_Min_Box(data, role="i"):
    """
    Data premise: minimum value of boxplot at tick `role`.
    """
    y = data["_y_title_"]
    tick = data.get(f"_{role}_", f"X Value {role}")
    chart_data = data["chart"].get("data", [{}])[0].get("data", [])

    min_val = None
    for entry in chart_data:
        if str(entry.get("x")) == str(tick):
            min_val = entry.get("min")
            min_val = str(round(float(min_val), 2))
            break
    if min_val is None:
        min_val = "Unknown"

    true_templates = [
        f"The minimum value of {y} at {tick} is {min_val}",
        f"At x-tick {tick}, the lowest observed value of {y} is {min_val}",
        f"For {y}, the minimum value at {tick} equals {min_val}",
    ]
    true_output = {"tag": f"DP_min_val_{role}", "text": random.choice(true_templates), "truth": True}

    # Distractor: swap with max/median/third quartile
    distractor_val = random.choice([
        entry.get("max"), entry.get("median"), entry.get("third_quartile")
    ]) if chart_data else "-0"
    distractor_val = str(round(float(distractor_val), 2))

    false_templates = [
        f"The minimum value of {y} at {tick} is {distractor_val}",
        f"At x-tick {tick}, the lowest observed value of {y} is {distractor_val}",
        f"For {y}, the minimum value at {tick} equals {distractor_val}",
    ]
    false_outputs = [{"tag": f"DP_min_val_{role}", "text": t, "truth": False} for t in false_templates]

    return [true_output] + false_outputs


def DP_max_box(data, role="i"):
    """
    Data premise: maximum value of boxplot at tick `role`.
    """
    y = data["_y_title_"]
    tick = data.get(f"_{role}_", f"X Value {role}")
    chart_data = data["chart"].get("data", [{}])[0].get("data", [])

    max_val = None
    for entry in chart_data:
        if str(entry.get("x")) == str(tick):
            max_val = entry.get("max")
            max_val = str(round(float(max_val), 2))
            break
    if max_val is None:
        max_val = "Unknown"

    true_templates = [
        f"The maximum value of {y} at {tick} is {max_val}",
        f"At x-tick {tick}, the highest observed value of {y} is {max_val}",
        f"For {y}, the maximum value at {tick} equals {max_val}",
    ]
    true_output = {"tag": f"DP_max_val_{role}", "text": random.choice(true_templates), "truth": True}

    # Distractor: swap with min/median/first quartile
    distractor_val = random.choice([
        entry.get("min"), entry.get("median"), entry.get("first_quartile")
    ]) if chart_data else "-0"
    distractor_val = str(round(float(distractor_val), 2))

    false_templates = [
        f"The maximum value of {y} at {tick} is {distractor_val}",
        f"At x-tick {tick}, the highest observed value of {y} is {distractor_val}",
        f"For {y}, the maximum value at {tick} equals {distractor_val}",
    ]
    false_outputs = [{"tag": f"DP_max_val_{role}", "text": t, "truth": False} for t in false_templates]

    return [true_output] + false_outputs

def DP_Line_Count(data):
    """
    Data premise: how many lines exist in the chart.
    """
    line_count = data["chart"]['number_of_ds']

    true_templates = [
        f"There exist {line_count} lines in the chart",
        f"In the chart, there are {line_count} lines",
        f"{line_count} lines are displayed in the chart",
    ]
    true_output = {"tag": "DP_line", "text": random.choice(true_templates), "truth": True}

    # distractor: swap with a wrong count
    distractor_count = str(int(line_count) + random.choice([-2, -1, 1, 2]))
    false_templates = [
        f"There exist {distractor_count} lines in the chart",
        f"In the chart, there are {distractor_count} lines",
        f"{distractor_count} lines are displayed in the chart",
    ]
    false_outputs = [{"tag": "DP_line", "text": t, "truth": False} for t in false_templates]

    return [true_output] + false_outputs


def DP_Leg_Count(data):
    """
    Data premise: how many legends exist in the chart.
    """
    leg_count = len(data["chart"]['legend_labels'])

    true_templates = [
        f"There exist {leg_count} legends in the chart",
        f"In the chart, there are {leg_count} legends",
        f"{leg_count} legends are displayed in the chart",
    ]
    true_output = {"tag": "DP_leg", "text": random.choice(true_templates), "truth": True}

    # distractor: swap with a wrong count
    distractor_count = str(int(leg_count) + random.choice([-2, -1, 1, 2]))
    false_templates = [
        f"There exist {distractor_count} legends in the chart",
        f"In the chart, there are {distractor_count} legends",
        f"{distractor_count} legends are displayed in the chart",
    ]
    false_outputs = [{"tag": "DP_leg", "text": t, "truth": False} for t in false_templates]

    return [true_output] + false_outputs


def DP_Mark_Count(data):
    """
    Data premise: how many mark labels exist in the chart.
    """
    mark_count = len(data["chart"].get("mark_label", []))

    true_templates = [
        f"There exist {mark_count} mark labels in the chart",
        f"In the chart, there are {mark_count} mark labels",
        f"{mark_count} mark labels are displayed in the chart",
    ]
    true_output = {"tag": "DP_mark", "text": random.choice(true_templates), "truth": True}

    # distractor: swap with a wrong count
    distractor_count = str(int(mark_count) + random.choice([-2, -1, 1, 2]))
    false_templates = [
        f"There exist {distractor_count} mark labels in the chart",
        f"In the chart, there are {distractor_count} mark labels",
        f"{distractor_count} mark labels are displayed in the chart",
    ]
    false_outputs = [{"tag": "DP_mark", "text": t, "truth": False} for t in false_templates]

    return [true_output] + false_outputs

def get_fake_tick(chart_vars):
    """
    Generate a fake tick for distractor premises in box plots.
    """
    x_ticks = chart_vars.get("x_ticks", [])
    all_txt = chart_vars.get("all_txt", [])
    xmin, xmax = chart_vars.get("xmin"), chart_vars.get("xmax")
    # Case 1: numeric ticks
    if all(isinstance(t, (int, float)) for t in x_ticks if t is not None):
        try:
            # sample a random float within range but not in x_ticks
            fake = None
            attempts = 0
            while fake is None or fake in x_ticks:
                fake = round(random.uniform(float(xmin), float(xmax)), 2)
                attempts += 1
                if attempts > 20:  # fallback to guaranteed fake
                    fake = max(x_ticks) + 1
                    break
            return fake
        except Exception:
            return "FakeTick_Num"
    # Case 2: categorical / string ticks
    else:
        candidates = [t for t in all_txt if t not in x_ticks]
        if candidates:
            return random.choice(candidates)
        return f"FakeCat_{random.randint(1, 999)}"

def DP_FQ_exist(data, role="i"):
    """
    Data premise: existence of lower quartile at tick `role`.
    - If role == "i"/"j"/"k": use that tick.
    - If role == "all": generate existence premises for all ticks in the chart.
    """
    chart_vars = data["chart"]
    chart_data = chart_vars.get("data", [{}])[0].get("data", [])
    y_title = data["_y_title_"]
    if y_title == '!Example Y Title' and 'y_title' in chart_vars:
        y_title = chart_vars['y_title']
    outputs = []
    def make_outputs_for_tick(tick_true):
        fq_val = None
        for entry in chart_data:
            if str(entry.get("x")) == str(tick_true):
                fq_val = entry.get("first_quartile")
                if fq_val is not None:
                    fq_val = str(round(float(fq_val), 2))
                break
        true_templates = [
            f"The box plot for {y_title} at {tick_true} has a lower quartile value",
            f"At x-tick {tick_true}, the lower quartile of {y_title} is defined",
            f"Lower quartile data exists for {y_title} at {tick_true}",
        ]
        true_output = {"tag": f"DP_FQ_exist_{role}", "text": random.choice(true_templates), "truth": True}
        false_templates = [
            f"The box plot for {y_title} at {get_fake_tick(chart_vars)} has a lower quartile value",
            f"At x-tick {get_fake_tick(chart_vars)}, the lower quartile of {y_title} is defined",
            f"Lower quartile data exists for {y_title} at {get_fake_tick(chart_vars)}",
        ]
        false_outputs = [{"tag": f"DP_FQ_exist_{role}", "text": t, "truth": False} for t in false_templates]
        return [true_output] + false_outputs
    # --- Case 1: role = all → loop over all x_ticks
    if role == "all":
        for tick in chart_vars.get("x_ticks", []):
            # print('tick', tick )
            outputs.extend(make_outputs_for_tick(tick))
        return outputs
    # --- Case 2: role = i/j/k → single tick from data
    tick_true = data.get(f"_{role}_")
    if not tick_true or "X Value" in str(tick_true):
        tick_true = random.choice(chart_vars.get("x_ticks", ["some x-tick"]))
    return make_outputs_for_tick(tick_true)

def DP_FQ_val(data, role="i"):
    """
    Math premise: actual numeric value of lower quartile (25th percentile)
    at tick `role`. Supports both single tick ('i', 'j', etc.) and 'all'.
    """
    chart_vars = data["chart"]
    chart_data = chart_vars.get("data", [{}])[0].get("data", [])
    y_title = data.get("_y_title_", "!Example Y Title")
    if y_title == "!Example Y Title" and "y_title" in chart_vars:
        y_title = chart_vars["y_title"]

    outputs = []

    def make_outputs_for_tick(tick_true):
        # Find true FQ value
        fq_val = None
        for entry in chart_data:
            if str(entry.get("x")) == str(tick_true):
                fq_val = entry.get("first_quartile")
                if fq_val is not None:
                    fq_val = str(round(float(fq_val), 2))
                break
        if fq_val is None:
            fq_val = "Unknown"
        # --- True premise
        true_templates = [
            f"The lower quartile value of {y_title} at {tick_true} is {fq_val}",
            f"At x-tick {tick_true}, the first quartile of {y_title} equals {fq_val}",
            f"The 25th percentile value for {y_title} at {tick_true} is {fq_val}",
        ]
        true_output = {"tag": f"DP_FQ_val_{role}", "text": random.choice(true_templates), "truth": True}
        # --- False distractor: choose other statistic (min/median/max)
        entry = next((e for e in chart_data if str(e.get("x")) == str(tick_true)), None)
        if entry:
            distractor_val = [entry.get("min"), entry.get("median"), entry.get("max")]
            if distractor_val is not None:
                distractor_val = [str(round(float(c), 2)) for c in distractor_val]
            else:
                distractor_val = "WrongVal"
        else:
            distractor_val = "WrongVal"
        false_templates = [
            f"The lower quartile value of {y_title} at {tick_true} is {distractor_val[0]}",
            f"At x-tick {tick_true}, the first quartile of {y_title} equals {distractor_val[1]}",
            f"The 25th percentile value for {y_title} at {tick_true} is {distractor_val[2]}",
        ]
        false_outputs = [{"tag": f"DP_FQ_val_{role}", "text": t, "truth": False} for t in false_templates]
        return [true_output] + false_outputs
    # --- Case 1: "all" → apply across all ticks
    if role == "all":
        for tick in chart_vars.get("x_ticks", []):
            outputs.extend(make_outputs_for_tick(tick))
        return outputs
    # --- Case 2: single tick (i/j/k)
    tick_true = data.get(f"_{role}_")
    if not tick_true or "X Value" in str(tick_true):
        tick_true = random.choice(chart_vars.get("x_ticks", ["some x-tick"]))

    return make_outputs_for_tick(tick_true)

def DP_TQ_exist(data, role="i"):
    """
    Data premise: existence of upper quartile (75th percentile) at tick `role`.
    - If role == "i"/"j"/"k": single tick premise.
    - If role == "all": generate for all ticks in the chart.
    """
    chart_vars = data["chart"]
    chart_data = chart_vars.get("data", [{}])[0].get("data", [])
    y_title = data.get("_y_title_", "!Example Y Title")
    if y_title == "!Example Y Title" and "y_title" in chart_vars:
        y_title = chart_vars["y_title"]

    outputs = []

    def make_outputs_for_tick(tick_true):
        tq_val = None
        for entry in chart_data:
            if str(entry.get("x")) == str(tick_true):
                tq_val = entry.get("third_quartile")
                if tq_val is not None:
                    tq_val = str(round(float(tq_val), 2))
                break

        # --- True templates
        true_templates = [
            f"The box plot for {y_title} at {tick_true} has an upper quartile value",
            f"At x-tick {tick_true}, the upper quartile of {y_title} is defined",
            f"Upper quartile data exists for {y_title} at {tick_true}",
        ]
        true_output = {"tag": f"DP_TQ_exist_{role}", "text": random.choice(true_templates), "truth": True}

        false_templates = [
            f"The box plot for {y_title} at {get_fake_tick(chart_vars)} has an upper quartile value",
            f"At x-tick {get_fake_tick(chart_vars)}, the upper quartile of {y_title} is defined",
            f"Upper quartile data exists for {y_title} at {get_fake_tick(chart_vars)}",
        ]
        false_outputs = [{"tag": f"DP_TQ_exist_{role}", "text": t, "truth": False} for t in false_templates]

        return [true_output] + false_outputs

    # --- Case 1: “all” → all x_ticks
    if role == "all":
        for tick in chart_vars.get("x_ticks", []):
            outputs.extend(make_outputs_for_tick(tick))
        return outputs

    # --- Case 2: specific tick
    tick_true = data.get(f"_{role}_")
    if not tick_true or "X Value" in str(tick_true):
        tick_true = random.choice(chart_vars.get("x_ticks", ["some x-tick"]))

    return make_outputs_for_tick(tick_true)

def DP_TQ_val(data, role="i"):
    """
    Math premise: actual numeric value of upper quartile (75th percentile)
    at tick `role`. Supports both single tick ('i', 'j', etc.) and 'all'.
    """
    chart_vars = data["chart"]
    chart_data = chart_vars.get("data", [{}])[0].get("data", [])
    y_title = data.get("_y_title_", "!Example Y Title")
    if y_title == "!Example Y Title" and "y_title" in chart_vars:
        y_title = chart_vars["y_title"]

    outputs = []

    def make_outputs_for_tick(tick_true):
        # Find true TQ value
        tq_val = None
        for entry in chart_data:
            if str(entry.get("x")) == str(tick_true):
                tq_val = entry.get("third_quartile")
                if tq_val is not None:
                    tq_val = str(round(float(tq_val), 2))
                break

        if tq_val is None:
            tq_val = "Unknown"

        # --- True premise
        true_templates = [
            f"The upper quartile value of {y_title} at {tick_true} is {tq_val}",
            f"At x-tick {tick_true}, the third quartile of {y_title} equals {tq_val}",
            f"The 75th percentile value for {y_title} at {tick_true} is {tq_val}",
        ]
        true_output = {"tag": f"DP_TQ_val_{role}", "text": random.choice(true_templates), "truth": True}

        # --- False distractor: choose other statistic (min/median/max)
        entry = next((e for e in chart_data if str(e.get("x")) == str(tick_true)), None)
        distractor_val = "WrongVal"
        if entry:
            candidates = [entry.get("min"), entry.get("median"), entry.get("max")]
            candidates = [c for c in candidates if c is not None and str(c) != str(tq_val)]
            if candidates:
                distractor_val = [str(round(float(c), 2)) for c in candidates]

        false_templates = [
            f"The upper quartile value of {y_title} at {tick_true} is {distractor_val[0]}",
            f"At x-tick {tick_true}, the third quartile of {y_title} equals {distractor_val[1]}",
            f"The 75th percentile value for {y_title} at {tick_true} is {distractor_val[2]}",
        ]
        false_outputs = [{"tag": f"DP_TQ_val_{role}", "text": t, "truth": False} for t in false_templates]

        return [true_output] + false_outputs

    # --- Case 1: "all" → apply across all x-ticks
    if role == "all":
        for tick in chart_vars.get("x_ticks", []):
            outputs.extend(make_outputs_for_tick(tick))
        return outputs

    # --- Case 2: single tick (i/j/k)
    tick_true = data.get(f"_{role}_")
    if not tick_true or "X Value" in str(tick_true):
        tick_true = random.choice(chart_vars.get("x_ticks", ["some x-tick"]))

    return make_outputs_for_tick(tick_true)

def DP_scatter_series(data):
    """
    Data premise: ensure scatter series exists for a legend across X/Y points.
    """
    y_title = data.get("_y_title_", "Y-axis")
    x_title = data.get("_x_title_", "X-axis")
    legend_true = data.get("_legend_", "!Example Legend")
    chart_vars = data["chart"]

    # Assume scatter data points exist if both axes have values
    true_templates = [
        f"The chart contains scatter values for {legend_true} across both {x_title} and {y_title}.",
        f"Valid {y_title} values are paired with {x_title} values in the series {legend_true}.",
        f"Data points for {legend_true} exist covering the relationship between {x_title} and {y_title}."
    ]
    true_output = {"tag": "DP_scatter", "text": random.choice(true_templates), "truth": True}

    # Distractors: wrong legend or missing data
    wrong_legend = random.choice(chart_vars.get("legend_labels", ["OtherSeries"]))
    false_templates = [
        f"The chart contains no valid {y_title} vs {x_title} data points for {legend_true}.",
        f"Scatter values are missing for {legend_true} across {x_title} ticks.",
        f"The series {wrong_legend} rather than {legend_true} shows {y_title} vs {x_title} data."
    ]
    false_outputs = [{"tag": "DP_scatter", "text": t, "truth": False} for t in false_templates]

    return [true_output] + false_outputs



def RP_59(data):
    """
    Reasoning premise for QID 59:
    "Is the value of <y_title> at <i> less than that at <j>?"
    Answer may be 'yes' or 'no'.
    """
    y = data["_y_title_"]
    i = data["_i_"]
    j = data["_j_"]
    answer = str(data.get("answer", "")).lower()

    # --- True case templates: y_i < y_j ---
    less_templates = [
        f"The value of {y} at x-tick {i} is less than that at x-tick {j}",
        f"For {y}, the value at {i} is lower than the value at {j}",
        f"At x-tick {i}, the {y} value is smaller than at x-tick {j}",
    ]

    # --- True case templates: y_i > y_j ---
    greater_templates = [
        f"The value of {y} at x-tick {i} is greater than that at x-tick {j}",
        f"For {y}, the value at {i} is higher than the value at {j}",
        f"At x-tick {i}, the {y} value is larger than at x-tick {j}",
    ]
    if answer == "yes":  # y_i < y_j
        true_outputs = [{"tag": "RP_59", "text": t, "truth": True} for t in less_templates]
        false_outputs = [{"tag": "RP_59", "text": t, "truth": False} for t in greater_templates]
    else:  # "no" → y_i >= y_j
        true_outputs = [{"tag": "RP_59", "text": t, "truth": True} for t in greater_templates]
        false_outputs = [{"tag": "RP_59", "text": t, "truth": False} for t in less_templates]
    return [random.choice(true_outputs)] + false_outputs

def RP_62(data):
    y = data["_y_title_"]
    legend = data["_legend_"]
    i = data["_i_"]
    j = data["_j_"]
    answer = data['answer']

    # Templates for each logical form
    less_than = f"The value of {y} for {legend} at x-tick {i} is less than that at x-tick {j}"
    greater_than = f"The value of {y} for {legend} at x-tick {i} is greater than that at x-tick {j}"

    diff_ij_pos = f"The difference of values of {y} for {legend} at x-tick {i} and x-tick {j} is greater than zero"
    diff_ij_neg = f"The difference of values of {y} for {legend} at x-tick {i} and x-tick {j} is less than zero"

    diff_ji_pos = f"The difference of values of {y} for {legend} at x-tick {j} and x-tick {i} is greater than zero"
    diff_ji_neg = f"The difference of values of {y} for {legend} at x-tick {j} and x-tick {i} is less than zero"

    if str(answer).lower() == "yes":  # y_i < y_j
        
        true_ret = [
            {"tag": "RP_62", "text": less_than, "truth": True},
            {"tag": "RP_62", "text": diff_ji_pos, "truth": True},
            {"tag": "RP_62", "text": diff_ij_neg, "truth": True}]
        false_ret = [
            {"tag": "RP_62", "text": greater_than, "truth": False},
            {"tag": "RP_62", "text": diff_ij_pos, "truth": False},
            {"tag": "RP_62", "text": diff_ji_neg, "truth": False},
        ]
        return [random.choice(true_ret)] + false_ret

    else:  # answer = "no" → y_i > y_j
        true_ret = [
            {"tag": "RP_62", "text": greater_than, "truth": True},
            {"tag": "RP_62", "text": diff_ij_pos, "truth": True},
            {"tag": "RP_62", "text": diff_ji_neg, "truth": True},
        ]
        false_ret = [
            {"tag": "RP_62", "text": less_than, "truth": False},
            {"tag": "RP_62", "text": diff_ji_pos, "truth": False},
            {"tag": "RP_62", "text": diff_ij_neg, "truth": False},
        ]
        return [random.choice(true_ret)] + false_ret

def RP_63(data):
    _x_title_ = data['_x_title_']
    _y_title_ = data['_y_title_']
    _i_ = data['_i_']
    _j_ = data['_j_']
    RP63 = [
        f"The difference in {_x_title_} between {_i_} and {_j_} is greater than the largest difference between any two consecutive {_y_title_} values.",
        f"The maximum difference in {_x_title_} for any two consecutive {_y_title_} values is between {_i_} and {_j_}."
    ]
    return random.choice(RP63)

def RP_65(data):
    _xi_, _yi_ = data['_xi_,_yi_']
    _xj_, _yj_ = data['_xj_,_yj_']
    _xk_, _yk_ = data['_xk_,_yk_']
    RP65 = [
        f'The sum of the values at ({_xi_} , {_yi_}) and ({_xj_} , {_yj_}) is greater than the value at ({_xk_} , {_yk_})', 
        f'If the plot values at ({_xi_} , {_yi_}) and ({_xj_} , {_yj_}) are added together, the sum is greater than the value at ({_xk_} , {_yk_}).'
    ]
    return random.choice(RP65)


def RP_68(data):
    y = data["_y_title_"]
    legend1 = data["_legend1_"]
    legend2 = data["_legend2_"]
    i = data["_i_"]
    j = data["_j_"]
    xi_extra = data["_xi_extra_"]
    xj_extra = data["_xj_extra_"]
    answer = str(data.get("answer", "")).lower()

    # --- Templates ---
    # Positive claim: legend1 diff > legend2 diff
    greater_template = f"The difference in {y} for {legend1} between x-ticks {i} and {j} is greater than the difference for {legend2} between x-ticks {xi_extra} and {xj_extra}"
    greater_alt1 = f"The gap in {y} values for {legend1} ({i} vs {j}) exceeds the gap for {legend2} ({xi_extra} vs {xj_extra})"
    greater_alt2 = f"For {y}, the absolute change in {legend1} between {i} and {j} is larger than that of {legend2} between {xi_extra} and {xj_extra}"

    # Negative claim: legend1 diff ≤ legend2 diff
    lesser_template = f"The difference in {y} for {legend1} between x-ticks {i} and {j} is less than the difference for {legend2} between x-ticks {xi_extra} and {xj_extra}"
    lesser_alt1 = f"The gap in {y} values for {legend1} ({i} vs {j}) does not exceed the gap for {legend2} ({xi_extra} vs {xj_extra})"
    lesser_alt2 = f"For {y}, the absolute change in {legend1} between {i} and {j} is smaller than that of {legend2} between {xi_extra} and {xj_extra}"

    if answer == "yes":
        true_ret = [
            {"tag": "RP_68", "text": greater_template, "truth": True},
            {"tag": "RP_68", "text": greater_alt1, "truth": True},
            {"tag": "RP_68", "text": greater_alt2, "truth": True},
        ]
        false_ret = [
            {"tag": "RP_68", "text": lesser_template, "truth": False},
            {"tag": "RP_68", "text": lesser_alt1, "truth": False},
            {"tag": "RP_68", "text": lesser_alt2, "truth": False},
        ]
    else:  # "no"
        true_ret = [
            {"tag": "RP_68", "text": lesser_template, "truth": True},
            {"tag": "RP_68", "text": lesser_alt1, "truth": True},
            {"tag": "RP_68", "text": lesser_alt2, "truth": True},
        ]
        false_ret = [
            {"tag": "RP_68", "text": greater_template, "truth": False},
            {"tag": "RP_68", "text": greater_alt1, "truth": False},
            {"tag": "RP_68", "text": greater_alt2, "truth": False},
        ]

    # Keep 1 true + all false, like RP_62 design
    return [random.choice(true_ret)] + false_ret

def RP_72(data):
    """
    Reasoning premise for QID 72:
    Sum of two legends' values compared to a third legend's value.
    """
    y = data.get("_y_title_", "Y-axis")
    x = data.get("_x_title_", "X-axis")
    l1 = data.get("_legend1_", "Legend1")
    l2 = data.get("_legend2_", "Legend2")
    l3 = data.get("_legend3_", "Legend3")
    i = data.get("_i_", "i")
    j = data.get("_j_", "j")
    k = data.get("_k_", "k")
    answer = str(data.get("answer", "")).lower()

    # --- true premises if answer == "yes"
    true_templates = [
        f"The sum of {y} for {l1} at {i} and {y} for {l2} at {j} is greater than {y} for {l3} at {k}",
        f"If {y} values of {l1} at {i} and {l2} at {j} are added, their sum exceeds the value of {l3} at {k}",
        f"For every {x}, the combined value of {l1} and {l2} is larger than the value of {l3}",
    ]
    # --- false premises (negations)
    false_templates = [
        f"The sum of {y} for {l1} at {i} and {y} for {l2} at {j} is less than {y} for {l3} at {k}",
        f"If {y} values of {l1} at {i} and {l2} at {j} are added, their sum is smaller than the value of {l3} at {k}",
        f"For every {x}, the combined value of {l1} and {l2} is not greater than the value of {l3}",
    ]

    if answer == "yes":
        true_outputs = [{"tag": "RP_72", "text": t, "truth": True} for t in true_templates]
        false_outputs = [{"tag": "RP_72", "text": t, "truth": False} for t in false_templates]
    else:  # "no"
        true_outputs = [{"tag": "RP_72", "text": t, "truth": True} for t in false_templates]
        false_outputs = [{"tag": "RP_72", "text": t, "truth": False} for t in true_templates]

    # keep the 1:true + many:false structure
    return [random.choice(true_outputs)] + false_outputs

def RP_166(data):
    """
    Compare medians at tick i vs j.
    """
    y = data["_y_title_"]
    i = data["_i_"]
    j = data["_j_"]
    answer = str(data.get("answer", "")).lower()

    less_templates = [
        f"The median value of {y} at x-tick {i} is less than that at x-tick {j}",
        f"For {y}, the median at {i} is lower than at {j}",
    ]
    greater_templates = [
        f"The median value of {y} at x-tick {i} is greater than that at x-tick {j}",
        f"For {y}, the median at {i} is higher than at {j}",
    ]

    if answer == "yes":  # i < j
        true_outputs = [{"tag": "RP_166", "text": t, "truth": True} for t in less_templates]
        false_outputs = [{"tag": "RP_166", "text": t, "truth": False} for t in greater_templates]
    else:  # "no" → i >= j
        true_outputs = [{"tag": "RP_166", "text": t, "truth": True} for t in greater_templates]
        false_outputs = [{"tag": "RP_166", "text": t, "truth": False} for t in less_templates]

    return [random.choice(true_outputs)] + false_outputs


def RP_167(data):
    """
    Compare upper quartiles at tick i vs j.
    """
    y = data["_y_title_"]
    i = data["_i_"]
    j = data["_j_"]
    answer = str(data.get("answer", "")).lower()

    less_templates = [
        f"The upper quartile of {y} at x-tick {i} is less than that at x-tick {j}",
        f"For {y}, the third quartile at {i} is lower than at {j}",
    ]
    greater_templates = [
        f"The upper quartile of {y} at x-tick {i} is greater than that at x-tick {j}",
        f"For {y}, the third quartile at {i} is higher than at {j}",
    ]

    if answer == "yes":  # i < j
        true_outputs = [{"tag": "RP_167", "text": t, "truth": True} for t in less_templates]
        false_outputs = [{"tag": "RP_167", "text": t, "truth": False} for t in greater_templates]
    else:  # "no" → i >= j
        true_outputs = [{"tag": "RP_167", "text": t, "truth": True} for t in greater_templates]
        false_outputs = [{"tag": "RP_167", "text": t, "truth": False} for t in less_templates]

    return [random.choice(true_outputs)] + false_outputs

def RP_168(data):
    """
    Compare lower quartiles at tick i vs j.
    """
    y = data["_y_title_"]
    i = data["_i_"]
    j = data["_j_"]
    answer = str(data.get("answer", "")).lower()

    less_templates = [
        f"The lower quartile of {y} at x-tick {i} is less than that at x-tick {j}",
        f"For {y}, the first quartile at {i} is lower than at {j}",
    ]
    greater_templates = [
        f"The lower quartile of {y} at x-tick {i} is greater than that at x-tick {j}",
        f"For {y}, the first quartile at {i} is higher than at {j}",
    ]

    if answer == "yes":  # i < j
        true_outputs = [{"tag": "RP_168", "text": t, "truth": True} for t in less_templates]
        false_outputs = [{"tag": "RP_168", "text": t, "truth": False} for t in greater_templates]
    else:  # "no" → i >= j
        true_outputs = [{"tag": "RP_168", "text": t, "truth": True} for t in greater_templates]
        false_outputs = [{"tag": "RP_168", "text": t, "truth": False} for t in less_templates]

    return [random.choice(true_outputs)] + false_outputs

def RP_146(data):
    """
    Reasoning premise: determine whether any two x-ticks share equal IQRs.
    Uses the box-plot data to infer logical truth based on QA answer.
    """
    chart_vars = data["chart"]
    chart_data = chart_vars.get("data", [{}])[0].get("data", [])
    y_title = data.get("_y_title_", chart_vars.get("y_title", "Unknown Y"))
    answer = str(data.get("answer", "no")).lower()

    # --- Compute IQRs ---
    iqrs = {}
    for entry in chart_data:
        x = entry.get("x")
        fq, tq = entry.get("first_quartile"), entry.get("third_quartile")
        if fq is not None and tq is not None:
            iqrs[x] = round(float(tq) - float(fq), 3)

    # --- Find any equal pairs ---
    equal_pairs = []
    ticks = list(iqrs.keys())
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            if abs(iqrs[ticks[i]] - iqrs[ticks[j]]) < 1e-3:
                equal_pairs.append((ticks[i], ticks[j]))

    if answer == "yes":
        true_texts = [
            f"At least two x-ticks have equal interquartile range for {y_title}",
            f"Some categories share identical IQRs for {y_title}",
        ]
        false_texts = [
            f"All x-ticks have distinct interquartile ranges for {y_title}",
            f"No two x-ticks share equal IQRs for {y_title}",
        ]
    else:
        true_texts = [
            f"All x-ticks have distinct interquartile ranges for {y_title}",
            f"No two x-ticks share equal IQRs for {y_title}",
        ]
        false_texts = [
            f"At least two x-ticks have equal interquartile range for {y_title}",
            f"Some categories share identical IQRs for {y_title}",
        ]

    return (
        [{"tag": "RP_146", "text": random.choice(true_texts), "truth": True}] +
        [{"tag": "RP_146", "text": t, "truth": False} for t in false_texts]
    )


def RP_169(data):
    """
    Compare maximum values at tick i vs j.
    """
    y = data["_y_title_"]
    i = data["_i_"]
    j = data["_j_"]
    answer = str(data.get("answer", "")).lower()

    less_templates = [
        f"The maximum value of {y} at {i} is less than that at {j}",
        f"For {y}, the highest value at {i} is lower than at {j}",
    ]
    greater_templates = [
        f"The maximum value of {y} at {i} is greater than that at {j}",
        f"For {y}, the highest value at {i} is higher than at {j}",
    ]

    if answer == "yes":  # i < j
        true_outputs = [{"tag": "RP_169", "text": t, "truth": True} for t in less_templates]
        false_outputs = [{"tag": "RP_169", "text": t, "truth": False} for t in greater_templates]
    else:  # "no" → i >= j
        true_outputs = [{"tag": "RP_169", "text": t, "truth": True} for t in greater_templates]
        false_outputs = [{"tag": "RP_169", "text": t, "truth": False} for t in less_templates]

    return [random.choice(true_outputs)] + false_outputs

def RP_170(data):
    """
    Compare minimum values at tick i vs j.
    """
    y = data["_y_title_"]
    i = data["_i_"]
    j = data["_j_"]
    answer = str(data.get("answer", "")).lower()

    less_templates = [
        f"The minimum value of {y} at {i} is less than that at {j}",
        f"For {y}, the lowest value at {i} is lower than at {j}",
    ]
    greater_templates = [
        f"The minimum value of {y} at {i} is greater than that at {j}",
        f"For {y}, the lowest value at {i} is higher than at {j}",
    ]

    if answer == "yes":  # i < j
        true_outputs = [{"tag": "RP_170", "text": t, "truth": True} for t in less_templates]
        false_outputs = [{"tag": "RP_170", "text": t, "truth": False} for t in greater_templates]
    else:  # "no" → i >= j
        true_outputs = [{"tag": "RP_170", "text": t, "truth": True} for t in greater_templates]
        false_outputs = [{"tag": "RP_170", "text": t, "truth": False} for t in less_templates]

    return [random.choice(true_outputs)] + false_outputs


def RP_18(data):
    """
    Reasoning premise: do line count and legend count match?
    Uses the 'answer' field to determine truth assignment.
    """
    answer = str(data.get("answer", "")).lower()

    true_templates_equal = [
        "The number of lines equals the number of legends",
        "The count of lines matches the count of legends",
        "Equal quantities of lines and legends are present",
    ]
    false_templates_equal = [
        "The number of lines does not equal the number of legends",
        "There is a mismatch in the count of lines and legends",
        "Lines and legends are present in unequal numbers",
    ]

    if answer == "yes":  # equality holds
        true_outputs = [{"tag": "RP_18", "text": t, "truth": True} for t in true_templates_equal]
        false_outputs = [{"tag": "RP_18", "text": t, "truth": False} for t in false_templates_equal]
    else:  # "no" → inequality holds
        true_outputs = [{"tag": "RP_18", "text": t, "truth": True} for t in false_templates_equal]
        false_outputs = [{"tag": "RP_18", "text": t, "truth": False} for t in true_templates_equal]

    return [random.choice(true_outputs)] + false_outputs
   
def RP_18a(data):
    """
    Reasoning premise for QID 18a:
    "Is the number of lines equal to the number of mark labels?"
    Uses the 'answer' field to set truth assignment.
    """
    answer = str(data.get("answer", "")).lower()

    equal_templates = [
        "The number of lines equals the number of mark labels",
        "The line count is identical to the count of mark labels",
        "An equal number of lines and mark labels are displayed",
    ]
    not_equal_templates = [
        "The number of lines does not equal the number of mark labels",
        "There is a disparity between the count of lines and mark labels",
        "Lines and mark labels count do not match",
    ]

    if answer == "yes":  # equality holds
        true_outputs = [{"tag": "RP_18a", "text": t, "truth": True} for t in equal_templates]
        false_outputs = [{"tag": "RP_18a", "text": t, "truth": False} for t in not_equal_templates]
    else:  # "no" → inequality holds
        true_outputs = [{"tag": "RP_18a", "text": t, "truth": True} for t in not_equal_templates]
        false_outputs = [{"tag": "RP_18a", "text": t, "truth": False} for t in equal_templates]
    return [random.choice(true_outputs)] + false_outputs


def RP_35(data):
    """
    Reasoning premise for monotonic increase of y over x.
    """
    y = data.get("_y_title_", "Y-axis")
    x = data.get("_x_title_", "X-axis")
    answer = str(data.get("answer", "")).lower()

    true_templates = [
        f"The values of {y} monotonically increase over the {x}",
        f"As {x} increases, the values of {y} consistently rise",
        f"For every step along the {x}, the values of {y} are greater than the previous step",
    ]
    false_templates = [
        f"The values of {y} do not monotonically increase over the {x}",
        f"As {x} increases, the values of {y} fluctuate or decrease at some points",
        f"A consistent upward trend in {y} over {x} is not observed",
    ]

    if answer == "yes":  # monotonic increase holds
        true_outputs = [{"tag": "RP_35", "text": t, "truth": True} for t in true_templates]
        false_outputs = [{"tag": "RP_35", "text": t, "truth": False} for t in false_templates]
    else:  # "no"
        true_outputs = [{"tag": "RP_35", "text": t, "truth": True} for t in false_templates]
        false_outputs = [{"tag": "RP_35", "text": t, "truth": False} for t in true_templates]

    return [random.choice(true_outputs)] + false_outputs

def RP_116(data):
    """
    Reasoning premise for QID 116:
    "Is the <legend> monotonically increasing?"
    Answer may be "yes" or "no".
    """
    legend = data["_legend_"]
    answer = str(data.get("answer", "")).lower()

    # --- True case templates: monotonic increase ---
    inc_templates = [
        f"The y-axis values for the legend {legend} monotonically increase over the x-axis values.",
        f"For the legend {legend}, each successive value on the y-axis is greater than the previous as x increases.",
        f"A continuous upward trend is observed in y-axis values for the legend {legend}.",
    ]

    # --- False case templates: not monotonic (decreasing or fluctuating) ---
    not_inc_templates = [
        f"The y-axis values for the legend {legend} do not monotonically increase over the x-axis values.",
        f"For the legend {legend}, the values across the y-axis are not strictly increasing.",
        f"No consistent upward trend is observed in y-axis values for the legend {legend}.",
    ]

    if answer == "yes":  # monotonic increase holds
        true_outputs = [{"tag": "RP_116", "text": t, "truth": True} for t in inc_templates]
        false_outputs = [{"tag": "RP_116", "text": t, "truth": False} for t in not_inc_templates]
    else:  # "no" → not increasing
        true_outputs = [{"tag": "RP_116", "text": t, "truth": True} for t in not_inc_templates]
        false_outputs = [{"tag": "RP_116", "text": t, "truth": False} for t in inc_templates]

    return [random.choice(true_outputs)] + false_outputs

def RP_117(data):
    """
    Reasoning premise for QID 117:
    "Is the <legend> monotonically decreasing?"
    Answer may be "yes" or "no".
    """
    legend = data["_legend_"]
    answer = str(data.get("answer", "")).lower()

    # --- True case templates: monotonic decrease ---
    dec_templates = [
        f"The y-axis values for the legend {legend} monotonically decrease over the x-axis values.",
        f"For the legend {legend}, each successive value on the y-axis is smaller than the previous as x increases.",
        f"A consistent downward trend is observed in y-axis values for the legend {legend}.",
    ]

    # --- False case templates: not monotonic decreasing ---
    not_dec_templates = [
        f"The y-axis values for the legend {legend} do not monotonically decrease over the x-axis values.",
        f"For the legend {legend}, the values across the y-axis are not strictly decreasing.",
        f"No consistent downward trend is observed in y-axis values for the legend {legend}.",
    ]

    if answer == "yes":  # monotonic decrease holds
        true_outputs = [{"tag": "RP_117", "text": t, "truth": True} for t in dec_templates]
        false_outputs = [{"tag": "RP_117", "text": t, "truth": False} for t in not_dec_templates]
    else:  # "no" → not decreasing
        true_outputs = [{"tag": "RP_117", "text": t, "truth": True} for t in not_dec_templates]
        false_outputs = [{"tag": "RP_117", "text": t, "truth": False} for t in dec_templates]

    return [random.choice(true_outputs)] + false_outputs


def RP_121(data):
    legend = data["_legend_"]
    answer = str(data.get("answer", "")).lower()

    true_templates = [
        f"{legend} has a correlation value greater than 0 but less than or equal to 0.5",
        f"The correlation for {legend} is low positive, between 0 and 0.5.",
        f"The Pearson r for {legend} falls in the low positive range (0 < r ≤ 0.5)."
    ]
    false_templates = [
        f"{legend} has a correlation value outside the low positive range (0 < r ≤ 0.5).",
        f"The correlation for {legend} is not between 0 and 0.5.",
        f"The Pearson r for {legend} lies in a range inconsistent with low positive correlation."
    ]

    if answer == "yes":
        return [{"tag": "RP_121", "text": random.choice(true_templates), "truth": True}] + \
               [{"tag": "RP_121", "text": t, "truth": False} for t in false_templates]
    else:
        return [{"tag": "RP_121", "text": random.choice(false_templates), "truth": True}] + \
               [{"tag": "RP_121", "text": t, "truth": False} for t in true_templates]

def RP_122(data):
    legend = data["_legend_"]
    answer = str(data.get("answer", "")).lower()

    true_templates = [
        f"{legend} has correlation value greater than 0.5 but less than or equal to 1.",
        f"The Pearson r for {legend} is in the high positive range (0.5 < r ≤ 1).",
        f"The correlation of {legend} is strongly positive, above 0.5."
    ]
    false_templates = [
        f"{legend} does not have correlation value greater than 0.5 but less than or equal to 1.",
        f"The Pearson r for {legend} is not in the high positive range.",
        f"The correlation for {legend} lies outside 0.5 < r ≤ 1."
    ]

    if answer == "yes":
        return [{"tag": "RP_122", "text": random.choice(true_templates), "truth": True}] + \
               [{"tag": "RP_122", "text": t, "truth": False} for t in false_templates]
    else:
        return [{"tag": "RP_122", "text": random.choice(false_templates), "truth": True}] + \
               [{"tag": "RP_122", "text": t, "truth": False} for t in true_templates]


def RP_123(data):
    legend = data["_legend_"]
    answer = str(data.get("answer", "")).lower()

    true_templates = [
        f"{legend} has correlation value greater than or equal to -0.5 but less than 0.",
        f"The Pearson r for {legend} is in the low negative range (-0.5 ≤ r < 0).",
        f"The correlation of {legend} is weakly negative, between -0.5 and 0."
    ]
    false_templates = [
        f"{legend} does not have correlation value greater than or equal to -0.5 but less than 0.",
        f"The Pearson r for {legend} is not in the low negative range.",
        f"The correlation for {legend} lies outside -0.5 ≤ r < 0."
    ]

    if answer == "yes":
        return [{"tag": "RP_123", "text": random.choice(true_templates), "truth": True}] + \
               [{"tag": "RP_123", "text": t, "truth": False} for t in false_templates]
    else:
        return [{"tag": "RP_123", "text": random.choice(false_templates), "truth": True}] + \
               [{"tag": "RP_123", "text": t, "truth": False} for t in true_templates]


def RP_124(data):
    legend = data["_legend_"]
    answer = str(data.get("answer", "")).lower()

    true_templates = [
        f"{legend} has correlation value greater than or equal to -1 but less than -0.5.",
        f"The Pearson r for {legend} is in the high negative range (-1 ≤ r < -0.5).",
        f"The correlation of {legend} is strongly negative, below -0.5."
    ]
    false_templates = [
        f"{legend} does not have correlation value greater than or equal to -1 but less than -0.5.",
        f"The Pearson r for {legend} is not in the high negative range.",
        f"The correlation for {legend} lies outside -1 ≤ r < -0.5."
    ]

    if answer == "yes":
        return [{"tag": "RP_124", "text": random.choice(true_templates), "truth": True}] + \
               [{"tag": "RP_124", "text": t, "truth": False} for t in false_templates]
    else:
        return [{"tag": "RP_124", "text": random.choice(false_templates), "truth": True}] + \
               [{"tag": "RP_124", "text": t, "truth": False} for t in true_templates]

def MP_Median(data, role="i"):
    """
    State the median value for a given tick in a box plot.
    """
    y = data["_y_title_"]
    tick = data.get(f"_{role}_", f"X Value {role}")
    chart_data = data["chart"].get("data", [{}])[0].get("data", [])

    # Find the median from chart data
    median_val = None
    for entry in chart_data:
        if str(entry.get("x")) == str(tick):
            median_val = entry.get("median")
            median_val = str(round(float(median_val), 2))
            break

    if median_val is None:
        median_val = 0

    true_templates = [
        f"The median value of {y} at {tick} is {median_val}",
        f"For {y}, the median at {tick} equals {median_val}",
        f"At x-tick {tick}, the median of {y} is {median_val}",
    ]
    true_output = {"tag": f"MP_median_{role}", "text": random.choice(true_templates), "truth": True}

    # Distractors: swap with other quartiles or random numbers
    distractor_val = random.choice([
        entry.get("min"), entry.get("max"), entry.get("first_quartile"),
        entry.get("third_quartile"), round(float(median_val) + random.uniform(1, 3), 3)
    ]) if chart_data else "-0"
    distractor_val = str(round(float(distractor_val), 2))

    false_templates = [
        f"The median value of {y} at {tick} is {distractor_val}",
        f"For {y}, the median at {tick} equals {distractor_val}",
        f"At x-tick {tick}, the median of {y} is {distractor_val}",
    ]
    false_outputs = [{"tag": f"MP_median_{role}", "text": t, "truth": False} for t in false_templates]

    return [true_output] + false_outputs


def MP_PC_all(data):
    """
    Math Premises for Pearson correlation calculation — rewritten for interpretive clarity.
    These describe the reasoning chain a human analyst would follow while computing correlation.
    """
    chart_vars = data["chart"]
    X = chart_vars.get("x_ticks", "X values")
    Y = chart_vars.get("y_ticks", "Y values")

    # --- True, semantically meaningful premises ---
    true_steps = [
        f"The average of {X} represents the central tendency of the horizontal values.",
        f"The average of {Y} represents the central tendency of the vertical values.",
        f"For each point, deviations of {X} and {Y} from their respective means are determined.",
        f"Positive deviations in {X} often correspond to positive deviations in {Y}, indicating co-movement.",
        f"The products of these paired deviations are summed to measure joint variability.",
        f"This sum is normalized by the square root of the product of the individual squared deviations.",
        f"The resulting ratio defines the Pearson correlation coefficient r = cov(X,Y)/(σX·σY).",
        f"A high positive r implies that {Y} increases as {X} increases; a negative r implies the opposite."
    ]
    true_outputs = [{"tag": "MP_PC", "text": t, "truth": True} for t in true_steps]

    # --- False, but plausible distractors (near-miss reasoning errors) ---
    false_steps = [
        f"The correlation is measured by summing the differences of {X} and {Y} directly, without centering by their means.",
        f"The deviations of {X} and {Y} are combined by addition rather than multiplication, losing sign information.",
        f"The normalization step divides by the sum instead of the square root of the product of variances.",
        f"The mean of {Y} is mistakenly used in computing deviations for {X}.",
        f"The Pearson r is computed without accounting for the standard deviations of {X} and {Y}."
    ]
    false_outputs = [{"tag": "MP_PC", "text": t, "truth": False} for t in false_steps]

    return true_outputs + false_outputs


def MP_146(data):
    """
    Math premise: compute and state the interquartile range (IQR = TQ - FQ)
    for each x-tick in a box plot. Does not draw any conclusion yet.
    """
    chart_vars = data["chart"]
    chart_data = chart_vars.get("data", [{}])[0].get("data", [])
    y_title = data.get("_y_title_", chart_vars.get("y_title", "Unknown Y"))

    outputs = []
    for entry in chart_data:
        x = entry.get("x")
        fq = entry.get("first_quartile")
        tq = entry.get("third_quartile")

        if fq is None or tq is None:
            continue
        try:
            iqr_val = round(float(tq) - float(fq), 3)
        except Exception:
            iqr_val = "Unknown"

        text = f"The interquartile range (TQ−FQ) for {y_title} at {x} is {iqr_val}"
        outputs.append({"tag": "MP_146", "text": text, "truth": True})

    # --- Add a distractor: wrong arithmetic or swapped quartiles ---
    if len(chart_data) >= 1:
        entry = random.choice(chart_data)
        x = entry.get("x", "some tick")
        fq, tq = entry.get("first_quartile", 0), entry.get("third_quartile", 0)
        if fq is not None and tq is not None:
            wrong_val = round(float(tq) + float(fq), 3)  # intentionally wrong operation
            text = f"The interquartile range (TQ−FQ) for {y_title} at {x} is {wrong_val}"
            outputs.append({"tag": "MP_146", "text": text, "truth": False})

    return outputs



# q_ids = ["59", "62", "63", "65", "72", "68", "146", "166", "167", "168", "169", "170", "18", "18a", "35", "116", "117", "121", "122", "123", "124"]
# print(q_ids, '\n')



QID_PATTERNS = {
    '63': r'Is the difference between the value of (?P<_y_title_>.+) at (?P<_i_>.+) and (?P<_j_>.+) greater than the difference between any two (?P<_x_title_>.+)\?',
    '65': r'Is the sum of the value of (?P<_y_title_>.+) in (?P<_i_>.+) and (?P<_j_>.+) greater than the maximum value of (?P<_y_title_extra_>.+) across all (?P<_x_title_>.+)\?',
    '59': r'Is the value of (?P<_y_title_>.+) at (?P<_i_>.+) less than that at (?P<_j_>.+)\?',
    '72': r'Is it the case that in every (?P<_x_title_>.+), the sum of the value of (?P<_y_title_>.+) for (?P<_legend1_>.+) and (?P<_legend2_>.+) is greater than the value of (?P<_y_title_extra_>.+) for (?P<_legend3_>.+)\?',
    '62': r'Is the value of (?P<_y_title_>.+) for (?P<_legend_>.+) at (?P<_i_>.+) less than that at (?P<_j_>.+)\?',
    '68': r'Is the difference between the value of (?P<_y_title_>.+) for (?P<_legend1_>.+) at (?P<_i_>.+) and at (?P<_j_>.+) greater than the difference between the value of (?P<_y_title_extra_>.+) for (?P<_legend2_>.+) at (?P<_xi_extra_>.+) and at (?P<_xj_extra_>.+)\?',
    '146': r'Does any (?P<_x_title_>.+) have equal interyers-quartile range\?',
    '166': r'Is the value of median of (?P<_y_title_>.+) at (?P<_i_>.+) less than that at (?P<_j_>.+)\?',
    '167': r'Is the value of upper quartile of (?P<_y_title_>.+) at (?P<_i_>.+) less than that at (?P<_j_>.+)\?',
    '169': r'Is the maximum value of (?P<_y_title_>.+) at (?P<_i_>.+) less than that at (?P<_j_>.+)\?',
    '168': r'Is the value of lower quartile of (?P<_y_title_>.+) at (?P<_i_>.+) less than that at (?P<_j_>.+)\?',
    '170': r'Is the minimum value of (?P<_y_title_>.+) at (?P<_i_>.+) less than that at (?P<_j_>.+)\?',
    '18': r'Is the number of lines equal to the number of legend labels\?',
    '18a': r'Is the number of lines equal to the number of mark labels\?',
    '35': r'Does the (?P<_y_title_>.+) monotonically increase over the (?P<_x_title_>.+)\?',
    '116': r'Is the (?P<_legend_>.+) monotonically increasing\?',
    '117': r'Is the (?P<_legend_>.+) monotonically decreasing\?',
    '121': r'Does (?P<_legend_>.+) have low positive correlation\?',
    '122': r'Does (?P<_legend_>.+) have high positive correlation\?',
    '123': r'Does (?P<_legend_>.+) have low negative correlation\?',
    '124': r'Does (?P<_legend_>.+) have high negative correlation\?'
}

def init_data_dict():
    data = {
        '_y_title_': '!Example Y Title',
        '_x_title_': '!Example X Title',

        '_xi_': '!X Value for i',
        '_yi_': '!Y Value for i',
        '_xi_,_yi_': ('!X Value for i', '!Y Value for i'),

        '_xj_': '!X Value for j',
        '_yj_': '!Y Value for j',
        '_xj_,_yj_': ('!X Value for j', '!Y Value for j'),

        '_xk_': '!X Value for k',
        '_yk_': '!Y Value for k',
        '_xk_,_yk_': ('!X Value for k', '!Y Value for k'),

        '__val__': '!Value of data series',
        '__L__': '!Legend L',

        '_legend_': '!Example Legend Label',
        '_fqval_': '!First Whisker Val',
        '_tqval_': '!Third Whisker Val',

        '_max_val_': '!Maximum value of data',
        '_min_val_': '!Minimum value of data',
        '_max_val_leg_': '!Maximum value of data with legend',
        '_min_val_leg_': '!Minimum value of data with legend',

        '__S__': '!Set of scatter pts',
        '__n__': -1,
        '__M__': '!Median of set',

        'line_match': False,
        'mark_match': False,
    }
    return data


def extract_info_from_question(qa_obj, chart_vars=None):
    """
    Extract placeholder info from QA question string using regex patterns.
    Optionally enrich with chart_vars.
    """
    qid = str(qa_obj.get("QID"))
    question = qa_obj.get("question", "")

    # start from global defaults
    data = copy.deepcopy(init_data_dict())

    pattern = QID_PATTERNS.get(qid)
    if not pattern:
        return data   # fallback: defaults only

    match = re.match(pattern, question, re.IGNORECASE)
    if match:
        extracted = match.groupdict()
        # overlay extracted values into the base dict
        for k, v in extracted.items():
            data[k] = v.strip(" '")
    else:
        print(f"[WARN] QID {qid} question did not match pattern:\n  Q: {question}\n  P: {pattern}")
    return data



# 1. Registry of premise chains
PREMISE_CHAINS = {
    "59": [lambda d: DP_val(d, "i"), lambda d: DP_val(d, "j"), RP_59],
    "62": [lambda d: DP_val_leg(d, "i"), lambda d: DP_val_leg(d, "j"), RP_62],
    "63": [lambda d: DP_val(d, "i"), lambda d: DP_val(d, "j"), RP_63],
    "65": [lambda d: DP_val(d, "i"), lambda d: DP_val(d, "j"), DP_max_val, RP_65],
    "72": [lambda d: DP_val_leg_all(d, ("legend1", "legend2")), lambda d: DP_val_leg_all(d, ("legend3",)), RP_72],
    "68": [lambda d: DP_val_leg(d, "i"), lambda d: DP_val_leg(d, "j"), RP_68],
    "146": [lambda d: DP_FQ_exist(d, "all"), lambda d: DP_TQ_exist(d, "all"), 
            lambda d: DP_FQ_val(d, "all"), lambda d: DP_TQ_val(d, "all"), MP_146,  RP_146, ],
    "166": [lambda d: DP_val(d, "i"), lambda d: DP_val(d, "j"), 
            lambda d: MP_Median(d, "i"), lambda d: MP_Median(d, "j"), RP_166],
    "167": [lambda d: DP_TQ_exist(d, "i"),lambda d: DP_TQ_exist(d, "j"),
            lambda d: DP_TQ_val(d, "i"),lambda d: DP_TQ_val(d, "j"),RP_167],
    "168": [lambda d: DP_FQ_exist(d, "i"),lambda d: DP_FQ_exist(d, "j"),
            lambda d: DP_FQ_val(d, "i"),lambda d: DP_FQ_val(d, "j"),RP_168],
    "169": [lambda d: DP_max_box(d, "i"),lambda d: DP_max_box(d, "j"), RP_169],
    "170": [lambda d: DP_Min_Box(d, "i"),lambda d: DP_Min_Box(d, "j"), RP_170],
    "18": [DP_Leg_Count, DP_Line_Count, RP_18],
    "18a": [DP_Mark_Count, DP_Line_Count, RP_18a],
    "35": [DP_val, RP_35],
    "116": [DP_val_leg, RP_116],
    "117": [DP_val_leg, RP_117],
# }
# PREMISE_CHAINS = {
    "121": [DP_scatter_series, MP_PC_all, RP_121],
    "122": [DP_scatter_series, MP_PC_all, RP_122],
    "123": [DP_scatter_series, MP_PC_all, RP_123],
    "124": [DP_scatter_series, MP_PC_all, RP_124],
}

# PREMISE_CHAINS = {
#     "146": [lambda d: DP_FQ_exist(d, "all"), lambda d: DP_TQ_exist(d, "all"), 
#             lambda d: DP_FQ_val(d, "all"), lambda d: DP_TQ_val(d, "all"), RP_146, ],
#     }

# 2. Executor
def build_premise_chain(json_obj, qid, question_text, image_id, chart_obj):
    # print('-'*80)
    # print('-'*80)
    # print('build premise chain for qid', qid)
    # print('-'*80)
    chain = PREMISE_CHAINS.get(str(qid))
    if not chain:
        return None

    premises = []
    # print('\n json_obj')
    # for k in json_obj : 
    #     print(k, json_obj[k])
    data = extract_info_from_question(json_obj)
    
    data['answer'] = json_obj['answer']
    data['chart'] = chart_obj
    # print('-'*80)
    # print('\n data')
    # for k in data : 
    #     print(k, ' ', data[k])
    # print('-'*80)
    for func in chain:
        if callable(func):
            # print('Function : ', func)
            output = func(data)
            # print('output')
            # for k in output : 
            #     print(k)
            if isinstance(output, list):  # multiple premises
                for o in output:
                    premises.append(o)
            else:
                premises.append(o)
        else:
            premises.append({"type": "Misc", "text": str(func), "truth": True})

    return {
        "image_id": image_id,
        "QID": qid,
        "question": question_text,
        "premises": premises
    }
    # print(aefaefef)

# 3. Saver
def save_premise_chain(chain_obj, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{chain_obj['image_id']}__{chain_obj['QID']}.json"
    with open(out_dir / fname, "w") as f:
        json.dump(chain_obj, f, indent=2)


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.replace("\n", " ").replace("  ", " ").strip()

def normalize_chain(chain_obj):
    for p in chain_obj["premises"]:
        p["text"] = normalize_text(p["text"])
    return chain_obj

done_qid = set()


def save_chain_by_type(chain_obj, out_root: Path):
    """Save each premise into its DP/MP/RP folder and per-image JSONL shard."""
    qid = chain_obj["qid"]
    qa_id = chain_obj["qa_id"]
    pmc_id = chain_obj["pmc_id"]

    for p in chain_obj["premises"]:
        tag = p.get("tag", "")
        if not tag:
            continue
        prefix = tag.split("_")[0].upper()
        if prefix not in {"DP", "MP", "RP"}:
            continue

        subdir = out_root / prefix
        subdir.mkdir(parents=True, exist_ok=True)

        obj = {"qa_id": qa_id, "qid": qid, "pmc_id": pmc_id, "premise": p}
        fp = subdir / f"{pmc_id}.jsonl"
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")


def summarize_per_image(stats_per_image):
    """Return a dataframe summarizing per-image premise stats."""
    rows = []
    # print('in stats_per_image', stats_per_image)
    for img, s in stats_per_image.items():
        rows.append({
            "pmc_id": img,
            "questions": s["questions"],
            "premises_total": s["premises"],
            "DP": s["DP"],
            "MP": s["MP"],
            "RP": s["RP"],
            "unique_qids": len(s["qids"])
        })
    return pd.DataFrame(rows).sort_values("premises_total", ascending=False)


def summarize_per_qid(stats_per_qid):
    """Return a dataframe summarizing per-qid premise stats."""
    rows = []
    for qid, s in stats_per_qid.items():
        rows.append({
            "qid": qid,
            "questions": s["questions"],
            "premises_total": s["premises"],
            "DP": s["DP"],
            "MP": s["MP"],
            "RP": s["RP"],
            "avg_chain_len": statistics.mean(s["chain_lengths"]) if s["chain_lengths"] else 0
        })
    return pd.DataFrame(rows).sort_values("premises_total", ascending=False)


def print_stats(stats, df_image, df_qid):
    print("\n===== GLOBAL EDA =====")
    print(f"Questions processed : {stats['questions_total']}")
    print(f"Chains created      : {stats['chains_created']}")
    print(f"Total premises      : {stats['premises_total']}")
    print(f"Avg chain length    : {statistics.mean(stats['chain_lengths']):.2f}")
    print(f"Median chain length : {statistics.median(stats['chain_lengths']):.2f}")
    print(f"Min/Max chain len   : {min(stats['chain_lengths'])} / {max(stats['chain_lengths'])}")
    print("\nPremises by type:")
    for k, v in stats["premises_by_type"].most_common():
        print(f" - {k}: {v}")

    print("\n===== PER IMAGE SUMMARY =====")
    print(df_image.head(10).to_string(index=False))

    print("\n===== PER QID SUMMARY =====")
    print(df_qid.head(10).to_string(index=False))


def get_done_qa_ids(out_dir: Path):
    """Scan DP/MP/RP subfolders for existing qa_ids to support resume."""
    done_ids = set()
    for sub in ["DP", "MP", "RP"]:
        subdir = out_dir / sub
        if not subdir.exists():
            continue
        for fp in subdir.glob("*.jsonl"):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "qa_id" in obj:
                            done_ids.add(obj["qa_id"])
                    except json.JSONDecodeError:
                        continue
    print(f"[Resume] Found {len(done_ids):,} completed QA records across DP/MP/RP.")
    return done_ids


def get_done_qa_ids(out_dir: Path, rebuild_stats: bool = False):
    """
    Scan existing DP/MP/RP shards to recover done QA IDs.
    Optionally rebuild statistics from already saved data.

    Returns:
        done_qa_ids (set)
        stats (dict or None)
    """
    done_ids = set()
    stats = None
    ckpt_file = out_dir / "resume_state.json"

    # --- Load previous checkpoint if exists ---
    if ckpt_file.exists() and not rebuild_stats:
        try:
            with open(ckpt_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            done_ids = set(ckpt.get("done_qa_ids", []))
            stats = ckpt.get("stats_global", None)
            print(f"[Resume] Loaded checkpoint with {len(done_ids):,} done QA IDs.")
            return done_ids, stats
        except Exception as e:
            print(f"[Resume] Failed to read checkpoint ({e}), rescanning shards...")

    # --- Full rescan: build from DP/MP/RP subfolders ---
    print("[Resume] Scanning DP/MP/RP folders to recover progress...")
    stats = {
        "questions_total": 0,
        "chains_created": 0,
        "premises_total": 0,
        "premises_by_type": Counter(),
        "chain_lengths": []
    }
    meta = {}
    for sub in ["DP", "MP", "RP"]:
        subdir = out_dir / sub
        if not subdir.exists():
            continue

        for fp in tqdm(subdir.glob("*.jsonl")):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    print('---')
                    qa_id = obj['qa_id']
                    qid = obj['qid']
                    pmc_id = obj['pmc_id']
                    tag = obj['premise']['tag']
                    text = obj['premise']['text']
                    truth =  obj['premise']['truth']
                    print(qa_id, qid, pmc_id)
                    print(tag, truth)
                    print(text)
                    print('---')

                    if pmc_id in meta : 
                        meta[pmc_id]['qa_id'].add(qa_id)
                        meta[pmc_id]['qid'].add(qid)
                        meta[pmc_id]['tag'].add(tag)
                    else : 
                        meta[pmc_id] = {
                            'qa_id' : set([qa_id]),
                            'qid' : set([qid]),
                            'tag' : set([tag])}
                    if not qa_id:
                        continue
                    done_ids.add(qa_id)

                    premise = obj.get("premise", {})
                    tag = premise.get("tag", "")
                    if tag:
                        stats["premises_by_type"][tag] += 1
                        stats["premises_total"] += 1
                    # Optionally add dummy chain stats
                    stats["questions_total"] += 1
                    stats["chains_created"] += 1
                    stats["chain_lengths"].append(1)
            #         if len(done_ids) > 5 :
            #             break
            # break
    print(f"[Resume] Found {len(done_ids):,} completed QA records across DP/MP/RP.")
    # print('stats', stats)
    # print('ddd')
    # print('meta', meta)
    # print(sdfs)
    return done_ids, stats


def generate_all_chains(qa_list, out_dir: Path):
    """
    Main orchestrator: generates DP/MP/RP chains per QA,
    saves per-image JSONL, and collects hierarchical statistics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["DP", "MP", "RP"]:
        (out_dir / sub).mkdir(exist_ok=True)


    # --- Resume: collect processed qa_ids
    done_qa_ids, saved_stats = get_done_qa_ids(out_dir, rebuild_stats=False)
    
    # --- Aggregators ---
    stats_global = {
        "questions_total": 0,
        "chains_created": 0,
        "premises_total": 0,
        "premises_by_type": Counter(),
        "chain_lengths": []
    }
    
    # If checkpoint or rebuild provided previous stats
    if saved_stats:
        stats_global = saved_stats
        print(f"[Resume] Initialized global stats from checkpoint.")
        print(stats_global)

    stats_per_image = defaultdict(lambda: {"questions": 0, "premises": 0, "DP": 0, "MP": 0, "RP": 0, "qids": set()})
    stats_per_qid = defaultdict(lambda: {"questions": 0, "premises": 0, "DP": 0, "MP": 0, "RP": 0, "chain_lengths": []})

    skipped, new = 0, 0
    for qa in tqdm(qa_list, desc="Generating DP/RP/MP chains"):
        qid = qa.get("QID")
        image_id = qa.get("PMC_ID")
        qa_id = qa.get("qa_id")
        question_text = qa.get("question")
        # print('->>', qid, image_id, qa_id, '\n', question_text)
        # --- Skip logic for resume
        if qa_id in done_qa_ids:
            skipped += 1
            # print('skipped' ,skipped)
            continue

        if not qid or str(qid) not in PREMISE_CHAINS:
            # print('qid X')
            continue

        chart_fp = P.chart_json_dir / f"{image_id}.json"
        # print('chart_fp', chart_fp)
        if not chart_fp.exists():
            # print('chart_fp X')
            continue
            
        with open(chart_fp, "r", encoding="utf-8") as fh:
            chart_obj = json.load(fh)
            # print('chart_obj', len(chart_obj))
            
        chart_vars = extract_chart_variables(chart_obj)
        # print('chart_vars', chart_vars)
        try:    
            chain_obj = build_premise_chain(qa, qid, question_text, image_id, chart_vars)
            # print('chain_obj', len(chain_obj))
            
            if not chain_obj:
                # print('chain_obj X')
                continue

            chain_record = {
                "qa_id": qa_id,
                "qid": qid,
                "pmc_id": image_id,
                "premises": chain_obj["premises"]
            }

            save_chain_by_type(chain_record, out_dir)
            new += 1

            # --- Stats update
            plen = len(chain_record["premises"])
            stats_global["questions_total"] += 1
            stats_global["chains_created"] += 1
            stats_global["premises_total"] += plen
            stats_global["chain_lengths"].append(plen)

            img_stats = stats_per_image[image_id]
            qid_stats = stats_per_qid[qid]
            img_stats["questions"] += 1
            img_stats["premises"] += plen
            img_stats["qids"].add(qid)
            qid_stats["questions"] += 1
            qid_stats["premises"] += plen
            qid_stats["chain_lengths"].append(plen)

            for p in chain_record["premises"]:
                tag = p["tag"]
                stats_global["premises_by_type"][tag] += 1
                prefix = tag.split("_")[0].upper()
                if prefix in {"DP", "MP", "RP"}:
                    img_stats[prefix] += 1
                    qid_stats[prefix] += 1

        except Exception as e:
            print('Exception occured : ', qid, image_id)
            print('\n chart Vars ')
            for k in chart_vars :
                print(k, chart_vars[k])
            
            print('\n qa obj ')
            for k in qa : 
                print(k, qa[k])

            print(f"[Error] {qa.get('qa_id')} → {e}")
            continue 

    print(f"\n[Resume] Skipped {skipped:,} existing QAs; generated {new:,} new.")

    # --- Summaries ---
    df_image = summarize_per_image(stats_per_image)
    df_qid = summarize_per_qid(stats_per_qid)
    print_stats(stats_global, df_image, df_qid)

    # --- Save CSV summaries ---
    df_image.to_csv(out_dir / "summary_per_image.csv", index=False)
    df_qid.to_csv(out_dir / "summary_per_qid.csv", index=False)

    return stats_global, df_image, df_qid


def parallel_generate_all_chains(qa_data, out_dir, num_chunks=8):
    step_size = math.ceil(len(qa_data) / num_chunks)
    futures = []

    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        for i in range(num_chunks):
            start = i * step_size
            end = min((i + 1) * step_size, len(qa_data))
            sub_data = qa_data[start:end]
            sub_out_dir = out_dir / f"chunk_{i}"

            sub_out_dir.mkdir(parents=True, exist_ok=True) 
            futures.append(executor.submit(generate_all_chains, sub_data, sub_out_dir))

        # Optional: collect results (e.g., stats)
        all_stats = []
        for future in as_completed(futures):
            result = future.result()  # or handle errors
            all_stats.append(result)

    return all_stats



# -------------------------------------
# Check Utils 
# -------------------------------------

def superscript_to_int(superscript_str):
    """Convert superscript string to integer."""
    superscript_map = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", 
                       "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
    return int("".join(superscript_map.get(char, '') for char in superscript_str))

def convert_to_numeric_or_string(value):
    # Patterns to match
    patterns = [
        (r'^(\d+)$', lambda x: int(x)),  # Integer
        (r'^(\d+\.\d+)$', lambda x: float(x)),  # Float
        (r'^(\d+(\.\d+)?[eE][+-]?\d+)$', lambda x: float(x)),  # Scientific notation
        (r'^(\d+(\.\d+)?)[×x*]10\^([+-]?\d+)$', lambda x: float(x.replace('×', 'e').replace('x', 'e').replace('*', 'e').replace('^', ''))),  # Custom scientific notation with "x" or "*"
        (r'^(\d{1,3}(,\d{3})*(\.\d+)?)$', lambda x: float(x.replace(',', ''))),  # Float with commas
        # Superscript scientific notation (e.g., "2.5×10⁵")
        (r'^(\d+(\.\d+)?)[×x*]10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$', lambda x: float(x.split('×')[0] + 'e' + str(superscript_to_int(x.split('×')[1][2:])))),
    ]
    
    for pattern, converter in patterns:
        match = re.match(pattern, value)
        if match:
            try:
                return converter(match.group(0))
            except ValueError:
                pass  # If conversion fails, try the next pattern
    
    # Fallback: return the original value as string if no pattern matches
    return value

def analyze_array(values):
    all_numeric = all(isinstance(value, (int, float)) for value in values)
    if all_numeric:
        return (min(values), max(values))
    else:
        return 'categorical'
    
def extract_txt_getRect(tb, filtered_ids ):
    results = []
    for block in tb:
        if block["id"] in filtered_ids:
            text = block["text"]
            polygon = block["polygon"]
            x = polygon["x0"]
            y = polygon["y0"]
            w = polygon["x1"] - polygon["x0"]  # Assuming x1 and x0 are the horizontal bounds
            h = polygon["y2"] - polygon["y0"]  # Assuming y2 and y0 are the vertical bounds
            results.append({
                "id": block["id"],
                "text": text,
                "bounding_box": {"x": x, "y": y, "width": w, "height": h}
            })
    return results

def get_combined_centroid(tb, filtered_ids):
    sum_x = 0
    sum_y = 0

    for block in tb:
        if block["id"] in filtered_ids:
            polygon = block["polygon"]
            centroid_x = (polygon["x0"] + polygon["x1"] + polygon["x2"] + polygon["x3"]) / 4
            centroid_y = (polygon["y0"] + polygon["y1"] + polygon["y2"] + polygon["y3"]) / 4
            sum_x += centroid_x
            sum_y += centroid_y
    
    filtered_count = len(filtered_ids)
    return (sum_x / filtered_count, sum_y / filtered_count)

def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def assign_axis_title(xax_ctr, yax_ctr, axs_titles, plot_bb):
    assignments = {'x_title': None, 'y_title': None}
    # Case when axis centroids are not provided
    if not xax_ctr or not yax_ctr:
        # Determine top-left-most title to assign as y_title
        titles_sorted_by_position = sorted(axs_titles, key=lambda t: (t["bounding_box"]["y"], t["bounding_box"]["x"]))
        if len(titles_sorted_by_position) > 1:
            assignments['y_title'] = titles_sorted_by_position[0]["text"]
            assignments['x_title'] = titles_sorted_by_position[1]["text"]
        elif titles_sorted_by_position:
            # When there's only one title, use plot_bb to decide
            title = titles_sorted_by_position[0]
            title_centroid = ((title["bounding_box"]["x"] + title["bounding_box"]["width"] / 2),
                              (title["bounding_box"]["y"] + title["bounding_box"]["height"] / 2))
            plot_centroid = (plot_bb["x0"] + plot_bb["width"] / 2, plot_bb["y0"] + plot_bb["height"] / 2)
            # If title is above or to the left of plot centroid, it's y_title, else x_title
            if title_centroid[0] < plot_centroid[0] or title_centroid[1] < plot_centroid[1]:
                assignments['y_title'] = title["text"]
            else:
                assignments['x_title'] = title["text"]
    else:
        # Original logic for assigning axis titles based on proximity
        for title in axs_titles:
            title_centroid = ((title["bounding_box"]["x"] + title["bounding_box"]["width"] / 2),
                              (title["bounding_box"]["y"] + title["bounding_box"]["height"] / 2))
            dist_to_x = distance(title_centroid, xax_ctr) if xax_ctr else float('inf')
            dist_to_y = distance(title_centroid, yax_ctr) if yax_ctr else float('inf')
            
            if dist_to_x < dist_to_y:
                closest_axis = 'x_title'
            else:
                closest_axis = 'y_title'
            assignments[closest_axis] = title["text"]
    return assignments


# -------------------------------------
# Checks
# -------------------------------------

def check1(js_obj):
    if 'task1' in js_obj and js_obj['task1'] is not None and \
               'output' in js_obj['task1'] and js_obj['task1']['output'] is not None and \
               'chart_type' in js_obj['task1']['output'] and \
               js_obj['task1']['output']['chart_type'] is not None:
        return True 
    return False

def check2(js_obj):
    if 'task2' in js_obj and js_obj['task2'] is not None and \
               'output' in js_obj['task2'] and js_obj['task2']['output'] is not None and \
               'text_blocks' in js_obj['task2']['output'] and \
               len(js_obj['task2']['output']['text_blocks']) > 0:
        return True 
    return False

def check3(js_obj):
    if 'task3' in js_obj and js_obj['task3'] is not None and \
               'output' in js_obj['task3'] and js_obj['task3']['output'] is not None and \
               'text_roles' in js_obj['task3']['output'] and \
               len(js_obj['task3']['output']['text_roles']) > 0:
        return True 
    return False

def check4(js_obj):
    if 'task4' in js_obj and js_obj['task4'] is not None and \
               'output' in js_obj['task4'] and js_obj['task4']['output'] is not None and \
               'axes' in js_obj['task4']['output'] and \
               (len(js_obj['task4']['output']['axes']['x-axis']) > 0 or\
                len(js_obj['task4']['output']['axes']['y-axis'])) > 0:
        return True 
    return False

def check5(js_obj):
    if 'task5' in js_obj and js_obj['task5'] is not None and \
               'output' in js_obj['task5'] and js_obj['task5']['output'] is not None and \
               'legend_pairs' in js_obj['task5']['output'] and \
               len(js_obj['task5']['output']['legend_pairs']) > 0:
        return True 
    return False

def check6(js_obj):
    if 'task6' in js_obj and js_obj['task6'] is not None and \
               'output' in js_obj['task6'] and js_obj['task6']['output'] is not None and \
               'data series' in js_obj['task6']['output'] and \
               len(js_obj['task6']['output']['data series']) > 0:
        return True 
    return False


# -------------------------------------
# Chart Vars
# -------------------------------------
def extract_chart_variables(js_obj):
    chart_variables = {
        'chart_type':    None,
        'all_txt':       None,
        'xmin':          None, 
        'xmax':          None, 
        'ymin':          None, 
        'ymax':          None,
        'x_title':       None, 
        'y_title':       None,
        'x_ticks':       None,
        'y_ticks':       None, 
        'categorical':   None,
        'number_of_ds':  None, 
        'legend_labels': None,
        'mark_label':    None, 
        'data' :         None
    }
    chart_variables = {}

    # Task 1: Chart Type
    if check1(js_obj) : 
        chart_type = js_obj['task1']['output']['chart_type']
        chart_variables['chart_type'] = chart_type

    
    if check2(js_obj) : 
        tb = js_obj['task2']['output'].get('text_blocks', [])
        chart_variables['all_txt'] = [block['text'] for block in tb]

    # Axis Titles
    if check3(js_obj) and check4(js_obj) : 
        trole = js_obj['task3']['output'].get('text_roles', [])
        # print('===trole=====', trole)
        axs_title_id = [block['id'] for block in trole if block['role'] == 'axis_title']
        axs_titles = extract_txt_getRect(tb, axs_title_id)
        mark_label_id = [block['id'] for block in trole if block['role'] == 'mark_label'] 
        # print('===mark_label_id=====', mark_label_id)
        if len(mark_label_id) :
            all_txt = js_obj['task2']['output'].get('text_blocks', [])
            chart_variables['mark_label'] = [
                block['text']
                for k in mark_label_id
                for block in all_txt
                if block['id'] == k
            ]
            # print('====mark_label====', mark_label)
    # Task 4: Axis Information
 
        xax_id = [block['id'] for block in js_obj['task4']['output'].get('axes', {}).get('x-axis', [])]
        xax_ctr = get_combined_centroid(tb, xax_id) if len(xax_id) > 0 else None
    
        yax_id = [block['id'] for block in js_obj['task4']['output'].get('axes', {}).get('y-axis', [])]
        yax_ctr = get_combined_centroid(tb, yax_id) if len(yax_id) > 0 else None

        # Analyze ticks for both axes
        yticks = [convert_to_numeric_or_string(block['text']) for block in tb if block['id'] in yax_id]
        chart_variables['y_ticks'] = yticks if len(yax_id) > 0 else None
        yrange = analyze_array(yticks) if len(yax_id) > 0 else None
    
        xticks = [convert_to_numeric_or_string(block['text']) for block in tb if block['id'] in xax_id]
        chart_variables['x_ticks'] = xticks if len(xax_id) > 0 else None
        xrange = analyze_array(xticks) if len(xax_id) > 0 else None

        # Handle categorical vs numerical axis data
        if isinstance(xrange, tuple):
            chart_variables['xmin'], chart_variables['xmax'] = xrange
            chart_variables['categorical'] = False
        else:
            chart_variables['categorical'] = True
            # Ensure random choice doesn't pick the same text for both min and max
            if chart_variables['all_txt']:
                chart_variables['xmin'] = chart_variables['xmax'] = random.choice(chart_variables['all_txt'])

        if isinstance(yrange, tuple):
            chart_variables['ymin'], chart_variables['ymax'] = yrange
        
        axs_assignments = assign_axis_title(xax_ctr, yax_ctr, axs_titles, js_obj['task4']['output']['_plot_bb'])
        chart_variables.update(axs_assignments)


    # Task 5: Legend Information
    if check5(js_obj) : 
        lid = [block['id'] for block in js_obj['task5']['output']['legend_pairs']]
        llbls = [block['text'] for block in tb if block['id'] in lid]
        chart_variables['legend_labels'] = llbls
    if check6(js_obj) : 
        num_ds = len(js_obj['task6']['output']['data series'])
        chart_variables['number_of_ds'] = num_ds
        chart_variables['data'] = js_obj['task6']['output']['data series']
    return chart_variables


# -------------------------------------
# TEMPLATES
# -------------------------------------
SP_TEMPLATES = {
    "SP0": "The type of chart is {chart_type}.",
    "SP1": "The dependant axis is labeled as {y_title}.",
    "SP2": "The independant axis is labeled as {x_title}.",
    "SP3": "The dependant axis ranges from a minimum of {ymin} to a maximum of {ymax} in {y_title}.",
    "SP4": "The independant axis ranges from a minimum of {xmin} to a maximum of {xmax} in {x_title}.",
    "SP5": "The independant axis is categorical with the labels {x_ticks}.",
    "SP6": "Tick marks corresponding to specified {x_title} values are present on the independant axis.",
    "SP7": "Tick marks corresponding to specified {y_title} values are present on the dependant axis.",
    "SP8": "The chart contains a legend that differentiates between the {number_of_ds} data series.",
    "SP9": "Each data series in the legend corresponds to a unique representation on the chart "
           "(e.g., color, pattern, line type) and has the labels {legend_labels}.",
}

CHART_TYPES = [
    "line", "bar", "horizontal bar", "vertical bar",
    "scatter", "pie", "area", "heatmap",
    "vertical box", "horizontal box"
]

# -------------------------------------
# GENERATION
# -------------------------------------
def generate_structure_premises(vars_dict, templates=SP_TEMPLATES, false_per_true=3):
    premises = []

    for key, template in templates.items():
        variable_names = re.findall(r"{(.*?)}", template)
        if not all(vars_dict.get(v) is not None for v in variable_names):
            continue

        # --- special case: SP4/SP5 pair ---
        if key in ("SP4", "SP5"):
            if key == "SP5":
                continue  # only handle once in SP4 branch

            if vars_dict["categorical"]:
                # SP5 True, SP4 False
                values5 = {v: vars_dict[v] for v in re.findall(r"{(.*?)}", templates["SP5"])}
                premises.append(f"SP5: True : " + templates["SP5"].format(**values5))
                values4 = {v: vars_dict[v] for v in re.findall(r"{(.*?)}", templates["SP4"])}
                premises.append(f"SP4: False : " + templates["SP4"].format(**values4))

                # add SP5 distractors
                for _ in range(false_per_true):
                    false_vals = {v: random.choice(vars_dict.get("all_txt", ["?"])) for v in values5}
                    premises.append(f"SP5: False : " + templates["SP5"].format(**false_vals))

            else:
                # SP4 True, SP5 False
                values4 = {v: vars_dict[v] for v in re.findall(r"{(.*?)}", templates["SP4"])}
                premises.append(f"SP4: True : " + templates["SP4"].format(**values4))
                values5 = {v: vars_dict[v] for v in re.findall(r"{(.*?)}", templates["SP5"])}
                premises.append(f"SP5: False : " + templates["SP5"].format(**values5))

                # add SP4 distractors
                for _ in range(false_per_true):
                    false_vals = {v: random.choice(vars_dict.get("all_txt", ["?"])) for v in values4}
                    premises.append(f"SP4: False : " + templates["SP4"].format(**false_vals))

            continue

        # --- general case ---
        values = {v: vars_dict[v] for v in variable_names}
        premises.append(f"{key}: True : " + template.format(**values))

        for _ in range(false_per_true):
            if key == "SP0":
                false_chart_type = random.choice([ct for ct in CHART_TYPES if ct != vars_dict["chart_type"]])
                premises.append(f"{key}: False : The type of chart is {false_chart_type}.")
            else:
                false_vals = {v: random.choice(vars_dict.get("all_txt", ["?"])) for v in variable_names}
                premises.append(f"{key}: False : " + template.format(**false_vals))

    return premises



# -------------------------------------
# WRAPPER
# -------------------------------------
def process_all_charts(cjson_dir: Path, out_dir: Path, seed=1337):
    """
    Process all chart JSONs and write premises per chart into .txt files.
    Returns EDA stats.
    """
    random.seed(seed)  # reproducibility for false premises
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in cjson_dir.iterdir() if f.suffix == ".json"]
    print(f'found total files {len(files)}')

    # --- tracking stats ---
    total_images = 0
    premise_counts = Counter()         # per SP key
    true_count = 0
    false_count = 0
    per_image_counts = []

    for f in tqdm(files, desc="Generating SP"):
        out_file = out_dir / (f.stem + ".txt")
        # if out_file.exists():
            # continue

        with f.open("r", encoding="utf-8") as fh:
            js_obj = json.load(fh)
        # print(f'js_obj loaded {len(js_obj)}')
        chart_vars = extract_chart_variables(js_obj)
        # print(f'chart_vars loaded {len(chart_vars)}')
        
        desc = generate_structure_premises(chart_vars)
        if not desc:
            continue

        # save
        out_file.write_text("\n".join(desc), encoding="utf-8")

        # stats
        total_images += 1
        per_image_counts.append(len(desc))
        for line in desc:
            # format: "SP#: True/False : sentence"
            sp_key, label, *_ = line.split(":", 2)
            premise_counts[sp_key.strip()] += 1
            if "True" in label:
                true_count += 1
            elif "False" in label:
                false_count += 1

    # --- summary EDA ---
    print("\n===== Structure Premises EDA =====")
    print(f"Total images processed : {total_images}")
    print(f"Total premises created : {sum(per_image_counts)}")
    print(f" - True premises       : {true_count}")
    print(f" - False premises      : {false_count}")
    print("\nPer SP type:")
    for sp, cnt in sorted(premise_counts.items()):
        print(f"  {sp}: {cnt}")

    if per_image_counts:
        print("\nPremises per image:")
        print(f"  Min: {min(per_image_counts)}")
        print(f"  Max: {max(per_image_counts)}")
        print(f"  Avg: {statistics.mean(per_image_counts):.2f}")
        print(f"  Median: {statistics.median(per_image_counts):.2f}")

    return {
        "images": total_images,
        "premises_total": sum(per_image_counts),
        "true": true_count,
        "false": false_count,
        "per_sp": dict(premise_counts),
        "premises_per_image": per_image_counts,
    }


def fix_text_files(directory):
    updated_files_count = 0  # Initialize the counter for updated files
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Ensure we're only processing text files
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            corrected_lines = []
            needs_update = False  # Flag to track if the current file needs updating
            for line in lines:
                if line.startswith("SP"):
                    corrected_lines.append(line.strip())
                else:
                    needs_update = True  # Set flag to True if a line doesn't start with "SP"
                    corrected_lines[-1] += " " + line.strip()
            
            if needs_update:
            #     # Only write back to the file if an update is needed
                corrected_text = '\n'.join(corrected_lines)
                with open(filepath, 'w') as file:
                    file.write(corrected_text)
                
                updated_files_count += 1  # Increment the counter
                print(f"Processed and updated {filename}")
    
    # After processing all files, print the total number of files updated
    print(f"Total files updated: {updated_files_count}")


def eda_on_text_files(directory):
    sp_frequencies = Counter()
    true_false_counts = {'True': 0, 'False': 0}
    vocabulary = set()
    all_words = []
    
    # Step 1: Read the Text Files
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                for line in file:
                    # Step 2: Extract Information
                    sp_category = re.match(r"(SP\d+):", line)
                    truth_value = "True" if ": True :" in line else "False"
                    words = re.findall(r"\b\w+\b", line)
                    all_words.extend(words)
                    
                    if sp_category:
                        sp_frequencies[sp_category.group(1)] += 1
                        true_false_counts[truth_value] += 1
                        vocabulary.update(words)
    
    # Step 3: Analyze Data - Already done during extraction
    
    # Step 4: Plot Results
    # Frequency per SP category
    plt.figure(figsize=(10, 6))
    plt.bar(sp_frequencies.keys(), sp_frequencies.values())
    plt.title('Frequency per SP Category')
    plt.xlabel('SP Category')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
    
    # True vs. False count
    plt.figure(figsize=(6, 4))
    plt.bar(true_false_counts.keys(), true_false_counts.values(), color=['blue', 'red'])
    plt.title('True vs. False Statements')
    plt.xlabel('Statement Truth Value')
    plt.ylabel('Count')
    plt.show()
    
    # Optional: Print vocabulary size
    print(f"Vocabulary size: {len(vocabulary)}")
    return vocabulary, all_words



if __name__ == "__main__":
    import multiprocessing



    SEED = 1337
    random.seed(SEED)
    np.random.seed(SEED)



    validate_paths(P)
    summarize_dataset(P)

    filter_ids = None # set(line.strip() for line in open(P.filter_list))
    qa_data = load_qa_jsons(P.save_dir / "cleaned_jsons", filter_ids)

    print(qa_data[0:3])



    multiprocessing.freeze_support()  # Optional, useful for frozen executables


    ################################
    ## Run Generate_all_chains
    ################################
    '''
    out_dir = P.save_dir / "PremiseChains_4"
    stats = generate_all_chains(qa_data[:100000], out_dir)
    '''
        
    ################################
    ## Run Generate_all_chains Concurrent 
    ################################    
    out_dir = P.save_dir / "PremiseChains_1"
    import time 
    a = time.time()
    stats = parallel_generate_all_chains(qa_data, out_dir)
    # stats = generate_all_chains(qa_data[:100000], out_dir)
    print('-->> time', time.time() - a )

        
    ################################
    ## Run SP 
    ################################


    # # ----

    # all_extract = [extract_info_from_question(q) for q in questions]
    # print(all_extract[0])


    # cjson_dir = P.chart_json_dir
    # out_dir = P.save_dir/'Structure'
    # print(cjson_dir, out_dir)
    # stats = process_all_charts(cjson_dir, out_dir)
    
    # Specify the directory containing your text files
    ################################
    ### Fix Run \n strp
    ################################
    # directory = out_dir
    # fix_text_files(directory)


    ################################
    ### Vocab EDA 
    ################################
    # Specify the directory containing your text files
    # directory = '/path/to/your/text/files'
    # vocab, all_words = eda_on_text_files(directory)
