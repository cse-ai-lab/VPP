# Visual Premise Proving (VPP)

**VPP** is a benchmark for *premise-grounded* visual reasoning over scientific figures (charts/plots/diagrams), designed to evaluate whether models can chain correct intermediate premises into a final answer.

> **Status:** Active development  
> **Primary repo:** cse-ai-lab/VPP  
> **Maintainer:** @crazysal (edit these)

---

## 🔥 What makes VPP different?

- **Premise chains:** Each question is associated with a set of intermediate premises that can be evaluated.
- **Fine-grained evaluation:** Score models on *which* premises they get right, not just final accuracy.
- **Scientific visual focus:** Targets plots, multi-panel figures, and caption-linked evidence.

---

## 📄 Paper

- **Paper (PDF):** *TBD*
- **arXiv:** *TBD*
- **BibTeX:**

```bibtex
@misc{vpp2026,
  title={Visual Premise Proving (VPP)},
  author={...},
  year={2026},
  note={arXiv: TBD}
}
````

---

## 📦 Dataset

* **Download:** *TBD*
* **Format:** JSONL with fields like `id`, `qa_id`, `qid`, `image`, `tag`, `truth`, `source`, `question`, `premises`, etc.
* **Splits:** train / val / test

### Quick example (illustrative)

```json
{
  "id": "example_0001",
  "qa_id": "pmc123__tagA",
  "qid": "qid_001",
  "image": "images/pmc123_fig2.png",
  "question": "Is the blue curve increasing after x=10?",
  "premises": [
    {"tag": "SP", "text": "The x-axis ranges from 0 to 20.", "truth": true},
    {"tag": "DP", "text": "After x=10 the blue curve slopes upward.", "truth": true}
  ]
}
```

---

## 🧪 Baselines

| Model    | Modality    | Setup           | Notes       |
| -------- | ----------- | --------------- | ----------- |
| InternVL | Vision+Text | zero-shot / SFT | add details |
| Qwen-VL  | Vision+Text | zero-shot / SFT | add details |
| Gemini   | Vision+Text | API eval        | add details |

---

## 📊 Leaderboard (placeholder)

| Rank | Model | Final Acc | Premise Acc | Chain-F1 | Notes |
| ---- | ----- | --------- | ----------- | -------- | ----- |
| 1    | TBD   | TBD       | TBD         | TBD      | TBD   |

**Metrics (examples):**

* Final answer accuracy
* Premise-level accuracy
* Chain consistency / chain-F1
* Dedup-true, chained-true (your custom metrics)

---

## 🧰 Code

* **Training / inference:** `scripts/`
* **Evaluation:** `eval/`
* **Data tools:** `etl/`

### Reproduce (placeholder)

```bash
# Install
pip install -r requirements.txt

# Evaluate a model on test
python eval/run_eval.py --split test --preds preds.jsonl --out metrics.json
```

---

## 🗺️ Roadmap

* [ ] Finalize dataset release packaging
* [ ] Baseline runs: InternVL / Qwen-VL / Gemini
* [ ] Add ablations: premise-only vs image-only vs full chain
* [ ] Public code + docs polish

---

## 📜 License

* Code: *TBD (MIT/Apache-2.0 recommended)*
* Data: *TBD (CC BY 4.0 often used for datasets; ensure compliance)*

---

## 🙏 Acknowledgements

* Figure sources: *TBD*
* Inspired by: premise-grounded evaluation + chain-based reasoning ideas

