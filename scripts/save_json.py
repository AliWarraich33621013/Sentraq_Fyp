import json


def main() -> None:
    payload = {
        "metadata": {"source": "file", "note": "User-provided transcript text"},
        "summary": (
            "A motivational, step-by-step guide to reducing overthinking and making better, faster decisions. "
            "It opens with reframes: confidence follows action, not the other way around; stop waiting for approval, "
            "certainty, or ease. The talk defines decision fatigue and ego depletion—too many micro-choices drain mental energy, "
            "so batch small decisions and make high‑stakes choices early. It introduces Bezos’s Type‑1 vs Type‑2 framework: "
            "treat irreversible/high‑stakes decisions slowly and reversible/low‑stakes ones quickly; most people invert this. "
            "It argues emotions precede logic (Damasio): name the driving feeling, test whether it’s trustworthy, then reason. "
            "It adds practical tools: the 10‑10 rule (how you’ll feel in 10 minutes/10 months/10 years), regret simulation (choose the path you’ll respect), "
            "and identity alignment (decide as your future self). Finally: decide, then move within 5 minutes—action reduces anxiety. "
            "Overall, prioritize clarity over perfection, protect cognitive energy, and build momentum through decisive action."
        ),
        "abuse_detection": {"overall_severity": 0.0, "flags": []},
        "stats": {"num_sentences": 140, "num_chunks": 7, "total_chars": 12000},
    }
    with open("out_from_text.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()


