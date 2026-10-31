"""Development-only (prompt_dev, 199 claims) checks for the mitigation plan. Never run on dev / SciFact-Open.

1. Re-score M1's existing ledgers with alternative contradiction rules (no new LLM calls).
2. M1 vs B1 on the 60 pilot claims used to revise prompts vs the 139 untouched ones.
3. Where M1's wrong NEI on gold-SUPPORTS claims come from.
4. B1 / M1 agreement (for the fallback idea of using the ledger as a check on B1).
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oblig.ledger.aggregation import scifact_v1  # noqa: E402
from oblig.ledger.schema import CONTRADICTED, SATISFIED, Ledger  # noqa: E402

P = ROOT / "to_transfer/pilot/predictions"
LABELS = ("SUPPORTS", "REFUTES", "NEI")


def load(name):
    recs = [json.loads(l) for l in open(P / f"pilot_{name}.jsonl")]
    return {r["claim_id"]: r for r in recs}


def macro_f1(pairs):
    f1s = []
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in pairs)
        fp = sum(g != c and p == c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / 3, f1s


def unsupported(pairs):
    nei = [p for g, p in pairs if g == "NEI"]
    return sum(p != "NEI" for p in nei) / len(nei)


def show(name, pairs):
    m, (s, r, n) = macro_f1(pairs)
    print(f"  {name:<44} macro-F1 {m:.3f}  (S {s:.2f} / R {r:.2f} / NEI {n:.2f})  unsupported {unsupported(pairs):.3f}")


def docs(led, o):
    return {e.doc_id for e in led.evidence(o.obligation_id)}


def verdict(led, discount):
    """scifact_v1 with CONTRADICTED statuses for which discount(led, o) is True treated as unresolved."""
    core = [o for o in led.obligations if o.criticality == "core"]
    if not core:
        return "NEI"
    st = {o.obligation_id: led.status(o.obligation_id) for o in led.obligations}
    for o in led.obligations:
        if st[o.obligation_id] == CONTRADICTED and discount(led, o):
            st[o.obligation_id] = "UNSUPPORTED"
    if any(s == CONTRADICTED for s in st.values()):
        return "REFUTES"
    if not all(st[o.obligation_id] == SATISFIED for o in core):
        return "NEI"
    return "SUPPORTS" if set.intersection(*[docs(led, o) for o in core]) else "NEI"


def subjects(led):
    return [o for o in led.obligations if o.type == "subject"]


RULES = {
    "V0 frozen rule (sanity: must equal M1)": lambda led, o: False,
    "V1 subject obligations cannot be contradicted": lambda led, o: o.type == "subject",
    "V2 only relation/qualifier contradictions count": lambda led, o: o.type in ("subject", "outcome"),
    "V3 contradicting abstract cited for the subject": lambda led, o: bool(subjects(led)) and not any(
        docs(led, o) & docs(led, s) for s in subjects(led)),
    "V4 ... and the subject SATISFIED there": lambda led, o: bool(subjects(led)) and not any(
        docs(led, o) & docs(led, s) for s in subjects(led) if led.status(s.obligation_id) == SATISFIED),
}

m1, b1, llm, m2 = load("m1"), load("b1"), load("m1_llmagg"), load("m2")
ids = sorted(m1)
assert len(ids) == 199 and set(ids) == set(b1) == set(llm) == set(m2)
gold = {i: m1[i]["gold_label"] for i in ids}

print("1. M1 ledgers re-scored (prompt_dev, 199 claims)")
show("B1 (direct RAG)", [(gold[i], b1[i]["pred_label"]) for i in ids])
show("M1-LLMagg", [(gold[i], llm[i]["pred_label"]) for i in ids])
for name, rule in RULES.items():
    preds = {i: verdict(Ledger.from_dict(m1[i]["ledger"]), rule) for i in ids}
    if name.startswith("V0"):
        assert all(preds[i] == m1[i]["pred_label"] for i in ids), "re-scoring does not reproduce M1"
    changed = [(gold[i], m1[i]["pred_label"], preds[i]) for i in ids if preds[i] != m1[i]["pred_label"]]
    fixed = sum(g == p for g, _, p in changed)
    broken = sum(g == o for g, o, _ in changed)
    show(name, [(gold[i], preds[i]) for i in ids])
    if changed:
        print(f"    changed {len(changed)}: fixed {fixed}, broken {broken}; {dict(Counter((g, o, p) for g, o, p in changed))}")
print("  same rules on M2's final ledgers:")
for name, rule in RULES.items():
    preds = {i: verdict(Ledger.from_dict(m2[i]["ledger"]), rule) for i in ids}
    if name.startswith("V0"):
        assert all(preds[i] == m2[i]["pred_label"] for i in ids), "re-scoring does not reproduce M2"
    show("M2 " + name.split(" ", 1)[0], [(gold[i], preds[i]) for i in ids])

print("\n2. Prompt-revision claims (pilot 60) vs untouched development claims")
pilot = set(json.loads(l)["claim_id"] for l in open(P / "pilot_b1_sc5.jsonl"))
for label, sub in (("pilot 60", [i for i in ids if i in pilot]), ("other 139", [i for i in ids if i not in pilot])):
    print(f" {label} (n={len(sub)}, gold {dict(Counter(gold[i] for i in sub))})")
    show("B1", [(gold[i], b1[i]["pred_label"]) for i in sub])
    show("M1", [(gold[i], m1[i]["pred_label"]) for i in sub])

print("\n3. M1 gold-SUPPORTS -> NEI: statuses of the core obligations that were not SATISFIED")
miss = Counter()
for i in ids:
    if gold[i] == "SUPPORTS" and m1[i]["pred_label"] == "NEI":
        led = Ledger.from_dict(m1[i]["ledger"])
        for o in led.obligations:
            if o.criticality == "core" and led.status(o.obligation_id) != SATISFIED:
                miss[(o.type, led.status(o.obligation_id))] += 1
print(" ", dict(miss.most_common()))

print("\n4. B1 / M1 agreement")
agree = [i for i in ids if b1[i]["pred_label"] == m1[i]["pred_label"]]
acc = sum(gold[i] == b1[i]["pred_label"] for i in agree) / len(agree)
dis = [i for i in ids if i not in set(agree)]
print(f"  agree on {len(agree)}/199 ({len(agree) / 199:.0%}), accuracy there {acc:.3f}; "
      f"disagree on {len(dis)}: B1 right {sum(gold[i] == b1[i]['pred_label'] for i in dis)}, "
      f"M1 right {sum(gold[i] == m1[i]['pred_label'] for i in dis)}")
show("B1, NEI where M1 disagrees", [(gold[i], b1[i]["pred_label"] if i in set(agree) else "NEI") for i in ids])

print("\n5. Anatomy of M1's REFUTES verdicts (input to the Auditor fix; added 30 Oct, 18:30 IST)")
from oblig.data.scifact import load_corpus, load_eval_set  # noqa: E402

claims = {c.id: c for c in load_eval_set("prompt_dev", load_corpus().keys())}
kinds, right_subject, per_claim, n_contra = Counter(), Counter(), [], Counter()
for i in ids:
    led = Ledger.from_dict(m1[i]["ledger"])
    contra = [o for o in led.obligations if led.status(o.obligation_id) == CONTRADICTED]
    per_claim.append(len(contra))
    if m1[i]["pred_label"] != "REFUTES":
        continue
    n_contra[(gold[i] == "REFUTES", len(contra))] += 1
    subs = subjects(led)
    sub_status = {led.status(s.obligation_id) for s in subs}
    if gold[i] == "REFUTES":
        right_subject[" / ".join(sorted(str(x) for x in sub_status)) or "no subject obligation"] += 1
        continue
    gold_docs = set(claims[i].evidence)
    on_gold = any(docs(led, o) & gold_docs for o in contra)
    if not subs:
        kind = "no subject obligation"
    elif CONTRADICTED in sub_status:
        kind = "subject itself contradicted (spill-over)"
    elif not any(docs(led, o) & docs(led, s) for o in contra for s in subs):
        kind = "contradicting abstract not cited for the subject (off-subject)"
    else:
        kind = "same abstract as a satisfied/partial subject (on-subject)"
    kinds[(gold[i], kind, "cites a gold abstract" if on_gold else "no gold abstract")] += 1
wrong = sum(kinds.values())
print(f"  M1 REFUTES: {sum(m1[i]['pred_label'] == 'REFUTES' for i in ids)}, wrong {wrong}")
for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
    print(f"    wrong, gold {k[0]}: {k[1]}; {k[2]}: {v}")
print(f"  correct REFUTES, subject obligation status: {dict(right_subject)}")
print(f"  REFUTES by number of contradicted obligations (correct?, n): {dict(sorted(n_contra.items()))}")
print(f"  contradicted obligations per claim: {sum(per_claim) / len(per_claim):.2f} "
      f"(claims with any: {sum(x > 0 for x in per_claim)})")
