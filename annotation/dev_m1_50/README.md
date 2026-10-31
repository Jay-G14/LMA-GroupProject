# Obligation-status annotation (dev_m1_50)

Source run: `dev_m1` (m1, dev); 50 claims, 165 obligations; sample seed 13.

Two annotators work **independently** and do not open `key.json` (the model's answers) until scoring.

1. Open `packet.md`. For each claim, read the abstracts listed under it.
2. In `sheet.csv`, fill your column (`annotator_A_status` or `annotator_B_status`) for every obligation with
   its true status, judged **only on the abstracts that obligation was audited on** (column `audited_on`):

   Statuses:
   - SATISFIED: a cited sentence establishes the condition.
   - CONTRADICTED: a cited sentence states the opposite of the condition for the same subject (for example a decrease where the condition requires an increase). A different finding, a different topic or missing information is not a contradiction.
   - PARTIAL: the evidence addresses the condition but is incomplete or not specific enough.
   - UNSUPPORTED: the abstracts do not address the condition, or address something else.

3. In `claims.csv`, fill `annotator_A_covers` / `annotator_B_covers` with `yes` if the obligations together
   cover every condition the claim states, else `no`.
4. Afterwards, discuss disagreements and fill `adjudicated_status` / `adjudicated_covers` where you differed.
5. Run `python scripts/score_annotation.py --name dev_m1_50`.
