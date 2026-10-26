# Clinical Ethics, Patient Safety & Algorithmic Accountability Statement

> **Course:** DSC306 Data Mining — COMSATS University Islamabad  
> **Project:** Project 11 — Patient Record Deduplication Tool (National Patient Master Index)  
> **Authorship:** *Team-Authored Document (Members A, B, C) — To be reviewed, adapted, and approved by the team.*  
> **Status:** Academic Clinical Governance & Ethical Safeguards Specification (Roadmap Section 7.12)

---

## 1. Executive Summary & Purpose

Patient master index deduplication is not merely an algorithmic pattern-recognition task; it is a **high-stakes clinical safety intervention**. In hospital health systems, inaccurate patient identity management directly degrades clinical decision-making, patient safety, and medical confidentiality.

This ethics statement documents:
1. **Who could be harmed** if the system produces false positives or false negatives.
2. The psychological risks of **automation bias** among hospital clerical staff.
3. The **concrete architectural and interface safeguards** engineered into the application to mitigate these hazards.

---

## 2. Who Could Be Harmed by Misclassification?

```
                                  CLINICAL RISKS
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
   FALSE POSITIVE (False Merge)                            FALSE NEGATIVE (Missed Merge)
   ┌────────────────────────────────┐                     ┌────────────────────────────────┐
   │ • Blood transfusion reaction   │                     │ • Redundant diagnostic CTs     │
   │ • Mismatched drug allergies    │                     │ • Missed chronic conditions    │
   │ • Privacy / data confidentiality│                    │ • Delayed clinical care        │
   │ • Misattributed legal liability│                     │ • Fragmented medical history   │
   └────────────────────────────────┘                     └────────────────────────────────┘
```

### 2.1 The Hazard of False Merges (Over-Matching / False Positives)
Merging two distinct individuals into a single consolidated chart represents the most catastrophic failure mode in clinical informatics:
- **Fatal Transfusion Incompatibilities:** If Patient A (Blood Group O-) and Patient B (Blood Group AB+) are erroneously merged, an emergency trauma team consulting the consolidated chart may infuse incompatible blood products, resulting in acute intravascular hemolytic reaction and death.
- **Contaminated Medication & Allergy Histories:** Documented anaphylactic allergies (e.g. penicillin, sulfonamides) belonging to one patient may be erased, or an allergy belonging to another person may cause clinicians to withhold first-line life-saving antibiotics.
- **Privacy Violations & Stigma (HIPAA / Data Protection):** Blending medical charts exposes highly confidential health information (psychiatric diagnoses, reproductive history, infectious disease statuses, substance use history) to unauthorized parties or family members sharing the chart.
- **Legal and Financial Misattribution:** Insurance claims, billing liabilities, and legal documentation become inextricably tangled across two citizens.

### 2.2 The Hazard of Missed Duplicates (Under-Matching / False Negatives)
Failing to link existing patient records leads to fragmented care across hospital departments:
- **Redundant Radiation and Invasiveness:** When past encounters are unlinked, patients are subjected to unnecessary repeat ionizing radiation (duplicate CT scans) or repeated invasive diagnostic biopsies.
- **Missed Clinical Baselines:** An emergency physician assessing a diabetic patient may be unaware of prior cardiac catheterizations, baseline renal dysfunction (serum creatinine), or surgical implants documented in an unlinked departmental chart.
- **Therapeutic Non-Adherence:** Disjointed records obscure prescription histories, increasing the risk of adverse drug-drug interactions or polypharmacy.

### 2.3 The Risk of Clerical Automation Bias
Records clerks working in understaffed hospital registration desks face immense time pressure (processing dozens of emergency admissions per hour). Clerks may exhibit **automation bias**—uncritically rubber-stamping high-probability algorithmic suggestions without independently verifying physical identity documents or questioning anomalous demographic combinations.

---

## 3. Engineered Interface & Architectural Safeguards

To prevent these clinical harms, the application enforces seven layers of defense:

| Defense Layer | Technical Implementation | Ethical Mitigation |
|---|---|---|
| **1. Strict Human-in-the-Loop** | **NEVER Auto-Merge.** System only generates candidate recommendations. Screen 6 requires explicit clerk decision with a two-step confirmation dialog. | Eliminates silent, automated catastrophic chart mergers. |
| **2. Plain-English Reasons** | Every prediction surfaces plain-English evidence **FOR** and **AGAINST** duplicate status (e.g. *"Same date of birth"*, *"Phone numbers differ"*). Zero raw model jargon (no node IDs, Gini, or scientific floats). | Combats automation bias by forcing clerks to weigh concrete clinical evidence. |
| **3. Blood Group Red Alert** | Automated clinical hazard detector in [`pages/6_Merge_Review.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/6_Merge_Review.py) fires an unmissable red alert if recorded blood groups conflict ($A+$ vs $B-$). | Prevents fatal transfusion incompatibility errors before merge execution. |
| **4. National ID Red Alert** | Automated identity collision detector fires an unmissable red alert if both charts contain different valid 13-digit Pakistani CNICs. | Prevents collapsing distinct citizens into a single identity. |
| **5. Exact Reversible Undo** | SQLite Merge Store ([`src/merge_store.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/merge_store.py)) stores full pre-merge JSON snapshots. Single-click **Undo** restores both original records **byte-for-byte**. | Ensures that any erroneous merge can be immediately and completely reversed. |
| **6. Audit Trail & Attribution** | Every merge, reject, and deferral logs timestamp, clerk ID/note, model name, and probability in SQLite (`data/merges.db`). | Ensures accountability, traceability, and defensibility during hospital audits. |
| **7. Universal Academic Banner** | Mandatory banner rendered at the top of every screen:<br>`⚠️ Student project. Not fit for clinical, financial or legal use. All data is synthetic.` | Prevents accidental operational deployment in real healthcare facilities. |

---

## 4. Data Provenance & Anonymization

1. **100% Synthetic Data Generation:** All 50,000 records in the patient register were synthetically generated using rule-based stochastic processes anchored to Pakistani demographic distributions (PBS census tables, NADRA CNIC formats, canonical city names).
2. **Zero Protected Health Information (PHI):** No real patient charts, medical records, or genuine citizen identities were accessed or utilized at any stage of this project.
3. **Reproducible Generation Mechanism:** All generator parameters, noise injection rates, and corruption probabilities are documented in [`docs/generation_mechanism.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/generation_mechanism.md) and tied to `CFG.RANDOM_SEED = 42`.

---

## 5. Team Governance & Review Instructions

> **Note for Team Members A, B, and C:**  
> This ethics statement is an open academic governance document. Prior to final viva submission, the team must hold a joint review to:
> - Review hospital legal standards applicable to provincial health departments in Pakistan (e.g. Punjab Healthcare Commission standards).
> - Confirm that all team members can articulate and defend the trade-off between False Alarm Cost ($C_{FP}$) and Missed Duplicate Cost ($C_{FN}$) during oral examination.
