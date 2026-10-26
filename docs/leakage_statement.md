# Data Leakage Prevention Statement

> **Course:** DSC306 Data Mining — COMSATS University Islamabad  
> **Project:** Project 11 — Patient Record Deduplication Tool  
> **Specification:** Section 7.2 of `PROJECT_ROADMAP.md` & Phase 2 Gate Requirements  
> **Scope:** Formal boundary definition of observable versus unobservable information at inference time.

---

## 1. The Core Mining Task & The Records Clerk's Information Horizon

The objective of this data mining project is to assist a hospital records clerk by presenting a ranked worklist of candidate duplicate pairs $(r_i, r_j)$. For each pair, the machine learning model must estimate the probability that both records belong to the same true patient:

$$P(\text{duplicate} = 1 \mid \mathbf{x}_{ij})$$

where $\mathbf{x}_{ij}$ is a vector of pairwise similarity, distance, and agreement features computed strictly from the observed attributes of record $i$ and record $j$.

### The Fundamental Rule of Honest Mining
> [!IMPORTANT]
> **The Information Horizon Rule:**  
> A feature or transformation is **permissible if and only if** its information would be physically observable to a records clerk examining two unmerged hospital charts in real time. Any information that requires knowledge of the future, knowledge of internal generator state, knowledge of true entity clusters, or knowledge of downstream human actions constitutes **Data Leakage** and is strictly disqualified.

---

## 2. Inventory of Excluded & Prohibited Information

The following items are completely excluded from candidate blocking, feature extraction, and model training:

| # | Excluded Fact / Object | Source | Why It Constitutes Leakage | Mechanism of Enforcement |
|---|---|---|---|---|
| **1** | `true_entity_id` | `data/truth/ground_truth.csv` | **Omniscient Oracle:** Contains the true cluster identity of every record. Joining this column to candidate pairs yields a trivial $100\%$ accuracy model that learns nothing. | Physically isolated in `data/truth/`; never imported into `src/clean.py`, `src/blocking.py`, or `src/features.py`. |
| **2** | Generator Fault Log | `data/truth/fault_log.csv` | **Synthetic Generation Metadata:** Records exactly which fields were corrupted (e.g. typos, sentinels, transposed digits). Real hospital systems provide no error annotations. | Segregated in `data/truth/`; used solely in Phase 3 audit verification to assess detector recall/precision. |
| **3** | Post-Merge Identifiers | Application state (`merged_into`, `canonical_id`) | **Outcome Leakage:** A merged identifier or canonical cluster ID is the *consequence* of a clerk's decision, not an antecedent feature. | Exists solely in the application's runtime SQLite/JSON merge store (`data/merge_store.json`), never in training matrices. |
| **4** | Human Clerk Decisions | UI Action Log (`Merge`, `Reject`, `Defer`) | **Feedback Leakage:** Future manual review outcomes cannot be used to predict current or past candidate pairs. | Recorded strictly in audit trail tables for reversible undo operations. |
| **5** | Cluster Cardinality | Entity Graph ($k$ records per entity) | **Frequency Leakage:** Knowing whether a patient has 1, 2, 3, or 4 records in the database reveals duplicate status before comparison occurs. | Prohibited; features are strictly pairwise $(r_i, r_j)$ without global frequency lookups by true entity. |
| **6** | Record ID Ordering | `record_id` integer sequence | **Index Correlation:** If record IDs were generated sequentially during duplication (e.g. clone immediately following original), row index would predict duplicates. | **Explicit Shuffle:** In `src/generate_data.py`, `record_id` values were randomly permuted across the entire register. Tests in `tests/test_generate_data.py` assert zero rank correlation ($r < 0.01$). |
| **7** | Synthetic Generation Artefacts | Trailing whitespace, synthetic noise patterns | **Shortcut Learning:** If synthetic duplicate copies carry systematic syntax differences not present in clean records (e.g., trailing spaces on typos), the classifier exploits the artefact. | Thorough cleaning in `src/clean.py` strips all whitespace, standardizes case, and harmonizes encodings across all records before feature extraction. |
| **8** | Future / Time-Travel Data | Future registrations ($t > t_{\text{query}}$) | **Temporal Leakage:** In prospective deployment, a clerk evaluating a record in 2021 cannot know what records the patient might generate in 2024. | **Strict Temporal Split:** Records are split by earliest registration date. Candidate pairs are formed strictly within train (early) or test (late) partitions. |

---

## 3. Why Pair-Level Target Labels Cannot Be Computed from `register_raw.csv` Alone

A fundamental concept in data mining and relational database theory is that:
$$\text{Single-Table Register} \not\equiv \text{Pairwise Training Relation}$$

The raw hospital register `data/raw/register_raw.csv` is an **unlinked flat table** of 54,373 individual records. Each row represents a single administrative encounter or profile with no pointers, foreign keys, or cluster groupings linking it to any other row.

### Mathematical Proof of Under-Specification:
Let the raw dataset be $R = \{r_1, r_2, \dots, r_n\}$. The space of all possible unordered candidate pairs is:
$$\mathcal{P} = \left\{ (r_i, r_j) \mid 1 \le i < j \le n \right\}, \quad |\mathcal{P}| = \frac{n(n-1)}{2} \approx 1.478 \times 10^9$$

To train a supervised classifier, each pair must have an assigned binary ground-truth label:
$$y_{ij} \in \{0, 1\}, \quad \text{where } y_{ij} = \mathbb{I}(\text{Entity}(r_i) = \text{Entity}(r_j))$$

1. **No Target Column in $R$:** There is no column in `register_raw.csv` containing $y_{ij}$ or `Entity(r)`.
2. **Deterministic Rules Fail:** As demonstrated in Phase 1 sanity testing, a deterministic rule such as:
   $$y_{ij} = \mathbb{I}(\text{DOB}_i = \text{DOB}_j \land \text{Phone}_i = \text{Phone}_j)$$
   achieves an unacceptably low recall of **15.72%** (finding only 1,533 of 9,754 true duplicate pairs) while also returning false positives from shared household phones and twins.
3. **The Necessity of an External Oracle:** Without an external ground truth table (`data/truth/ground_truth.csv`) generated during simulation or produced by expert clinical chart review, pair-level target labels **do not exist within the raw data**.
4. **Leakage Implication:** Any pipeline that attempts to derive $y_{ij}$ by "clustering raw names" and then evaluates models on those self-generated labels commits circular validation (evaluating the model against its own assumptions). We enforce strict separation: `ground_truth.csv` provides labels for training and testing, but is never accessible during feature computation.

---

## 4. Deep-Dive on `source_system`: Legitimate Signal vs. Sampling Leakage

The attribute `source_system` records the hospital subsystem from which each record originated:
$$\text{source\_system} \in \{\text{OPD}, \text{Lab}, \text{Radiology}, \text{Pharmacy}, \text{Emergency}, \text{Legacy}\}$$

Because duplicate patient records in hospitals are heavily driven by departmental silos, `source_system` is a critical attribute that requires nuanced handling.

```
       ┌──────────────┐         Cross-Departmental Transfer         ┌──────────────┐
       │     OPD      │ ──────────────────────────────────────────> │  Emergency   │
       │ (REC0012040) │                                             │ (REC0048119) │
       └──────────────┘                                             └──────────────┘
              │                                                            │
              ▼                                                            ▼
   ┌──────────────────────┐                                     ┌──────────────────────┐
   │ John Khan, DOB 1985  │                                     │ J. Khan, DOB 1985    │
   └──────────────────────┘                                     └──────────────────────┘
              ▲                                                            ▲
              └─────────────────── SAME PERSON ? ──────────────────────────┘
```

### 4.1 Why `source_system` is a Legitimate Real-World Feature
In clinical informatics, patients rarely register twice within the same department on the same day. Duplicate records almost exclusively arise when a patient registered in **OPD** is later admitted through **Emergency**, or receives bloodwork in **Lab**, or underwent imaging in **Radiology**, or migrated from an old **Legacy** database.

Therefore, the pairwise transition feature:
$$\Delta_{\text{source}} = (\text{source}_i, \text{source}_j)$$
or an indicator of cross-system presence:
$$\mathbb{I}(\text{source}_i \neq \text{source}_j)$$
reflects authentic hospital workflow dynamics. A model that knows a pair crosses between OPD and Emergency has legitimate prior reason to expect differing levels of data completeness (e.g. missing father's name or missing phone in Emergency).

---

### 4.2 The Leakage Risk: Sampling Distribution Mismatch
The danger arises not from the feature itself, but from **how negative candidate pairs are sampled** during training data preparation:

1. **The Synthetic Duplication Pattern:** When duplicate records are injected, clones are assigned to different hospital departments (e.g., OPD $\to$ Lab, or Legacy $\to$ Emergency). Consequently, a high percentage (e.g. $70-85\%$) of true duplicate pairs cross source systems.
2. **The Naive Negative Sampling Flaw:** If negative pairs are sampled naively by pairing random records within the same department or block without controlling for department distribution, the negative pairs will exhibit a drastically lower cross-system rate.
3. **The Spurious Shortcut:** Under this distribution mismatch, a decision tree or neural network will discover that the rule:
   $$\text{IF } \text{source}_i \neq \text{source}_j \implies \text{PREDICT DUPLICATE}$$
   yields high apparent accuracy without inspecting patient names, dates of birth, or addresses.
4. **Failure at Deployment:** In production, when a clerk compares two different patients who happen to have attended different departments, the model will falsely classify them as the same person.

---

### 4.3 Mitigation Protocol & Planned Experimental Validation

To guarantee absolute methodological honesty, we implement a strict three-step protocol:

#### Protocol 1: Balanced Cross-System Negative Sampling
During candidate pair generation (Phase 6), negative pairs (non-duplicates) must be drawn such that their cross-system marginal distribution matches the candidate population:
$$P(\text{source}_i = S_a, \text{source}_j = S_b \mid y=0) \approx P(\text{source}_i = S_a, \text{source}_j = S_b \mid \text{Blocked Candidates})$$
This prevents the classifier from exploiting source system discordance as an artificial shortcut.

#### Protocol 2: Ablation Study (With vs. Without `source_system`)
In Phase 8 and Phase 9 (Model Training and Evaluation), every classifier family will be trained and evaluated under two distinct configurations:
1. **Model A (Full Feature Set):** Includes demographic similarity features PLUS pairwise source system transition encodings.
2. **Model B (Demographic-Only Feature Set):** Excludes `source_system` entirely, relying strictly on names, DOB, age, CNIC, phone, city, address, and blood group.

We will record both models in `reports/metrics.json` and publish a comparative ROC-AUC and F1-score table in `reports/evaluation_report.md`. If Model A outperforms Model B by an unrealistic margin while exhibiting brittle transfer on same-system duplicates, `source_system` will be demoted to an explanatory display badge in the UI rather than an active model feature.

#### Protocol 3: Automated Invariant Unit Tests
A dedicated unit test in `tests/test_pairs.py` will assert that feature extraction produces zero NaN leakage, that no ground-truth columns are present in the feature matrix $\mathbf{X}$, and that permutation of `record_id` columns yields strictly identical similarity vectors.

---

## 5. Traceability & Sign-Off

This leakage statement guarantees adherence to:
- Section A of the Common Project Specification (DSC306).
- Section 7.2 of `PROJECT_ROADMAP.md`.
- Individual student viva defensibility: every team member can articulate why each excluded field constitutes leakage and how the pipeline prevents it.
