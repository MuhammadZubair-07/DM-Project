# Clerk-Facing Explanation Guide & Phrase Dictionary

**National Patient Master Index (NPMI) Record Linkage System**  
**Document Version:** 1.0.0  
**Phase:** 10 (Roadmap Section 7.10, Lecture 3 & 4)  
**Target Audience:** Medical Records Clerks, Clinical Registrars, Hospital Informatics Quality Officers

---

## 1. Executive Summary & Clinical Philosophy

When medical records clerks review potential duplicate patient charts, they operate under strict time constraints and carry high clinical liability. A false positive duplicate merge risks contaminating patient drug allergy profiles, surgical history, and active prescriptions. Conversely, a false negative split leaves vital medical histories fragmented across separate files.

Standard machine learning outputs—such as numerical confidence scores, tree split thresholds (`dob_diff_days_log <= 5.8527`), internal node IDs (`Node 3`), impurity measures (`gini = 0.32`), or mathematical inequality operators (`<=`, `>=`)—are completely unusable and counterproductive in a hospital records department.

### The Plain-English Mandate
The NPMI Explanation Engine (`src/explain.py`) translates complex pairwise machine learning models (Pruned Decision Tree, Categorical Naive Bayes, $k$-Nearest Neighbours, and Multi-Layer Perceptron) into **plain-English, actionable clinical explanations**.

**Guiding Design Rules:**
1. **Zero Technical Jargon:** Never expose feature names, node numbers, Gini impurities, inequality symbols, or multi-decimal floats to records clerks.
2. **Threshold-to-Word Translation:** Convert numerical comparison margins into natural language qualifiers (*"nearly identical"*, *"very close"*, *"moderately similar"*, *"differing"*).
3. **Balanced Reasons:** Every review presentation must supply both reasons **FOR** a match and reasons **AGAINST** a match, preventing confirmation bias.
4. **Graceful Missingness Handling:** Explicitly explain unrecorded attributes from Emergency intake or Legacy paper systems without flagging false contradictions.
5. **Cost-Derived Confidence Bands:** Confidence labels are tied directly to Bayesian cost-optimal decision boundaries, not arbitrary decimal heuristics.
6. **Merge Screen Support:** Every explanation provides per-field `agree`, `conflict`, or `unknown` flags to drive the Screen 6 survivorship merge screen.

---

## 2. Explanation Architecture & Structured Interface

Every call to `explain(pair, model)` returns a standardized dictionary structured as follows:

```python
{
    "verdict": "Duplicate Match" | "Potential Duplicate" | "Distinct Patients",
    "probability": float,          # Calibrated posterior probability in [0.0, 1.0]
    "confidence_label": str,       # Cost-derived tier: 'Very likely duplicate', 'Worth reviewing', 'Unlikely duplicate'
    "reasons_for": list[str],      # Plain-English sentences supporting duplicate identity
    "reasons_against": list[str],  # Plain-English sentences opposing duplicate identity
    "field_comparisons": dict,     # Per-field agree/conflict/unknown status tags for Screen 6
    "clerk_summary": str,          # One-sentence overall synthesis
    "model_name": str,             # Display name of the active classifier
    "model_family": str,           # Identifier: decision_tree, naive_bayes, knn, mlp
    "operating_threshold": float,  # Cost-optimal review boundary (theta_1* = 0.20)
    "strict_threshold": float      # Cost-optimal auto-merge boundary (theta_2* = 0.8333)
}
```

### Cost-Derived Confidence Tiers
In alignment with Decision D6 and the Evaluation Report, confidence bands are derived from the clinical misclassification cost matrix:

| Posterior Probability ($p$) | Confidence Label | Action Recommendation | Clinical Workflow |
|:---|:---|:---|:---|
| **$p \ge 0.8333$** ($\theta_2^*$) | **Very likely duplicate** | High-Confidence Merge | Recommended for automated merge or rapid 1-click confirmation by clerk. |
| **$0.20 \le p < 0.8333$** ($\theta_1^* \dots \theta_2^*$) | **Worth reviewing** | Manual Clerk Inspection | Enters the Clerk Review Worklist for side-by-side demographic comparison. |
| **$p < 0.20$** ($< \theta_1^*$) | **Unlikely duplicate** | Reject Duplicate | Flagged as distinct patients; dismissed from duplicate queues without clerk intervention. |

---

## 3. Feature-to-Phrase Dictionary

The explanation engine utilizes a curated phrase catalogue mapping demographic comparison features into clerk-facing statements:

| Internal Feature Code | Evaluation Context | Plain-English Explanation (Clerk-Facing) |
|:---|:---|:---|
| `gower_similarity` | $\ge 0.95$ | *"Overall demographic profile across recorded fields is nearly identical."* |
| `gower_similarity` | $0.82 \dots 0.95$ | *"Overall demographic agreement across recorded patient attributes is very high."* |
| `gower_similarity` | $< 0.82$ | *"Overall demographic agreement across recorded fields is below the threshold for duplicates."* |
| `dob_diff_days_log` | Days diff $= 0$ | *"Same date of birth (YYYY-MM-DD)."* |
| `dob_diff_days_log` | Days diff $\le 347$ | *"Dates of birth are very close (possible clerical year or keystroke error)."* |
| `dob_diff_days_log` | Days diff $> 347$ | *"Dates of birth differ significantly across calendar years."* |
| `dob_diff_days_log` | Sentinel value $-1.0$ | *"Date of birth is missing on one record so it could not be compared."* |
| `nid_hamming_le_1` | Exact or $\le 1$ digit diff | *"National identity card numbers match or differ by at most one single digit typo."* |
| `nid_hamming_le_1` | Differ $> 1$ digit | *"National identity card numbers do not match."* |
| `nid_hamming_le_1` | Sentinel value $-1.0$ | *"National ID is missing on one record so it could not be compared."* |
| `phone_last7_equal` | Last 7 digits match | *"Mobile phone numbers match on the final 7 subscriber digits."* |
| `phone_edit_distance`| Differ substantially | *"Phone numbers differ."* |
| `phone_edit_distance`| Sentinel value $-1.0$ | *"Phone number is missing on one record so it could not be compared."* |
| `father_exact` | Match $= 1$ | *"Father names are identical."* |
| `father_exact` | Differ $= 0$ | *"Father names differ."* |
| `father_exact` | Sentinel value $-1.0$ | *"Father name is unrecorded on one record so it could not be compared."* |
| `city_equal` | Match $= 1$ | *"Both registrations originate from the same city."* |
| `city_equal` | Differ $= 0$ | *"Patients are registered in different cities."* |
| `address_token_jaccard` | Word overlap $> 0.45$| *"Residential addresses share multiple matching street or locality words."* |
| `address_token_jaccard` | Word overlap $\le 0.45$| *"Residential addresses share few or no common locality words."* |
| `blood_group_equal` | Match $= 1$ | *"Recorded blood groups match."* |
| `blood_group_equal` | Differ $= 0$ | *"Recorded blood groups conflict."* |
| `visit_count_diff_log` | Comparable volume | *"Hospital visit frequencies are comparable in volume."* |
| `visit_count_diff_log` | Discrepant volume | *"Hospital encounter visit counts differ substantially."* |

---

## 4. Model-Specific Reasoning Engines

The explanation system provides custom explanation logic tailored to the mechanics of each model family:

```
                                  [ Pair of Records ]
                                           │
                                ┌──────────┴──────────┐
                                │   Model Family?     │
                                └──────────┬──────────┘
             ┌─────────────────┬───────────┴───────────┬─────────────────┐
             ▼                 ▼                       ▼                 ▼
     [ Decision Tree ]  [ Naive Bayes ]             [ k-NN ]           [ MLP ]
             │                 │                       │                 │
    Decision Path     Log-Likelihood Ratio    Historical Precedent  Joint Feature
    Child vs Parent     LLR_j = log(p1/p0)      k-Nearest Pairs      Activation
    Probability Shift   Top Pos / Neg Ranks     "4 of 5 were same"    Synthesis
```

### 1. Pruned Decision Tree Explainer
- **Mechanics:** Traces the exact path from root to leaf via `clf.tree_.decision_path`.
- **Probability Direction:** For each internal split, the engine compares the duplicate class probability of the current node ($P_{\text{parent}}$) against the next child ($P_{\text{child}}$).
  - If $P_{\text{child}} \ge P_{\text{parent}}$: The test condition increased duplicate likelihood $\implies$ Appended to `reasons_for`.
  - If $P_{\text{child}} < P_{\text{parent}}$: The test condition reduced duplicate likelihood $\implies$ Appended to `reasons_against`.
- **Missing Value Sentinels:** Splits testing $\le -0.5$ indicate missing demographic data (e.g. trauma patient lacking CNIC). Reaching this branch gracefully explains that the attribute was unrecorded rather than reporting a false conflict.

### 2. Categorical Naive Bayes Explainer
- **Mechanics:** Computes the exact Log-Likelihood Ratio (LLR) contribution for every feature $j$ given discretized category bin $c$:
  $$\text{LLR}_j = \ln P(X_j = c \mid \text{Duplicate}) - \ln P(X_j = c \mid \text{Distinct})$$
- **Evidence Balance:**
  - Features with $\text{LLR}_j > +0.20$ provide statistical evidence in favor of duplicate identity $\implies$ Top 4 ranked in `reasons_for`.
  - Features with $\text{LLR}_j < -0.20$ provide statistical evidence against duplicate identity $\implies$ Top 4 ranked in `reasons_against`.
- **Clerk Summary:** Generates an evidence ledger (e.g. *"Naive Bayes evidence balance: 4 primary factors favor duplicate status, while 1 attribute argues against."*).

### 3. $k$-Nearest Neighbours ($k$-NN) Explainer
- **Mechanics:** Transforms the pair into the standardized 15-dimensional metric space and queries the 5 closest historical training pairs via `clf.kneighbors`.
- **Historical Consensus Phrasing:** Directly reports the proportion of nearest reference pairs that were verified duplicates:
  - If $\ge 3$ of 5 were duplicates: *"Most similar past pairs: 4 of 5 were the same person."*
  - If $< 3$ of 5 were duplicates: *"Most similar past pairs: 4 of 5 were distinct persons."*
- **Demographic Corroboration:** Accompanied by direct comparison of recorded demographic fields (DOB, first name, phone, city) to clarify *why* the pair lies in that geometric neighborhood.

### 4. Multi-Layer Perceptron (MLP) Explainer
- **Mechanics:** Evaluates joint feature alignments across Gower similarity, dates of birth, phone digits, and locality words, presenting a high-level clinical summary of nonlinear neural activation agreement.

---

## 5. Worked Clinical Case Studies

### Case Study 1: High-Confidence True Duplicate with Name Typo
- **Patient Chart 1:** `Muhammad Khan`, DOB: `1985-06-15`, CNIC: `42101-1234567-1`, Phone: `03009876543`, Karachi.
- **Patient Chart 2:** `Mohammad Khan`, DOB: `1985-06-15`, CNIC: `42101-1234567-1`, Phone: `03009876543`, Karachi.

```json
{
  "verdict": "Duplicate Match",
  "probability": 1.0,
  "confidence_label": "Very likely duplicate",
  "reasons_for": [
    "Overall demographic profile across recorded fields is nearly identical.",
    "Same date of birth (1985-06-15).",
    "First names are nearly identical (Muhammad / Mohammad).",
    "Family names are identical (Khan).",
    "National identity cards (CNIC) are identical.",
    "Phone numbers are identical."
  ],
  "reasons_against": [
    "No conflicting demographic attributes detected between records."
  ],
  "clerk_summary": "Decision tree identified 6 primary factors supporting duplicate status and 0 conflicting attributes."
}
```

---

### Case Study 2: Emergency Room Intake with Missing National ID
- **Patient Chart 1 (ER Trauma):** `Fatima Bibi`, DOB: `1980-01-01`, CNIC: *Unrecorded (`""`)*, Phone: `03339876543`, Lahore.
- **Patient Chart 2 (OPD):** `Fatima Bibi`, DOB: `1980-01-01`, CNIC: `35201-9876543-2`, Phone: `03339876543`, Lahore.

```json
{
  "verdict": "Duplicate Match",
  "probability": 1.0,
  "confidence_label": "Very likely duplicate",
  "reasons_for": [
    "Overall demographic profile across recorded fields is nearly identical.",
    "Same date of birth (1980-01-01).",
    "First names are identical (Fatima).",
    "Family names are identical (Bibi).",
    "Phone numbers are identical."
  ],
  "reasons_against": [
    "National ID is missing on one record so it could not be compared."
  ],
  "clerk_summary": "Decision tree identified 5 primary factors supporting duplicate status and 1 conflicting attributes."
}
```
*Clinical Utility:* The clerk immediately understands that the absence of CNIC was due to unrecorded intake data rather than an identity discrepancy.

---

### Case Study 3: Hard Negative Sibling Pair
- **Patient Chart 1:** `Ali Khan`, Father: `Tariq Khan`, DOB: `1990-03-10`, Phone: `03001234567`, Karachi.
- **Patient Chart 2:** `Ahmed Khan`, Father: `Tariq Khan`, DOB: `1995-11-20`, Phone: `03001234567`, Karachi.

```json
{
  "verdict": "Distinct Patients",
  "probability": 0.0,
  "confidence_label": "Unlikely duplicate",
  "reasons_for": [
    "Overall demographic agreement across recorded patient attributes is very high.",
    "Family names are identical (Khan).",
    "Phone numbers are identical."
  ],
  "reasons_against": [
    "Dates of birth differ significantly across calendar years (1990-03-10 vs 1995-11-20).",
    "First names are completely different (Ali vs Ahmed)."
  ],
  "clerk_summary": "Decision tree identified 3 primary factors supporting duplicate status and 2 conflicting attributes."
}
```
*Clinical Utility:* Prevents disastrous sibling merges caused by shared family phones and addresses by clearly presenting the differing birth years and distinct first names.

---

### Case Study 4: $k$-NN Explanation with Nearest Historical Pairs
- **Patient Pair:** Two emergency encounters with matching phone and address, minor first name variation.

```json
{
  "verdict": "Duplicate Match",
  "probability": 1.0,
  "confidence_label": "Very likely duplicate",
  "reasons_for": [
    "Most similar past pairs: 5 of 5 were the same person.",
    "First names are nearly identical (Muhammad / Mohammad).",
    "Same date of birth (1985-06-15).",
    "Both patients registered in Karachi."
  ],
  "reasons_against": [
    "No conflicting demographic attributes detected between records."
  ],
  "clerk_summary": "Nearest-neighbour historical consensus: 5 of 5 closest comparisons in the reference register were confirmed duplicates."
}
```

---

## 6. Screen 6 Merge Screen Integration

When a records clerk opens the merge review screen (Screen 6 in Streamlit UI), `explain()` provides the structured `field_comparisons` dictionary across all 13 core demographic fields.

### Status Tag Meanings:
- **`agree` (Green Card):** Attributes match exactly or fall within accepted clerical typo tolerances (e.g. Jaro-Winkler $\ge 0.85$, final 7 digits of phone match, or CNIC check-digit typo).
- **`conflict` (Red Card):** Attributes contradict each other (e.g. conflicting birth years, different residential cities, conflicting blood groups).
- **`unknown` (Amber Card):** One or both patient charts lack recorded data for this attribute.

### Field Comparison Structure:
```python
{
    "first_name":     {"status": "agree",   "value_1": "Fatima",         "value_2": "Fatima",         "detail": "Exact match"},
    "family_name":    {"status": "agree",   "value_1": "Bibi",           "value_2": "Bibi",           "detail": "Exact match"},
    "father_name":    {"status": "agree",   "value_1": "Ghulam Nabi",    "value_2": "Ghulam Nabi",    "detail": "Exact match"},
    "gender":         {"status": "agree",   "value_1": "F",              "value_2": "F",              "detail": "Exact match"},
    "date_of_birth":  {"status": "agree",   "value_1": "1980-01-01",     "value_2": "1980-01-01",     "detail": "Exact match"},
    "age_years":      {"status": "agree",   "value_1": "46",             "value_2": "46",             "detail": "Exact match"},
    "national_id":    {"status": "unknown", "value_1": "-",              "value_2": "35201-9876543-2", "detail": "Unrecorded on one or both records"},
    "phone":          {"status": "agree",   "value_1": "03339876543",    "value_2": "03339876543",    "detail": "Exact match"},
    "city":           {"status": "agree",   "value_1": "Lahore",         "value_2": "Lahore",         "detail": "Exact match"},
    "address":        {"status": "agree",   "value_1": "Block C Model",  "value_2": "Block C Model",  "detail": "Exact match"},
    "blood_group":    {"status": "agree",   "value_1": "O+",             "value_2": "O+",             "detail": "Exact match"},
    "insurance_tier": {"status": "agree",   "value_1": "Silver",         "value_2": "Silver",         "detail": "Exact match"},
    "visit_count":    {"status": "agree",   "value_1": "5",              "value_2": "5",              "detail": "Exact match"}
}
```

The UI uses these status tags to render side-by-side demographic comparisons, allowing the clerk to select the golden survivorship value with a single radio-button click.
