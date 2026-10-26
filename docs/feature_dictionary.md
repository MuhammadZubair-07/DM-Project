# Feature Dictionary: Pairwise Comparison Vector

> **Course:** DSC306 Data Mining — COMSATS University Islamabad  
> **Project:** Project 11 — Patient Record Deduplication Tool  
> **Specification:** Section 7.6 of `PROJECT_ROADMAP.md` & Phase 6 Core Deliverable  
> **Scope:** Definitive reference for all 58 pairwise comparison features, measurement types, mathematical definitions, missing-value treatments, and clerk-facing explanation templates.

---

## 1. Overview & Conceptual Architecture

In patient record deduplication, machine learning models do not operate on single flat records. Instead, they classify **candidate pairs of records** $(r_i, r_j)$ into:
- **Match ($y = 1$):** Both records belong to the exact same human entity.
- **Non-Match ($y = 0$):** Records represent distinct individuals (including siblings, twins, parent-child, or unrelated namesakes).

The input to each classifier is a **comparison vector** $\mathbf{x}_{ij} \in \mathbb{R}^{58}$ where each component measures attribute agreement, typographical distance, phonetic similarity, or presence.

```
Record A: [ "Muhammad Khan",  DOB: 1985-05-12,  Phone: 03001234567,  CNIC: 3520112345671 ]
Record B: [ "Mohammad Khaan", DOB: 1985-12-05,  Phone: 03001234567,  CNIC: missing       ]
                                    │
                         PAIRWISE COMPARISON VECTOR
                                    ▼
[ fn_jaro_winkler: 0.933, dob_day_month_swapped: 1.0, phone_exact: 1.0, nid_both_present: 0.0, ... ]
                                    │
                          DOWNSTREAM CLASSIFIER
                                    ▼
P(Duplicate = 1) = 0.962 ──> "Same birthdate with swapped day/month, identical phone, spelling variant"
```

---

## 2. Missing Value Policy by Model Family

Missing demographic values are common in clinical data (e.g. emergency trauma encounters lacking national IDs or phones). Missingness is handled explicitly across three distinct model families:

| Model Family | Treatment Strategy | Justification (Lecture 4) |
|---|---|---|
| **Decision Trees & Ensembles** | **Sentinel Value (`0.0` or `-1.0`) + Explicit `both_present` Indicator** | Tree splitters naturally partition on binary presence flags (e.g., `phone_both_present <= 0.5`) or handle missingness as a distinct branch without numeric distortion. |
| **Naive Bayes (Categorical / Binned)** | **Dedicated "Missing" Bin (Bin index `-1`)** | Probabilistic models require discrete frequency counts. Assigning unobserved comparisons to a `"missing"` category allows Laplace smoothing to estimate $P(\text{missing} \mid \text{duplicate})$ versus $P(\text{missing} \mid \text{non-duplicate})$. |
| **$k$-Nearest Neighbours ($k$-NN)** | **Gower Metric with the Delta Rule ($\delta_{ijk}$)** | Geometric distance metrics distort nearest-neighbour neighbourhoods if missing values are imputed arbitrarily. The Gower delta rule dynamically skips unobserved attributes in the denominator. |

---

## 3. Comprehensive Feature Catalogue

The following table documents all 58 features in the comparison vector:

| # | Feature Name | Category | Scale | Range | Meaning in Plain Words | Clerk-Facing Explanation Phrase |
|---|---|---|---|---|---|---|
| 1 | `fn_jaro_winkler` | First Name | Ratio | $[0, 1]$ | Jaro-Winkler similarity giving weight to matching prefix characters. | "First names have strong prefix and character agreement." |
| 2 | `fn_levenshtein` | First Name | Ratio | $[0, 1]$ | Normalized Levenshtein edit similarity (1 - edit_dist / max_len). | "First names differ by very few insertion/deletion edits." |
| 3 | `fn_exact` | First Name | Binary | $\{0, 1\}$ | Exact string match between given names. | "First names are identical." |
| 4 | `fn_soundex_equal` | First Name | Binary | $\{0, 1\}$ | Soundex phonetic codes of given names are identical. | "First names sound identical phonetically." |
| 5 | `fn_initial_equal` | First Name | Binary | $\{0, 1\}$ | Initial first letters of given names match. | "First names share the same initial letter." |
| 6 | `fn_both_present` | First Name | Binary | $\{0, 1\}$ | Both records contain a non-empty first name. | "Both records record a first name." |
| 7 | `fam_jaro_winkler` | Family Name | Ratio | $[0, 1]$ | Jaro-Winkler similarity on family / clan name. | "Family names have strong character agreement." |
| 8 | `fam_levenshtein` | Family Name | Ratio | $[0, 1]$ | Normalized Levenshtein edit similarity on family name. | "Family names differ by minor typographical typos." |
| 9 | `fam_exact` | Family Name | Binary | $\{0, 1\}$ | Exact string match on family name. | "Family names are identical." |
| 10 | `fam_soundex_equal` | Family Name | Binary | $\{0, 1\}$ | American Soundex phonetic code match on family name. | "Family names sound identical phonetically (e.g. Khan/Khaan)." |
| 11 | `fam_initial_equal` | Family Name | Binary | $\{0, 1\}$ | Initial first letter of family name matches. | "Family names share the same starting initial." |
| 12 | `fam_both_present` | Family Name | Binary | $\{0, 1\}$ | Both records contain a non-empty family name. | "Both records record a family name." |
| 13 | `full_name_jaro_winkler` | Full Name | Ratio | $[0, 1]$ | Jaro-Winkler similarity across full compound name. | "Full compound names match closely across words." |
| 14 | `full_name_levenshtein` | Full Name | Ratio | $[0, 1]$ | Normalized edit similarity on full compound name. | "Full compound names have minimal character edits." |
| 15 | `full_name_ngram_jaccard` | Full Name | Ratio | $[0, 1]$ | Character 2-gram (bigram) Jaccard overlap on full name. | "Full names share a high proportion of two-letter segments." |
| 16 | `full_name_tfidf_cosine` | Full Name | Ratio | $[0, 1]$ | Character 3-gram TF-IDF cosine similarity on full name. | "Full names share significant three-letter subword frequencies." |
| 17 | `name_swap_flag` | Full Name | Binary | $\{0, 1\}$ | First and family names appear inverted / transposed. | "First and family names appear inverted between the two records." |
| 18 | `father_jaro_winkler` | Father Name | Ratio | $[0, 1]$ | Jaro-Winkler similarity on father / guardian name. | "Father names have strong character agreement." |
| 19 | `father_levenshtein` | Father Name | Ratio | $[0, 1]$ | Normalized edit similarity on father name. | "Father names differ by minor spelling variations." |
| 20 | `father_exact` | Father Name | Binary | $\{0, 1\}$ | Exact string match on father / guardian name. | "Father names are identical." |
| 21 | `father_both_present` | Father Name | Binary | $\{0, 1\}$ | Both records contain an observed father name. | "Both records contain a recorded father name." |
| 22 | `dob_exact` | Date of Birth | Binary | $\{0, 1\}$ | Exact match on complete birth date (YYYY-MM-DD). | "Dates of birth are exactly identical." |
| 23 | `dob_day_equal` | Date of Birth | Binary | $\{0, 1\}$ | Exact match on day of the month (DD). | "Birthdays fall on the same day of the month." |
| 24 | `dob_month_equal` | Date of Birth | Binary | $\{0, 1\}$ | Exact match on birth month (MM). | "Birthdays fall in the same calendar month." |
| 25 | `dob_year_equal` | Date of Birth | Binary | $\{0, 1\}$ | Exact match on birth year (YYYY). | "Birth years are identical." |
| 26 | `dob_day_month_swapped` | Date of Birth | Binary | $\{0, 1\}$ | Day and month appear transposed (DD/MM vs MM/DD). | "Dates of birth show an inverted day and month (e.g. 05-12 vs 12-05)." |
| 27 | `dob_one_digit_edit` | Date of Birth | Binary | $\{0, 1\}$ | Compact birthdate digits differ by at most 1 digit edit. | "Dates of birth differ by only a single digit typo." |
| 28 | `dob_diff_days_log` | Date of Birth | Ratio | $[0, \infty)$ | Natural log of absolute difference in days: $\ln(1 + |\Delta t|)$. | "Dates of birth have a small elapsed interval between them." |
| 29 | `dob_both_present` | Date of Birth | Binary | $\{0, 1\}$ | Both records contain a valid, parsed birth date. | "Both records record a valid date of birth." |
| 30 | `age_diff_abs` | Age | Ratio | $[0, 110]$ | Absolute difference in recorded age in years. | "Recorded ages differ by $|\Delta \text{age}|$ years." |
| 31 | `age_diff_log` | Age | Ratio | $[0, \infty)$ | Natural log of age difference: $\ln(1 + |\Delta \text{age}|)$. | "Recorded ages show minimal relative divergence." |
| 32 | `age_both_present` | Age | Binary | $\{0, 1\}$ | Both records have an uncorrupted age value. | "Both records record a valid age." |
| 33 | `nid_exact` | National ID | Binary | $\{0, 1\}$ | Exact 13-digit match on national identity number. | "National identity cards (CNIC) are identical." |
| 34 | `nid_hamming_le_1` | National ID | Binary | $\{0, 1\}$ | 13-digit IDs differ at no more than 1 digit position. | "National identity cards differ by at most one single check digit." |
| 35 | `nid_levenshtein` | National ID | Ratio | $[0, 1]$ | Normalized edit similarity on 13-digit ID string. | "National identity cards share a high degree of digit sequence alignment." |
| 36 | `nid_both_present` | National ID | Binary | $\{0, 1\}$ | Both records contain a complete 13-digit national ID. | "Both records record a national identity card (CNIC)." |
| 37 | `phone_exact` | Phone | Binary | $\{0, 1\}$ | Exact match on complete 11-digit mobile phone. | "Mobile phone numbers are exactly identical." |
| 38 | `phone_last7_equal` | Phone | Binary | $\{0, 1\}$ | Match on subscriber number (last 7 digits, ignoring network prefix). | "Phone numbers match on the final 7 subscriber digits." |
| 39 | `phone_edit_distance` | Phone | Ratio | $[0, 1]$ | Normalized edit similarity on phone digits. | "Phone numbers differ by only 1 or 2 minor digit typos." |
| 40 | `phone_both_present` | Phone | Binary | $\{0, 1\}$ | Both records contain a valid 11-digit phone number. | "Both records record a mobile phone number." |
| 41 | `gender_equal` | Gender | Binary | $\{0, 1\}$ | Cleaned gender codes match exactly (M or F). | "Genders match." |
| 42 | `gender_both_present` | Gender | Binary | $\{0, 1\}$ | Both records contain a valid binary gender (M/F). | "Both records record an unambiguous gender." |
| 43 | `city_equal` | City | Binary | $\{0, 1\}$ | Canonical city names match exactly. | "Patients are registered in the same canonical city." |
| 44 | `city_both_present` | City | Binary | $\{0, 1\}$ | Both records contain a recognized canonical city. | "Both records record a valid residential city." |
| 45 | `address_token_jaccard` | Address | Ratio | $[0, 1]$ | Jaccard word-overlap similarity on address text. | "Addresses share multiple street or locality tokens." |
| 46 | `address_token_cosine` | Address | Ratio | $[0, 1]$ | Cosine similarity on bag-of-words address frequencies. | "Address text exhibits high word-frequency similarity." |
| 47 | `address_both_present` | Address | Binary | $\{0, 1\}$ | Both records contain non-empty address text. | "Both records record an address." |
| 48 | `blood_group_equal` | Blood Group | Binary | $\{0, 1\}$ | ABO / Rh blood group matches exactly. | "Blood groups match." |
| 49 | `blood_group_both_present` | Blood Group | Binary | $\{0, 1\}$ | Both records have an observed blood group. | "Both records record an ABO/Rh blood type." |
| 50 | `insurance_tier_diff` | Insurance | Ordinal | $[0, 1]$ | Normalized ordinal rank difference $|r_1 - r_2| / 3.0$. | "Insurance coverage tiers show an ordinal rank distance." |
| 51 | `insurance_tier_sim` | Insurance | Ordinal | $[0, 1]$ | Ordinal agreement score: $1.0 - \text{diff}$. | "Insurance coverage tiers are identical or adjacent." |
| 52 | `insurance_tier_both_present` | Insurance | Binary | $\{0, 1\}$ | Both records have an explicit insurance tier. | "Both records record an insurance coverage tier." |
| 53 | `visit_count_diff_abs` | Utilization | Ratio | $[0, \infty)$ | Absolute difference in recorded visit encounters. | "Patient visit counts differ by $|\Delta v|$ encounters." |
| 54 | `visit_count_diff_log` | Utilization | Ratio | $[0, \infty)$ | Natural log of visit count difference: $\ln(1 + |\Delta v|)$. | "Patient utilization histories show comparable scale." |
| 55 | `visit_count_both_present` | Utilization | Binary | $\{0, 1\}$ | Both records record an uncorrupted visit count. | "Both records record hospital visit counts." |
| 56 | `source_system_same` | Administrative | Binary | $\{0, 1\}$ | Both records originated from the same department. | "Records originated from the exact same hospital department." |
| 57 | `source_system_both_present` | Administrative | Binary | $\{0, 1\}$ | Both records have a recorded hospital department. | "Both records record an originating department." |
| 58 | `gower_similarity` | Composite | Ratio | $[0, 1]$ | Overall Gower agreement score across 11 core attributes using the delta rule. | "Overall demographic agreement score across all observed fields." |

---

## 4. Mathematical Formulations of Key Metrics

### 4.1 Character $n$-gram Jaccard (Lecture 4)
For two normalized strings $s_1$ and $s_2$, let $\mathcal{G}_n(s)$ denote the set of all overlapping character substrings of length $n$ (default $n=2$):
$$J_n(s_1, s_2) = \frac{|\mathcal{G}_n(s_1) \cap \mathcal{G}_n(s_2)|}{|\mathcal{G}_n(s_1) \cup \mathcal{G}_n(s_2)|}$$
- **Viva Intuition:** If $s_1 = \text{"muhammad"}$ and $s_2 = \text{"mohammad"}$, bigrams overlap heavily (`"ha"`, `"am"`, `"mm"`, `"ma"`, `"ad"`), yielding $J_2 \approx 0.64$, whereas completely distinct names yield $0.0$.

### 4.2 Character 3-gram TF-IDF Cosine
Let $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{|\mathcal{V}|}$ represent the frequency vectors of character 3-grams occurring in $s_1$ and $s_2$:
$$\text{Cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{g \in \mathcal{V}} u_g v_g}{\sqrt{\sum_{g} u_g^2} \sqrt{\sum_{g} v_g^2}}$$
- **Viva Intuition:** Captures word-level phonetic contours by assessing subword frequency alignment, resilient to localized character transpositions.

### 4.3 Ordinal Attribute Distance (Lecture 4)
For an ordinal attribute with ordered ranks $r \in \{0, 1, \dots, R-1\}$ (where $\text{None}=0, \text{Basic}=1, \text{Standard}=2, \text{Premium}=3$, and $R=4$):
$$d_{\text{ordinal}}(r_1, r_2) = \frac{|r_1 - r_2|}{R - 1} = \frac{|r_1 - r_2|}{3.0}$$
$$s_{\text{ordinal}}(r_1, r_2) = 1.0 - d_{\text{ordinal}}(r_1, r_2)$$

### 4.4 Overall Gower Similarity with the Delta Rule (Lecture 4)
Gower's general similarity coefficient combines heterogeneous attribute types (nominal, ordinal, interval, ratio) while natively accommodating missing values through an indicator weight $\delta_{ijk}$:
$$S_{\text{Gower}}(r_i, r_j) = \frac{\sum_{k=1}^K \delta_{ijk} s_k(r_i, r_j)}{\sum_{k=1}^K \delta_{ijk}}$$
where:
$$\delta_{ijk} = \begin{cases} 1 & \text{if attribute } k \text{ is observed in both } r_i \text{ and } r_j \\ 0 & \text{if attribute } k \text{ is missing in either record} \end{cases}$$
- **Viva Intuition:** If an emergency record has no phone number, $\delta_{\text{phone}} = 0$. The phone comparison is completely omitted from both numerator and denominator, preventing the patient from being penalized for missing administrative data.
