# Streamlit Application Manual QA Test Checklist

**National Patient Master Index (NPMI) Record Linkage System**  
**Phase:** 12 (Screens 3, 4, 5 Implemented; Foundation, Screens 1 & 2 Active; Screen 6 Placeholder)  
**Target Audience:** Students, Instructors, Viva Examiners, and QA Reviewers

---

## 1. Pre-Flight & Single-Command Startup

| Check # | Test Item | Verification Procedure | Expected Result | Status |
|:---:|:---|:---|:---|:---:|
| **PF-1** | One-Command Launch | Run `streamlit run app.py` (or `.venv\Scripts\streamlit.exe run app.py`) from repository root. | App launches cleanly without dependency errors; browser opens at `http://localhost:8501`. | [x] |
| **PF-2** | Navigation Layout | Check Streamlit sidebar navigation. | Sidebar displays all pages: `Home` (`app.py`), `1 Data and Quality`, `2 Model Performance`, `3 Single Prediction`, `4 Batch Scoring`, `5 Settings and Cost Matrix`, `6 Merge Review`. | [x] |
| **PF-3** | Zero External API Calls | Disconnect internet connection and refresh app. | App runs completely locally from shipped models (`models/*.joblib`), data (`data/raw/`), and reports (`reports/`). | [x] |

---

## 2. Mandatory Global Safeguards (All Screens)

| Check # | Test Item | Verification Procedure | Expected Result | Status |
|:---:|:---|:---|:---|:---:|
| **GS-1** | Disclaimer Banner Visibility | Navigate to `app.py` and each of the six pages in `pages/`. | **Every single page** displays the warning banner at the top:<br>`⚠️ Student project. Not fit for clinical, financial or legal use. All data is synthetic.` | [x] |
| **GS-2** | Help Sidebar | Expand the *"How to Use This Portal"* sidebar widget on any page. | Clerk guidance panel expands, outlining the 6-step hospital deduplication workflow. | [x] |
| **GS-3** | Sample Data Selector | Click the `Demo (~5k)`, `Sample (~2k)`, and `Full Register (~50k)` buttons in the sidebar. | Active dataset label updates dynamically; page reruns and updates metrics accordingly. | [x] |
| **GS-4** | Zero Ground Truth Leakage | Inspect data previews on Screens 1, 3, and 4. | `true_entity_id`, cluster IDs, and generator corruption logs are **never** rendered or accessible in the UI. | [x] |
| **GS-5** | Zero Technical Jargon | Inspect explanations and metric cards across all pages. | No internal node numbers (`Node 4`), Gini impurities (`gini=0.32`), feature codes (`dob_diff_days_log`), or raw mathematical inequality symbols (`<=`). | [x] |

---

## 3. Screen 1: Data Ingestion, Quality Audit & Cleaning Provenance

**Page Path:** `pages/1_Data_and_Quality.py`

| Check # | Test Item | Verification Procedure | Expected Result | Status |
|:---:|:---|:---|:---|:---:|
| **S1-1** | Default Demo Ingestion | Open Screen 1 without changing settings. | Successfully ingests `data/raw/demo_register.csv` by default, displaying ~5,000 records, 16 columns, and registration span. | [x] |
| **S1-2** | Raw Data Preview | Expand *"View Raw Register Data Preview"*. | Shows the first 10 rows of raw demographic data; all column values are uncleaned raw strings. | [x] |
| **S1-3** | Custom CSV Upload & Validation | Select *"Upload New Hospital Register"* and upload a non-compliant CSV missing required fields. | Friendly red error alert appears explaining which columns are missing and what demographic data they represent. | [x] |
| **S1-4** | Leakage Rejection on Upload | Upload a CSV containing a column named `true_entity_id`. | Upload is halted with a clear leakage violation warning alert. | [x] |
| **S1-5** | Quality Audit Report Table | Scroll to *"2. Data Quality Audit (Lecture 2 Categories)"*. | Summary cards show ~19,773 detected anomalies across 8 distinct Lecture 2 categories (`duplicate`, `inconsistent encoding`, `sentinel value`, `invalid value`, `missing value`, `measurement error`, `outlier`, `collection error`). | [x] |
| **S1-6** | Audit Category Filter | Select *"Inconsistent encoding"* in the category filter selectbox. | Audit table filters dynamically to show non-canonical cities, dirty gender encodings, non-ISO dates, phone formats, and name casing noise. | [x] |
| **S1-7** | Before/After Repair Table | Scroll to *"3. Cleaning & Standardization Provenance"*. | Table presents exact numbers before and after repair (e.g. non-canonical cities: 2,324 before $\to$ 0 after; dirty gender: 1,410 before $\to$ 0 after). | [x] |
| **S1-8** | Automated Validation Fail | Examine the *"Raw Register Validation"* card. | Explicitly displays **FAILED** status with failure breakdown across invalid dates, sentinels, and non-canonical gender. | [x] |
| **S1-9** | Automated Validation Pass | Examine the *"Cleaned Register Validation"* card. | Explicitly displays **PASSED** status (15 of 15 assertions passed, 0 violations). | [x] |
| **S1-10** | Stevens Data Dictionary | Expand *"5. Data Dictionary & Measurement Scale Classification"*. | Table shows all attributes classified by measurement scale (nominal, ordinal, interval, ratio), discrete/continuous, and role. | [x] |
| **S1-11** | Clean CSV Download Button | Click the *"Download Cleaned Register CSV"* button. | Browser initiates download of `register_clean.csv` containing standardized fields and provenance columns. | [x] |

---

## 4. Screen 2: Prospective Model Performance & Scientific Rigor

**Page Path:** `pages/2_Model_Performance.py`

| Check # | Test Item | Verification Procedure | Expected Result | Status |
|:---:|:---|:---|:---|:---:|
| **S2-1** | Candidate Blocking Cards | Scroll to *"1. Upstream Candidate Blocking Efficiency"*. | Metric cards display: Total All-Pairs: $1,478,184,378$, Candidate Pairs: $68,916$, Reduction Ratio: $99.9953\%$, Pairs Completeness: $94.48\%$. | [x] |
| **S2-2** | Benchmark Table Baseline Adjacency | Inspect the *"Prospective Test Set Model Benchmark Table"*. | **MANDATORY CHECK:** The `Majority Baseline (%)` column (**99.46%**) is located **directly adjacent** to the `Test Accuracy (%)` column for every model! | [x] |
| **S2-3** | End-to-End Recall Accounting | Inspect the `End-to-End Recall (%)` column. | End-to-end recall is reported as **94.48%** (penalizing the 538 true duplicates pruned upstream by blocking), rather than naive 100%. | [x] |
| **S2-4** | Interactive Diagnostic Tabs | Click through ROC, PR, Confusion Matrix, and Calibration tabs. | Visualizations render cleanly with AUC values and Platt calibration curves. | [x] |
| **S2-5** | Leakage & Split Findings | Read the scientific experiment cards. | Clearly documents sequential ID leakage failure, random vs temporal split inflation, and seed 999 stability. | [x] |

---

## 5. Screen 5: Decision Policy, Threshold Tuning & Cost Matrix

**Page Path:** `pages/5_Settings_and_Cost_Matrix.py`

| Check # | Test Item | Verification Procedure | Expected Result | Status |
|:---:|:---|:---|:---|:---:|
| **S5-1** | Active Model Switcher | Select different models in the classifier dropdown (Decision Tree, Naive Bayes, k-NN, MLP). | Session state updates immediately; model info badge and description update dynamically. | [x] |
| **S5-2** | Cost Matrix Editing | Edit $C_{FP}$ (e.g. from 5.0 to 10.0) and $C_{FN}$ (e.g. from 20.0 to 30.0). | Units (PKR / pair) and assumptions remain visible; theoretical optimal threshold $\theta^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ recomputes live. | [x] |
| **S5-3** | Threshold Slider & Reset Buttons | Move the threshold slider, then click *"Reset to Cost-Optimal"*. | Slider updates smoothly; reset button instantly sets threshold back to the theoretical $\theta^*$ value. | [x] |
| **S5-4** | Live Metric Simulation | Move threshold from 0.20 to 0.50. | Live cards update instantaneously via `evaluate.py:compute_cost_and_thresholds()`: Expected cost per 1k pairs, recall, precision, worklist queue size, and estimated clerk hours. | [x] |
| **S5-5** | Live Confusion Matrix | Inspect the live confusion matrix table. | Shows exact breakdown of $TP$, $FP$, $FN$, $TN$ counts based on the chosen threshold. | [x] |
| **S5-6** | Baseline Adjacency on Screen 5 | Inspect the clinical accuracy trade-off table. | Overall Accuracy has the **Majority Baseline (99.46%)** in the adjacent column! | [x] |
| **S5-7** | Accuracy Given Up | Check the accuracy trade-off callout. | Quantifies the minimal accuracy given up compared to naive 0.50 to guarantee high recall. | [x] |
| **S5-8** | Global Session Persistence | Change model and threshold on Screen 5, then navigate to Screen 3 and Screen 4. | Screens 3 and 4 immediately reflect the updated active model and review threshold without resetting. | [x] |

---

## 6. Screen 3: Single Pair Duplicate Prediction & Clinical Explanation

**Page Path:** `pages/3_Single_Prediction.py`

| Check # | Test Item | Verification Procedure | Expected Result | Status |
|:---:|:---|:---|:---|:---:|
| **S3-1** | Easy Duplicate Pre-Fill | Click *"🟢 Easy Duplicate (Casing, Typo & Phone Reformat)"*. | Both charts populate with matching patient records; verdict displays 🟢 **Duplicate Match (High Confidence)**; probability is $\ge 90\%$. | [x] |
| **S3-2** | Hard Negative Siblings Pre-Fill | Click *"🔴 Hard Negative (Twin Sisters / Same Father & Address)"*. | Charts populate with twin sisters sharing father, address, and phone; verdict displays ⚪ **Distinct Patients (Unlikely Duplicate)**. | [x] |
| **S3-3** | Subtle Duplicate Pre-Fill | Click *"🟡 Subtle Duplicate (Transposed Surname & Phone Digit)"*. | Charts populate with transposed names; verdict displays 🟡 **Potential Duplicate (Needs Review)**. | [x] |
| **S3-4** | Emergency Intake Pre-Fill | Click *"⚠️ Emergency Intake (Unrecorded CNIC & Missing Phone)"*. | Charts populate with missing CNIC and phone; explainer notes missing attributes gracefully without treating them as conflicts. | [x] |
| **S3-5** | Cleaning & Feature Execution | Click *"🚀 Compare & Explain Pair"*. | Runs the production `clean_patient_register` pipeline and `explain()` engine cleanly. | [x] |
| **S3-6** | Zero Technical Jargon | Read the bulleted reasons in *"Reasons Supporting Match"* and *"Reasons Opposing Match"*. | Contains only plain-English sentences (e.g. *"Dates of birth are identical"*, *"Same surname"*). Zero node numbers, Gini values, or raw threshold numbers. | [x] |
| **S3-7** | Field-by-Field Alignment Table | Scroll to *"4. Field-by-Field Alignment"*. | Table renders all 13 attributes with color-coded badges (🟢 Agree, 🔴 Conflict, 🟡 Unknown/Missing) and clinical observations. | [x] |

---

## 7. Screen 4: Batch Deduplication & Prioritized Worklist

**Page Path:** `pages/4_Batch_Scoring.py`

| Check # | Test Item | Verification Procedure | Expected Result | Status |
|:---:|:---|:---|:---|:---:|
| **S4-1** | Sample Batch Loading | Click *"📁 Load sample_batch.csv (1,801 records)"*. | Successfully loads the 1,801-record register and validates demographic schema. | [x] |
| **S4-2** | File Upload Handling | Upload a custom CSV register. | System validates columns, warns if files exceed recommended 10,000 records, and strips ground-truth columns if present. | [x] |
| **S4-3** | Pipeline Progress Bar | Click *"⚡ Run Deduplication Pipeline"*. | Progress bar progresses across 4 stages (Cleaning $\to$ 5-Pass Blocking $\to$ Feature Extraction $\to$ ML Scoring) with informative status updates. | [x] |
| **S4-4** | File-Specific Blocking Stats | Examine *"3. Blocking Statistics for THIS Register"*. | Metric cards display Total Records ($1,801$), All-Pairs Space ($1,620,900$), Candidate Pairs ($7,871$), and Reduction Ratio ($99.5144\%$). | [x] |
| **S4-5** | Ranked Worklist Sorting | Inspect the worklist table. | Candidate pairs are strictly sorted by `duplicate_probability` descending with confidence badges. | [x] |
| **S4-6** | One-Line Plain Reason | Check the *"Clerk Explanation"* column. | Every candidate pair shows a concise plain-English explanation (e.g. *"Identical CNIC; exact same DOB; same city"*). Zero jargon. | [x] |
| **S4-7** | Dynamic Worklist Filtering | Test the filters: Confidence Tier, Facility, City, and Min Probability slider. | Table filters dynamically in real-time. | [x] |
| **S4-8** | Text Search | Enter a patient name (e.g. *"Muhammad"*) into the search box. | Worklist filters to candidate pairs matching that search query. | [x] |
| **S4-9** | CSV Export | Click *"📥 Download Full Scored Worklist (CSV)"*. | Initiates browser download of complete ranked CSV with probabilities, confidence tiers, and clerk explanations. | [x] |

---

## 8. Screen 6: Reversible Golden Record Merge Review

**Page Path:** `pages/6_Merge_Review.py`

| Check # | Test Item | Verification Procedure | Expected Result | Status |
|:---:|:---|:---|:---|:---:|
| **S6-1** | Clerk Guidance Panel | Open Screen 6 and expand the clerk workflow guide. | Guidance panel details the 5-step review, survivorship, and reversible merge protocol. | [x] |
| **S6-2** | Queue Sourcing | Switch between Screen 4 Worklist, Screen 3 Pair, and Pre-Curated Demo Pairs. | Candidate charts update cleanly; unreviewed queue progress is displayed. | [x] |
| **S6-3** | Side-by-Side Highlighting | Inspect the field comparison section. | Uses **colour AND text/icons** (🟢 Agree, 🔴 Conflict, 🟡 Unknown/Missing) for full clinical accessibility. | [x] |
| **S6-4** | Low Confidence Warning | Select a pair with probability below $\theta_{strict}^*$ ($0.8333$). | Displays prominent warning advising manual clerk verification before merging. | [x] |
| **S6-5** | Blood Group Hazard Alert | Select Demo Pair 2 (Conflicting Blood Groups: A+ vs B-). | Red alert fires: *"🚨 CRITICAL CLINICAL HAZARD: Conflicting Blood Groups Detected! Transfusion incompatibility hazard!"* | [x] |
| **S6-6** | National ID Hazard Alert | Select Demo Pair 3 (Distinct Valid CNICs: 35201 vs 35202). | Red alert fires: *"🚨 CRITICAL CITIZEN CONFLICT: Conflicting Valid National IDs (CNIC)!"* | [x] |
| **S6-7** | Survivorship Suggestions | Examine default winning value selections. | Automatically suggests non-missing/non-sentinel values, newer registration contact details, and complete CNICs with clear rationale notes. | [x] |
| **S6-8** | Survivorship Custom Edit | Select *"Custom Manual Edit"* on any field and enter custom text. | Text input appears; custom text becomes part of the final Golden Record with `"Manual Clerk Edit"` provenance. | [x] |
| **S6-9** | Merge Confirmation Step | Click *"🔀 Merge Records (Golden Record)"*. | Step 2 confirmation box appears displaying the exact Golden Record preview and prompt for clerk audit notes. | [x] |
| **S6-10** | Merge Execution & SQLite Store | Click *"✅ Confirm & Execute Reversible Merge"*. | Merge is committed to `data/merges.db`; displays success alert with unique Merge ID (`MRG-...`). | [x] |
| **S6-11** | Reject Distinct Patients | Click *"❌ Reject (Confirm Distinct Patients)"*. | Records decision in SQLite `decisions` table; pair is removed from active review worklist in Screen 4. | [x] |
| **S6-12** | Deferral Queueing | Click *"⏸️ Defer (Flag for Senior Review)"*. | Records deferral with timestamp in SQLite `decisions` table; returns pair to queue. | [x] |
| **S6-13** | Exact Byte-for-Byte Undo | In the Audit History log, click *"↩️ Undo Merge"* on an active merge. | Status updates to *"Undone"*; original records are restored **byte-for-byte** to pre-merge state. | [x] |
| **S6-14** | Export Labelled Decisions | Click *"📥 Download Labelled Decisions (CSV)"*. | Initiates browser download of complete labelled audit registry (`merge` = 1, `reject` = 0, `defer` = -1). | [x] |

---

## 9. Automated Test Execution

To execute the complete automated test suite across all application modules and SQLite merge stores:
```bash
.venv\Scripts\pytest.exe -v tests/test_app.py tests/test_merge_store.py
```
*Expected Result:* 23 passed tests in < 15.0 seconds.

