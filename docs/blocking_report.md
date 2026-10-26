# Blocking Strategy and Scaling Evaluation Report

> **Course:** DSC306 Data Mining — COMSATS University Islamabad
> **Project:** Project 11 — Patient Record Deduplication Tool
> **Topic:** Tractable Pairwise Matching & Blocking Challenge (Section 7.5 & Assigned Challenge)
> **Evaluated Population:** `data/clean/register_clean.csv` (1,478,184,378 all-pairs space)

---

## 1. Executive Summary & Assigned Challenge Objective

In patient record deduplication, comparing all possible record pairs is quadratic: for $n = 54,373$ records, an exhaustive all-pairs comparison requires **1,478,184,378 pairwise comparisons** (~1.48 billion). At 1,000 comparisons per second, brute-force matching would take **over 17 days of continuous compute** and cannot be deployed in a responsive hospital web application.

### Pre-Registered Gate Requirement:
- **Pre-registered Target Pairs Completeness:** **$\ge 92.0\%$** true duplicate retention.
- **Hard Ceiling on Candidate Comparisons:** **$\le 1,000,000$ pairs** ($RR \ge 99.93\%$).

### Results Achieved by Union Scheme (B1 + B2 + B3 + B4 + B5):
- **Candidate Comparisons:** **823,534 pairs** (down from 1,478,184,378).
- **Reduction Ratio (RR):** **99.944287%** of all comparisons eliminated.
- **Pairs Completeness (Recall):** **94.48%** (9,216 of 9,754 true duplicate pairs retained).
- **Pairs Quality (Precision):** **1.12%** (a ~1,700x positive-class enrichment over all-pairs space).
- **Total Runtime:** **1.442 seconds** on pure Python dictionaries without external record-linkage dependencies.

---

## 2. Scheme Evaluation & Performance Matrix

| Blocking Scheme | Key Definition | Candidate Comparisons | Reduction Ratio (%) | True Pairs Found | Pairs Completeness (%) | Pairs Quality (%) | Runtime (s) |
|---|---|---|---|---|---|---|---|
| **B1 (Soundex+Yr)** | `Soundex(family_name) + birth_year` | 583,742 | 99.96051% | 5,470 / 9,754 | **56.08%** | 0.94% | 0.406s |
| **B2 (Fn3+YrMo)** | `first_name[:3] + DOB YYYY-MM` | 25,202 | 99.998295% | 3,537 / 9,754 | **36.26%** | 14.03% | 0.229s |
| **B3 (Phone7)** | `phone[-7:]` | 7,608 | 99.999485% | 5,906 / 9,754 | **60.55%** | 77.63% | 0.13s |
| **B4 (NID)** | `national_id (13 digits)` | 7,194 | 99.999513% | 6,087 / 9,754 | **62.41%** | 84.61% | 0.201s |
| **B5 (SNM w=5)** | `SNM window w=5 on surname+firstname` | 217,482 | 99.985287% | 1,900 / 9,754 | **19.48%** | 0.87% | 0.271s |
| **B1 + B3** | `B1 | B3` | 587,922 | 99.960227% | 8,039 / 9,754 | **82.42%** | 1.37% | 0.536s |
| **B1 + B3 + B4** | `B1 | B3 | B4` | 589,134 | 99.960145% | 9,071 / 9,754 | **93.0%** | 1.54% | 0.738s |
| **B1+B2+B3+B4** | `B1 | B2 | B3 | B4` | 609,935 | 99.958738% | 9,146 / 9,754 | **93.77%** | 1.5% | 0.966s |
| **UNION (B1-B5)** | `Union of B1, B2, B3, B4, B5` | 823,534 | 99.944287% | 9,216 / 9,754 | **94.48%** | 1.12% | 1.442s |

---

## 3. Empirical Scaling Experiment (5k to 54k Records)

To prove that our blocking pipeline scales sub-quadratically, we evaluated candidate counts across increasing subsets:

| Subset Size ($n$) | All-Pairs Space $n(n-1)/2$ | Candidate Pairs Generated | Reduction Ratio (%) | Pairs Completeness (%) | Runtime (s) |
|---|---|---|---|---|---|
| **5,000** | 12,497,500 | 24,939 | 99.800448% | 95.18% | 0.077s |
| **10,000** | 49,995,000 | 60,333 | 99.879322% | 95.43% | 0.446s |
| **25,000** | 312,487,500 | 226,517 | 99.927512% | 95.27% | 0.767s |
| **54,373** | 1,478,184,378 | 823,534 | 99.944287% | 94.48% | 1.985s |

> **Growth Law:** As the register grows 10.9-fold (from 5,000 to 54,373 records), brute-force comparisons grow **118.3-fold** (from 12.5M to 1.48B), whereas candidate comparisons grow only **33.0-fold** (from 24.9k to 823.5k), demonstrating clear sub-quadratic scaling.

---

## 4. Missed-Duplicate Analysis

Across the full register of 54,373 records, the ground truth contains **9,754 true duplicate pairs**. The Union blocking scheme successfully captured **9,216** of them, missing exactly **538 pairs** (a missed rate of **5.52%**).

### Failure Attribution Breakdown:

| Defeat Category | Affected Pairs | Proportion of Missed (%) | Mechanism of Defeat Across All 5 Keys |
|---|---|---|---|
| **Phone Omitted or Digit Error** | 538 | 100.0% | Defeats B3 (`phone[-7:]`); caused by Emergency department phone omissions or digit transpositions. |
| **National ID Missing or Typo** | 538 | 100.0% | Defeats B4 (`national_id`); caused by Legacy department omissions or synthetic check digit errors. |
| **DOB Discrepancy or Missing** | 390 | 72.5% | Defeats B1 (birth year) and B2 (birth year-month); caused by unrecoverable day-month transpositions or missing DOB. |
| **Surname Soundex Mismatch** | 286 | 53.2% | Defeats B1; caused by leading letter phonetic mutations (e.g. Qureshi vs Kureshi) or severe typos. |
| **Simultaneous Multi-Field Corruption** | 538 | 100.0% | Three or more critical fields simultaneously corrupted across the pair, defeating all passes simultaneously. |

---

## 5. What Blocking Costs Us in Missed Duplicates

### Plain-English Assessment for Hospital Stakeholders:
In a hospital setting, deduplicating patient records has a direct clinical and administrative impact. A patient who visits the Emergency department after a car accident might have no national ID recorded and a blank phone number. If their previous inpatient surgery was registered under an OPD record with a slightly misspelled surname and a transposed birth date, the blocking stage will not bring these two records together into the candidate worklist.

**What is the exact clinical cost?**
Out of every 100 true duplicate pairs in our hospital register, our blocking strategy safely delivers **94.48%** to the records clerk's screen for pairwise classification, while **5.52%** (538 pairs) are filtered out before comparison.

**Why is this trade-off justified?**
1. **Tractability:** Catching that remaining 5.52% of duplicates would require opening wide windows or relaxing blocking constraints, causing candidate comparisons to balloon from 823,534 toward tens of millions of pairs, overwhelming the hospital server and making the interactive clerk UI unusable.
2. **Data Governance Remedy:** The true root cause of these 538 missed pairs is upstream data quality—specifically emergency registrations without phone numbers and legacy migrations without CNICs. Hospital data entry policies requiring at least one validated anchor identifier (either biometric CNIC or mobile SMS verification) would immediately elevate blocking completeness to >99%.

### Concrete Case Studies of Missed Pairs:

#### Case Study 1: Records `REC0009474` vs `REC0018262`
- **Demographics:** `Butt Ksobia vs Sobia Butt` | DOB: `1941-01-10 vs 1941-10-01`
- **Identifiers:** Phone: `03445492524 vs ` | CNIC: `1314556831209 vs `
- **Why Every Pass Failed:** Phone mismatch/missing, NID mismatch/missing, DOB mismatch/missing, Surname Soundex mismatch.

#### Case Study 2: Records `REC0001848` vs `REC0048468`
- **Demographics:** `Raza Hashmi vs Raza Hgshmi` | DOB: `1975-12-18 vs 1957-12-18`
- **Identifiers:** Phone: `03409551571 vs ` | CNIC: ` vs `
- **Why Every Pass Failed:** Phone mismatch/missing, NID mismatch/missing, DOB mismatch/missing.

#### Case Study 3: Records `REC0008912` vs `REC0029672`
- **Demographics:** `Nawaz Agha Nadi vs Nadia Nawaz` | DOB: `1999-05-02 vs 1999-02-05`
- **Identifiers:** Phone: ` vs 03221334943` | CNIC: `3399748967165 vs 3399742967165`
- **Why Every Pass Failed:** Phone mismatch/missing, NID mismatch/missing, DOB mismatch/missing, Surname Soundex mismatch.

#### Case Study 4: Records `REC0031973` vs `REC0037548`
- **Demographics:** `Osana Ahmxd vs Sana Ahmed` | DOB: `1981-10-19 vs 1981-10-19`
- **Identifiers:** Phone: `03036857954 vs 03036857934` | CNIC: `3982686649949 vs 3932686649949`
- **Why Every Pass Failed:** Phone mismatch/missing, NID mismatch/missing, Surname Soundex mismatch.

#### Case Study 5: Records `REC0019100` vs `REC0042535`
- **Demographics:** `Mumhammed Khawaa vs Muhammad Khawaja` | DOB: `1939-03-06 vs 1939-03-06`
- **Identifiers:** Phone: ` vs 03189632761` | CNIC: `5110883998014 vs 5110883948014`
- **Why Every Pass Failed:** Phone mismatch/missing, NID mismatch/missing, Surname Soundex mismatch.

---

## 6. Block-Size Distribution and Oversized Block Guard

In traditional blocking, common Pakistani surnames (e.g. Khan, Ahmed, Ali) create massive blocks that generate an explosion of comparisons (k*(k-1)/2). To protect the deduplication pipeline, we enforce a configurable max_block_size guard (B_max = 250).

### Pass B1 Block Distribution Metrics:
- **Total Unique Blocks Formed:** 4,942
- **Singleton Blocks (Size = 1, Zero Comparisons):** 1,265 (25.6%)
- **Active Candidate Blocks (Size >= 2):** 3,677
- **Median Block Size (p50):** 6 records
- **95th Percentile Block Size (p95):** 39 records
- **99th Percentile Block Size (p99):** 55 records
- **Maximum Block Size Encountered:** 79 records
- **Oversized Blocks Suppressed (> B_max):** 0
- **Comparisons Safely Pruned by Guard:** 0 comparisons

### Top 5 Largest Blocks in Pass B1:
| Rank | Blocking Key (Soundex_BirthYear) | Block Size ($k$) | Potential Comparisons $k(k-1)/2$ |
|---|---|---|---|
| 1 | `K500_2003` | 79 records | 3,081 |
| 2 | `K500_1937` | 77 records | 2,926 |
| 3 | `K500_1948` | 74 records | 2,701 |
| 4 | `K500_1939` | 73 records | 2,628 |
| 5 | `H250_1936` | 70 records | 2,415 |

---

## 7. Viva Defence & Theoretical Justification

### Viva Q1: Why did you implement blocking from scratch rather than using `recordlinkage`?
**Answer:** In medical and enterprise data engineering, external libraries often obscure memory allocations, cannot handle domain-specific Pakistani naming patterns, and introduce unvetted transitive dependencies. Implementing candidate generation via pure Python dictionary hash tables ($O(n)$ insertion and lookup) gives us exact control over block size guards, determinism, and execution profiling.

### Viva Q2: What is the difference between Reduction Ratio (RR) and Pairs Completeness (PC)?
**Answer:** Reduction Ratio measures computational efficiency: $RR = 1 - (\text{candidate comparisons} / \text{all-pairs space})$. Our union scheme eliminates $99.944\%$ of all possible pairs. Pairs Completeness measures recall against ground truth: $PC = \text{true pairs retained} / \text{total true pairs}$. Our union scheme captures $94.48\%$ of all true duplicate entities.

### Viva Q3: Why is Sorted-Neighbourhood Method (SNM) complementary to standard blocking?
**Answer:** Standard hash blocking (B1-B4) uses hard partitions: if an error occurs in the blocking key, the records fall into completely different buckets. Sorted-Neighbourhood Method (B5) sorts records by a compound key and slides a window of size $w=5$. Even if a typo occurs in the middle of a name, the records sort adjacent to one another and are successfully compared.

---

## 8. Artifacts and Visualization Summary

The following figures were generated and saved into `reports/figures/`:
1. `reports/figures/blocking_scaling.png`: Scaling curve comparing candidate comparisons vs $O(n^2)$ space across 5k, 10k, 25k, and 54k records.
2. `reports/figures/block_size_distribution.png`: Histogram and CDF of block sizes illustrating the long-tail distribution.
3. `reports/figures/blocking_tradeoff.png`: Completeness vs candidate volume trade-off across individual and combined blocking schemes.
