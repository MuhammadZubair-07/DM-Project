# Usability Test Protocol & Observer Script

> **Course:** DSC306 Data Mining — COMSATS University Islamabad  
> **Project:** Project 11 — Patient Record Deduplication Tool (National Patient Master Index)  
> **Target Audience:** External Evaluator / Medical Records Clerk / Viva Examiner  
> **Test Facilitator Guide:** *Administer this test with an unassisted user (someone unfamiliar with the codebase). Do NOT explain machine learning concepts or tell the participant which buttons to click. Observe silently, time each task, and document friction points.*

---

## 1. Test Participant & Session Information

- **Participant Name / ID:** `____________________________________________`
- **Background / Role:** `[ ] Student   [ ] Informatics Staff   [ ] Healthcare Clerk   [ ] Instructor`
- **Date & Time:** `____________________________________________`
- **Facilitator Name:** `____________________________________________`
- **Starting System State:** Fresh repository launch (`run.bat` or `streamlit run app.py`)

---

## 2. Participant Briefing Script

> *"Welcome. Today you will be evaluating a clinical patient record deduplication system designed for hospital registration desks. The application helps hospital staff detect when a patient has multiple duplicate medical charts registered under slightly different names, typos, or formats.
> 
> Please think out loud as you navigate through the system. We are testing the usability and clarity of the interface, not your knowledge. If you get stuck or confused, that is a flaw in our interface design, not in your ability. You will be asked to complete four routine tasks."*

---

## 3. Structured Usability Tasks

### Task 1: Find the Candidate Worklist and Identify Top Duplicate Pair
- **Scenario:** You are a morning intake clerk. You need to review all newly identified potential duplicate patients.
- **Goal:** Navigate to the batch scoring screen, load the sample batch register, run the deduplication pipeline, and identify the top-ranked duplicate pair.
- **Target Page:** `Screen 4: Batch Scoring`

| Checkpoint | Target Action | Success? | Time Spent |
|---|---|:---:|:---:|
| 1.1 | Finds and clicks *"4 Batch Scoring"* in sidebar navigation. | [ ] | _____ sec |
| 1.2 | Clicks *"Load sample_batch.csv"* or uploads register. | [ ] | _____ sec |
| 1.3 | Clicks *"⚡ Run Deduplication Pipeline"* and waits for progress bar. | [ ] | _____ sec |
| 1.4 | Identifies candidate reduction ratio (e.g. 99.5%) and worklist table. | [ ] | _____ sec |
| 1.5 | Reads out the primary reason for the top-ranked duplicate pair. | [ ] | _____ sec |

**Observer Notes & Where Participant Got Stuck:**
```
[Record hesitations, confusion over column labels, button clicks, or error handling]
```

---

### Task 2: Investigate a Suspicious Patient Pair
- **Scenario:** A doctor contacts medical records stating that two patient charts may belong to twin sisters or the same person.
- **Goal:** Use the single prediction tool to compare two charts and explain why the system recommends keeping them separate or merging them.
- **Target Page:** `Screen 3: Single Prediction`

| Checkpoint | Target Action | Success? | Time Spent |
|---|---|:---:|:---:|
| 2.1 | Finds and clicks *"3 Single Prediction"* in sidebar navigation. | [ ] | _____ sec |
| 2.2 | Clicks quick-load button *"🔴 Hard Negative (Twin Sisters)"*. | [ ] | _____ sec |
| 2.3 | Clicks *"🚀 Compare & Explain Pair"*. | [ ] | _____ sec |
| 2.4 | Understands the verdict badge (*Distinct Patients / Unlikely Duplicate*). | [ ] | _____ sec |
| 2.5 | Explains in their own words the reasons FOR and AGAINST without asking facilitator. | [ ] | _____ sec |

**Observer Notes & Where Participant Got Stuck:**
```
[Record whether reasons were clear or if any machine learning jargon confused the user]
```

---

### Task 3: Adjust Hospital Misclassification Cost Policy
- **Scenario:** The hospital chief medical officer orders that missing a duplicate patient chart is now considered four times more dangerous to clinical safety than wasting clerk review time.
- **Goal:** Update the hospital cost matrix and observe how the cost-optimal decision threshold and workload queue size respond.
- **Target Page:** `Screen 5: Settings and Cost Matrix`

| Checkpoint | Target Action | Success? | Time Spent |
|---|---|:---:|:---:|
| 3.1 | Finds and clicks *"5 Settings and Cost Matrix"* in sidebar navigation. | [ ] | _____ sec |
| 3.2 | Locates the Cost of Missed Duplicate ($C_{FN}$) input. | [ ] | _____ sec |
| 3.3 | Modifies cost values and observes live theoretical threshold $\theta^*$ change. | [ ] | _____ sec |
| 3.4 | Adjusts the threshold slider and notices the live expected cost recompute. | [ ] | _____ sec |
| 3.5 | Uses the *"🎯 Reset to Cost-Optimal"* button to restore policy alignment. | [ ] | _____ sec |

**Observer Notes & Where Participant Got Stuck:**
```
[Record whether the units (PKR) and clinical assumptions were readily understood]
```

---

### Task 4: Execute a Golden Record Merge and Reversible Undo
- **Scenario:** You have verified that two patient charts belong to the same citizen. You must consolidate them into a surviving Golden Record, and then demonstrate how an erroneous merge can be completely undone.
- **Goal:** Review side-by-side attributes, customize a survivorship value, complete the two-step merge confirmation, and then undo the merge in the audit log.
- **Target Page:** `Screen 6: Merge Review`

| Checkpoint | Target Action | Success? | Time Spent |
|---|---|:---:|:---:|
| 4.1 | Finds and clicks *"6 Merge Review"* in sidebar navigation. | [ ] | _____ sec |
| 4.2 | Reviews side-by-side attributes and understands 🟢 Agree / 🔴 Conflict tags. | [ ] | _____ sec |
| 4.3 | Modifies at least one attribute using the radio buttons or custom edit. | [ ] | _____ sec |
| 4.4 | Clicks *"🔀 Merge Records"*, enters a clerk note, and confirms Step 2. | [ ] | _____ sec |
| 4.5 | Scrolls to *"5. Reversible Merge Audit Log"* and clicks *"↩️ Undo Merge"*. | [ ] | _____ sec |
| 4.6 | Confirms the merge status updated to *"Undone"* and original records restored. | [ ] | _____ sec |

**Observer Notes & Where Participant Got Stuck:**
```
[Record any hesitations during Step 2 confirmation or locating the Undo button]
```

---

## 4. Standard System Usability Scale (SUS) Questionnaire

*Completed by participant immediately following task execution (1 = Strongly Disagree, 5 = Strongly Agree):*

| # | Question | Rating (1 - 5) |
|---|---|:---:|
| 1 | I think that I would like to use this system frequently. | `[ ]` |
| 2 | I found the system unnecessarily complex. | `[ ]` |
| 3 | I thought the system was easy to use. | `[ ]` |
| 4 | I think that I would need the support of a technical person to use this system. | `[ ]` |
| 5 | I found the various functions in this system were well integrated. | `[ ]` |
| 6 | I thought there was too much inconsistency in this system. | `[ ]` |
| 7 | I would imagine that most people would learn to use this system very quickly. | `[ ]` |
| 8 | I found the system very cumbersome to use. | `[ ]` |
| 9 | I felt very confident using the system. | `[ ]` |
| 10 | I needed to learn a lot of things before I could get going with this system. | `[ ]` |

$$\text{SUS Score} = \left[ (Q_1-1) + (5-Q_2) + (Q_3-1) + (5-Q_4) + (Q_5-1) + (5-Q_6) + (Q_7-1) + (5-Q_8) + (Q_9-1) + (5-Q_{10}) \right] \times 2.5$$

- **Calculated SUS Score:** `________ / 100` *(Industry benchmark: > 68 is above average)*

---

## 5. Qualitative Feedback & Viva Defense Insights

1. **What was the clearest / most helpful feature of the application?**  
   `____________________________________________________________________________________`
2. **What was the most confusing or difficult step?**  
   `____________________________________________________________________________________`
3. **Did you notice the academic warning banner at the top of each screen?**  
   `[ ] Yes   [ ] No`
4. **Were the plain-English reasons understandable without statistical training?**  
   `[ ] Yes   [ ] Somewhat   [ ] No`
