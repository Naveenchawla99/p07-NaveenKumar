# P07 — Term-Deposit Call Prioritisation: Report

## 1. The decision

The model ranks a bank's customers by their predicted probability of subscribing to a term deposit. The **campaign manager** uses this ranking to decide whom to phone: the call centre can make only a fixed number of calls per day (k), so the team calls the k highest-ranked customers. A call to a non-subscriber wastes a slot that could have reached a subscriber, while customers who are never called cost nothing. The model therefore only needs to rank well at the top of the list.

## 2. The data

- **Dataset:** UCI Bank Marketing (bank-additional-full), 41,188 rows, 20 features + target (`subscribed`).
- **Base rate:** 11.27% of customers subscribe, so the data are imbalanced.
- **Missingness:** no NaN values, but "unknown" is used as a missing code: `default` 8,597, `education` 1,731, `housing` 990, `loan` 990, `job` 330, `marital` 80. These were recoded to NaN (no statistics used) and imputed inside the pipeline.
- **Duplicates:** 12 exact duplicate rows (0.03%), dropped before splitting so no duplicate pair could straddle train and test, leaving 41,176 rows.
- **Leakage removed:** `duration` (call length) is only known after the call has ended, so it was dropped.
- **Groups:** there is no customer ID, so repeat customers cannot be identified; rows are treated as independent.
- **Time:** rows are ordered chronologically (May 2008 – Nov 2010), and success rates drift strongly across periods. The main results use a random stratified 80/20 split (32,940 train / 8,236 test); the consequences of this choice are tested in Section 6.

## 3. The metric and why

**Primary metric: Average Precision (AP).** Because calls are capped, a wasted call on a non-subscriber directly displaces a call to a likely subscriber, while the scores of customers who are never called do not matter. AP measures how well subscribers are concentrated at the top of the ranking across every call budget, which is what determines each day's deposits.

**Baseline:** a no-skill dummy classifier scores a cross-validated AP of **0.114**, essentially the base rate (0.113).

**Secondary metrics:**
- **Precision@10%:** the share of calls that convert; the number the campaign manager experiences.
- **Recall@10%:** the share of all would-be subscribers reached within the call budget.
- **ROC-AUC:** overall ranking quality, comparable with published work; optimistic under imbalance because it rewards ranking the many easy negatives.
- **Brier score:** calibration of the predicted probabilities, relevant if they are used for forecasting (no-skill baseline ≈ 0.10).

## 4. The search

Two model families were each tuned with `RandomizedSearchCV` (30 candidates each, scored by AP, same CV folds, training data only):

- **Logistic regression:** C sampled log-uniformly (0.001–100).
- **Histogram gradient boosting:** learning rate and L2 penalty sampled log-uniformly; max leaf nodes and min samples per leaf sampled as integers.

Both searches also tuned two **preprocessing choices**: categorical imputation (most frequent vs a separate "missing" category) and numeric scaling (standardise vs none).

**Results:**
- **Logistic regression:** best CV AP **0.453**. 22 of 30 configurations lie within one standard error of the best, so the response surface is flat. Only very strong regularisation (C < 0.01) clearly hurt; the imputation strategy made no difference; scaling gave a small, non-significant gain.
- **Gradient boosting:** best CV AP **0.466**. 19 of 30 configurations lie within one standard error of the best. High learning rates (> 0.15) performed worst; scaling and imputation had no visible effect, as expected for a tree model.

**Are the families distinguishable?** On the same five folds, gradient boosting beat logistic regression in **every fold** (gap 0.007–0.019). The mean gap is **0.013 AP** with a fold-to-fold standard deviation of 0.005 and a standard error of **0.002**, about six standard errors. The raw fold scores vary much more (≈ 0.44–0.49), but both models rise and fall together, so the paired comparison removes that shared noise. The families are therefore distinguishable. Two caveats: CV folds share training data, so the SE is somewhat optimistic, and the gain is small in absolute terms. **Gradient boosting was selected** as the final model; logistic regression remains a credible, more interpretable alternative.

## 5. The result

| | AP |
|---|---|
| Cross-validation (best HGB) | **0.466 ± 0.022** (std over 5 folds) |
| Test set (scored once) | **0.484** (test base rate 0.113) |
| Gap (test − CV) | **+0.018** = 0.8 CV standard deviations |

The test score falls within normal fold-to-fold variation, so there is no sign that the search overfit the training data.

**Secondary metrics (test):** Precision@10% 0.519 · Recall@10% 0.460 · ROC-AUC 0.809 · Brier 0.076.

**Operating point: call the top 10% (823 customers).** This comes from the business constraint in Section 1, not from tuning on the test set.

| | Not called | Called |
|---|---|---|
| **Would not subscribe** | 6,912 | 396 (wasted calls) |
| **Would subscribe** | 501 (missed) | 427 (found) |

**Capacity table** (test set: 8,236 customers, 928 would subscribe):

| Call budget | Calls made | Subscribers found | Wasted calls | Calls per subscriber | % of all subscribers reached | Random calling finds |
|---|---|---|---|---|---|---|
| top 5% | 411 | 253 | 158 | 1.6 | 27% | 46 |
| top 10% | 823 | 427 | 396 | 1.9 | 46% | 93 |
| top 15% | 1,235 | 573 | 662 | 2.2 | 62% | 139 |
| top 20% | 1,647 | 614 | 1,033 | 2.7 | 66% | 186 |
| top 30% | 2,470 | 681 | 1,789 | 3.6 | 73% | 278 |
| top 50% | 4,118 | 775 | 3,343 | 5.3 | 84% | 464 |

Calling the top 10% finds **4.6× as many subscribers** as random calling and reaches 46% of all subscribers with fewer than two calls per subscriber. Returns drop sharply after about 15%: moving from 15% to 20% costs 412 extra calls for only 41 more subscribers (about 10 calls each), which is worse than the random average of about 9. **Recommended budget: 10–15% of customers.**

## 6. The five checks, and what would change the conclusion

1. **Baselines.** Every dummy strategy scores AP ≈ [___]; the model's margin over the best dummy is [___]. The `most_frequent` dummy reaches [___]% accuracy while finding zero subscribers, which confirms that accuracy would have been a misleading metric.
2. **Single feature.** `euribor3m` alone reaches a CV AP of 0.375, about **74% of the full model's lift** over the baseline, and the top five single features are all macro-economic indicators. This is not a target leak, because these statistics are published before each call (unlike `duration`, which was removed). It is, however, a **time proxy**: the indicators take only [___] distinct values and map to specific months, so they mostly tell the model *which campaign period* a customer belongs to. Customer-level features (`poutcome`, `pdays`, `age`, `job`) carry genuine but weaker signal.
3. **Split stability.** Across five different split seeds, test AP = [___] ± [___]. The reported 0.484 sits [___] standard deviations from that mean, so it [is / is not] a lucky split.
4. **Shuffled labels.** Rerunning the whole procedure, including both searches, on permuted training labels gives a best CV AP of [___] (logistic regression) and [___] (gradient boosting), against a baseline of 0.113. [The scores collapse to the baseline, so there is no leak in the pipeline or the search.]
5. **Groups and time.** No groups exist (no customer ID). Cross-validation used [___]. Training on the earliest 80% of rows and testing on the latest 20% gives AP [___] (lift [___]×), compared with 0.484 (lift 4.3×) on the random split.

### The assumption that would invalidate the conclusion

**That future campaigns look like the 2008–2010 periods the model was trained and tested on.** The model's strongest signal is the economic period, not the individual customer, and a random split places every period in both training and test, so the reported AP measures performance on *familiar* periods. If the next campaign runs under different economic conditions, the ranking could lose much of its value and the capacity table would overstate how many subscribers each call finds.

**How to test it:**
- **Offline:** forward-chaining evaluation. Train on all months up to *t*, test on month *t+1*, repeat, and compare the lift per month with the random-split result (the chronological check in item 5 is a first step). Also refit the model without the five macro-economic features to measure how much purely customer-level signal remains.
- **In deployment:** run a pilot in which most of the daily budget goes to the model's top-ranked customers and a small random sample is called as a control. If the model's conversion rate is not clearly above the control group's, the ranking does not transfer to the new period and should not drive the call list.
