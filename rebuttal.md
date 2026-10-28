# Review

### Weaknesses

1. The causal chain is claimed not shown. The internal evidence is a static linear probe off a single layer, which is correlational. The authors will have to show the counterfactual to make a causal claim.
2. The unseen set is built by selecting for model failure, which makes the central findings partly circular. (see sec 4.2, it drops any query the model gets right in any of the 11 trials). This leaves only the ones it consistently fails (line 188)
3. It "fails unseen, scale doesnt help", which is their third finding, is partly baked in rather than discovered (see above).
4. The most important design choice gets the least space. the unseen set is about 6 lines in sec 4.2 while the seen set is roughly 5x longer, even though unseen carries both headline claims
5. Exposure is exact string match (line 172), so it misses paraphrase, translation, implied exposure (unseen really means this surface string wasnt matched, not that the fact is absent from training).
6. Line 243 says calibration collapses and CoT doesnt resolve it, but they never check whether the reasoning is causal or just post hoc. CoT is toggled by a prompt and scored only off the final answer line, no truncation or corruption test, no activation patching, no logit lens.

### Key questions

1. Can you curate unseen independent of model success, then re-report the self verify numbers?
2. Why 11 generations?

---

# Rebuttal

```markdown
We thank the reviewer for the constructive feedback. We address each point below.

# W1: Lack of Causal Validation for the Proposed Causal Chain
Our goal is not to propose a new interpretability or causal validation method, but to introduce a meta-benchmark and framework that unifies existing research along a single causal chain. The h→p and p→y transitions have been extensively studied through uncertainty estimation [1–3], hallucination detection [4–6], truthfulness and linear probing [5,7,8], and mechanistic probing [9,10]. We organize these directions according to the stages they address (Sections 2 and 3).

Our core contribution is to ground the previously unverified starting point, D→h, through verified seen/unseen exposure and head/tail stratification. Prior work analyzed h, p, and y without knowing what the model encountered during training. Our verified exposure supplies this missing denominator and strengthens the basis of those analyses. The reviewer’s suggested counterfactual validation, such as retraining without specific facts or activation patching, is valuable follow-up work that our exposure-grounded framework makes possible.

# W2: Potential Circularity in Failure-Based Unseen Set Construction
Open-ended filtering does not select questions that the model fails to answer; it removes potential information contamination. Although the unseen set is initially separated by the training cutoff, this criterion alone cannot guarantee genuinely unseen answers.

For example, (i) an event may occur after the cutoff while its answer was announced years earlier, as in “Which countries will host the 2026 FIFA World Cup?” Alternatively, (ii) a new event may have an answer based on time-invariant knowledge, such as geography or geology, as in “At which plate boundary was the epicenter of the May 2025 earthquake located?” Open-ended filtering removes such cases to produce a higher-confidence unseen set.

# W3: Potential Bias from Failure-Based Unseen Evaluation
Filtering uses open-ended questions, whereas evaluation uses MCQs. Thus, failure during filtering does not guarantee failure during evaluation. Models may still answer correctly by eliminating distractors, and this ability can improve with scale or CoT prompting.

Indeed, Unseen Accuracy exceeds the 25% random baseline and rises from 33.88% for 7B to 43.64% for 32B (Table 2). Our claim is not that unseen performance cannot improve, but that a substantial 30–40 percentage-point gap from seen performance persists despite this improvement.

# W4: Insufficient Unseen Set Details
We would appreciate clarification on whether this concern refers specifically to the amount of detail provided about the unseen set. For concision, we focused on its two central construction criteria: cutoff-based temporal filtering and open-ended filtering. If the reviewer has a more specific concern about the construction or description, we would be happy to address it in the revision.

# W5: Unseen vs. Unmatched Exposure
We acknowledge that exact matching cannot capture paraphrased, translated, or implied exposure. However, our unseen set does not rely on string matching. It consists of events occurring strictly after OLMo 2’s training cutoff, so descriptions of these events cannot appear in the corpus in any form or language. Cases whose answers may overlap with pre-cutoff knowledge are further removed through open-ended filtering (Section 4.2). String matching is used only for seen exposure, where an exact match indisputably verifies presence in the corpus.

# W6: Lack of Causal Validation for CoT Reasoning
Our goal is not to propose a new interpretability method, but to present a framework and benchmark that organize prior work on D→h→p→y along the axis of verified training exposure. Combining mechanistic methods such as activation patching or the logit lens with our exposure conditions would constitute a substantial independent follow-up study. Our work establishes the foundation for such research, while limiting the present CoT analysis to behavioral evidence.

Within this scope, we compared item-level changes in correctness with and without CoT. A correctness flip is binary only under deterministic decoding, so we use greedy top-1 decoding. These results therefore need not match the K=11 expected accuracy in Table 5.

| Exposure | Both (%) | Loss (%) | Gain (%) | Flip (%) | ΔAccuracy (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seen-head | 88.7 | 2.7 | 2.7 | 5.8 | +0.0 |
| Seen-tail | 67.4 | 10.0 | 6.4 | 19.5 | −3.6 |
| Unseen | 38.8 | 7.1 | 12.1 | 33.2 | +5.0 |

*Flip = (loss + gain) / (both + loss + gain). Other percentages are calculated over all questions.*

CoT exhibits three exposure-dependent regimes:

- **Seen-head:** Loss and gain are symmetric (2.7% each), yielding no net effect. CoT rarely changes outcomes for robust knowledge.

- **Seen-tail:** Loss exceeds gain, producing a net decrease of 3.6 points. For knowledge subject to a retrieval bottleneck, intermediate CoT tokens may disrupt fragile recall paths and eliminate otherwise reachable answers. This pattern supports concerns about post-hoc reasoning at the behavioral level.

- **Unseen:** CoT yields a net gain of 5.0 points but the highest flip rate (33.2%), gaining 12.1% while losing 7.1%. This instability is consistent with CoT increasing Norm. P on unseen questions (28.18→52.83) without improving calibration. Answers are extensively rearranged beneath the apparent confidence increase, with instability growing as exposure weakens (5.8%→19.5%→33.2%).

This analysis does not establish mechanistic causality, but provides exposure-conditioned behavioral evidence about whether CoT reliably leads to the answer. We will add it to the appendix.

# Q1: Need for Independent Unseen Evaluation
Following the reviewer’s suggestion, we reran the self-verification experiments on the unfiltered unseen set. Overall metrics increased slightly, likely due to residual contamination, but the main trends—calibration collapse on unseen content and its persistence under scaling—remain unchanged. This suggests that the cutoff-based construction already captures the exposure boundary reasonably well.

P1, P2, and P3 denote zero-shot, few-shot, and zero-shot CoT, respectively. The few-shot CoT experiment could not be completed during the rebuttal period but is currently running; we will share its results during discussion. Due to the character limit, Seen results are averaged across head and tail to focus on the seen–unseen comparison.

| Model | Metric | Seen P1 | Unseen P1 | Seen P2 | Unseen P2 | Seen P3 | Unseen P3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7B | Accuracy | 31.14 | 60.07 | 28.14 | 62.96 | 34.37 | 62.37 |
|  | Precision | 14.37 | 16.51 | 12.99 | 21.65 | 14.20 | 27.95 |
|  | Recall | 14.37 | 16.53 | 13.00 | 24.46 | 13.58 | 32.27 |
|  | F1 | 14.37 | 16.52 | 13.00 | 22.28 | 13.59 | 27.96 |
| 13B | Accuracy | 59.12 | 56.25 | 59.03 | 56.92 | 39.47 | 60.73 |
|  | Precision | 53.48 | 41.15 | 53.61 | 40.60 | 52.67 | 48.00 |
|  | Recall | 59.41 | 57.57 | 59.31 | 56.53 | 35.61 | 50.34 |
|  | F1 | 54.99 | 45.01 | 55.12 | 44.26 | 37.53 | 45.23 |
| 32B | Accuracy | 81.72 | 65.59 | 82.18 | 65.94 | 81.34 | 64.28 |
|  | Precision | 77.90 | 47.56 | 78.07 | 45.50 | 80.81 | 54.62 |
|  | Recall | 82.10 | 54.33 | 82.72 | 53.65 | 86.87 | 62.87 |
|  | F1 | 79.01 | 49.30 | 79.39 | 47.45 | 81.83 | 55.56 |

# Q2: About 11 Generations
Self-consistency studies [11–13] typically use a fixed sample count, such as 10–64, to balance accuracy and computational cost. Following [11], we generated 10 stochastic samples and added one deterministic greedy-decoding response for a more informative evaluation, yielding 11 candidate responses.

# References
[1] https://arxiv.org/abs/2207.05221  
[2] https://arxiv.org/abs/2406.13415  
[3] https://arxiv.org/abs/2302.09664  
[4] https://arxiv.org/abs/2405.19648  
[5] https://arxiv.org/abs/2304.13734  
[6] https://arxiv.org/abs/2410.02707  
[7] https://arxiv.org/abs/2102.12452  
[8] https://arxiv.org/abs/2212.03827  
[9] https://arxiv.org/abs/2004.12265  
[10] https://arxiv.org/abs/2202.05262  
[11] https://arxiv.org/abs/2402.13212  
[12] https://arxiv.org/abs/2203.11171  
[13] https://arxiv.org/abs/2401.10480
```

---

# 추가할 표

P4: CoT few-shot

| Model | Metric | Seen P1 | Unseen P1 | Seen P2 | Unseen P2 | Seen P3 | Unseen P3 | Seen P4 | Unseen P4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7B | Accuracy | 31.14 | 60.07 | 28.14 | 62.96 | 34.37 | 62.37 | 28.02 | 58.96 |
|  | Precision | 14.37 | 16.51 | 12.99 | 21.65 | 14.20 | 27.95 | 20.30 | 28.28 |
|  | Recall | 14.37 | 16.53 | 13.00 | 24.46 | 13.58 | 32.27 | 14.77 | 26.34 |
|  | F1 | 14.37 | 16.52 | 13.00 | 22.28 | 13.59 | 27.96 | 15.54 | 25.54 |
| 13B | Accuracy | 59.12 | 56.25 | 59.03 | 56.92 | 39.47 | 60.73 | 60.63 | 62.07 |
|  | Precision | 53.48 | 41.15 | 53.61 | 40.60 | 52.67 | 48.00 | 72.03 | 51.97 |
|  | Recall | 59.41 | 57.57 | 59.31 | 56.53 | 35.61 | 50.34 | 66.13 | 66.15 |
|  | F1 | 54.99 | 45.01 | 55.12 | 44.26 | 37.53 | 45.23 | 64.06 | 54.43 |
| 32B | Accuracy | 81.72 | 65.59 | 82.18 | 65.94 | 81.34 | 64.28 | 81.05 | 64.20 |
|  | Precision | 77.90 | 47.56 | 78.07 | 45.50 | 80.81 | 54.62 | 81.18 | 55.19 |
|  | Recall | 82.10 | 54.33 | 82.72 | 53.65 | 86.87 | 62.87 | 85.39 | 61.89 |
|  | F1 | 79.01 | 49.30 | 79.39 | 47.45 | 81.83 | 55.56 | 81.31 | 55.46 |