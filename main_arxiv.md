# PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers

**Authors**: Ngoc Phan Phuoc Loc, Toan Huynh La Viet, Thanh Tran Khanh, Duy A Nguyen, Tuan Anh Nguyen Pham, Thanh Nguyen, Nitesh V. Chawla, Wray Buntine, Kok-Seng Wong, Khoa D. Doan, Binh T. Nguyen

**Affiliations**: VinUniversity, University of Illinois Urbana-Champaign, University of Notre Dame, Monash University

---

## Abstract

The rapid growth in submissions to machine learning venues has strained the scientific peer-review system and intensified interest in LLM-based automated peer reviewers. However, how good these systems are actually, especially compared to human reviewers at catching scientific gaps, remains poorly understood. In this work, we introduce **PRISM** (**P**eer **R**eview **I**ntelligence via **S**tructured **M**ulti-dimensional assessment), a benchmarking framework that evaluates review quality across four dimensions: **Depth of Analysis**, **Novelty Assessment**, **Flaw Identification & Major Issues Prioritization**, and **Multi-dimensional Constructiveness**. Unlike most existing evaluations based on surface-level metrics like ROUGE and BLEU, or unconstrained LLM-as-a-judge prompting that conflates fluency with rigor, PRISM grounds each dimension in argument mining, retrieval-augmented verification, and consensus-based scoring. We apply PRISM to benchmark five leading automated reviewer systems and human reviewers on a stratified corpus of reviews from ICLR, ICML, and NeurIPS. The results reveal that LLMs can match or beat human reviewers on individual dimensions: comparable depth of analysis, stronger novelty verification, and highly accurate critique prioritization. However, no single system consistently matches the balanced performance of the human baseline across all dimensions at once. Each exhibits a distinct specialization profile with characteristic blind spots---failure modes that aggregate metrics miss entirely. The implication is that *LLM reviewers are best understood as targeted supplements to human review, effective within specific dimensions, but unreliable as standalone replacements.* Our demo and key results can be found at [https://prism-benchmark.github.io/](https://prism-benchmark.github.io/).

---

## 1. Introduction

Scientific peer review is under mounting strain. Submission volumes at major machine learning venues have grown at an incredible rate: NeurIPS received 15,671 submissions in 2024, surging to 21,575 in 2025 [neurips2024, neurips2025blog], while ICML saw a 44.9% year-on-year jump between 2023 and 2024 alone, followed by a further 25.4% increase in 2025 [icml2023stats, icml2024stats, icml2025stats]. This exponential growth severely strains the reviewer pool and complicates paper-to-reviewer matching, prompting venues to introduce new load-management and quality-control mechanisms, such as ICML's recent author self-ranking policies [icml2026policy_blog]. Furthermore, reviewing at several ML conferences is becoming mandatory with short deadlines, creating additional pressure on reviewers, particularly when assignments are not well aligned with their expertise. In response, Large Language Models (LLMs) have moved rapidly from proofreading aids to autonomous reviewer agents capable of drafting comprehensive critiques and their deployment is no longer theoretical [chang-etal-2025-treereview, gao2024reviewer2optimizingreviewgeneration, yu-etal-2024-automated-SEA, zhu-etal-2025-deepreview, cyclereviewer]. Estimates indicate that 17--21% of reviews at recent top-tier venues already involve LLM assistance [liang2023largelanguagemodelsprovide, Wang_2024, iclr2026policy], prompting venues to adopt a wide range of policies from outright bans to mandatory disclosure [icml2026policy].

This reality raises an important question:*Are LLMs sufficient reviewers to evaluate scientific work -- and, critically, are they better at identifying gaps in a paper than human reviewers who increasingly work under time constraints and review overload?* Answering this question is particularly important when growing evidence suggests that human review quality and reliability may be degrading under mounting pressures. For example, the NeurIPS consistency experiment [beygelzimer2023neurips] suggested that as many as 23% of acceptance decisions may change depending purely on reviewer assignment.

We address this by introducing a benchmark to evaluate both LLM-generated and human reviews, grounded by official reviewer guidelines of established machine learning venues (e.g., ICLR, NeurIPS). A high-quality peer review must go beyond mere summarization to satisfy four core duties: evaluating technical soundness, contextualizing originality, diagnosing critical errors, and providing actionable feedback. Accordingly, our benchmark evaluates whether the reviewers can fulfill these mandates across four dimensions:

- **RQ1** **Depth of Analysis:** Do reviewers engage with a paper's methodological and empirical claims in depth, or do they default to surface-level assessment?
- **RQ2** **Novelty Assessment:** Are reviewers' novelty judgments grounded in prior literature, or do they rely on unverified or factually incorrect assertions?
- **RQ3** **Flaw Identification & Major Issues Prioritization:** How accurately and comprehensively do reviewers detect critical scientific flaws, and do they correctly prioritize fatal methodological concerns over minor textual anomalies?
- **RQ4** **Multi-dimensional Constructiveness:** How actionable, solution-oriented, and professionally calibrated is the reviewers' feedback?

We call this benchmark **PRISM** (**P**eer **R**eview **I**ntelligence via **S**tructured **M**ulti-dimensional assessment). Each dimension is operationalized through a dedicated evaluation pipeline, which is grounded in argument mining, retrieval-augmented verification, and consensus-based scoring.

We then apply PRISM to compare five leading automated reviewer systems---TreeReview [chang-etal-2025-treereview], Reviewer2 [gao2024reviewer2optimizingreviewgeneration], SEA-E [yu-etal-2024-automated-SEA], DeepReview [zhu-etal-2025-deepreview], and CycleReviewer [cyclereviewer]---and human reviewers on a stratified corpus of papers drawn from ICLR, ICML, and NeurIPS. This analysis yields the following insights:

- **RQ1:** CycleReviewer and DeepReview match human analytical depth; TreeReview falls into a surface-level trap, over-indexing on presentation anomalies.
- **RQ2:** SEA-E outperforms human reviewers on grounded novelty verification; other systems exhibit measurable novelty hallucination.
- **RQ3:** Reviewer2 leads in flaw recall as a high-sensitivity scanner; LLMs broadly achieve near-perfect critical issue prioritization, demonstrating a cognitive alignment comparable to human reviewers.
- **RQ4:** DeepReview produces the most actionable feedback, though a constructiveness gap relative to human reviewers persists across all systems.

No single system dominates across all four dimensions: each excels in a distinct niche while leaving structured gaps invisible to aggregate metrics. This positions LLM reviewers as powerful, task-matched specialists---effective where deployed deliberately, but not yet near general-purpose replacements for human reviewers.

In summary, the key contributions of this work are:

- **PRISM: A Multi-dimensional Benchmarking Framework.** We introduce PRISM, a structured evaluation framework with four dedicated pipelines that operationalizes RQ1--RQ4, probing scientific reviewer competence beyond surface-level prose.
- **Comprehensive Evaluation Corpus.** We curate a dataset of manuscripts and expert human reviews spanning ICLR, ICML, and NeurIPS, establishing a robust, consensus-driven reference for benchmarking automated reviewer systems.
- **Systematic Human-vs-LLM Analysis.** We benchmark five leading LLM reviewer systems across all four dimensions, revealing distinct specialization profiles and structured failure modes invisible to aggregate metrics.
- **Actionable Deployment Guidance.** We derive evidence-based recommendations for deploying LLM reviewers, identifying which systems best fit which roles within a human-assisted review pipeline.

---

## 2. Related Work

**LLM-based Reviewer Systems.** The rapid progress of large language models has spawned a growing family of specialized automated reviewing systems. One line of work improves review quality through structured reasoning: TreeReview [chang-etal-2025-treereview] decomposes evaluation into a hierarchical tree of questions that are recursively refined and aggregated, while DeepReview [zhu-etal-2025-deepreview] emulates the slow, deliberate thinking process of expert reviewers. A complementary line focuses on optimizing the generation pipeline itself: Reviewer2 [gao2024reviewer2optimizingreviewgeneration] trains a two-stage model that first predicts review aspects and then conditions generation on them, and SEA [yu-etal-2024-automated-SEA] standardizes heterogeneous review data before fine-tuning dedicated evaluation and analysis modules. Multi-agent collaboration offers yet another angle; CycleReviewer [cyclereviewer] pairs a research agent with a reviewer agent in an iterative preference-training loop. While these systems demonstrate impressive linguistic fluency, their corresponding evaluation protocols predominantly rely on generic n-gram metrics or monolithic LLM-as-a-judge scoring applied to the review as a whole. Although some works evaluate multiple criteria, these macro-level assessments are structurally blind to the granular logic of the critique: they cannot verify whether individual claims are substantiated by grounded premises, nor can they cross-check novelty assertions against retrieved prior literature.

**Evaluation of AI-Generated Reviews.** Evaluating AI-generated reviews is a distinct challenge from generating them. Early work relied on lexical overlap metrics---ROUGE [lin-2004-rouge] and BLEU [papineni-etal-2002-bleu]---that reward surface similarity with reference reviews but are blind to scientific reasoning quality and factual correctness [novikova-etal-2017-need]. [liang2023largelanguagemodelsprovide] advanced beyond surface metrics by measuring point-level overlap between LLM and human feedback, finding comparable coverage but systematic gaps in methodological depth. The LLM-as-judge paradigm [liu-etal-2023-g, zheng2023judging] offers richer evaluation, but introduces well-documented biases---position [zheng2023large], verbosity [saito2023verbosity], and self-enhancement [panickssery2024llm]---that are especially problematic when scientific rigor, not linguistic fluency, is the target. ReviewEval [garg-etal-2025-revieweval] is the most structured prior framework, defining six evaluation dimensions including depth of analysis, constructiveness, and guideline adherence; however, it relies on end-to-end LLM rubric prompting to assign scores, and the benchmark covers only 16 papers and three reviewer systems. DeepReview-Bench have introduced large-scale evaluation sets (e.g., 1,000+ samples), but their scope is largely restricted to a single venue (ICLR). RottenReviews [ebrahimi2025rottenreviews] and the focus-level framework of [focuslevel2025] study failure patterns and distributional biases in LLM reviews, but neither provides a reusable, per-review scoring protocol. [dycke2026automatic] focused on faults in reasoning.

**PRISM** departs from all prior frameworks by deploying dedicated, verifiable pipelines for each dimension---argument mining for depth, retrieval-augmented claim verification for novelty, consensus-weighted scoring for flaw identification, severity atomization for prioritization, and semantic rule matching for constructiveness---rather than relying on rubric-prompted LLM judging. In addition, PRISM benchmarks five leading automated reviewer systems across a diverse, stratified corpus of 1,000 papers spanning five venue-years (ICLR 2024--2026, ICML 2025, and NeurIPS 2025), and each pipeline is rigorously operationalized rather than superficially assessed.

---

## 3. The PRISM Framework

**PRISM** evaluates reviews across four independent pipelines designed to target the specific failure modes of LLMs in scientific discourse. Rather than asking an LLM judge for a holistic rating---which risks conflating stylistic fluency with scientific rigor---each of the pipelines in our framework decomposes the evaluation into structured evidence-extraction tasks: the LLM identifies and classifies discrete evidence units, while final scores are computed analytically. This approach ensures the evaluation is traceable and allows for precise control over metric formulation. The subsequent sections detail the computational formulations and workflows for each dimension.

[Figure: PRISM_overview.pdf]

---

### 3.1 Depth of Analysis

A high-quality review is characterized not only by the presence of critical claims, but also by the substantive evidence supporting them [hua-etal-2019-argument]. We define *Depth of Analysis* (DoA) as the degree to which a reviewer substantiates their judgments with objective, well-grounded premises: a shallow review relies on generic assertions, while a strong critique backs each argument with evidence.

**Pipeline.** We extract the core review sections (Summary, Strengths, Weaknesses) and break them into Argumentative Discourse Units (ADUs) [peldzusadu]. Each ADU is classified along two axes: (i) *argumentative role*---*Claim* (a point of contention or conclusion) or *Premise* (supporting evidence)---and (ii) *aspect topic* (Novelty, Methodology, Experiments, or Clarity). Identified premises are then assessed for *grounding level* $g(p) \in \{0,1,2\}$: Level 0 (Vague/Generic), Level 1 (Internal---references the manuscript directly), or Level 2 (External---references broader scientific literature).

**Score Formulation.** Let $A$ be the set of all ADUs, $P \subseteq A$ the subset classified as premises, and $g_{\max} = 2$ as the maximum grounding level. We define the *Premise Ratio* $R_{\mathrm{prem}} = |P|/|A|$ (evidence coverage) and the *normalized Average Grounding Score* $S_{\mathrm{depth}} = \frac{1}{g_{\max}|P|}\sum_{p \in P}g(p) \in [0,1]$ (evidence quality). DoA is defined as the harmonic mean:

$$
\mathrm{DoA} = \frac{2 \cdot R_{\mathrm{prem}} \cdot S_{\mathrm{depth}}}{R_{\mathrm{prem}} + S_{\mathrm{depth}}},
$$

which penalizes the imbalance: a review must excel in both the *proportion* and the *rigorousness* of its evidence to score highly. If $|P|=0$, DoA $=0$ by definition. Although aspect labels do not factor into the DoA score themselves, they reveal where reviewers direct their effort---toward substantive dimensions or surface-level concerns.

---

### 3.2 Novelty Assessment

In scientific peer review, novelty is the degree to which a paper introduces non-trivial findings---such as new ideas, methods, data, or perspectives---relative to existing knowledge [novelty1, novelty2, novelty3]. A genuine novelty judgment, therefore, requires situating the paper's claimed contributions within the prior literature. Our pipeline operationalizes this by verifying whether a reviewer's novelty comments are supported or refuted by retrievable prior work [zhang2026opennovelty].

**Pipeline.** The pipeline proceeds in three stages. ***Extraction***: a constrained LLM extracts the paper's core task, contribution anchors, and key terms, along with the set of verbatim novelty claims $\mathcal{C} = \{c_1,\ldots,c_n\}$ from the review. ***Retrieval***: we construct deterministic Semantic Scholar queries using the extracted anchors. Results are filtered for prior publications, duplication, and diversified via Maximal Marginal Relevance to form a candidate pool $\mathcal{B} = \{b_1,\ldots,b_k\}$. ***Verification***: for each claim-candidate pair $(c_i, b_j)$, an LLM judge compares the review claim against both the paper context (abstract + introduction) and the candidate's prior work (title + abstract). It returns a discrete evidence-support score $s(c_i, b_j) \in \{-2,-1,0,+1,+2\}$ ranging from *contradicted* to *fully supported*.

**Score Formulation.** Because each claim is evaluated against multiple candidates, we aggregate scores using a relevance-weighted top-3 policy ($\mathcal{T}_i$) rather than maximum pooling. This choice mitigates optimistic inflation from a single spuriously favorable match and better preserves the evidence ranking induced by retrieval. Let $r_j$ denote the retrieval relevance of candidate $b_j$; the per-claim score is

$$
S_{\mathrm{claim}}(c_i) = \frac{\sum_{j \in \mathcal{T}_i} s(c_i,b_j)\,r_j}{\sum_{j \in \mathcal{T}_i} r_j}.
$$

At the review level, we compute the mean claim score $\bar{S} = \frac{1}{n}\sum_{i=1}^{n}S_{\mathrm{claim}}(c_i)$ and derive three normalized metrics---

$$
NS(R) = \frac{\bar{S}+2}{4}, \quad
SR(R) = \frac{|\{c_i : S_{\mathrm{claim}}(c_i) \ge 1\}|}{n}, \quad
SSR(R) = \frac{|\{c_i : S_{\mathrm{claim}}(c_i) = 2\}|}{n},
$$

where $NS \in [0,1]$ is the overall normalized score, $SR$ and $SSR$ measure the fraction of claims with partial and strict literature support, respectively. Together, these metrics distinguish well-grounded critiques from partial matches or unsupported hallucinations.

---

### 3.3 Flaw Identification & Major Issues Prioritization

Effective peer review requires both accurate diagnosis of scientific errors and clear structural organization. We define *Flaw Identification* as the ability to detect genuine methodological weaknesses in a manuscript while filtering minor surface-level issues. Because the absolute number of flaws in any manuscript is unobservable, we establish a relative "ground truth" using a consensus mechanism that merges findings from both verified human and LLM reviewers. Furthermore, since authors prioritize issues encountered early in a reviewing text [NDCG], we treat the burial of critical flaws beneath trivial formatting complaints as a significant failure in review quality.

**Pipeline.** The pipeline proceeds in two stages. ***Extraction***: we isolate the critical review sections (Summary, Weaknesses, Questions) from both the human and LLM reviews; an LLM parses them concurrently to extract distinct flaw arguments---specific criticisms regarding the manuscript. ***Consensus Verification***: grounded in the actual paper context, an LLM judge evaluates all extracted flaws, discarding invalid or hallucinated critiques; verified findings from both reviewer types are merged into a consensus ground truth and classified by severity into *Critical* (e.g., methodological errors, flawed proofs) or *Minor* (e.g., typos, formatting issues). ***Positional Recovery***: valid flaws are mapped back to their original sequential position within the review text, forming the ranked ordering used to compute the prioritization score.

**Score Formulation.** We represent the consensus sets of Critical and Minor flaws as $F_{\mathrm{true}}^{C}$ and $F_{\mathrm{true}}^{M}$, respectively. The subsets of these valid flaws successfully identified by the reviewer under evaluation are denoted as $F_{\mathrm{rev}}^{C}$ and $F_{\mathrm{rev}}^{M}$. **Diagnostic coverage** is measured by severity-stratified recall:

$$
\text{Critical/Minor Recall} = \frac{|F_{\mathrm{true}}^{C/M} \cap F_{\mathrm{rev}}^{C/M}|}{|F_{\mathrm{true}}^{C/M}|}.
$$

**Structural ranking** quality is measured by the normalized Critique Prioritization Score ($nCPS$), inspired by NDCG [NDCG]. We assign severity weights $w_i \in \{2,1\}$ for Critical/Minor flaws and let $p_i$ be the position of the $i$-th valid flaw in the review:

$$
nCPS = \frac{CPS}{iCPS}, \quad
CPS = \sum_{i=1}^{k} \frac{w_i}{\log_2(p_i + 1)},
$$

where $iCPS$ is the ideal score (all Critical flaws preceding Minor), so an $nCPS$ approaches 1 indicates optimal prioritization.

---

### 3.4 Multi-Dimensional Constructiveness

While identifying flaws is essential, a review's real value lies in its ability to help authors improve. To measure this, we introduce the *Multi-Dimensional Constructiveness* metric, which quantifies the helpfulness of feedback. Grounded in discourse taxonomies like DISAPERE [kennard-etal-2022-disapere], our framework systematically decomposes constructiveness into informational and social dimensions.

**Pipeline.** An LLM judge first breaks the review into Atomic Review Comments (ARCs), the smallest independent units of critique or suggestion. Each ARC ($c_j$) is then rated on a scale from 0 to 2 across five dimensions: **Actionability ($D_1$):** does the comment provide clear, implementable guidance rather than vague opinions?; **Specificity ($D_2$):** does it pinpoint concrete elements, such as specific sections or equations?; **Justification ($D_3$):** are assertions backed by logical reasoning or empirical evidence?; **Solution ($D_4$):** does the reviewer propose a path for improvement instead of just highlighting a problem?; **Tone ($D_5$):** is the language professional and encouraging? This dimension penalizes hostility, which can demoralize authors without improving scientific quality [hyland2020antithetical, rao2022civility].

**Score Formulation.** For a review $R$ with $n$ ARCs $\{c_1,\ldots,c_n\}$, the Comment-Level Constructiveness $CLC(c_j) = \frac{1}{10}\sum_{k=1}^{5} D_k(c_j) \in [0,1]$ normalizes the five dimension scores, and the Mean Constructiveness Score $MCS(R) = \frac{1}{n}\sum_{j=1}^{n} CLC(c_j)$ averages over all comments. This formulation ensures that to achieve a perfect $MCS$ of $1.0$, a reviewer must consistently deliver specific, well-justified, actionable and professionally toned feedback across all constituent comments.

---

## 4. Experiment and Analysis

### 4.1 Evaluation Setting

**Dataset selection.** PRISM is evaluated on 200 manuscripts per venue-year across five conference splits---**ICLR 2024**, **ICLR 2025**, **ICLR 2026**, **ICML 2025**, and **NeurIPS 2025**---stratified by decision category (*Reject*, *Poster*, *Spotlight*, *Oral*) and topic. Sampling preserves each venue's original score distribution, ensuring the benchmark reflects natural acceptance dynamics while remaining tractable for end-to-end multi-system evaluation.

| Venue (Year) | Oral | Spotlight | Poster | Reject | Total |
|---|---|---|---|---|---|
| ICLR 2024 | 50 | 45 | 51 | 54 | 200 |
| ICLR 2025 | 29 | 37 | 62 | 72 | 200 |
| ICLR 2026 | 26 | - | 80 | 94 | 200 |
| ICML 2025 | 50 | 39 | 59 | 52 | 200 |
| NeurIPS 2025 | 50 | 50 | 50 | 50 | 200 |
| **Total** | 205 | 171 | 302 | 322 | **1000** |

[Figure: keyword_cloud.pdf]

**Reviewer baselines and implementations.** We evaluate five automated reviewer systems spanning two paradigms---*supervised fine-tuning* (SEA-E [yu-etal-2024-automated-SEA], CycleReviewer [cyclereviewer], DeepReview [zhu-etal-2025-deepreview]) and *prompting-based* (Reviewer2 [gao2024reviewer2optimizingreviewgeneration], TreeReview [chang-etal-2025-treereview])---and human reviewers.

**LLM-as-a-Judge implementation.** We adopt the LLM-as-a-Judge paradigm, using Gemini 2.5 Flash Lite [gemini2025team] as our evaluation engine for all metric extraction and scoring tasks.

### 4.2 Result Analysis: LLMs vs Human-Reviewer Baselines

| Baselines | Depth of Analysis | Novelty Assessment | Flaw Identification (Critical) | Flaw Identification (Minor) | Prioritization | Constructiveness |
|---|---|---|---|---|---|---|
| **Human** | $0.494 \pm 0.063$ | $0.787 \pm 0.199$ | $0.343 \pm 0.157$ | $0.281 \pm 0.078$ | $0.973 \pm 0.046$ | $0.566 \pm 0.066$ |
| CycleReviewer | $\mathbf{0.484} \pm 0.134$ | $0.784 \pm 0.212$ | $0.240 \pm 0.295$ | $0.186 \pm 0.140$ | $0.971 \pm 0.100$ | $0.527 \pm 0.111$ |
| DeepReview | $0.483 \pm 0.135$ | $0.759 \pm 0.209$ | $0.332 \pm 0.298$ | $0.228 \pm 0.147$ | $0.967 \pm 0.072$ | $\mathbf{0.634} \pm 0.086$ |
| Reviewer2 | $0.377 \pm 0.129$ | $0.787 \pm 0.218$ | $\mathbf{0.591} \pm 0.297$ | $\mathbf{0.459} \pm 0.177$ | $0.975 \pm 0.040$ | $0.575 \pm 0.104$ |
| SEA | $0.389 \pm 0.154$ | $\mathbf{0.833} \pm 0.203$ | $0.222 \pm 0.257$ | $0.247 \pm 0.127$ | $\mathbf{0.977} \pm 0.070$ | $0.498 \pm 0.091$ |
| TreeReview | $0.359 \pm 0.122$ | $0.811 \pm 0.201$ | $0.272 \pm 0.294$ | $0.332 \pm 0.148$ | $0.972 \pm 0.061$ | $0.485 \pm 0.122$ |

Table: Macro-Average Performance Across 5 Conferences compared to Human.

### 4.3 Depth of Analysis

| System | Prem. Ratio | Grounding |
|---|---|---|
| **Human** | $0.567 \pm 0.092$ | $0.475 \pm 0.065$ |
| CycleReviewer | $\mathbf{0.614} \pm 0.186$ | $0.438 \pm 0.136$ |
| DeepReview | $0.596 \pm 0.177$ | $0.444 \pm 0.137$ |
| Reviewer2 | $0.398 \pm 0.181$ | $0.431 \pm 0.141$ |
| SEA | $0.381 \pm 0.209$ | $\mathbf{0.462} \pm 0.126$ |
| TreeReview | $0.333 \pm 0.148$ | $0.450 \pm 0.114$ |

[Figure: doa_heatmap_all.pdf]

The human ground-truth establishes the benchmark with the highest overall DoA score ($0.494$). Among the automated systems, **DeepReview** ($0.483$) and **CycleReviewer** ($0.484$) closely match the human standard. Their good performance is primarily driven by a robust *Premise Ratio* ($\approx 0.60$), meaning they consistently substantiate their claims, successfully compensating for the slight gap in absolute Grounding scores.

While Grounding scores remain consistent across humans and LLMs ($0.431$--$0.475$), the DoA disparity is primarily driven by the Premise Ratio. While baselines like TreeReview fall short, CycleReviewer ($0.614$) and DeepReview ($0.596$) successfully close the gap by matching or exceeding the human baseline ($0.567$) in consistently substantiating their claims. Furthermore, aspect distributions show that cognitive alignment is heavily architecture-dependent. Advanced pipelines (DeepReview, CycleReviewer, Reviewer2, SEA) mirror human intuitive focus by dedicating the vast majority of their grounded premises to Methodology and Experimental Design, while keeping *Clarity* strictly proportional to human levels ($\sim 7-12\%$). By contrast, TreeReview disproportionately squanders $\sim 24\%$ of its overall effort on formatting issues at the expense of methodological rigor---a degradation in evaluative depth recently observed in in-the-wild LLM peer reviews [llmsurfacebias]. With these results, the "surface-level trap" is thus not an inherent LLM flaw, but rather an artifact of reasoning frameworks that lack explicit, domain-specific constraints.

***Key Insight:** Human reviewers's analytical depth has both a high Premise Ratio and cognitive alignment that prioritizes core methodology over surface-level formatting. To perform comparably to human reviewers, the best-performing LLMs primarily rely on generating highly robust premises, effectively using structural completeness to compensate for their slight gaps in empirical grounding.*

### 4.4 Novelty Assessment

In contrast to the human-dominated Depth of Analysis, Novelty Assessment yields uniformly high evidence-grounding scores across automated baselines. All automated systems operate within the $0.750$ to $0.830$ range, meaning that many of their extracted novelty claims can be matched to supportive prior-work evidence under the PRISM retrieval-and-verification pipeline. Importantly, this metric does not certify the manuscript's objective novelty or full human-level agreement; it measures how well the claims a reviewer chose to make are grounded in retrieved literature. Accordingly, a review can score highly on Novelty Assessment while still differing from human reviewers in claim selection, evidence choice, or calibration. Within this evidence-grounding perspective, **SEA** achieves the highest macro-average score of $\mathbf{0.833}$, slightly above the human baseline ($0.787$), suggesting that structured prompting helps models articulate novelty claims that are retrievably justifiable.

[Figure: novelty_detailed_analysis_a.pdf]
[Figure: novelty_detailed_analysis_b.pdf]

Review systems diverge considerably in their novelty stance. SEA endorses novelty in 79% of claims---far above the human rate of 59%---reflecting a tendency to agree with authors rather than scrutinize their contributions. In contrast, DeepReview adopts the most skeptical lens (39% *Novel*, 33% *Not novel*), suggesting its multi-step reasoning positively searches for counter-evidence. In parallel, a consistent cross-reviewer pattern emerges: claims labeled *Not novel* or *Somewhat novel* attract markedly stronger literature groundings, compared with *Novel* claims. This aligns well with a natural reviewing dynamic---*a reviewer who challenges authors' novelty statements would cite prior works to substantiate that critique, whereas agreements would require little external justification*. Importantly, the pattern holds consistently across reviewer pipelines and human, confirming it is an intrinsic property of the reviewing task itself, rather than an LLM artifact.

***Key Insight:** While automated reviewers back their novelty claims with solid evidence, this reflects a tendency to select easily verifiable claims rather than true human-level judgment. Additionally, both models and humans follow a natural reviewing pattern: negative novelty judgments are consistently backed by much stronger evidence than positive ones.*

### 4.5 Flaw Identification & Major Issues Prioritization

[Figure: flaw_diverging.pdf]
[Figure: constructiveness_parallel.pdf]

Table 1 reveals distinct specialization profiles in diagnostic precision. **Reviewer2** stands out as an exhaustive flaw scanner, achieving the highest recall for both Critical ($\mathbf{0.591}$) and Minor ($\mathbf{0.459}$) issues---substantially exceeding the human baseline ($0.343$ and $0.281$, respectively). This suggests that structured LLM pipelines can systematically surface vulnerabilities that time-constrained human reviewers may overlook. By contrast, **DeepReview** and the Human baseline maintain more conservative, targeted diagnostic patterns, trading raw recall for precision.

[Figure: flaw_diverging.pdf] contextualizes raw recall by decomposing extracted flaws into valid and hallucinated counts. Reviewer2 recovers an exceptionally high volume of valid flaws at a low hallucination rate (${\sim}3.3\%$), while CycleReviewer's high hallucination rate (${\sim}18.5\%$) signals a fundamental precision deficit. Critically, hallucinations are strictly confined to minor issues across every system: no reviewer---human or LLM---fabricates a fatal methodological breakdown, ensuring that Critical flaw flags remain factually grounded. Complementary aspect-level analysis further shows that both LLMs and humans dynamically adapt their diagnostic focus by severity---concentrating on core methodology for Critical flaws while shifting toward presentation and clarity for Minor anomalies.

Notably, all systems---including humans---achieve near-identical nCPS scores ($\approx 0.97$), suggesting that prioritization of critical over minor flaws may reflect a near-universal baseline behavior rather than a discriminating capability at current performance levels.

***Key Insight:** Certain LLMs act as high-sensitivity scanners, catching more critical flaws than human reviewers. However, structuring a review by severity (putting critical issues first) is a standard behavior across all evaluated systems and humans, not a unique advantage of any single model.*

### 4.6 Multi-Dimensional Constructiveness

The Multi-Dimensional Constructiveness Score evaluation reveals that LLMs can emulate, and in some cases exceed, the professional and supportive tone expected in academic peer review. While human reviewers establish a solid constructiveness baseline of $0.566$, **DeepReview** significantly outperforms both human reviewers and other LLMs, achieving the highest score of $\mathbf{0.634}$. This suggests that DeepReview's multi-stage reasoning pipeline is exceptionally effective at not only identifying weaknesses but also formulating specific, actionable and professionally communicated suggestions for author improvement.

The sub-dimension decomposition reveals an intriguing divergence. Both humans ($1.725$) and **CycleReviewer** ($1.897$) excel at *Specificity* (D2), yet human reviewers show a surprising shortfall in *Solution* provision (D4 $= 0.470$)---they identify problems but rarely propose fixes. **DeepReview** fills this gap most convincingly, leading on both *Actionability* (D1 $= 1.414$) and *Solution* (D4 $= 0.784$): it does not merely flag issues but formulates explicit, implementable improvements. **Reviewer2**'s elevated *Justification* score (D3 $= 0.939$) may partly reflect its verbose style rather than genuine reasoning depth, as its low *Solution* rate (D4 $= 0.266$) leaves critiques largely unactionable. On *Tone* (D5), LLMs generally stay neutral-to-encouraging; DeepReview ($1.726$) is the most professional, avoiding the dismissive register of some humans.

***Key Insight:** Helpful feedback does not emerge automatically from LLMs; it requires specific system design. Purpose-built pipelines (like DeepReview) go beyond simply pointing out errors to offer actionable, professional solutions---a level of constructive feedback that standard models and even human reviewers rarely provide.*

---

## 5. Conclusion & Future Work

PRISM demonstrates that LLM peer reviewers are specialized tools rather than general-purpose replacements for human expertise. Each system excels in a specific niche but exhibits distinct blind spots across other dimensions.

**Actionable deployment recommendations.** Since no single system dominates all four dimensions, we recommend a targeted ensemble deployment rather than a standalone approach: use **Reviewer2** for exhaustive flaw scanning (highest diagnostic recall); use **DeepReview** for constructive feedback drafting (highest actionability and solution density); use **SEA** for novelty-grounding checks (highest literature support rate). Ultimately, these systems are most effective as specialist co-pilots within a human-assisted pipeline rather than autonomous reviewers.

**Limitations.** Our primary evaluation pipeline relies on `Gemini 2.5 Flash Lite` as the core judge model. While we conducted preliminary robustness checks using an alternative model (Xiaomi `MiMo V2.5 Pro` [mimo2026v25pro]) on a subset of the data to verify metric stability, a comprehensive multi-judge study across diverse LLM families remains necessary to fully eliminate judge-specific biases. Furthermore, the benchmark corpus covers ML/AI venues only, and PRISM may require recalibration for other scientific domains.

**Future work.** We identify three priority directions: (1) *Cross-domain generalization*---recalibrating PRISM for clinical medicine, social sciences, and pure mathematics. (2) *Judge robustness*---systematic study of inter-judge agreement across LLM judge families and human raters. (3) *Human validation*---correlating PRISM scores with post-review author satisfaction or acceptance decision outcomes to confirm that the metrics capture meaningful review quality.

---

## Acknowledgment

This research is funded by CAIR, College of Engineering & Computer Science, VinUniversity, Hanoi, Vietnam. The work of Duy A. Nguyen was supported in part by a PhD fellowship from the VinUni-Illinois Smart Health Center, VinUniversity, Hanoi, Vietnam.

---

## References

1. Paper Copilot. NeurIPS 2024 Statistics. 2024. [neurips2024]
2. Communications Chairs 2025. Reflections on the 2025 review process from the program committee chairs. 2025. [neurips2025blog]
3. Paper Copilot. ICML 2023 statistics. 2023. [icml2023stats]
4. Paper Copilot. ICML 2024 statistics. 2024. [icml2024stats]
5. Paper Copilot. ICML 2025 statistics. 2025. [icml2025stats]
6. Weijie Su and Buxin Su. Introducing ICML 2026 policy for self-ranking in reviews. January 2026. [icml2026policy_blog]
7. Yuan Chang et al. TreeReview: A dynamic tree of questions framework for deep and efficient LLM-based scientific peer review. EMNLP 2025. [chang-etal-2025-treereview]
8. Zhaolin Gao et al. Reviewer2: Optimizing review generation through prompt generation. 2024. [gao2024reviewer2optimizingreviewgeneration]
9. Jianxiang Yu et al. Automated peer reviewing in paper SEA: Standardization, evaluation, and analysis. EMNLP 2024 Findings. [yu-etal-2024-automated-SEA]
10. Minjun Zhu et al. DeepReview: Improving LLM-based paper review with human-like deep thinking process. ACL 2025. [zhu-etal-2025-deepreview]
11. Yixuan Weng et al. Cycleresearcher: Improving automated research via automated review. ICLR 2025. [cyclereviewer]
12. Weixin Liang et al. Can large language models provide useful feedback on research papers? A large-scale empirical analysis. NEJM AI, 2024. [liang2023largelanguagemodelsprovide]
13. Lei Wang et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 2024. [Wang_2024]
14. ICLR 2026 Program Chairs. Policies on large language model usage at ICLR 2026. 2025. [iclr2026policy]
15. ICML. ICML 2026 policy for LLM use in reviewing. 2026. [icml2026policy]
16. Alina Beygelzimer et al. Has the machine learning review process become more arbitrary as the field has grown? The NeurIPS 2021 consistency experiment. arXiv, 2023. [beygelzimer2023neurips]
17. Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. ACL 2004. [lin-2004-rouge]
18. Kishore Papineni et al. BLEU: a method for automatic evaluation of machine translation. ACL 2002. [papineni-etal-2002-bleu]
19. Jekaterina Novikova et al. Why we need new evaluation metrics for NLG. EMNLP 2017. [novikova-etal-2017-need]
20. Yang Liu et al. G-Eval: NLG evaluation using GPT-4 with better human alignment. EMNLP 2023. [liu-etal-2023-g]
21. Lianmin Zheng et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS 2023. [zheng2023judging]
22. Chujie Zheng et al. Large language models are not robust multiple choice selectors. ICLR 2024. [zheng2023large]
23. Keita Saito et al. Verbosity bias in preference labeling by large language models. NeurIPS 2023 Workshop. [saito2023verbosity]
24. Arjun Panickssery et al. LLM evaluators recognize and favor their own generations. NeurIPS 2024. [panickssery2024llm]
25. Madhav Krishan Garg et al. ReviewEval: An evaluation framework for AI-generated reviews. EMNLP 2025 Findings. [garg-etal-2025-revieweval]
26. Sajad Ebrahimi et al. Rottenreviews: Benchmarking review quality with human and LLM-based judgments. CIKM 2025. [ebrahimi2025rottenreviews]
27. Hyungyu Shin et al. Mind the blind spots: A focus-level evaluation framework for LLM reviews. EMNLP 2025. [focuslevel2025]
28. Nils Dycke and Iryna Gurevych. Automatic reviewers fail to detect faulty reasoning in research papers. arXiv, 2025. [dycke2026automatic]
29. Xinyu Hua et al. Argument mining for understanding peer reviews. NAACL 2019. [hua-etal-2019-argument]
30. Andreas Peldszus and Manfred Stede. From argument diagrams to argumentation mining in texts: A survey. Int. J. Cogn. Inform. Nat. Intell., 2013. [peldzusadu]
31. Peter P. Morgan. Originality, novelty and priority: Three words to reckon with in scientific publishing. Canadian Medical Association Journal, 1985. [novelty1]
32. Shubhanshu Mishra and Vetle I. Torvik. Quantifying conceptual novelty in the biomedical literature. D-Lib Magazine, 2016. [novelty2]
33. Yi Zhao and Chengzhi Zhang. A review on the novelty measurements of academic papers. Scientometrics, 2025. [novelty3]
34. Ming Zhang et al. OpenNovelty: An LLM-powered agentic system for verifiable scholarly novelty assessment. arXiv, 2026. [zhang2026opennovelty]
35. Kalervo Jarvelin and Jaana Kekalainen. Cumulated gain-based evaluation of IR techniques. ACM TOIS, 2002. [NDCG]
36. Neha Nayak Kennard et al. DISAPERE: A dataset for discourse structure in peer review discussions. NAACL 2022. [kennard-etal-2022-disapere]
37. Ken Hyland and Feng Kevin Jiang. "This work is antithetical to the spirit of research": An anatomy of harsh peer reviews. Journal of English for Academic Purposes, 2020. [hyland2020antithetical]
38. Rahul Tony Rao and Beth Bareham. Regression towards the mean---a plea for civility in peer review. BMJ, 2022. [rao2022civility]
39. Gheorghe Comanici et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. 2025. [gemini2025team]
40. Wenqing Wu et al. Impact of large language models on peer review opinions from a fine-grained perspective. arXiv, 2026. [llmsurfacebias]
41. Xiaomi MiMo Team. MiMo-V2.5-Pro. 2026. [mimo2026v25pro]
42. Jaime Carbonell and Jade Goldstein. The use of MMR, diversity-based reranking for reordering documents and producing summaries. SIGIR 1998. [carbonell1998use]
43. Jacob Cohen. Statistical Power Analysis for the Behavioral Sciences. 2nd edition, 1988. [cohen1988statistical]

---

## Appendix A: Formal Problem Definition

The fundamental challenge in benchmarking automated peer reviewers lies in the highly subjective, domain-specific, and unstructured nature of scientific critiques. While existing literature often treats LLMs as either pure text generators or generic evaluators, assessing a scientific peer review requires measuring cognitive depth rather than mere linguistic fluency.

To systematically evaluate this, we formalize the peer review benchmarking process. Let $P$ denote a submitted scientific manuscript. In our setting, an LLM-based reviewer baseline $\mathcal{M}$ processes $P$ to generate an automated review, denoted as $R_{LLM} = \mathcal{M}(P)$. Simultaneously, we possess a high-quality human expert review $R_{human}$ corresponding to the same manuscript $P$, which serves as our reference.

The central problem addressed in this work is to construct a multi-dimensional evaluation function, denoted as $\mathcal{E}$. Rather than relying on superficial n-gram matching metrics (like ROUGE) or unconstrained prompting, our framework requires $\mathcal{E}$ to process the generated review, the human reference, and the original paper to output a comprehensive capability profile:

$$ \mathcal{S} = \mathcal{E}(R_{LLM}, R_{human}, P) $$

where $\mathcal{S}$ represents a set of quantitative scores spanning diverse cognitive aspects. The goal of our benchmarking protocol is to design $\mathcal{E}$ such that it accurately measures the analytical gap between $R_{LLM}$ and $R_{human}$, specifically penalizing superficial summarization, hallucinated flaws, ungrounded novelty claims and un-actionable feedback.

---

## Appendix B: Experimental Details

### B.1 Dataset Selection

We evaluate PRISM on a stratified benchmark drawn from five major conference splits: **ICLR 2024**, **ICLR 2025**, **ICLR 2026**, **ICML 2025**, and **NeurIPS 2025**. For each venue-year, we construct a representative subset of exactly 200 manuscripts, stratified across their final decision categories (*Reject*, *Poster*, *Spotlight*, *Oral*) and cover various topics. During the sampling process, we strictly preserve the original score distribution of a full conference pool. As a result, the number of papers within each decision tier organically reflects the natural acceptance dynamics and quality distribution of each specific venue. This approach ensures comprehensive outcome coverage and high-fidelity review-quality diversity, while keeping the benchmark tractable for end-to-end multi-system evaluation.

### B.2 Reviewer Baselines and Implementations

#### Taxonomy of Baseline LLM Reviewers

We compare human reviews against five automated reviewer systems: **SEA-E**, **DeepReview**, **Reviewer2**, **CycleReviewer**, and **TreeReview**. These systems span two broad paradigms.

**Supervised fine-tuning methods.** SEA-E [yu-etal-2024-automated-SEA] is a structured evaluation model trained to output review components such as summaries, strengths, weaknesses, and questions. CycleReviewer [cyclereviewer] is optimized through an iterative preference-based training framework in which a reviewer model is progressively refined from win/lose comparisons. DeepReview [zhu-etal-2025-deepreview] uses a multi-stage reasoning pipeline that explicitly models multi-perspective analysis, and reliability checking before producing a final review.

**Prompting-based methods.** Reviewer2 [gao2024reviewer2optimizingreviewgeneration] is based on a two-stage rubric-driven process that first generates aspect-specific questions and then answers them to synthesize the final review. TreeReview [chang-etal-2025-treereview] follows a hierarchical reasoning strategy, decomposing the review into a tree of sub-questions and aggregating leaf-level evidence into a complete critique. In our experiments, prompting-based baselines are executed under a standardized backbone configuration to isolate the effect of the prompting framework rather than confounding it with backbone choice.

#### Baseline Implementation and Configuration

**SEA-E** operates as a structured evaluation model, utilizing the model *ECNU-SEA/SEA-E* to generate comprehensive review components such as summaries, strengths, weaknesses, and numerical ratings. To accommodate full-length manuscripts, the engine is configured with a 70,000-token context window. The pipeline processes a batch size of 4 papers simultaneously, generating the final critique with the maximum output length capped at 8,000 tokens. To ensure a balance between analytical diversity and factual coherence, the generation hyperparameters are strictly configured with a temperature of 0.7 and a top-p of 0.9.

**CycleReviewer** utilizes a model *WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B* optimized through an iterative preference-based reasoning framework. All inference workloads are executed on NVIDIA RTX A5000 GPUs. The model employs a 24,000-token context window to accommodate and process complete manuscript texts. For each manuscript, the system executes 2 to 3 iterative refinement passes to progressively enhance the review quality. The 8B engine operates on a single GPU configuration. It generates critiques with a maximum generation length of 3,000 tokens. Generation hyperparameters are configured with a temperature of 0.7, top-p of 0.9, top-k of 50, and a repetition penalty of 1.2.

**DeepReview** utilizes the *WestlakeNLP/DeepReviewer-14B* core reasoning engine alongside a retrieval-augmented subsystem powered by OpenScholar. All inference workloads are executed on NVIDIA RTX A5000 GPUs. Aligning with the original architecture, the retrieval module employs *Llama-3.1-OpenScholar-8B* (configured with a 70,000-token context limit) for evidence synthesis and *Qwen-2.5-3B-Instruct* (configured with a 10,000-token context limit) for query processing. For each manuscript, the system transforms generated questions into search keywords to retrieve approximately 30 candidate papers, utilizing a dedicated reranking model to select the top 10 most relevant sources for grounding. The core 14B engine operates across two GPUs via tensor parallelism. It processes a batch size of 8 papers with a maximum generation length of 7,000 tokens. Generation hyperparameters are configured with a temperature of 0.8, top-p of 0.9, top-k of 50, and a repetition penalty of 1.2.

**Reviewer2** framework originally operates on a two-stage prompting methodology utilizing custom checkpoints (*GitBag/Reviewer2_Mp* and *GitBag/Reviewer2_Mr*). However, due to the suboptimal generation quality observed from these native models during our preliminary evaluations, we replace the underlying generation engine with the open-weights *Qwen-3.5-14B* model. Crucially, we strictly retain Reviewer2's official prompt templates to preserve the methodological integrity of their two-stage pipeline. All inference workloads are executed on NVIDIA RTX A5000 GPUs. To process extensive manuscripts, the engine is configured with an 80,000-token context window. The pipeline processes a batch size of 4 papers simultaneously. During Phase 1, the model generates specific review questions, and in Phase 2, it synthesizes the final critique based on these generated prompts, with the maximum generation length capped at 7,000 tokens. Across both stages, the generation hyperparameters are strictly configured with a temperature of 0.8, top-p of 0.9, top-k of 50, and a repetition penalty of 1.2.

**TreeReview** models the peer review process as a hierarchical and bidirectional question-answering framework. While the original implementation utilizes GPT-4o, we standardized the backbone to *Qwen3-14B* to ensure a fair comparison across all prompting-based baselines. All inference workloads are executed on NVIDIA RTX A5000 GPUs. To accommodate the full paper text alongside the dynamically expanding tree of sub-questions, the engine is configured with an 80,000-token context window. The pipeline processes a batch size of 4 papers simultaneously. Following its core logic, the system recursively decomposes high-level review objectives into fine-grained sub-questions and aggregates answers from leaf to root to synthesize the final critique, with the maximum output length capped at 7,000 tokens. Across all reasoning stages, the generation hyperparameters are strictly set to a temperature of 0.8, top-p of 0.9, top-k of 50, and a repetition penalty of 1.2.

### B.3 Review Generation Process

Before applying our evaluation framework, we must first generate the corresponding AI reviews. For each of the 1,000 papers in our dataset, we provide the complete textual content---including all sections and tables represented in text form---to all five LLM reviewer baselines. Figures and other visual elements are excluded, as the LLM reviewers considered in this study do not yet reliably support multimodal (vision-language) understanding. Each model then independently generates a complete peer review based on its respective methodology. This process yields a comprehensive corpus of 5,000 automated reviews, which serves as the primary testbed for all subsequent PRISM evaluations.

---

## Appendix C: PRISM Evaluation Framework: Pipeline Details

[Figure: detailed_flow.pdf]

### C.1 PRISM Judge Setup

To compute the diverse evaluation metrics defined in our framework PRISM, we adopt the LLM-as-a-Judge paradigm. We deploy **Gemini 2.5 Flash Lite** (`gemini-2.5-flash-lite`) [gemini2025team] as the core evaluation engine for all metric extraction and scoring tasks. To ensure strict reproducibility and minimize generation variance, we explicitly configure the model parameters by setting the temperature to 0.0 and top-$p$ to 0.95, without utilizing top-$k$ sampling. During the evaluation phase, the model is strictly prompted with our standardized rubrics to systematically extract arguments, verify flaw validity against the ground truth, and compute component scores for both human and automated reviewers equally.

### C.2 Running Example: Depth of Analysis

To illustrate the full Depth of Analysis pipeline, consider the following excerpt from a raw review: *"3 seeds is too few to get any statistical confidence, especially without doing independent hyperparameter sweeps for each baseline. While in the past this has been standard, as a field we continually have shown that the statistical power of our experiments are laughably poor. The performance of the proposed goal-conditioned RL algorithm on the most challenging tasks was less than 50%. QRL assumes deterministic dynamics of the environment, while TD InfoNCE learns without such assumption."*

Processing this text through our three-phase framework yields the following structured output:

- **Claim:** *"3 seeds is too few to get any statistical confidence..."* $\rightarrow$ **Aspect:** Experimental Design & Evaluation.
- **Premise 1:** *"While in the past this has been standard, as a field we continually have shown that the statistical power of our experiments are laughably poor."* $\rightarrow$ **Aspect:** Experimental Design & Evaluation. $\rightarrow$ **Grounding Score: 0** (Generic/Vague).
- **Premise 2:** *"The performance of the proposed goal-conditioned RL algorithm on the most challenging tasks was less than 50%."* $\rightarrow$ **Aspect:** Experimental Design & Evaluation. $\rightarrow$ **Grounding Score: 1** (Internal).
- **Premise 3:** *"QRL assumes deterministic dynamics of the environment, while TD InfoNCE learns without such assumption."* $\rightarrow$ **Aspect:** Methodology & Theoretical Soundness. $\rightarrow$ **Grounding Score: 2** (External/Comparative).

The overall DoA score is defined as the harmonic mean of the Premise Ratio and the Normalized Grounding Score:

$$DoA = 2 \times \frac{R_{premise} \times S_{grounding}}{R_{premise} + S_{grounding}}$$

*Calculation for the Running Example:* In the excerpt above, the pipeline extracted a total of $4$ ADUs ($1$ Claim and $3$ Premises). The premises received grounding scores of $0$, $1$, and $2$.

$$R_{premise} = \frac{3}{4} = 0.75$$
$$S_{grounding} = \frac{0 + 1 + 2}{3 \times 2} = 0.5$$
$$DoA = 2 \times \frac{0.75 \times 0.5}{0.75 + 0.5} = 0.6$$

### C.3 Running Example: Novelty Assessment

To illustrate the full Novelty Assessment pipeline, consider a human review of a NeurIPS 2025 oral paper, *Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation*. Phase 1 extracts three novelty claims from distinct reviewers:

- **C1** (*not_novel*, from Reviewer 1's Weaknesses): *"The proposed method appears incremental, as the techniques involving attention weighting and logits fusion are already well-known and mainly borrowed from previous works."*
- **C2** (*novel*, from Reviewer 2's Strengths): *"The proposed two modules, attention reallocation and adaptive logits fusion, offer a novel and effective perspective to enhance MLLM performance in knowledge-intensive tasks."*
- **C3** (*somewhat_novel*, from Reviewer 4's Originality assessment): *"The paper proposes a novel approach to reallocate attentions and fuse knowledge, although similar ideas for attention reallocations have been introduced by earlier work."*

Phase 2 issues structured queries to the Semantic Scholar API, retrieving 20 candidate related works. The three most relevant per the top-3 relevance-weighted aggregation policy are: (RW1) MambaTrans: Multimodal Fusion Image Translation via LLM Priors; (RW2) Can Multimodal LLMs be Guided to Improve Industrial Anomaly Detection?; and (RW3) CAT+: Investigating and Enhancing Audio-Visual Understanding in LLMs.

Phase 3 evaluates each (claim, related-work) pair:

- **C1** (*not_novel*): RW1 is off-topic - Unsupported ($-2$). RW2 corroborates that attention weighting is a known technique - Supported ($+2$). RW3 similarly confirms these are established techniques - Supported ($+2$).
- **C2** (*novel*): RW1 again contains no information about the paper - Unsupported ($-2$). RW2 discusses a novel multi-expert framework for MLLM tasks - Supported ($+2$). RW3 aligns with the novelty claim - Supported ($+2$).
- **C3** (*somewhat_novel*): All three related works corroborate the nuanced stance - Supported ($+2$) for all.

The Novelty Verification Score is the mean aggregated score over all $K$ novelty claims:

$$\bar{s}(R) = \frac{1}{K} \sum_{k=1}^{K} s_k, \quad s_k = \sum_{j=1}^{3} w_j \cdot v_{k,j}, \quad NS(R) = \frac{\bar{s}(R) + 2}{4}$$

*Calculation:* With $K = 3$ claims and equal relevance weights ($w_j = 1/3$):

$$s_{C_1} = \frac{(-2) + 2 + 2}{3} = 0.667, \quad s_{C_2} = \frac{(-2) + 2 + 2}{3} = 0.667, \quad s_{C_3} = \frac{2 + 2 + 2}{3} = 2.0$$

$$\bar{s}(R) = \frac{0.667 + 0.667 + 2.0}{3} = 1.111, \quad NS(R) = \frac{1.111 + 2}{4} = \mathbf{0.778}$$

This example illustrates three key behaviors. First, the *somewhat_novel* claim (C3) achieves the highest per-claim score because its nuanced stance is confirmed by all retrieved evidence. Second, the *not_novel* (C1) and *novel* (C2) claims receive identical aggregated scores ($0.667$) despite opposing stances, because the same off-topic related work penalizes both equally. Third, the overall normalized score $NS(R)=0.778$ reflects a review whose novelty assessments are partially well-grounded but sensitive to the composition of the retrieval pool.

### C.4 Running Example: Flaw Identification & Major Issues Prioritization

To illustrate both the Flaw Identification and Prioritization pipelines, consider a reviewer evaluating a graph neural network paper. The **Ground Truth** flaw set, defined as the union of all valid flaws identified across all reviewers for this paper, consists of:

- **Critical flaws (GT):** FC1: *"Missing comparison to GraphSAGE baseline in Table 2."* FC2: *"Convergence proof contains a gap in Lemma 3: the Lipschitz assumption is invoked but never verified."* FC3: *"No ablation study on the effect of message-passing depth."*
- **Minor flaws (GT):** FM1: *"Inconsistent notation: A and $\tilde{A}$ used interchangeably in Eq. 4 and Eq. 5."* FM2: *"Figure 2 axes are unlabeled."* FM3: *"Related work omits Liu et al. (2022)."*

Reviewer $X$ produces the following flaw list, in the order they appear in the review:

1. **[Minor]** FM1 --- inconsistent notation $\checkmark$
2. **[Critical]** FC1 --- missing GraphSAGE comparison $\checkmark$
3. **[Minor]** FM2 --- Figure 2 axes unlabeled $\checkmark$
4. **[Critical]** FC2 --- convergence proof gap $\checkmark$

*Calculation for Flaw Identification:*

Reviewer $X$ matched 2 out of 3 critical flaws (missed FC3) and 2 out of 3 minor flaws (missed FM3):

$$\text{Critical Recall} = \frac{2}{3} \approx 0.667, \quad \text{Minor Recall} = \frac{2}{3} \approx 0.667$$

*Calculation for Major Issue Prioritization (nCPS):*

The $nCPS$ is computed over all $k$ GT-matched valid flaws identified by Reviewer $X$, ordered by their position of appearance in the review. The GT-matched valid flaws are: FM1 (Minor, position 1), FC1 (Critical, position 2), FM2 (Minor, position 3), FC2 (Critical, position 4).

$$CPS = \frac{1}{\log_2 2} + \frac{2}{\log_2 3} + \frac{1}{\log_2 4} + \frac{2}{\log_2 5} = 1.000 + 1.262 + 0.500 + 0.861 = 3.623$$

The ideal score $iCPS$ places all Critical flaws first, then Minor:

$$iCPS = \frac{2}{\log_2 2} + \frac{2}{\log_2 3} + \frac{1}{\log_2 4} + \frac{1}{\log_2 5} = 2.000 + 1.262 + 0.500 + 0.431 = 4.193$$

$$nCPS = \frac{3.623}{4.193} \approx \mathbf{0.864}$$

### C.5 Running Example: Multi-Dimensional Constructiveness

To illustrate the MCS pipeline, consider the following excerpt from a human review of a theoretical machine learning paper:
*"The paper lacks a clear comparison of its theoretical results (Table 1, Section 5) with prior related work. No experimental results. The toy example should correspond to the motivation example. Provide concrete toy examples illustrating setup and theorems, including specific distributions and query complexity bounds. A detailed comparison to existing results in the bandits literature is needed."*

Processing this text through Gemini yields the following Atomic Review Comments (ARCs):

- **ARC 1 (Weakness):** *"Lacks clear comparison of theoretical results (Table 1, Sec. 5) with prior related work."* $\rightarrow$ **Scores:** D1=1, D2=2, D3=1, D4=0, D5=1.
- **ARC 2 (Weakness):** *"No experimental results to validate theoretical findings."* $\rightarrow$ **Scores:** D1=2, D2=2, D3=0, D4=1, D5=1.
- **ARC 3 (Question):** *"Provide concrete toy examples with distributions and query complexity bounds."* $\rightarrow$ **Scores:** D1=2, D2=2, D3=0, D4=1, D5=2.
- **ARC 4 (Weakness):** *"Detailed comparison to existing bandit results needed."* $\rightarrow$ **Scores:** D1=2, D2=2, D3=0, D4=0, D5=1.

Aggregating dimension-wise over these 4 ARCs:

$$\overline{D_1} = 1.75, \quad \overline{D_2} = 2.00, \quad \overline{D_3} = 0.25, \quad \overline{D_4} = 0.50, \quad \overline{D_5} = 1.25$$

$$\text{MCS} = \frac{1.75 + 2.00 + 0.25 + 0.50 + 1.25}{10} = \mathbf{0.575}$$

---

## Appendix D: Metric Independence Analysis via Pearson Correlation

### D.1 Motivation and Objective

A key requirement for a multi-dimensional evaluation benchmark is that its constituent metrics should capture *distinct, non-overlapping* aspects of review quality. If two metrics were highly correlated, they would convey redundant information and effectively reduce the dimensionality of the evaluation, undermining the claim that different facets of peer review are independently assessed. To verify this property, we conduct a pairwise Pearson correlation analysis across the five evaluation dimensions of our benchmark: Depth of Analysis (DoA), Novelty Assessment (NS), Flaw Identification (Critical Recall, Minor Recall), Issue Prioritization (nCPS) and Multi-dimensional Constructiveness (MCS).

### D.2 Statistical Method

For two metric vectors $\mathbf{x} = (x_1, \ldots, x_n)$ and $\mathbf{y} = (y_1, \ldots, y_n)$ measured over $n$ paper-review samples, the Pearson correlation coefficient is defined as:

$$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

To assess whether an observed $r_{xy}$ differs significantly from zero, we apply the two-tailed *t*-test under $H_0: \rho = 0$. The test statistic is $t = r_{xy}\sqrt{n - 2}/\sqrt{1 - r_{xy}^2}$, which follows a Student's $t$-distribution with $n - 2$ degrees of freedom. Statistical significance alone is insufficient because large samples can render even trivially small correlations significant. We therefore assess the *practical magnitude* of each $|r_{xy}|$ using conventional thresholds [cohen1988statistical]: $|r| < 0.10$ (negligible), $0.10 \leq |r| < 0.30$ (small), $0.30 \leq |r| < 0.50$ (moderate) and $|r| \geq 0.50$ (large).

### D.3 Results and Discussion

[Figure: correlation_heatmap.png]

The results consistently show very weak inter-metric associations, with a maximum absolute coefficient of $|r|_{\max} = 0.193$.

**Cross-dimension independence.** The most critical finding concerns correlations *across* evaluation dimensions. **Novelty (NS)** shows no significant association with any flaw-related metric (Critical Recall: $r = -0.015$, $p = 0.40$; Minor Recall: $r = +0.021$, $p = 0.23$; nCPS: $r = +0.007$, $p = 0.70$), confirming that evaluating the novelty of reviewer claims is entirely decoupled from flaw detection ability. **DoA** likewise exhibits no meaningful linear relationship with Critical Recall ($r = -0.006$, $p = 0.72$), demonstrating that structural argumentation depth is independent of a reviewer's capacity to identify methodological flaws. The correlation between DoA and MCS is marginally significant ($r = +0.094$, $p < 0.001$), yet the effect size remains negligible ($r^2 < 0.01$).

**Overall assessment.** Seven of fifteen metric pairs show no statistically significant correlation ($p \geq 0.05$). All significant pairs have $|r| < 0.20$, placing them in the *negligible-to-small* range with shared variance below $4\%$ ($r^2 < 0.04$). These results collectively confirm that the six metrics are **empirically near-orthogonal**: each captures a distinct dimension of peer review quality, thereby justifying their joint use as a comprehensive multi-dimensional evaluation benchmark.

---

## Appendix E: Full Cross-Dataset Quantitative Results

### E.1 Statistical Significance Testing Protocol

To rigorously assess the performance differences between the LLM baselines and the human ground-truth, we conduct non-parametric statistical testing across all metrics. Given the non-normal distribution of the evaluation scores, we employ the **Wilcoxon signed-rank test** to compute the uncorrected $p$-values for paired comparisons. Furthermore, to stringently control the Family-Wise Error Rate (FWER), the **Holm-Bonferroni step-down correction** is applied independently for each LLM baseline within each specific evaluation dimension across the 5 conferences ($N=5$ comparisons per family).

Throughout the subsequent analysis, we report the effect size (rank-biserial correlation $r$) and denote Holm-corrected statistical significance as: **ns** ($p_{holm} \ge 0.05$), $*$ ($p_{holm} < 0.05$), $**$ ($p_{holm} < 0.01$), $***$ ($p_{holm} < 0.001$).

### E.2 Depth of Analysis

The granular DoA results highlight a stark contrast in the evidentiary capabilities of the evaluated baselines. TreeReview, SEA, and Reviewer2 yield consistently lower DoA scores across all five venues, with statistical significance ($p < 0.005$) underscoring their systematic deficiency in substantiating critiques compared to human reviewers. Conversely, CycleReviewer and DeepReview successfully bridge this gap. The consistent lack of statistical significance (ns) when compared to the human baseline proves that these models achieve a comparable level of analytical depth. As analyzed previously, this statistical parity is primarily driven by their robust internal grounding mechanisms and high premise ratios, which effectively compensate for the inherent limitations of standard LLM generation.

[Figure: grounding_dist_bar.pdf]

**The Illusion of Depth in Reviewer2.** Reviewer2 presents an intriguing paradox at the grounding level. Despite producing an overwhelming absolute volume of vague, unanchored statements (Score 0), it surprisingly generates more externally grounded premises (Score 2) than other LLM baselines. However, this marginal grounding advantage is entirely negated by its severely low Premise Ratio, which dilutes the overall analytical depth across all aspects.

[Figure: aspect_distribution_macro.png]

**Alignment with Human Priorities.** Most advanced baselines successfully mirror human intuitive focus, dedicating the largest proportion of their grounded premises to core technical components: Methodology ($\sim 50-56\%$) and Experimental Design ($\sim 27-31\%$). Notably, **Reviewer2** achieves the closest alignment to Human reviewers with the lowest JSD of $0.071$. Its premise distribution (Methodology $52.3\%$, Experiment $31.3\%$, Clarity $7.5\%$) closely traces the human pattern ($50.8\%$, $29.3\%$, $9.4\%$), demonstrating a well-calibrated allocation of critical effort. **DeepReview** achieves the lowest Clarity proportion ($7.0\%$) and the highest Methodology concentration ($56.8\%$), confirming the model's capacity to firmly anchor its feedback in the most critical dimensions.

| Reviewer | Novelty | Methodology | Experiment | Clarity | JSD $\downarrow$ | H (bits) $\uparrow$ |
|---|---|---|---|---|---|---|
| Human | 0.103 | 0.508 | 0.293 | 0.094 | --- | 1.524 |
| CycleReviewer | 0.072 | 0.513 | 0.281 | 0.134 | 0.090 | 1.287 |
| DeepReview | 0.071 | **0.568** | 0.291 | **0.070** | 0.092 | 1.124 |
| Reviewer2 | 0.089 | 0.523 | 0.313 | 0.075 | **0.071** | 1.325 |
| SEA | 0.061 | 0.548 | 0.275 | 0.117 | 0.094 | 1.223 |
| TreeReview | **0.125** | 0.447 | 0.199 | **0.229** | 0.111 | 1.442 |

Table: Macro-average aspect distribution (premise-level) and alignment with Human reviewers measured by Jensen-Shannon Divergence (JSD). JSD $\in [0,1]$; lower values indicate closer alignment. $H$ (bits) denotes Shannon entropy.

**The Surface-Level Trap.** Rather than being an inherent LLM limitation, the "surface-level trap" manifests when automated frameworks lack explicit, domain-specific evaluation constraints. This is starkly pronounced in **TreeReview**, which, despite its complex reasoning topology, allocates an excessive $22.9\%$ of its premise-level effort to Clarity, nearly $2.4\times$ the proportion of Human reviewers ($9.4\%$). Consequently, TreeReview records the highest JSD against Humans ($0.111$) and the lowest Methodology coverage ($44.7\%$), confirming that without strict dimensional guidance, its analytical distribution naturally diverges toward superficial nitpicking.

[Figure: per_aspect_doa_grouped_bar.pdf]

**Shared Prioritization of Core Technical Aspects.** A consistent pattern emerges across both human and LLM reviewers: DoA scores are substantially higher for Methodology and Experimental Design than for Novelty and Clarity. For Human reviewers, the Methodology aspect achieves the highest DoA Score ($0.510 \pm 0.156$), followed closely by Experiment ($0.456 \pm 0.207$), while Novelty ($0.357 \pm 0.322$) and Clarity ($0.266 \pm 0.268$) trail significantly behind. This pattern holds uniformly across all five baselines, confirming that both humans and LLMs inherently recognize the need to anchor their most substantive arguments in the methodological and experimental core of a paper.

**Evidence Density as the Key Differentiator.** Across all four aspects, CycleReviewer and DeepReview consistently achieve DoA scores closest to---and in several cases statistically indistinguishable from---Human reviewers. This cross-aspect parity is not coincidental: both systems maintain the highest Premise Ratios among LLM baselines across every aspect (e.g., CycleReviewer reaches $0.706$ and DeepReview $0.674$ on Methodology, both exceeding the Human value of $0.655$). In contrast, Reviewer2, SEA, and TreeReview show markedly lower Premise Ratios particularly on Novelty ($0.272$, $0.113$, $0.186$ respectively), producing the steepest per-aspect DoA drops and confirming that their aggregate weakness reflects a globally deficient evidentiary discipline.

**Summary.** In conclusion, achieving a human-level Depth of Analysis requires more than merely generating a high volume of text. Models that fall into the trap of unsupported verbosity (Reviewer2) or surface-level nitpicking (TreeReview) are severely penalized. To bridge the analytical gap, automated reviewers must systematically substantiate their claims and strictly prioritize core technical dimensions over formatting issues.

### E.3 Novelty Assessment

Novelty Assessment scores generally fall between $0.730$ and $0.870$ across venues, indicating that both human reviewers and automated baselines frequently produce novelty claims that the retrieval-and-verification procedure can resolve with substantial evidence. The large number of non-significant differences (ns) for models such as DeepReview, Reviewer2, and TreeReview suggests similar *evidence-grounding performance on this scalar metric*, not full claim-by-claim agreement with human reviewers.

**The Outperformance of SEA.** The most notable result is the performance of the **SEA** baseline. While SEA is not uniformly strongest on the other review dimensions, it achieves the highest novelty-assessment score in multiple venues and significantly exceeds the human baseline in ICLR 2025 ($p_{Holm} < 0.01$), ICLR 2026 ($p_{Holm} < 0.005$), and NeurIPS 2025 ($p_{Holm} < 0.01$). This suggests that SEA's structured generation style tends to produce novelty claims that are especially easy for the PRISM retrieval-and-verification pipeline to ground in prior work.

[Figure: mean_novelty_claims.png]

**Claim Volume and Generation Verbosity.** Review sources vary substantially in how many novelty claims they choose to make. The concatenated human-review bundle averages $5.1$ extracted novelty-related claims per paper. **DeepReview** is markedly more verbose at $8.3$ claims per generated review, reflecting a finer-grained decomposition of contributions and comparisons. By contrast, **SEA** ($3.0$ claims) and **Reviewer2** ($3.5$ claims) are much more conservative, concentrating their novelty discussion into fewer statements. This matters because downstream agreement depends not only on how claims are scored once extracted, but also on claim granularity and boundary choices at the extraction stage.

**Summary.** Taken together, these results show that LLM reviewers can produce novelty claims that are often well grounded under the PRISM retrieval-and-verification pipeline. However, this scalar score should not be read as evidence that LLMs can certify the objective novelty of a manuscript, nor that they fully replicate human novelty judgments.

### E.4 Flaw Identification & Prioritization

The detailed results unequivocally confirm the exhaustive diagnostic capability of **Reviewer2**. Across every single evaluated venue, Reviewer2 consistently achieves the highest Recall for both *Critical* (ranging from $0.506$ to $0.649$) and *Minor* flaws. More importantly, the statistical tests ($p_{Holm} < 0.005$) validate that this over-performance relative to the human baseline is structurally ingrained in the model's generation style, not merely a statistical artifact.

Conversely, other LLM baselines generally exhibit a diagnostic deficit compared to human experts. Models like CycleReviewer and SEA consistently underperform the human ground-truth in extracting both major and minor flaws across most venues. An interesting anomaly is **TreeReview**: while it struggles to detect fatal methodological errors, it frequently outperforms or matches humans in *Minor Flaw Identification*, further corroborating the "surface-level trap" finding. DeepReview, maintaining a highly conservative profile, yields scores that are statistically indistinguishable (ns) from human experts in several venues.

The granular breakdown of the Critique Prioritization Score ($nCPS$) solidifies a key observation: the ability to strategically rank flaws is a solved problem for modern LLMs. Across all conferences, every baseline achieves near-perfect scores ($nCPS > 0.96$), rendering them statistically indistinguishable from the human ground-truth.

[Figure: severity_aspect_distribution.png]

**Alignment on Critical Vulnerabilities.** When evaluating *Critical* flaws, human experts demonstrate a laser-focused approach, allocating an overwhelming $92.3\%$ of their critiques to core technical components ($56.5\%$ on Methodology and $35.8\%$ on Experimental Design). **DeepReview** exhibits a remarkably identical signature, dedicating $91.7\%$ of its critical flaw detection to Methodology and Experiments. Conversely, **TreeReview** broadens the scope of critical evaluation, proactively identifying critical presentation errors at a rate of $6.0\%$.

**The Adaptive Focus on Minor Flaws.** The distribution of *Minor* flaws reveals a compelling alignment between human experts and LLM baselines. The cognitive focus for minor flaws naturally shifts toward surface-level anomalies. Both human reviewers and LLMs significantly reduce their extreme scrutiny on Methodology, while substantially increasing their attention to *Clarity, Presentation & Reproducibility*. Remarkably, the automated baselines successfully mirror this contextual shift.

**Summary.** These findings highlight a sophisticated capability in modern automated peer review: LLMs can successfully mimic human intuition in identifying and categorizing different types of flaws, proving they possess a nuanced, human-like understanding of manuscript evaluation.

### E.5 Multi-Dimensional Constructiveness

DeepReview consistently achieves the highest MCS across all five conferences (ranging from $0.629$ to $0.635$), with statistical significance ($p_{Holm} < 0.05$) over the human baseline in every venue. Reviewer2 demonstrates competitive performance, while SEA and TreeReview consistently underperform.

| System | D1: Actionability | D2: Specificity | D3: Justification | D4: Solution | D5: Tone |
|---|---|---|---|---|---|
| **Human** | $1.105 \pm 0.411$ | $1.725 \pm 0.260$ | $0.759 \pm 0.465$ | $0.470 \pm 0.350$ | $1.589 \pm 0.346$ |
| CycleReviewer | $1.328 \pm 0.492$ | $\mathbf{1.897} \pm 0.206$ | $0.325 \pm 0.434$ | $0.401 \pm 0.385$ | $1.321 \pm 0.410$ |
| DeepReview | $\mathbf{1.414} \pm 0.294$ | $1.831 \pm 0.201$ | $0.580 \pm 0.369$ | $\mathbf{0.784} \pm 0.290$ | $\mathbf{1.726} \pm 0.286$ |
| Reviewer2 | $1.178 \pm 0.324$ | $1.784 \pm 0.252$ | $\mathbf{0.939} \pm 0.429$ | $0.266 \pm 0.248$ | $1.586 \pm 0.441$ |
| SEA | $0.909 \pm 0.384$ | $1.651 \pm 0.252$ | $0.478 \pm 0.413$ | $0.375 \pm 0.273$ | $1.593 \pm 0.245$ |
| TreeReview | $1.045 \pm 0.276$ | $1.532 \pm 0.353$ | $0.639 \pm 0.463$ | $0.357 \pm 0.345$ | $1.278 \pm 0.575$ |

Table: Detailed Constructiveness Sub-dimensions (D1-D5) across 5 Conferences evaluated on a raw scale of $[0, 2]$.

[Figure: d_score_heatmap.png]
[Figure: core_constructiveness_metrics.png]

To provide a more nuanced evaluation, we derive three auxiliary density metrics. Let $\mathbb{I}(\cdot)$ be an indicator function:

$$
AR(R) = \frac{1}{n} \sum_{j=1}^n \mathbb{I}(D_1(c_j) \ge 1), \quad
SD(R) = \frac{1}{n} \sum_{j=1}^n \mathbb{I}(D_4(c_j) = 2), \quad
CD(R) = \frac{1}{n} \sum_{j=1}^n \mathbb{I}(CLC(c_j) \ge 0.5)
$$

**The Solution Bottleneck.** The most profound divergence between human experts and top LLMs lies in the capacity to propose explicit improvements. The data reveals a systemic limitation in traditional peer review: while humans are highly proficient at pinpointing concrete flaws ($D_2$ Specificity $\approx 1.72$), they frequently fail to provide actionable remedies. This is evidenced by the human baseline's remarkably low Solution Density ($SD \approx 0.102$), indicating that only $10\%$ of human comments contain an explicit fix ($D_4 = 2$). In stark contrast, DeepReview fundamentally alters this paradigm, consistently achieving an $SD$ approaching $0.29$ and $D_4$ scores near $0.80$ across all venues.

**The Verbosity Paradox of Reviewer2.** Reviewer2 achieves exceptional Justification scores ($D_3 > 0.90$) largely as an artifact of its highly verbose, two-stage rubric-driven generation style. However, its abysmal Solution Density ($SD \approx 0.05$) and low Solution score ($D_4 \approx 0.26$) reveal a critical flaw: it exhaustively explains *why* something is wrong but almost never tells the author *how* to fix it.

**Actionability without Depth in CycleReviewer.** CycleReviewer exhibits a contrasting failure mode. It achieves the highest Specificity ($D_2 \approx 1.89$) and Actionability Ratio ($AR > 0.90$), meaning almost all of its comments reference concrete paper elements. Yet, it suffers a catastrophic drop in Justification ($D_3 \approx 0.32$) and Solution Density ($SD \approx 0.04$), pointing to shallow, "checklist-style" reviewing behavior.

**Tone and Professionalism.** Well-calibrated LLMs can systematically elevate the discourse of peer review. **DeepReview** ($D_5 \approx 1.72$) consistently outputs more professional, neutral, and encouraging feedback than the human baseline ($D_5 \approx 1.58$), effectively mitigating the dismissive or hostile language occasionally encountered in human peer reviews.

**Summary.** Human experts primarily act as *diagnosticians*, highly effective at pinpointing errors but lacking in actionable guidance. In contrast, **DeepReview** transcends these limitations, acting more as a *collaborator* by bridging the critical gap between identifying flaws and formulating explicit, professionally toned solutions.

### E.6 Review Sensitivity to Paper Quality: Accept vs. Reject Analysis

| Metric | Human | Reviewer2 | TreeReview | DeepReview | SEA | CycleReviewer |
|---|---|---|---|---|---|---|
| Novelty Score | $\mathbf{+0.051^{*}}$ | $+0.024$ | $+0.035$ | $+0.037$ | $+0.029$ | $+0.021$ |
| DoA Score | $-0.006$ | $-0.004$ | $-0.028$ | $-0.001$ | $-0.006$ | $-0.003$ |
| Critical Score | $\mathbf{-0.049^{***}}$ | $-0.058$ | $+0.043$ | $+0.005$ | $-0.005$ | $+0.018$ |
| Minor Score | $\mathbf{-0.018^{**}}$ | $-0.008$ | $-0.030$ | $-0.002$ | $-0.007$ | $-0.004$ |
| Prioritization Score | $\mathbf{+0.006^{***}}$ | $-0.004$ | $+0.001$ | $\mathbf{+0.011^{***}}$ | $-0.001$ | $-0.001$ |
| MCS | $-0.001$ | $+0.001$ | $+0.011$ | $-0.006$ | $+0.004$ | $-0.010$ |

Table: Metric differences across reviewers. Bold values indicate statistically significant results.

**Human Reviews Exhibit Strong Predictive Validity.** Although human reviewers evaluate manuscripts blindly without any knowledge of the final editorial outcome, their assessments exhibit a robust, statistically significant correlation with the eventual decisions. Manuscripts that are ultimately accepted garner blind reviews with significantly higher Novelty Scores ($\Delta = +0.051^{*}$) and tighter structural prioritization ($\Delta = +0.006^{***}$). Conversely, papers that are ultimately rejected accumulate substantially more critical diagnostic feedback, reflected in significantly worse scores for both Critical ($\Delta = -0.049^{***}$) and Minor ($\Delta = -0.018^{**}$) flaws.

**LLMs Exhibit Evaluative Invariance Across Quality Tiers.** In contrast to human reviewers, LLM reviewers exhibit a highly invariant and consistent evaluative pattern regardless of whether a paper is ultimately accepted or rejected. Across all five automated systems, the metric differences are predominantly uniform, with only DeepReview's Prioritization Score reaching statistical significance ($\Delta = +0.011^{***}$). Notably, this stability is most pronounced in the *Critical Score* dimension. Rather than being swayed by the overall quality of the submission, LLMs apply their internal diagnostic heuristics independently.

**Summary.** Human reviewers display high sensitivity, naturally adjusting the severity of their critiques based on the overall scientific merit of the submission. In contrast, LLMs function as invariant diagnostic scanners, generating reviews with remarkably stable metric distributions across all papers.

### E.7 Evaluator Robustness Across LLM Backends

[Figure: combined_6metrics.pdf]

To assess whether our metric framework is sensitive to the choice of evaluator LLM, we re-ran the full evaluation pipeline using Mimo v2.5 Pro [mimo2026v25pro] as an alternative backend and compared results against Gemini 2.5 Flash Lite across all six metrics. Gemini assigns slightly higher values on Depth of Analysis, Novelty Assessment and Constructiveness, while Mimo yields marginally higher scores on Minor Recall. Importantly, Prioritization score shows the smallest divergence between evaluators.

Despite absolute score offsets, the relative ordering of reviewer types is consistent between the two evaluators. Reviewer2 and DeepReview consistently achieve the highest Constructiveness and Critical Recall, while CycleReviewer and SEA rank lowest in most dimensions. For Novelty Assessment, both evaluators agree that SEA produces the highest scores. These results demonstrate that the proposed evaluation framework is robust to evaluator LLM substitution. No qualitative conclusion drawn from Gemini is reversed by Mimo.

---

## Appendix F: Qualitative Analysis & Case Studies

### F.1 Depth of Analysis

**Case 1: The Evidentiary Collapse --- Why Claim-Heavy Reviewers Fail.**

*Paper: NV-Embed: Generalist Text Embeddings from Decoder-Only LLMs* (ICLR 2025)

**Context.** NV-Embed proposes a generalist embedding model built on decoder-only LLMs, introducing (1) a latent attention layer replacing mean-pooling, and (2) a two-stage contrastive instruction-tuning pipeline. The model achieves top-1 performance on the MTEB benchmark.

| Reviewer | DoA | R_premise | Avg GS | Total Args | Premises | Claims |
|---|---|---|---|---|---|---|
| **Human** | 0.581 | 0.673 | 0.500 | 55 | 37 | 18 |
| DeepReview | **0.626** | 0.733 | 0.546 | 15 | 11 | 4 |
| Reviewer2 | 0.178 | 0.152 | 0.215 | 46 | 7 | 39 |

**Human: Dense Technical Grounding Across All Aspects.** Human reviewers produce 37 premises from 55 total arguments ($R_{\text{premise}} = 0.673$), with strong grounding quality ($\overline{\text{GS}} = 0.500$). Premises are component-specific, naming the paper's actual technical building blocks:

> *[GS=1.0, Methodology]* "The techniques used are: 1. latent attention layer that achieves better pooling/combination of the last layer embeddings."

> *[GS=1.0, Methodology]* "2. a two-stage contrastive instruction tuning method. First step tuning with in-batch negative and hard negative on retrieval datasets, and the second step tuning on non-retrieval datasets."

> *[GS=2.0, Experiment]* "The model achieves top performance on the MTEB benchmark."

**DeepReview: Exceeds Human DoA via Precision-to-Volume Economy.** DeepReview produces only 15 arguments, yet 11 are premises ($R_{\text{premise}} = 0.733$, exceeding Human's 0.673), with the highest average grounding score of any reviewer ($\overline{\text{GS}} = 0.546$).

> *[GS=1.0, Methodology]* "The authors introduce a novel latent attention layer for pooling embeddings, which outperforms traditional methods like average pooling and <EOS> token embedding."

> *[GS=2.0, Experiment]* "Ablation experiments in Table 2 confirm the contribution of the latent attention layer over alternative pooling strategies."

**Reviewer2: Self-Referential Grounding and Volume Inflation.** Reviewer2 generates 46 arguments but only 7 qualify as premises, and 4 of those 7 carry grounding score 0 ($\overline{\text{GS}} = 0.215$, barely above the minimum). The 39 claims are structured section-by-section summaries offering no independent analytical judgment:

> *[Claim, Methodology]* "The paper introduces NV-Embed, a generalist embedding model based on decoder-only large language models (LLMs), aimed at enhancing performance in downstream tasks such as retrieval..."

> *[GS=0.0, Methodology]* "The introduction of a latent attention layer for sequence pooling is theoretically grounded in dictionary learning concepts and is argued to mitigate information dilution compared to mean pooling or last token pooling."

The low grounding quality compounds the low premise ratio: $R_{\text{premise}} = 0.152$ and $\overline{\text{GS}} = 0.215$ together produce $\text{DoA} = 0.178$ --- a **70% drop** from Human's $0.581$.

**Key insight.** Reviewer2's failure on this paper illustrates a form of analytical failure beyond pure volume-inflation: even its few "premises" ground claims in the paper's own assertions rather than in independent analytical observations.

---

**Case 2: The Surface-Level Trap in Practice.**

*Paper: VLAP --- Visual-Language Alignment via Pre-trained Word Embeddings* (ICLR 2024)

**Context.** VLAP proposes a lightweight vision-language alignment method that maps visual representations directly into the pre-trained word embedding space of a frozen LLM using a single trainable linear layer. The design is deliberately minimal: the LLM and visual encoder remain frozen, with only the linear projection trained.

| Reviewer | DoA | R_premise | Avg GS | Total | Premises | Claims | % Clarity |
|---|---|---|---|---|---|---|---|
| **Human (mean, n=5)** | 0.567 | 0.614 | 0.527 | 14 | 8 | 6 | 2% |
| Reviewer2 | **0.551** | 0.569 | 0.534 | 51 | 29 | 22 | **0%** |
| CycleReviewer | 0.483 | 0.529 | 0.444 | 17 | 9 | 8 | **0%** |
| SEA | 0.417 | 0.500 | 0.357 | 14 | 7 | 7 | 29% |
| DeepReview | 0.263 | 0.759 | 0.159 | 29 | 22 | 7 | **0%** |
| **TreeReview** | 0.252 | 0.194 | 0.357 | 36 | 7 | 29 | **29%** |

**Reviewer consensus: Clarity is irrelevant on a paper with a simple design.** This is a particularly instructive paper for the surface-level trap because the simplicity of VLAP's design leaves almost no room for legitimate reproducibility criticism. Accordingly, **Human, DeepReview, CycleReviewer, and Reviewer2 all allocate 0%** of their premise budget to Clarity. Instead, they focus exclusively on the paper's technical formulation and experimental comparisons:

> *[GS=2.0, Novelty]* "Contrastive alignment in ALBEF, BLIP, and the first-stage alignment by BLIP2 includes image-text matching and image-grounded text generation." *(Human_2)*

> *[GS=2.0, Methodology]* "An optimal transport-based training objective is proposed to enforce the consistency of word assignments for paired multimodal data. This allows frozen LLMs to ground their word embedding space in visual data." *(Human_3)*

**SEA: Clarity premises praising, not criticizing.** SEA allocates 29% to Clarity, matching TreeReview's proportion. However, its two Clarity premises are *positive quality affirmations* about the paper, not complaints:

> *[GS=0.0, Clarity]* "The methodology is clearly explained, making it accessible and understandable, which is crucial for reproducibility and further research."

> *[GS=0.0, Clarity]* "The paper is well-structured, with comprehensive experiments and detailed analyses."

**TreeReview: reproducibility boilerplate displaces technical analysis.** TreeReview produces only 7 premises from 36 total arguments ($R_{\text{premise}} = 0.194$). Of these 7, **2 (29%) are Clarity premises**, both carrying GS=0 and targeting the same generic reproducibility axis:

> *[GS=0.0, Clarity]* "This omission hinders reproducibility and limits the ability of other researchers to build upon the work."

> *[GS=0.0, Clarity]* "This would greatly enhance the reproducibility of the method."

Neither statement names a specific omission. On a paper whose entire contribution is a single linear layer with two objectives, these statements convey essentially no analytical information.

**Key insight.** The surface-level trap on this paper is not caused by a genuinely unclear manuscript: four of the six reviewers independently assess that VLAP requires no Clarity criticism at all. The trap is therefore *triggered internally* by TreeReview's reviewing heuristic---a tendency to produce generic reproducibility premises regardless of whether the paper's design warrants them.

### F.2 Novelty Assessment

**Case 1: The Speculative Critique Trap**

Paper from ICLR 2025, which proposes **CONPAIR**, a contrastive compositional dataset and **EVOGEN**, a curriculum contrastive learning framework for improving compositional text-to-image generation in diffusion models.

Both reviewers correctly identify the paper's core contributions. The divergence arises not from a disagreement on *what* is novel, but from how reviewers handle *uncertain* assessments.

**Human reviewer** produces 10 claims with a nuanced mixture of stances. Claim C8 carries speculative criticism that the metric system cannot verify:

> *[C8, unclear]* "The ContraFusion model is compared against other methods using the T2I-CompBench dataset. However, it is trained on the Com-Diff dataset, which **likely overlaps** noticeably with the T2I-CompBench test set."

This concern about training/test overlap is plausible but entirely speculative: no related paper in the retrieved pool provides evidence for or against dataset overlap. Consequently, 7 of 11 related-paper comparisons for C8 return Unsupported or Insufficient.

**SEA reviewer** produces only 5 claims, all either *novel* or *somewhat_novel*, with no speculative or *unclear* claims. Each SEA claim targets a specific, verifiable contribution, and none raises unverifiable speculation.

**Insight --- Speculative Assessment Penalty.** This case illustrates a systematic pattern: Human reviewers raise *unclear*-stance concerns that are reasonable in context but impossible to corroborate through paper-pool evidence. The NS metric penalises such claims because they lack verifiable grounding. SEA tends to avoid this failure mode, focusing on claims directly grounded in the paper's methods. This finding does *not* imply SEA is a "better reviewer"; rather, it shows that SEA's positively-biased and evidence-anchored claim style is systematically rewarded by the evidence-grounded NS metric.

---

**Case 2: Coverage Without Sacrifice**

Paper from ICML 2025 examines a theoretical paper that characterises the expressivity of fixed-precision Transformer decoders using formal language theory. The paper establishes: (R1) without positional encoding (NoPE), fixed-precision Transformers can recognise only finite and co-finite languages; (R2) adding absolute positional encoding (APE) extends expressibility to cyclic languages; (R3) relaxing parameter bounds further allows recognition of letter-set languages.

**Human reviewer** makes 5 well-targeted claims, all centred on the paper's three core theoretical results. Three claims earn the maximum per-claim score $s_k=+2.0$ by directly naming the language classes established. Two claims are penalised to $s_k=+0.667$ for imprecision.

**DeepReview** produces 12 claims, $2.4\times$ more, at a nearly identical raw mean claim score ($\bar{s}=1.500$ vs. $1.467$). Its first 7 claims cover *all* of Human's 5 novelty dimensions, often with greater precision, while claims DR8--DR12 open an entirely new dimension absent from the human review: well-evidenced critical analysis of the paper's *scope limitations*.

> *[DR2, novel]* "Introducing absolute positional encoding extends their capabilities to recognizing **cyclic languages**, while allowing non-finite floating-point values further expands their expressivity to **letter-set languages**." $(s_k=+2.0)$

> *[DR10, not_novel]* "The paper's analysis of positional encoding is limited to absolute positional encoding (APE) and no positional encoding (NoPE). It does not explore **relative positional encodings**, which are commonly used in modern Transformer architectures." $(s_k=+2.0)$

Crucially, these limitation claims are *not_novel* in stance but still earn $s_k=+2.0$. This is because the novelty metric rewards *calibration*: the claim that "relative PE is not studied here" is verifiable against the related-work pool.

**Insight --- Coverage Expansion with Preserved Calibration.** DeepReview does not achieve a higher raw mean claim score $\bar{s}$ by avoiding criticism, but by *expanding coverage* while maintaining the same calibration quality as human reviewers. DeepReview's extra volume comes in two flavours: (i) *precision elaboration*---naming specific language classes and precision regimes more exactly than Human; and (ii) *gap enumeration*---identifying well-evidenced scope limitations that human reviewers do not consider.

### F.3 Flaw Identification & Major Issues Prioritization

**Case 1: Complementary Blind Spots --- The Equation-Level Scanner vs. the Practical Assessor.**

Paper from ICML 2025 Oral proposes a preconditioning-based optimizer for Domain Generalization that leverages the One-Step Generalization Ratio (OSGR) to dynamically balance parameter-wise gradient updates. The canonical flaw bank contains 52 entries, of which 34 are valid upon independent verification.

**What the LLM catches that all Humans miss: systematic equation-level scrutiny.** Reviewer2 identifies 9 unique valid flaws that none of the three human reviewers raise. These flaws share a distinctive pattern: they arise from systematically walking through the paper's mathematical derivations and cross-referencing theoretical claims against their practical implementation.

> *[LLM-only, Methodology]* "In the PAC-Bayes analysis, the prior $\pi$ is approximated using all data except the current batch. Why is this approximation valid, and what is the impact of this choice on the tightness of the generalization bound?"

> *[LLM-only, Methodology]* "In Corollary 3.2, the preconditioning factor $p_j$ is derived under the assumption that gradients are independent across parameters. However, in practice, gradients are often highly correlated."

**What Humans catch that the LLM misses: claim-evidence calibration and field norms.** Human reviewers identify 9 unique valid flaws that Reviewer2 entirely overlooks. These flaws target gaps between what the paper *claims* and what the *evidence* supports:

> *[Human-only, Methodology]* "The paper claims [the optimizer] promotes domain-invariant features, but it doesn't directly evaluate this claim by examining feature representations."

> *[Human-only, Methodology]* "The claim that uniformly distributed OSGR across parameters indicates better generalization... is stated as a conjecture rather than a theorem, and while intuitively supported, it's not rigorously demonstrated."

**Where the LLM hallucinates: asserting absence of content that exists.** Reviewer2 also generates several flaws that are *directly contradicted* by the paper. The most striking: the LLM claims "the paper does not adequately explain the computational cost relative to simpler optimizers," despite the paper explicitly reporting training times. These fabrications follow a consistent pattern: the LLM evaluates flaw claims in isolation without cross-referencing the manuscript's actual content.

**Key insight.** This case demonstrates that LLM and human reviewers operate as *complementary diagnostic instruments* with near-zero overlap. The LLM excels at systematic, equation-level verification---surfacing implicit assumptions and theory-practice gaps. Human reviewers excel at claim-evidence calibration---demanding quantitative backing for qualitative claims. Neither perspective subsumes the other: the union of their flaw sets produces substantially broader diagnostic coverage than either alone.

---

**Case 2: The Diagnostic Volume Advantage --- Broader Coverage Through Exhaustive Scanning.**

Paper from ICLR 2025 proposes an iterative data augmentation pipeline for fine-tuning LLMs on Operations Research (OR) tasks. The canonical flaw bank contains 28 entries, contributed equally by human reviewers (14 flaws) and Reviewer2 (14 flaws).

**Human reviewers: deep but concentrated coverage.** The four human reviewers produce 14 valid flaws, with a strong concentration on Experimental Design (7 of 14, 50%). Their critiques are precise and field-specific:

> *[Human, Experimental Design]* "No comparison with traditional OR solvers is provided. The paper claims to advance `automation of decision-making' but never shows whether LLM-based modeling is competitive with established OR methods."

**Reviewer2: broader aspect coverage with systematic gap enumeration.** Reviewer2 produces an equal number of valid flaws (14), but distributes them more evenly across aspect categories. Notably, 5 of its 14 flaws target Applicability and Limitations:

> *[LLM-only, Applicability]* "The scalability of the pipeline to large-scale optimization problems with thousands of variables and constraints is not evaluated."

> *[LLM-only, Methodology]* "The four validation checkers are entirely LLM-prompt-based. No formal algorithmic specification, error bounds, or coverage guarantees are provided for the checking pipeline."

**Key insight.** Reviewer2's exhaustive scanning style does not merely produce *more* flaws---it produces flaws in *different diagnostic categories* than human reviewers, systematically covering scope limitations and methodological formalization gaps that humans deprioritize. The resulting union of human and LLM flaw sets achieves broader aspect coverage than either alone.

### F.4 Multi-Dimensional Constructiveness

**Case: The Actionability Gap --- From Diagnosis to Prescription.**

*Paper: GenColor: A Diffusion-Based Framework for Color Enhancement in Digital Photography* (NeurIPS 2025)

**Context.** GenColor proposes a no-reference color enhancement pipeline consisting of three learned components: (1) a diffusion-based Color Generation Module, (2) a Texture Preservation Module, and (3) a post-processing Global Adjustment step.

| System | MCS | D1:Act | D2:Spec | D3:Just | D4:Sol | D5:Tone | ARCs |
|---|---|---|---|---|---|---|---|
| **Human** | 0.488 | 0.721 | 1.767 | 0.372 | 0.326 | 1.698 | 43 |
| **DeepReview** | **0.724** | **1.588** | **2.000** | 0.882 | **1.059** | 1.706 | 17 |
| Reviewer2 | 0.800 | 1.000 | 2.000 | **2.000** | 1.000 | **2.000** | 24 |
| CycleReviewer | 0.400 | 0.833 | 1.833 | 0.167 | 0.167 | 1.000 | 6 |
| TreeReview | 0.434 | 0.793 | 1.517 | 0.241 | 0.207 | 1.586 | 29 |

**Human: Precise Diagnosis, Absent Prescription.** Human reviewers produce 43 ARCs with respectable specificity ($\overline{D2} = 1.767$). However, D4 (Solution) averages only $0.326$: the majority of human ARCs identify a problem but stop short of prescribing a remedy.

> *[D1=1, D4=0]* "The method's near-deterministic nature raises concerns about user control and the ability to capture personalized styles."

> *[D1=1, D4=0]* "The proposed method has a relatively long runtime compared to lightweight comparison models."

**DeepReview: Prescriptive Constructiveness at Scale.** DeepReview produces only 17 ARCs, yet achieves $\overline{D1} = 1.588$ and $\overline{D4} = 1.059$, both substantially above Human. Critically, **six of its ARCs reach the maximum solution score D4=2**---meaning the feedback specifies not only *what* is missing but *how* to address it:

> *[D1=2, D4=2]* "Include comprehensive ablation studies by systematically removing or modifying components to evaluate their individual contribution."

> *[D1=2, D4=2]* "Provide a detailed analysis of the computational cost of each component and the overall pipeline, including training time, inference time, and memory requirements."

**Reviewer2: High MCS via Justification Inflation, Not Solutions.** Reviewer2 achieves the highest raw MCS on this paper (0.800), driven entirely by near-perfect D3 scores ($\overline{D3} = 2.000$). Its 24 ARCs are detailed, well-grounded observations---but they are *observations*, not directives. D1 averages only 1.000 and D4 averages 1.000, because every ARC points to a general direction at best without specifying an implementable fix.

**Key insight.** This case illustrates the central behavioral gap in constructiveness: even when human reviewers are technically perceptive (high D2, moderate D3), they default to *problem identification without resolution*. DeepReview's architectural orientation toward explicit remediation---reflected in D4 exceeding human baseline by $+0.733$ and D1 by $+0.867$---demonstrates that high constructiveness is not a matter of writing more, but of *closing the feedback loop* from critique to actionable prescription.

---

## Appendix G: Limitations

While PRISM provides a rigorous, multi-dimensional benchmarking framework for automated peer review, we acknowledge several limitations that highlight avenues for future research.

**Domain Generalization.** Our dataset comprises 1,000 manuscripts exclusively from premier machine learning and representation learning venues (ICLR, ICML, NeurIPS). The structural norms, citation densities, and evaluation criteria in these venues differ from those in other scientific disciplines (e.g., clinical medicine, humanities, or pure mathematics). Consequently, the current instantiation of PRISM may require recalibration before deployment in non-ML domains.

**LLM Dependency: Hallucination, Prompt Sensitivity, and Judge Bias.** A foundational premise of PRISM is delegating complex tasks---such as text atomization, fact-finding, and scoring---to frontier LLMs. It is well-documented that LLMs are heavily susceptible to hallucinations (fabricating non-existent critiques or citations) and prompt sensitivity (where minor structural variations in instructions yield divergent outputs). To actively mitigate these vulnerabilities, PRISM strictly departs from monolithic, single-prompt evaluation. By decomposing the framework into constrained, multi-phase pipelines and enforcing deterministic decoding, we significantly restrict the generation space and effectively filter out hallucinated noise. Nevertheless, this dependency introduces residual bottlenecks: atomizing isolated sentences inherently risks context loss, fact-finding remains bounded by the coverage of external retrieval APIs, and models acting as judges may still retain subtle internal priors for specific rhetorical styles.

In this work, our primary evaluation pipeline is instantiated using `Gemini 2.5 Flash Lite`. While we conducted preliminary robustness checks with an alternative model (Xiaomi `MiMo V2.5 Pro`) on a data subsample to confirm baseline metric stability, this single-judge dependency means we cannot fully rule out model-specific evaluation biases. Future work must not only develop robust uncertainty quantification to prevent edge-case extraction errors from cascading into downstream metrics, but also conduct comprehensive multi-judge studies across diverse LLM families to fully isolate and eliminate judge-specific priors.
