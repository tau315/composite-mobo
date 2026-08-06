# LLM-polished

# **Paper template: [Neurips](https://neurips.cc/Conferences/2026/CallForPapers)**

# **Target workshop: [AI4Mat](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/AI4Mat#tab-recent-activity)** 

# **Proposed Workshop Paper Outline**

## **Abstract**

The abstract should briefly establish the problem, methodological gap, contribution, and main empirical findings.

1. Introduce the importance of scientific design and optimization in applications such as material discovery, formulation design, and experimental process optimization.  
2. Explain that scientific design problems often involve optimizing several competing objectives simultaneously. Provide one concrete example, such as maximizing material performance while minimizing cost, toxicity, or environmental impact.  
3. Describe the primary computational challenges: experiments are expensive, the objective functions are black-box, and the design space may be high-dimensional.  
4. Introduce multi-objective Bayesian optimization (MOBO) as a general framework for efficiently solving such problems.  
5. Explain that, in many scientific applications, each objective has a known composite structure: a design first produces an intermediate, potentially high-dimensional scientific response, which is then transformed into one or more scalar objectives.  
6. State that most existing MOBO methods model only the final objective values and therefore do not exploit this intermediate structure.  
7. State the main objective of the paper: to investigate whether leveraging composite-function structure improves the performance of existing MOBO frameworks, particularly in high-dimensional design spaces.  
8. Briefly summarize the experimental scope, including the number of synthetic and real-world benchmark problems and the classes of MOBO methods considered.  
9. Report the principal quantitative result. For example, state how often composite-aware variants outperform their standard counterparts and, where possible, how much faster they reach a specified performance threshold.

---

## **1\. Introduction**

### **Paragraph 1: Scientific design**

Introduce scientific design as the task of selecting experimental or engineering parameters to produce systems with desirable properties. Motivate its relevance using applications such as material design, chemical formulation, protein engineering, or experimental process optimization.

### **Paragraph 2: Computational challenges**

Explain that scientific design problems commonly involve expensive evaluations, black-box relationships between inputs and outcomes, noisy observations, and high-dimensional design spaces.

Use one representative application throughout this paragraph to make the challenges concrete. For example, in formulation design, the input may contain concentrations of many ingredients, while each experimental evaluation requires physically preparing and measuring a formulation.

### **Paragraph 3: Bayesian optimization and MOBO**

Introduce Bayesian optimization as a sample-efficient approach for optimizing expensive black-box functions. Cite foundational Bayesian optimization work and relevant scientific applications, especially those related to formulation or material design.

Explain that Bayesian optimization has been extended to multi-objective problems, where the goal is to identify a set of Pareto-optimal designs rather than a single optimum. Cite foundational MOBO methods.

### **Paragraph 4: High-dimensional MOBO**

Discuss why conventional MOBO methods become difficult to apply in high-dimensional input spaces.

Introduce representative high-dimensional MOBO frameworks, such as MORBO and other relevant trust-region, decomposition, or scalable surrogate-based methods. Give a brief one- or two-sentence explanation of each approach rather than a complete technical description.

Focus only on the methods evaluated in the paper or those necessary to establish the literature gap.

### **Paragraph 5: Scalarization-based alternatives**

Introduce scalarization-based MOBO, particularly random or augmented Chebyshev scalarization.

Explain that scalarization transforms a multi-objective problem into a sequence of single-objective problems and can therefore be combined with existing high-dimensional Bayesian optimization methods.

Clarify why this family of methods is relevant to the comparisons performed in the paper.

### **Paragraph 6: Composite-function structure**

Explain that many scientific objectives are not observed as indivisible scalar black-box functions. Instead, evaluating a design often produces an intermediate response, such as a spectrum, image, time series, field, or concentration profile, from which one or more scalar objectives are computed.

Introduce Bayesian optimization of composite functions (BOCF), which exploits this structure in single-objective optimization.

Discuss prior extensions to multi-objective or multi-output settings. Identify the Max workshop paper as the closest related work, if appropriate.

Clearly distinguish the present study from that work. For example:

* The prior work primarily examined the effect of using a high-order Gaussian process to model structured intermediate outputs.  
* The present paper studies whether composite modeling improves a broader range of MOBO frameworks.  
* The present work emphasizes high-dimensional design spaces.  
* The study performs controlled comparisons between composite-aware methods and their non-composite counterparts.  
* The experiments include a broader or different collection of synthetic and real-world scientific benchmarks.

### **Paragraph 7: Contributions**

State the paper’s contributions explicitly. A possible structure is:

1. We formulate a common framework for incorporating composite-function models into several existing MOBO strategies.  
2. We conduct controlled comparisons between composite-aware methods and otherwise equivalent methods that model only the final objectives.  
3. We evaluate these methods on synthetic and real-world problems spanning low- and high-dimensional design spaces.  
4. We identify the problem characteristics under which exploiting composite structure provides the largest improvement.

### **Paragraph 8: Experimental scope**

Briefly introduce the methods and benchmark problems considered. Avoid extensive details, which belong in later sections.

Mention the number of synthetic benchmarks, the number of real-world benchmarks, the range of input dimensions, and the number of objectives.

### **Paragraph 9: Main findings**

Summarize the main results quantitatively.

Report how consistently composite-aware variants improve performance, whether the gains are larger in high-dimensional settings, and whether there are notable exceptions.

Where possible, report interpretable quantities such as:

* percentage improvement in hypervolume;  
* number of benchmarks on which composite modeling performs better;  
* number of evaluations required to reach a target hypervolume;  
* average acceleration in convergence.

---

## **2\. Problem Setting**

Begin with a short paragraph explaining that the section introduces the notation, assumptions, objective structure, and optimization goal.

Let the design variable be

\[  
\\mathbf{x}\\in\\mathcal{X}\\subseteq\\mathbb{R}^{d},  
\]

where (d) may be large.

Suppose there are (m) objectives. Define an intermediate response for objective (i) as

\[  
\\mathbf{h}\_i(\\mathbf{x})\\in\\mathbb{R}^{p\_i},  
\]

and define the corresponding scalar objective as

\[  
f\_i(\\mathbf{x}) \= g\_i\!\\left(\\mathbf{h}\_i(\\mathbf{x})\\right),  
\]

where (g\_i:\\mathbb{R}^{p\_i}\\rightarrow\\mathbb{R}) is known and inexpensive to evaluate.

Clearly explain:

* whether different objectives share the same intermediate response;  
* whether (\\mathbf{h}\_i) is observed directly;  
* whether the observations are noisy;  
* whether the mappings (g\_i) are deterministic and known;  
* whether all objectives are minimized or maximized;  
* whether evaluations of all objectives occur jointly.

Define the multi-objective optimization goal, the Pareto set, and the Pareto front.

State that the study compares two modeling strategies:

1. **Objective-space modeling**, which directly models (f\_1,\\ldots,f\_m).  
2. **Composite modeling**, which models the intermediate responses (\\mathbf{h}\_1,\\ldots,\\mathbf{h}\_m) and propagates their posterior samples through the known mappings (g\_i).

---

## **3\. Considered Methods**

Begin with a short paragraph listing the method families included in the comparison and explaining that each composite-aware method is paired with a corresponding non-composite baseline whenever possible.

Use one subsection for each method family.

### **3.1 \[Method Family 1\]**

Explain:

* the surrogate model;  
* the acquisition function;  
* how candidate points are selected;  
* how the method handles multiple objectives;  
* how it handles high-dimensional inputs.

Then describe its composite-aware variant and explain exactly what changes.

### **3.2 \[Method Family 2\]**

Use the same structure to make the comparison easy to follow.

### **3.3 \[Method Family 3\]**

Repeat as needed.

### **Method naming**

Use short, systematic abbreviations that make paired comparisons immediately recognizable. For example:

* **MORBO**: standard objective-space MORBO;  
* **MORBO-CF**: composite-aware MORBO;  
* **TS-TCH**: Thompson sampling with Chebyshev scalarization;  
* **TS-TCH-CF**: its composite-aware counterpart.

Alternatively, use a consistent prefix:

* **Direct-MORBO** versus **Composite-MORBO**;  
* **Direct-TCH** versus **Composite-TCH**.

Avoid using a dash alone to distinguish composite and non-composite methods in the text, because this may be difficult to interpret. The line style can still distinguish the variants in figures.

Keep the main-text descriptions concise. Move complete algorithmic details, derivations, and pseudocode to Appendix A.

---

## **4\. Numerical Experiments**

Begin with a short overview stating that the experiments include synthetic and real-world benchmarks in both low- and high-dimensional design spaces.

State the primary evaluation metric and explain that all composite-aware methods are compared against matched non-composite counterparts under the same evaluation budget.

### **4.1 Test Problems**

#### **4.1.1 Synthetic Problems**

List the synthetic benchmark functions and cite their original sources.

The main text does not need to reproduce every mathematical definition. Instead, for each benchmark, report:

* input dimension (d);  
* number of objectives (m);  
* intermediate-output dimension (p\_i);  
* whether the problem is categorized as low- or high-dimensional;  
* the scientific structure or behavior it is intended to represent;  
* why its composite structure may be informative.

Present the low-dimensional problems first, followed by the high-dimensional problems.

For each synthetic problem, briefly connect the construction to a real scientific setting. For example, an intermediate vector could represent a spectrum, spatial field, time series, or collection of material properties.

Place the complete mathematical definitions in Appendix B.

#### **4.1.2 Real-World Problems**

Use one paragraph for each real-world benchmark.

For every problem, specify:

* the scientific application;  
* the design variable (\\mathbf{x});  
* the input dimension (d);  
* the number of objectives;  
* the intermediate response associated with each objective;  
* the dimension of each intermediate response;  
* the known transformations (g\_i);  
* the evaluation cost or simulator used;  
* the source of the benchmark.

Cite the paper or repository from which each benchmark is obtained.

Again, present low-dimensional benchmarks before high-dimensional benchmarks if both classes are included.

Move implementation-specific and scientific details that are not necessary for understanding the experiment to Appendix B.

### **4.2 Evaluation Metrics and Experimental Settings**

Define the primary performance metric, such as dominated hypervolume or hypervolume regret.

If using hypervolume, specify:

* whether objectives are normalized;  
* the reference point;  
* whether a known or approximate Pareto front is used;  
* whether the reported quantity is hypervolume or hypervolume difference;  
* how comparisons are made across benchmarks with different objective scales.

Report the following experimental settings:

* number of independent random seeds;  
* number of initial observations;  
* initialization procedure;  
* total number of optimization iterations;  
* batch size, if applicable;  
* surrogate-model architecture;  
* observation-noise assumptions;  
* model-fitting procedure;  
* software packages and versions;  
* computational hardware, when relevant.

State that Gaussian-process hyperparameters are estimated by maximizing the marginal log likelihood, rather than referring to this simply as “MLE.” If priors are placed on the GP hyperparameters and included during fitting, describe the procedure as maximum a posteriori estimation.

Cite BoTorch, GPyTorch, and PyTorch as appropriate.

Report results using the mean across random seeds. Use shaded regions to represent either one standard deviation or a confidence interval, and state clearly which quantity is shown.

Use a consistent visual convention. For example:

* solid line: objective-space method;  
* dashed line: matched composite-aware variant.

Provide complete acquisition-optimization settings in Appendix C, including:

* number of raw samples;  
* number of optimizer restarts;  
* number of Monte Carlo samples;  
* trust-region settings;  
* scalarization-sampling procedure;  
* GP training settings.

---

## **5\. Results and Discussion**

The organization of this section should reflect the scientific questions rather than simply describing each figure.

### **5.1 Low-Dimensional Problems**

Summarize the performance of composite-aware and objective-space methods on the low-dimensional benchmarks.

Discuss:

* how often composite modeling improves performance;  
* the magnitude of the improvement;  
* whether the benefit occurs early or late in the optimization process;  
* whether the improvement is consistent across MOBO frameworks.

For cases in which composite modeling does not help, provide plausible explanations supported by the experiments. Possibilities include:

* the objective-space problem is already easy to model;  
* the intermediate response has much higher dimension than the final objective;  
* the intermediate-output model is misspecified;  
* the mapping (g\_i) discards most of the information in (\\mathbf{h}\_i);  
* the additional modeling complexity outweighs the information gained from the composite structure.

Avoid attributing outcomes to a cause unless it is supported by an ablation or diagnostic analysis.

### **5.2 High-Dimensional Problems**

Repeat the analysis for high-dimensional benchmarks.

Emphasize whether the relative value of composite modeling changes as the input dimension increases.

Discuss possible interactions between composite modeling and the high-dimensional optimization mechanism. For example, composite modeling may improve posterior accuracy while the trust-region or dimensionality-reduction component improves acquisition optimization.

Report quantitative aggregate results, such as:

* average final hypervolume rank;  
* median improvement over matched baselines;  
* number of benchmark–method pairs improved by composite modeling;  
* average number of evaluations saved in reaching a target performance level.

### **5.3 When Does Composite Modeling Help?**

If space permits, include a short synthesis subsection identifying the characteristics associated with successful composite modeling.

Potential factors include:

* input dimension;  
* intermediate-output dimension;  
* smoothness of the intermediate response;  
* complexity of the known transformation;  
* strength of correlation across intermediate outputs;  
* number of objectives;  
* available evaluation budget.

This subsection would strengthen the paper by providing practical guidance rather than only reporting benchmark rankings.

### **Figures**

Use one main figure for low-dimensional problems and one for high-dimensional problems.

Each figure may contain several subplots, but keep the number manageable and use consistent axis limits, labels, and legends.

Consider adding a compact summary table reporting final performance or normalized method ranks across all benchmarks. This may communicate the aggregate result more clearly than convergence plots alone.

---

## **6\. Conclusion**

Restate the central question: whether exploiting known composite structure improves high-dimensional multi-objective Bayesian optimization.

Summarize the main empirical findings and quantify the overall improvement.

State the principal practical conclusion, such as which types of problems and MOBO methods benefit most from composite modeling.

Acknowledge the principal limitations, including benchmark coverage, computational overhead, assumptions about observing intermediate responses, and dependence on known transformations.

Conclude with future directions, such as partial or noisy composite structure, shared intermediate responses, adaptive selection between direct and composite models, and validation in physical laboratory experiments.

---

# **Appendices**

## **Appendix A: Detailed Method Descriptions**

Include complete algorithmic details, acquisition functions, surrogate-model specifications, and pseudocode.

## **Appendix B: Detailed Benchmark Descriptions**

Provide mathematical definitions of all synthetic functions and full descriptions of the real-world problems.

## **Appendix C: Optimization and Implementation Configurations**

Report acquisition-optimization settings, GP-fitting settings, software details, and any benchmark-specific configurations.

## **Appendix D: Ablation and Diagnostic Studies**

Possible ablations include:

* composite versus objective-space modeling under identical acquisition functions;  
* effect of intermediate-output dimension;  
* effect of observation noise;  
* effect of the number of initial observations;

# Rough idea

**Abstract** 

- 1 sentence explaining the importance of scientific design  
- 1 sentence mentioning that it is usually optimizing multiple objectives at the same time; provide an example  
- 1 sentence explaining challenges of solving the problem – maybe its expensiveness, black-box, high-dimension.  
- 1 sentence explaining MOBO can be used as a generic solution.  
- 1-2 sentences explaining that some applications the objectives exhibit composite structure and MOBO ignoring that.   
- 1-2 sentences explaining our objective; investigate the benefits of leveraging CF in MOBO frameworks for high dimensional problems  
- 1 sentence explaining the number of test cases  
- 1 sentence explaining the summary of the finding, i.e. most MOBO frameworks when using composite performance better. You should also add the quantity number like converge how many times faster on average. 

**Introduction**

- One paragraph talking about scientific design  
- One paragraph explaining the properties of the problems: expensive, black-box, high-dimension \*\* You can pick one application and keep talking about it  
- One paragraph mentions standard BO (cite BO paper) and mentions its applications \[trying to mention the ones that related to scientific design; formulation design\] (with citations) and says that it has been extended to multi-objective (cite original MOBO paper).   
- One paragraph mentions a high-dimensional MOBO framework, briefly giving a 1-2 sentence summary for each framework. \[Citing those MORBO etc\]  
- One paragraph mentions an alternative method, chebyshev and how it can be extended to high-dimensional problems.  
- One paragraph mentioning the problems usually comes with composite structure. BOCF has been proposed for a singular objective (cite BOCF) and the idea has been extended to multi-output (cite Max workshop paper). Mention that Max workshop paper is the closest. However, in that work they studied the impact of HOGP. State what we are going to do and why this is significantly different from Max workshop.  
- One paragraph mentioning the methods considers and test cases (briefly)   
- One paragraph summarizing the results.

**Problem setting**

- **Have a preamble paragraph saying what will be covering in this section: notation, assumptions**  
- Describe the structure of the problems – clearly define notation x\\in R^{d}, d is big, h\_i, f\_i, g\_i. 

**Considered Methods**

- **Have a preamble paragraph saying what will be covering in this section: list methods**  
- Have one subsection for each method. Explain clearly what they do. Give a good abbreviation for methods – this will be used in the result section

**Numerical Experiments**

- **Have a preamble paragraph saying what will be covered in this section: saying we have both synthetic and real-world test cases, what comparison metric we will use.**  
- One subsection for test cases  
  - One subsubsection for synthetic problems – just mention the function names no need to write the equation, mention that this represents of real problem for example first function can be .. second function can be in the real life. If you consider two classes (low and high-dimension), mention low first and then follow by high.  
  - One subsubsection for real-world test cases: One paragraph for each problems. You should mention what are x, what is d, how many objectives and in each objective what is the dimension of intermediate output.  If you consider two classes (low and high-dimension), mention low first and then follow by high.  
  - Cite all papers you take their benchmarks from  
  - Mention that full explanation will be in Appendix  
- One subsection for comparison metric and experiment settings including number of seed, number of initial observations, number of iterations, how did we implement cite Botorch, pytorch. Mention that optimization configurations (raw sample, optimizer, monte carlo sample) will be in appendix. GP model is trained by maximizing MLE? Metric is presented in mean and standard deviate (denoted by shaded area). Methods with composite is denoted by dash while its counterpart without composite is denoted by solid line, etc.

**Result and Discussion**

- One paragraph for low dimension problem; summarizing what you found, using CF is better? Why? For the one that is not better why? Objective is too easy? Or problems are too complicated?  
- One paragraph for high dimensional problem; same thing as above  
- Have one figure with subplots for low dim problem and another one for high dim problem.

**Conclusion**

**Appendices**

- **Appendix A: Detailed explanation of methods**  
- **Appendix B: Detailed explanation of problem**  
- **Appendix C: Optimization configurations**  
- **Appendix D: Ablation if you have one.**