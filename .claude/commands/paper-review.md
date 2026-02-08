---
description: Read an academic paper PDF and produce methodology documentation for implementation
argument-hint: "<pdf-path> [--name <estimator-name>] [--confirm]"
---

# Paper Review

Read an academic paper (PDF) and produce structured methodology documentation
suitable for implementation. Uses multi-agent extraction to handle long papers
without overwhelming the context window.

## Arguments

Parse `$ARGUMENTS` to extract:
- **PDF path** (required): First positional argument. Must be a path ending in `.pdf`.
- `--name <name>` (optional): Estimator/method name for the output file slug.
  If omitted, derive from the paper title on page 1: first author's last name,
  lowercased, hyphenated with year (e.g., "callaway-santanna-2021").
  If year isn't on page 1, omit it from the slug.
- `--confirm` (optional flag): Pause after reconnaissance to let user adjust
  page ranges. Also pause after extraction to let user review before synthesis.

If no PDF path is provided or it doesn't end in `.pdf`, use AskUserQuestion to request it.

---

## Synthesis Output Template

This template is used by both the short-paper fast path and the Phase 3 synthesis agent:

```
# Paper Review: {Full Paper Title}

**Authors:** {author names}
**Citation:** {Author, I. (Year). Title. *Journal*, Vol(Issue), Pages.}
**PDF reviewed:** {pdf-path}
**Review date:** {today's date}

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries — copy the `## {EstimatorName}` section into the appropriate category in the registry.*

## {EstimatorName}

**Primary source:** {citation with DOI/URL if available}

**Key implementation requirements:**

*Assumption checks / warnings:*
- {assumption 1}
- {assumption 2}

*Estimator equation (Equation {N} in paper, as implemented):*

    {main equation with clear notation}

where:
- {symbol} = {meaning}

*With covariates / doubly robust (Equation {M}, if applicable):*

    {DR or covariate-adjusted equation}

*Standard errors (Section {X.Y}):*
- Default: {default SE method}
- Alternative: {alternative methods}
- Bootstrap: {type, weight distribution, recommended iterations}
- Clustering: {clustering level}

*Edge cases:*
- {case 1}: {detection} -> {handling}
- {case 2}: {detection} -> {handling}

*Algorithm (Algorithm {N} in paper):*
1. {step 1}
2. {step 2}

**Reference implementation(s):**
- R: {package}::{function}()
- Stata: {command}

**Requirements checklist:**
- [ ] {requirement 1}
- [ ] {requirement 2}

---

## Implementation Notes

### Data Structure Requirements
- {input data format}
- {required columns/variables}

### Computational Considerations
- {complexity}
- {memory requirements}
- {parallelization opportunities}

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| {param}   | {type} | {default} | {how to choose} |

### Relation to Existing diff-diff Estimators
- {how this relates to estimators already in the library}
- {which existing code could be reused}

---

## Gaps and Uncertainties

{Anything unclear, contradictions, missing details.
Include specific page references for the reader to consult.}
```

---

## Phase 1: Validation

**Step 1: Validate PDF and extract metadata**

Read page 1 of the PDF using the Read tool (pages: "1"). If it returns an error:

> Error: Cannot read PDF at {path}. Verify the file exists and is a valid PDF.
> If this is a scanned PDF without a text layer, this skill cannot process it.

If page 1 reads successfully but returns garbled or mostly empty content, warn via AskUserQuestion (options: Proceed / Cancel):

> Warning: This PDF may be a scanned document without a text layer.
> Extraction quality will be poor. Consider using an OCR tool first.

From page 1, extract: paper title, author names, abstract.

**Step 2: Derive output name**

If `--name` was provided, use it as the slug. Otherwise derive from the paper title:
- Take the first author's last name
- Lowercase it
- If a year is visible on page 1, append it with a hyphen (e.g., "callaway-santanna-2021")
- If no year, just use the name (e.g., "callaway-santanna")

**Step 3: Update .gitignore and create directories**

First, check if `.gitignore` already contains `.claude/paper-review/`. If not, add it directly after the `.claude/reviews/` line. Then create directories:

```bash
mkdir -p docs/methodology/papers
mkdir -p .claude/paper-review/sections
```

**Step 4: Check for existing output**

If `docs/methodology/papers/{paper-name}-review.md` exists, use AskUserQuestion:
- **Overwrite**: Replace the existing review
- **Cancel**: Stop. Tell the user to re-run with `--name <different-name>` to avoid the collision

**Step 5: Status message**

Tell the user: "Scanning paper structure..."

---

## Phase 1.5: Reconnaissance (Scout Agent)

Launch 1 sub-agent via Task tool (subagent_type: "general-purpose", model: "haiku", max_turns: 20).

Task description: "Determine paper structure and length"

Scout agent prompt — include the PDF path and these instructions:

> You are scanning an academic paper PDF to determine its structure. Do NOT extract content — only determine page boundaries.
>
> PDF path: {pdf-path}
>
> **Step 1: Find total pages via binary search.**
> Try reading a single page at a time using the Read tool (pages: "{N}"). A page exists if the Read tool returns content. A page does NOT exist if the Read tool returns an error (any error means the page is beyond the document).
>
> Pseudocode:
>
>     # Phase 1: Exponential probe to find upper bound
>     low = 1, probe = 64
>     if Read(page 1) fails: total_pages = 0, stop.
>     if Read(page probe) fails:
>         high = probe  # paper is shorter than probe
>     else:
>         while Read(page probe) succeeds:
>             low = probe
>             probe = probe * 2
>             if probe > 2000: break  # safety cap
>         high = probe
>     # Phase 2: Binary search within [low, high]
>     while high - low > 3:
>         mid = (low + high) // 2
>         if Read(page mid) succeeds: low = mid
>         else: high = mid
>     # Phase 3: Linear scan for exact last page
>     For N from low to high:
>       Try page N. If fails: last_page = N - 1. Break.
>     If all succeed: last_page = high.
>
> Example for a 12-page paper: page 64(fail)->low=1,high=64. Binary: 32(fail)->16(fail)->8(success)->12(success)->14(fail)->linear scan 12,13,14: 13(fail)->total_pages=12.
> Example for a 150-page paper: page 64(success)->128(success)->256(fail)->low=128,high=256. Binary: 192(fail)->160(fail)->144(success)->152(fail)->148(success)->linear scan 148,149,150,151: 151(fail)->total_pages=150.
>
> **Step 2: Find references section.**
> Read the last 15 pages of the paper (in one Read call if <=20 pages, or two calls). Scan backwards from the end looking for a heading line containing "References", "Bibliography", or "Works Cited" (case-insensitive — also match "REFERENCES", "references", etc.). Report the page number where that heading appears. If not found after scanning the last 15 pages, set references_start_page = total_pages + 1. (15 pages covers virtually all reference sections. If a paper has >15 pages of references, the user can correct via `--confirm`.)
>
> **Step 3: Report results.**
> End your response with exactly these three lines (the main context will parse them):
>
>     SCOUT_TOTAL_PAGES: {N}
>     SCOUT_REFERENCES_PAGE: {N}
>     SCOUT_CONTENT_PAGES: {references_start_page - 1}

After the scout completes, parse the three `SCOUT_` values from the agent's response text.

**Determine execution path:**
- content_pages <= 20: short-paper fast path (skip to next section)
- content_pages 21-35: 2 extraction agents
- content_pages 36-60: 3 extraction agents
- content_pages 61+: 4 extraction agents

Compute page ranges using the allocation table:

| content_pages | Agents | Ranges (with 5-page overlaps) |
|---------------|--------|-------------------------------|
| 21-35         | 2      | mid = content_pages // 2; A: 1 to mid+2, B: mid-2 to end |
| 36-60         | 3      | A: 1-20, B: 16-35, C: 31 to end |
| 61+           | 4      | A: 1-20, B: 16-35, C: 31-50, D: 46 to end |

"end" = references_start_page - 1 (or last page if references not found).

**If `--confirm`:** Present the scout results and computed page ranges via AskUserQuestion.
Options: Proceed with these ranges / Cancel. Let the user adjust ranges or override total pages
by selecting "Other" and providing custom values.

**Scout failure fallback:** If the scout agent fails or returns unparseable results, fall back to default page ranges (1-20, 16-35, 31-50) with 3 agents. Warn the user that ranges are heuristic.

---

## Short-Paper Fast Path (<=20 content pages)

If content_pages <= 20:

1. Read the entire paper in the main context using Read tool (pages: "1-{content_pages}")
2. Extract ALL information from all three extraction templates (Core Methodology + Estimation & Inference + Edge Cases) in a single pass
3. Write the output directly to `docs/methodology/papers/{paper-name}-review.md` using the Synthesis Output Template defined above
4. Skip Phases 2-4 entirely
5. Clean up: `rm -rf .claude/paper-review`
6. Report output path to user

---

## Phase 2: Parallel Extraction

Tell user: "Launching {N} extraction agents for pages 1-{content_pages}..."

Launch N sub-agents **simultaneously** via Task tool (subagent_type: "general-purpose", max_turns: 20). Do not specify a model — agents inherit the user's current model for equation transcription accuracy.

**Read tool constraint:** Max 20 pages per Read call. Each agent must chunk reads into <=20-page requests.

**Common instructions for all agents** — include in each agent's prompt:

> - Read your assigned pages of the PDF at {pdf-path} using the Read tool. Chunk into <=20-page reads.
> - **Stop before page {references_start_page}** — that's where references begin.
> - If any pages in your range don't exist (Read returns error), work with what's available and note the last readable page.
> - If you see a cross-reference to content outside your page range, note it as a gap rather than reading extra pages.
> - Use code blocks for all equations. Be precise with subscripts and notation.
> - **Preserve all equation numbers, algorithm numbers, theorem numbers, and section references** from the paper (e.g., "Equation 7", "Theorem 3.1", "Algorithm 2"). These are essential for cross-referencing during implementation.
> - If you encounter equations or formulas that are garbled, illegible, or appear to be embedded images rather than text, note them with **[UNREADABLE EQUATION on page {N}]** so the reader knows to check the source PDF manually.
> - Write output using the Write tool to your assigned file path.

### Agent A — Core Methodology

Task description: "Extract core methodology from paper"
Page range: {computed from scout}
Output file: `.claude/paper-review/sections/01-core-methodology.md`

Extraction template to include in the agent prompt:

> Extract information following this template and write to {output_file}:
>
> ## Core Methodology Extraction
> Paper: {title} | Pages read: {actual range}
>
> ### Model / Data Structure
> - Data structure (panel, cross-section, repeated cross-section)
> - Key variables (outcome, treatment, covariates, fixed effects)
> - Key notation: list every symbol and its meaning
>
> ### Identification
> - Identifying assumptions (parallel trends, conditional independence, etc.)
> - Target parameter (ATT, ATE, CATT, etc.) — exact definition as an equation
> - What must hold for the estimator to be consistent?
>
> ### Main Estimator Equation(s)
> - Write each estimator equation exactly as it appears
> - Include equation numbers from the paper (e.g., "Equation 7")
> - If multiple variants (regression adjustment, IPW, doubly robust), document each
> - Note which variant the authors recommend
>
> ### Comparison Groups
> - What serves as the control/comparison group?
> - Restrictions on which units can serve as controls
>
> ### Key Theorems / Propositions
> - List key results with theorem/proposition numbers (e.g., "Theorem 3.1")
> - What each establishes (consistency, asymptotic normality, efficiency, etc.)
> - Convergence rates if mentioned

### Agent B — Estimation & Inference

Task description: "Extract estimation and inference details from paper"
Page range: {computed from scout}
Output file: `.claude/paper-review/sections/02-estimation-inference.md`

Extraction template to include in the agent prompt:

> Extract information following this template and write to {output_file}:
>
> ## Estimation & Inference Extraction
> Paper: {title} | Pages read: {actual range}
>
> ### Algorithm / Computation Steps
> - Step-by-step algorithm for computing the estimator
> - If a named algorithm exists (e.g., "Algorithm 1"), transcribe it exactly with its label
> - Optimization problems to solve (OLS, weighted LS, LP, QP, etc.)
> - Computational complexity if discussed
>
> ### Standard Errors / Variance
> - Exact SE/variance formula with equation number (e.g., "Equation 12")
> - Analytical, bootstrap-based, or both?
> - Clustering level for cluster-robust SEs
> - Degrees of freedom / small-sample corrections
>
> ### Bootstrap Procedure (if applicable)
> - Bootstrap type (multiplier, pairs, block, wild cluster, etc.)
> - Weight distribution (Rademacher, Mammen, Webb, etc.)
> - Recommended number of iterations
> - Transcribe the bootstrap algorithm if provided (with its label)
>
> ### Aggregation (if applicable)
> - How group-time or cohort-specific effects are aggregated
> - Aggregation weight formulas with equation numbers
> - SE computation for aggregated quantities (delta method, bootstrap, influence function)
>
> ### Tuning Parameters (if applicable)
> - Hyperparameters and their selection methods (CV, information criterion, rule of thumb)
> - Recommended default values

### Agent C — Edge Cases & Practical Details

Task description: "Extract edge cases and practical details from paper"
Page range: {computed from scout}
Output file: `.claude/paper-review/sections/03-edge-cases-practical.md`

Extraction template to include in the agent prompt:

> Extract information following this template and write to {output_file}:
>
> ## Edge Cases & Practical Details Extraction
> Paper: {title} | Pages read: {actual range}
>
> ### Extensions / Variants
> - Multiple variants of the estimator?
> - Extensions for special data structures (unbalanced panels, staggered adoption, etc.)
> - Optional features or parameters
>
> ### Appendix Content (if your pages include an appendix)
> - Proofs that reveal implementation-relevant constraints
> - Supplementary algorithms or computational details
> - Additional simulation results not in the main text
>
> ### Simulation Findings
> - Simulation designs tested
> - What breaks the estimator (bias, poor coverage)?
> - Sample size / panel dimension requirements
> - Performance vs alternatives
>
> ### Edge Cases & Boundary Conditions
> - Small samples, few treated/control units
> - What happens when assumptions are violated
> - Rank deficiency or collinearity
> - Missing data handling
>
> ### Practical Recommendations
> - Author-recommended parameter values
> - Warnings about common misuse
> - When to use this vs alternatives
>
> ### Reference Implementations
> - Software packages mentioned (R, Stata, Python)
> - Package names and function names
> - Replication code references
>
> ### Relation to Other Methods
> - How this relates to other estimators in the literature
> - Under what conditions it reduces to simpler methods
> - Key advantages over alternatives

### Agent D — Extended Content (only for papers 61+ content pages)

Task description: "Extract extended appendix content from paper"
Page range: {computed from scout — typically the tail section}
Output file: `.claude/paper-review/sections/04-extended.md`

Same extraction template as Agent C, focused on appendix proofs, supplementary results, and implementation details that appear in extended appendices.

---

## Phase 2.5: Optional Review Pause (if --confirm)

After all extraction agents complete, if `--confirm` flag is set:

1. Tell the user the extraction files are ready at `.claude/paper-review/sections/`
2. Use AskUserQuestion:
   - **Proceed to synthesis**: Continue to Phase 3
   - **Abort**: Stop the skill. Extraction files are preserved at `.claude/paper-review/sections/` for manual use

---

## Phase 3: Synthesis

**Pre-check:** Verify which section files exist (use Glob for `.claude/paper-review/sections/*.md`).
If zero files were produced, report failure and stop — do not attempt synthesis.

Tell user: "Synthesizing paper review from {N} extraction files..."

Launch 1 sub-agent via Task tool (subagent_type: "general-purpose", max_turns: 20). Do not specify a model — inherits from parent.

Task description: "Synthesize paper review from extracted sections"

Synthesis agent prompt:

> You are synthesizing a paper review from extracted sections into a single output document.
>
> **Step 1:** Read these files (skip any that don't exist):
> - .claude/paper-review/sections/01-core-methodology.md
> - .claude/paper-review/sections/02-estimation-inference.md
> - .claude/paper-review/sections/03-edge-cases-practical.md
> - .claude/paper-review/sections/04-extended.md (if it exists)
>
> **Step 2:** Read docs/methodology/REGISTRY.md until you've seen the first complete `## ` estimator section (stop at the next `## ` heading or `---` separator). This gives you the target format.
>
> **Step 3: Handle overlapping content.** When the same equation, algorithm, or concept appears in multiple extraction files:
> - Prefer the version from the agent whose extraction template best matches the content (e.g., Agent A for equations, Agent B for algorithms)
> - If both contain unique details, merge them
> - Never include the same equation or algorithm twice
>
> **Step 4: Handle contradictions.** If extraction files contradict each other:
> - Note the contradiction in "Gaps and Uncertainties" with page numbers
> - Include both versions
> - Do NOT guess which is correct
>
> **Step 5: Preserve paper references.** Keep all equation numbers, algorithm numbers, theorem numbers, and section references from the extraction files. Annotate each item with its paper reference (e.g., "Equation 7", "Theorem 3.1", "Algorithm 2", "Section 4.2").
>
> **Step 6:** Write the synthesized document to docs/methodology/papers/{paper-name}-review.md using this template:
>
> {Include the full Synthesis Output Template from above, with all placeholders}
>
> Fill in all sections from the extraction files. For any section where no information was found, write "Not discussed in paper" rather than leaving it blank.
>
> Paper metadata:
> - Title: {title}
> - Authors: {authors}
> - PDF path: {pdf-path}
> - Paper name slug: {paper-name}
> - Today's date: {today's date}

---

## Phase 4: Cleanup & Report

**Step 1: Check synthesis success.** Read the output file at `docs/methodology/papers/{paper-name}-review.md` to verify it exists.

- If it exists: proceed to cleanup
- If it doesn't: skip cleanup, report failure, tell user extraction files are preserved at `.claude/paper-review/sections/`

**Step 2: Cleanup (success only).**

```bash
rm -rf .claude/paper-review
```

Remove the entire temp directory.

**Step 3: Report to user.**

> Paper review complete.
>
> Output: docs/methodology/papers/{paper-name}-review.md
>
> The file contains:
> - A Methodology Registry entry (ready to copy into REGISTRY.md)
> - Implementation notes and tuning parameters
> - Gaps and uncertainties flagged for manual review
>
> Recommended next steps:
> - Review the output, especially the "Gaps and Uncertainties" section
> - When implementing, copy the Registry Entry into docs/methodology/REGISTRY.md

---

## Error Handling

- **PDF doesn't exist or can't be read:** Report error, stop immediately. Do not create directories or launch agents.
- **Scanned PDF / no text layer:** Warn via AskUserQuestion (Proceed / Cancel).
- **Scout agent fails:** Fall back to default page ranges (1-20, 16-35, 31-50). Warn the user that ranges are heuristic.
- **All extraction agents fail (zero section files):** Report failure, stop. Do not attempt synthesis.
- **Some extraction agents fail:** Proceed with synthesis using available files. Synthesis agent notes gaps.
- **Synthesis agent fails:** Preserve extraction files at `.claude/paper-review/sections/`. Report failure, tell user files are available for manual review.
- **Existing output file:** Ask user to overwrite or cancel (Phase 1, Step 4).

---

## Examples

```
# Basic usage
/paper-review ~/papers/callaway-santanna-2021.pdf

# With explicit estimator name
/paper-review ~/papers/athey-imbens-2025.pdf --name TROP

# With confirmation pauses
/paper-review ~/papers/roth-2022.pdf --confirm --name PreTrendsPower
```

## Notes

- Papers >20 content pages use multi-agent extraction; <=20 use single-pass
- Scout agent uses haiku model (fast, cheap structural scanning)
- Extraction agents use the default model (equation transcription needs accuracy)
- Equation extraction is best-effort — complex multi-line equations may need manual verification
- The output is a starting point, not a final product — always review "Gaps and Uncertainties"
- Output files in docs/methodology/papers/ are intended to be committed to the repository
