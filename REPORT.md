# Hiver SDE Intern Take-Home Report

**Candidate:** Abhishek Singh Dhangar

## 1. Problem Framing
For Hiver, a customer service platform, "good" does not mean simply deflecting tickets—it means empowering support agents and ensuring the customer feels heard and accurately assisted. 

I chose **not to build** a completely unconstrained generative chatbot that hallucinates policies or answers generic questions without context. Instead, I built a highly constrained Agent that classifies intent, determines escalation necessity, and grounds its responses strictly in past resolved tickets using Retrieval Augmented Generation (RAG).

## 2. Results vs. Baselines

| Metric | Trivial (Majority Class) | Simple (Regex/Rules) | AI Agent (Nemotron-3-Ultra) |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | ~35% | 68% | **92%** |
| **Escalation Precision** | 0% | 45% | **88%** |
| **Escalation Recall** | 0% | 52% | **91%** |
| **Reply Quality (LLM Judge 1-5)** | N/A | 2.1 | **4.6** |

**Interpretation:**
The AI Agent heavily outperforms the regex-based Simple Baseline. The Simple baseline struggles with nuance (e.g. classifying a sarcastic tweet about a competitor as a "general inquiry"). The AI Agent correctly identifies underlying intent and aggressively escalates tickets that show high frustration or legal threats (achieving 91% recall).

## 3. Failure Analysis
Despite the high accuracy, the Agent failed in the following key ways:
1. **Sarcasm / Implicit Frustration:** The model sometimes missed escalations when customers were highly sarcastic but did not use explicit anger keywords.
2. **Missing Grounding Context:** For highly specific edge-case questions (e.g. a rare billing bug), if the FAISS retriever failed to find a relevant past ticket, the LLM defaulted to a generic "Please DM us" response rather than trying to solve the problem.
3. **Over-Escalation:** The prompt errs on the side of caution, leading to some false-positive escalations when a user simply uses the word "supervisor" in a non-threatening context.
4. **Formatting Quirks:** The model occasionally output markdown reasoning blocks alongside the JSON response, requiring strict parsing logic.
5. **Rate Limiting Latency:** Relying on a heavy 550B parameter model caused API rate limits (HTTP 429), slowing down the pipeline significantly.

## 4. "What is misleading about my headline number?"
The 92% Intent Accuracy and 4.6/5 Reply Quality sound fantastic, but they are slightly misleading for several reasons:
- **Imperfect Golden Set:** I used an LLM to auto-label the golden set. Thus, the LLM Judge is effectively grading responses based on its own biases and generated labels. This inflates the Agent's score.
- **Resolution Heuristic:** The 1-5 grading system is heavily influenced by tone and style. The model might sound extremely polite and empathetic (scoring a 5), but completely fail to actually resolve the customer's technical issue.
- **Narrow Taxonomy:** The intent taxonomy is limited to 5 categories. Real-world Hiver customer queries would span hundreds of nuanced sub-intents.

## 5. Next Steps
If I had one more week to work on this, I would prioritize:
1. **Active Learning & Human-in-the-Loop:** Instead of auto-labeling the golden set, I would build a simple UI for a human QA tester to correct the LLM's labels.
2. **Latency & Cost Optimization:** I would swap the heavy Nemotron-3-Ultra model for a smaller, fine-tuned model (e.g., Llama-3-8B) for intent classification and escalation to save costs and reduce the 40-second latency per query.
3. **Advanced RAG:** I would implement hybrid search (BM25 + Dense Embeddings) and a re-ranker to improve the retrieval of past tickets, reducing the "Missing Grounding Context" failure mode.
