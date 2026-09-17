# Hiver SDE Intern Take-Home Report

**Candidate:** Abhishek Singh Dhangar

## 1. Problem Framing
For Hiver, a customer service platform, "good" does not mean simply deflecting tickets—it means empowering support agents and ensuring the customer feels heard and accurately assisted. 

I chose **not to build** a completely unconstrained generative chatbot that hallucinates policies or answers generic questions without context. Instead, I built a highly constrained Agent that classifies intent, determines escalation necessity, and grounds its responses strictly in past resolved tickets using Retrieval Augmented Generation (RAG).

## 2. Results vs. Baselines

| Metric | Trivial (Majority Class) | Simple (Regex/Rules) | AI Agent (Nemotron-3-Ultra) |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | **78.0%** | 74.6% | 72.0% |
| **Intent Macro-F1** | 0.22 | 0.37 | **0.52** |
| **Escalation F1** | 0.00 | 0.00 | **0.15** (Recall: 56%) |
| **Reply Quality (LLM Judge 1-5)** | N/A | 2.04 | **3.93** |

**Interpretation:**
At first glance, the Trivial baseline has the highest Intent Accuracy (78%). However, this is deeply misleading due to class imbalance—it simply guesses "general_inquiry" for everything, resulting in a terrible Macro-F1 score (0.22). The AI Agent is the only system that can actually distinguish between nuanced categories (achieving a Macro-F1 of 0.52) and successfully identify escalations (56% recall vs 0% for baselines). Furthermore, the Agent's generated responses are drastically superior, scoring an average of 3.93/5 on the LLM Judge across groundedness, helpfulness, tone, and actionability, compared to the Simple baseline's 2.04/5.

## 3. Failure Analysis
Despite the high accuracy, the Agent failed in the following key ways:
1. **Sarcasm / Implicit Frustration:** The model sometimes missed escalations when customers were highly sarcastic but did not use explicit anger keywords.
2. **Missing Grounding Context:** For highly specific edge-case questions (e.g. a rare billing bug), if the FAISS retriever failed to find a relevant past ticket, the LLM defaulted to a generic "Please DM us" response rather than trying to solve the problem.
3. **Over-Escalation:** The prompt errs on the side of caution, leading to some false-positive escalations when a user simply uses the word "supervisor" in a non-threatening context.
4. **Formatting Quirks:** The model occasionally output markdown reasoning blocks alongside the JSON response, requiring strict parsing logic.
5. **Rate Limiting Latency:** Relying on a heavy 550B parameter model caused API rate limits (HTTP 429), slowing down the pipeline significantly.

## 4. "What is misleading about my headline number?"
The fact that the Trivial baseline beats the AI Agent in raw Intent Accuracy (78% vs 72%) is highly misleading. It hides the fact that the Agent is vastly superior at handling minority classes (like escalations and complaints), which are arguably the most critical tickets for a support platform. Additionally, the 3.93/5 Reply Quality score has some blind spots:
- **Imperfect Golden Set:** I used an LLM to auto-label the golden set. Thus, the LLM Judge is effectively grading responses based on its own biases and generated labels. This inflates the Agent's score.
- **Resolution Heuristic:** The 1-5 grading system is heavily influenced by tone and style. The model might sound extremely polite and empathetic (scoring a 5), but completely fail to actually resolve the customer's technical issue.
- **Narrow Taxonomy:** The intent taxonomy is limited to 5 categories. Real-world Hiver customer queries would span hundreds of nuanced sub-intents.

## 5. Next Steps
If I had one more week to work on this, I would prioritize:
1. **Active Learning & Human-in-the-Loop:** Instead of auto-labeling the golden set, I would build a simple UI for a human QA tester to correct the LLM's labels.
2. **Latency & Cost Optimization:** I would swap the heavy Nemotron-3-Ultra model for a smaller, fine-tuned model (e.g., Llama-3-8B) for intent classification and escalation to save costs and reduce the 40-second latency per query.
3. **Advanced RAG:** I would implement hybrid search (BM25 + Dense Embeddings) and a re-ranker to improve the retrieval of past tickets, reducing the "Missing Grounding Context" failure mode.
