# Decision Log

### Phase 1: Taxonomy & Golden Set
- **Decision:** I used a 5-category intent taxonomy (`general_inquiry`, `technical_support`, `billing_issue`, `complaint`, `feature_request`).
- **Rationale:** This covers the majority of standard SaaS/B2B support queries. I chose to use the LLM to auto-label the 150 golden set rows to save time, acknowledging this introduces some bias.

### Phase 2: Baselines
- **Decision:** I implemented two baselines: a Trivial baseline (always predicts majority class 'general_inquiry' and never escalates) and a Simple baseline (uses regex rules).
- **Rationale:** This provides a realistic floor to prove that the heavy LLM Agent is actually adding value over simple `if/else` logic.

### Phase 3: RAG Implementation
- **Decision:** I used `SentenceTransformers` (`all-MiniLM-L6-v2`) and `FAISS` for retrieval.
- **Rationale:** FAISS is extremely fast and works perfectly completely locally without needing a heavy vector database like Pinecone. MiniLM is lightweight enough to run on a CPU without a dedicated GPU.

### Phase 4: The Agent
- **Decision:** I forced the LLM to output strict JSON schemas for both intent classification and escalation logic.
- **Rationale:** In a production system like Hiver, you cannot rely on regex to parse LLM outputs. JSON schemas ensure the output can be reliably fed into downstream APIs or databases.

### Phase 5: LLM API Choice
- **Decision:** Integrated the NVIDIA `nemotron-3-ultra-550b-a55b` model as the primary LLM provider.
- **Rationale:** Handled API rate limiting by implementing an exponential backoff retry mechanism inside `llm_client.py`.

### Phase 6: Evaluation
- **Decision:** I used a 1-5 scale LLM Judge prompt to evaluate the final generated replies against the golden set reference notes.
- **Rationale:** Evaluating generative text via exact match (BLEU/ROUGE) is terrible for support replies. An LLM judge can evaluate empathy, brand voice, and correctness simultaneously.
