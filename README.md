# SmartExam Builder

SmartExam Builder is a demo-ready FastAPI system for AI-powered exam generation and grading in the Computer Networks domain. It generates Turkish questions from learning outcomes, retrieves relevant course material with a local RAG pipeline, improves output through an explicit Writer-Critic workflow, creates rubrics, and grades student answers.

The app runs out of the box with a deterministic mock LLM so demos and tests do not require network access. To use OpenAI for real LLM calls, set `SMARTEXAM_USE_MOCK_LLM=false` and provide `OPENAI_API_KEY`.

## Features

- Material ingestion from local folders or uploaded `.txt`, `.md`, and `.pdf` files
- Text extraction, chunking, metadata creation, embeddings, and FAISS indexing
- Learning outcome models for course, topic, id, text, and cognitive level
- Turkish multiple-choice and open-ended question generation
- Visible RAG response traces with retrieved chunk ids, source files, page numbers, scores, and previews
- Writer-Critic multi-agent workflow with revision loop
- Rubric generation for open-ended questions
- Automatic rubric-based grading with criterion-level feedback in Turkish
- Pydantic validation for structured LLM outputs
- Mockable LLM abstraction with retry and JSON repair logic for OpenAI responses
- Pytest coverage for schemas, chunking, retrieval, pipelines, and API smoke tests

## Folder Structure

```text
smart_exam_builder/
  app/
    main.py
    api/
    core/
    models/
    agents/
    services/
    rag/
    prompts/
    utils/
  data/
    raw/
    processed/
    samples/
  scripts/
  tests/
  README.md
  requirements.txt
  .env.example
  Dockerfile
```

## Quick Start

```bash
cd smart_exam_builder
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Demo Flow

1. Ingest sample materials:

```bash
curl -X POST http://127.0.0.1:8000/ingest-materials ^
  -F "course=Computer Networks"
```

2. Generate questions and show RAG visibility:

```bash
curl -X POST http://127.0.0.1:8000/generate-questions ^
  -H "Content-Type: application/json" ^
  -d "{\"learning_outcome_id\":\"LO1\",\"course\":\"Computer Networks\",\"difficulty\":\"medium\",\"question_type\":\"multiple_choice\",\"question_count\":2,\"top_k\":4}"
```

The response includes:

- `retrieved_chunks`: chunk id, source file, page number, chunk index, score, and preview text
- `questions[].source_chunks`: ids of chunks that influenced each question
- `writer_critic_trace`: Retriever, Writer, and Critic steps for presentation

3. Generate an open-ended question, then use it to generate a rubric:

```bash
curl -X POST http://127.0.0.1:8000/generate-questions ^
  -H "Content-Type: application/json" ^
  -d "{\"learning_outcome_id\":\"LO3\",\"question_type\":\"open_ended\",\"question_count\":1,\"top_k\":4}"
```

Send the returned question object to:

```text
POST /generate-rubric
```

4. Grade an answer:

```text
POST /grade-answer
```

with the returned question, generated rubric, and `student_answer`.

## CLI Demo

```bash
python scripts/demo_cli.py
```

This ingests the sample materials, generates an open-ended DHCP question, creates a rubric, and grades a sample answer.

## Using OpenAI

Edit `.env`:

```env
SMARTEXAM_USE_MOCK_LLM=false
OPENAI_API_KEY=your_api_key_here
SMARTEXAM_OPENAI_MODEL=gpt-4o-mini
```

All LLM calls go through `app/core/llm_client.py`, which validates responses with Pydantic. If the model returns invalid JSON, the client extracts/repairs JSON and retries.

## Writer-Critic Flow

`QuestionService` coordinates the workflow:

1. `RetrieverAgent` builds a query from the learning outcome and retrieves top-k source chunks.
2. `WriterAgent` drafts structured Turkish questions using the learning outcome and retrieved chunks.
3. `CriticAgent` checks alignment, clarity, answerability, difficulty, ambiguity, and distractor quality.
4. If rejected, `WriterAgent` revises once or twice using critic feedback.
5. The final response includes `writer_critic_trace` for easy explanation.

## RAG Implementation

The local RAG pipeline is intentionally simple:

- `loaders.py` extracts text from `.txt`, `.md`, and `.pdf`
- `text_chunker.py` creates retrieval-friendly chunks with metadata
- `embeddings.py` creates deterministic local embeddings
- `vector_store.py` stores vectors in FAISS when available, with a numpy fallback
- `retrieval.py` builds learning-outcome-aware queries

FAISS metadata is persisted under `data/processed/`. The generated question response exposes retrieved chunks so RAG behavior is visible during demos.

## Tests

```bash
pytest
```

Tests use the deterministic mock LLM and do not require `OPENAI_API_KEY`.

## Docker

```bash
docker build -t smart-exam-builder .
docker run --rm -p 8000:8000 smart-exam-builder
```

