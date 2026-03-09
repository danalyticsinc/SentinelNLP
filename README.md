# SentinelNLP

> **Enterprise text analytics API** — sentiment analysis, named entity recognition, document classification, keyword extraction, and Claude AI deep analysis. Zero heavy NLP dependencies.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Claude AI](https://img.shields.io/badge/Anthropic_Claude-AI_Analysis-8B5CF6)](https://anthropic.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)

---

## What It Does

| Feature | Detail |
|---|---|
| **Sentiment Analysis** | POSITIVE / NEGATIVE / NEUTRAL with confidence score |
| **Named Entity Recognition** | Persons, Organizations, Emails, URLs, Dates, Money, Phone numbers |
| **Document Classification** | Auto-classifies into: support ticket, product review, financial report, legal document, news article |
| **Keyword Extraction** | Top 10 keywords by frequency, stopword-filtered |
| **Readability Scoring** | Flesch Reading Ease score |
| **Batch Processing** | Analyze up to 50 texts in one request |
| **AI Deep Analysis** | Claude AI generates summary, tone, action items, risk flags |
| **Custom Classification** | Classify texts into your own categories via Claude AI |

---

## Quick Start

```bash
# Docker
export ANTHROPIC_API_KEY=sk-ant-...
docker build -t sentinelnlp . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY sentinelnlp

# Local
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API Reference

```bash
# Analyze single text
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The product quality is excellent and delivery was fast."}'

# Batch analyze
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great service!", "Terrible experience.", "Q3 revenue up 20%"]}'

# Custom classification with AI
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Server is down"], "categories": ["urgent", "billing", "feature_request"]}'

# Demo
curl http://localhost:8000/api/v1/demo
```

---

## Built By

Discovery Analytics Inc.
