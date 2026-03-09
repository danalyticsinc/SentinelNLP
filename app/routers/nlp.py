"""SentinelNLP — FastAPI router."""
import os
from fastapi import APIRouter, HTTPException, Query
from app.analyzers.text_analyzer import TextAnalyzer

router = APIRouter()
analyzer = TextAnalyzer()


@router.post("/analyze")
async def analyze(payload: dict, ai: bool = Query(default=False)):
    text = payload.get("text", "")
    if not text.strip():
        raise HTTPException(400, "No text provided.")
    result = analyzer.analyze(text)

    ai_result = None
    if ai and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from app.services.ai_nlp import AINLPEngine
            engine = AINLPEngine()
            ai_result = engine.deep_analyze(text)
        except Exception:
            pass

    return {
        "word_count": result.word_count,
        "sentence_count": result.sentence_count,
        "avg_sentence_length": result.avg_sentence_length,
        "language": result.language,
        "classification": result.classification,
        "readability_score": result.readability_score,
        "sentiment": {"label": result.sentiment.label, "score": result.sentiment.score, "confidence": result.sentiment.confidence},
        "entities": [{"text": e.text, "label": e.label} for e in result.entities],
        "keywords": result.keywords,
        "ai_analysis": ai_result,
    }


@router.post("/batch")
async def batch_analyze(payload: dict):
    texts = payload.get("texts", [])
    if not texts:
        raise HTTPException(400, "No texts provided.")
    if len(texts) > 50:
        raise HTTPException(400, "Max 50 texts per batch.")
    return {"results": [{"index": i, **{"sentiment": analyzer.analyze(t).sentiment.label, "keywords": analyzer.analyze(t).keywords[:5], "classification": analyzer.analyze(t).classification}} for i, t in enumerate(texts)]}


@router.post("/classify")
async def classify(payload: dict):
    texts = payload.get("texts", [])
    categories = payload.get("categories", [])
    if not texts or not categories:
        raise HTTPException(400, "Provide texts and categories.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "AI classification requires ANTHROPIC_API_KEY.")
    from app.services.ai_nlp import AINLPEngine
    engine = AINLPEngine()
    return {"results": engine.batch_classify(texts, categories)}


@router.get("/demo")
async def demo():
    samples = [
        "The new AI-powered dashboard is absolutely excellent. Our team loves the real-time insights and the performance is outstanding.",
        "System crashed again. This is the third time this week. Error code 500 keeps appearing. Terrible experience.",
        "Q3 revenue grew 23% YoY reaching $4.2M. Operating profit margin improved to 18%. Forecast remains positive.",
    ]
    return {"analyses": [{"text": t[:80] + "...", **{"sentiment": analyzer.analyze(t).sentiment.label, "classification": analyzer.analyze(t).classification, "keywords": analyzer.analyze(t).keywords[:5]}} for t in samples]}
