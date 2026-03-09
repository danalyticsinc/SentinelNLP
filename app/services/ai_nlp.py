"""SentinelNLP — Claude AI-powered deep text analysis."""
import os
import anthropic
import json
import re


class AINLPEngine:
    MODEL = "claude-opus-4-6"

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def deep_analyze(self, text: str) -> dict:
        message = self.client.messages.create(
            model=self.MODEL,
            max_tokens=1024,
            system="You are an enterprise NLP analyst. Analyze text and return structured JSON only.",
            messages=[{"role": "user", "content": f"""Analyze this text and return JSON with:
{{
  "summary": "1-2 sentence summary",
  "sentiment": "POSITIVE|NEGATIVE|NEUTRAL",
  "sentiment_reasoning": "why",
  "key_topics": ["topic1", "topic2", "topic3"],
  "tone": "formal|informal|urgent|neutral",
  "action_items": ["action if any"],
  "risk_flags": ["any concerning content"]
}}

Text: {text[:2000]}"""}],
        )
        raw = message.content[0].text
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"summary": raw[:300], "sentiment": "NEUTRAL", "key_topics": [], "tone": "neutral",
                "action_items": [], "risk_flags": []}

    def batch_classify(self, texts: list[str], categories: list[str]) -> list[dict]:
        results = []
        for text in texts[:10]:
            message = self.client.messages.create(
                model=self.MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": f"""Classify this text into one of: {categories}
Return JSON: {{"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
Text: {text[:500]}"""}],
            )
            raw = message.content[0].text
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                try:
                    results.append(json.loads(match.group()))
                    continue
                except Exception:
                    pass
            results.append({"category": categories[0], "confidence": 0.5, "reasoning": "fallback"})
        return results
