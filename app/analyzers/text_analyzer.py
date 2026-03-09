"""
SentinelNLP — Text Analyzer
Rule-based NLP: sentiment scoring, entity extraction, language detection,
keyword extraction, and document classification.
"""
import re
from dataclasses import dataclass, field


@dataclass
class SentimentResult:
    label: str          # POSITIVE | NEGATIVE | NEUTRAL
    score: float        # -1.0 to 1.0
    confidence: float   # 0.0 to 1.0


@dataclass
class Entity:
    text: str
    label: str          # PERSON | ORG | EMAIL | URL | DATE | MONEY | PHONE
    start: int
    end: int


@dataclass
class TextAnalysis:
    text: str
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    sentiment: SentimentResult
    entities: list[Entity]
    keywords: list[str]
    language: str
    classification: str
    readability_score: float


POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "outstanding",
    "superb", "brilliant", "perfect", "love", "best", "happy", "satisfied", "awesome",
    "recommend", "helpful", "efficient", "innovative", "reliable", "impressive",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "poor", "worst", "hate", "useless",
    "broken", "failed", "disappointing", "slow", "expensive", "frustrating",
    "unreliable", "buggy", "crash", "error", "problem", "issue", "complaint",
}

ENTITY_PATTERNS = [
    (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', "PERSON"),
    (r'\b[A-Z][A-Z0-9]+(?: [A-Z][A-Z0-9]+)*\b', "ORG"),
    (r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', "EMAIL"),
    (r'https?://[^\s]+', "URL"),
    (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', "DATE"),
    (r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)\b', "MONEY"),
    (r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', "PHONE"),
]

DOC_CLASSES = {
    "support_ticket": {"error", "bug", "issue", "problem", "fix", "help", "support", "broken", "crash"},
    "product_review": {"product", "review", "quality", "recommend", "purchase", "bought", "delivery"},
    "financial_report": {"revenue", "profit", "loss", "quarter", "fiscal", "earnings", "forecast", "growth"},
    "legal_document": {"agreement", "contract", "clause", "liability", "party", "terms", "herein", "whereas"},
    "news_article": {"according", "reported", "announced", "statement", "official", "government", "said"},
}


class TextAnalyzer:

    def analyze(self, text: str) -> TextAnalysis:
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sent_len = len(words) / max(len(sentences), 1)

        sentiment = self._sentiment(words)
        entities = self._extract_entities(text)
        keywords = self._keywords(words)
        language = self._detect_language(words)
        classification = self._classify(words)
        readability = self._readability(words, sentences)

        return TextAnalysis(
            text=text[:200] + "..." if len(text) > 200 else text,
            word_count=len(words),
            sentence_count=len(sentences),
            avg_sentence_length=round(avg_sent_len, 1),
            sentiment=sentiment,
            entities=entities,
            keywords=keywords,
            language=language,
            classification=classification,
            readability_score=readability,
        )

    def _sentiment(self, words: list[str]) -> SentimentResult:
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return SentimentResult("NEUTRAL", 0.0, 0.5)
        score = (pos - neg) / total
        confidence = min(1.0, total / max(len(words) * 0.1, 1))
        label = "POSITIVE" if score > 0.1 else "NEGATIVE" if score < -0.1 else "NEUTRAL"
        return SentimentResult(label, round(score, 3), round(confidence, 3))

    def _extract_entities(self, text: str) -> list[Entity]:
        entities = []
        seen = set()
        for pattern, label in ENTITY_PATTERNS:
            for match in re.finditer(pattern, text):
                if match.group() not in seen:
                    seen.add(match.group())
                    entities.append(Entity(match.group(), label, match.start(), match.end()))
        return entities[:20]

    def _keywords(self, words: list[str]) -> list[str]:
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                     "of", "with", "is", "was", "are", "were", "be", "been", "have", "has",
                     "it", "this", "that", "i", "we", "you", "he", "she", "they", "not"}
        freq = {}
        for w in words:
            if w not in stopwords and len(w) > 3:
                freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:10]]

    def _detect_language(self, words: list[str]) -> str:
        english_common = {"the", "and", "is", "in", "to", "of", "a", "that", "it", "for"}
        overlap = sum(1 for w in words if w in english_common)
        return "en" if overlap >= 2 else "unknown"

    def _classify(self, words: list[str]) -> str:
        word_set = set(words)
        scores = {cls: len(kws & word_set) for cls, kws in DOC_CLASSES.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def _readability(self, words: list[str], sentences: list[str]) -> float:
        if not sentences or not words:
            return 0.0
        avg_words = len(words) / len(sentences)
        avg_syllables = sum(self._syllables(w) for w in words) / max(len(words), 1)
        score = 206.835 - 1.015 * avg_words - 84.6 * avg_syllables
        return round(max(0, min(100, score)), 1)

    def _syllables(self, word: str) -> int:
        word = word.lower()
        count = len(re.findall(r'[aeiou]+', word))
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)
