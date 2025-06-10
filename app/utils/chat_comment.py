"""
utils/chat_comment.py
────────────────────────────────────────────────────────────────────────
ИИ-ассистент формирует маркетинговое сообщение о найденных объектах
(акцент на **автозапчасти**), не упоминая ChatGPT.
"""
import logging
from typing import Dict

from openai_config import get_client

logger = logging.getLogger(__name__)
_oai = get_client()


def summarize_detections(counts: Dict[str, int]) -> str:
    if not counts:
        return "Пока ничего значимого не обнаружено."

    items = ", ".join(f"{k}: {v}" for k, v in counts.items())
    system = (
        "Ты — профессиональный продавец автозапчастей. "
        "На экране показаны результаты компьютерного зрения."
    )
    user = (
        "Детектор нашёл следующее: " + items + ". "
        "Объясни за 1-2 предложения, какие запчасти или услуги могут "
        "понадобиться владельцу. Пиши на русском, без технического жаргона."
    )

    try:
        resp = _oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=80,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("summarize_detections: %s", e)
        return "— комментарий недоступен —"
