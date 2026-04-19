from __future__ import annotations

import json
import logging
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.models.schemas import (
    CritiqueResponse,
    GeneratedQuestion,
    GradingResult,
    MCQOption,
    QuestionSet,
    QuestionType,
    Rubric,
)
from app.utils.ids import short_uuid
from app.utils.validators import extract_json_text, loads_json_from_text

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    """Raised when an LLM response cannot be validated after retries."""


class LLMClient(Protocol):
    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        """Generate and validate a structured response."""


class OpenAIChatLLMClient:
    """Small OpenAI adapter that returns validated Pydantic objects."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.2, max_retries: int = 2) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        schema = response_model.model_json_schema()
        working_prompt = (
            f"{prompt}\n\n"
            "Return ONLY a valid JSON object matching this JSON Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise educational assessment assistant. "
                            "All user-facing educational content must be in Turkish. "
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": working_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            try:
                json_text = extract_json_text(content)
                return response_model.model_validate_json(json_text)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning("Invalid LLM JSON for %s on attempt %s: %s", response_model.__name__, attempt + 1, exc)
                working_prompt = self._repair_prompt(working_prompt, content, exc, schema)

        raise LLMOutputError(f"Could not validate {response_model.__name__}: {last_error}")

    @staticmethod
    def _repair_prompt(original_prompt: str, invalid_output: str, error: Exception, schema: dict[str, Any]) -> str:
        return (
            f"{original_prompt}\n\n"
            "The previous output was invalid. Repair it without adding commentary.\n"
            f"Validation error: {error}\n"
            f"Invalid output:\n{invalid_output}\n"
            f"Required JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}"
        )


class MockLLMClient:
    """Deterministic Turkish responses for tests and API demos without network access."""

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        payload = self._extract_request_payload(prompt)
        model_name = response_model.__name__

        if model_name == "QuestionSet":
            data = self._question_set(payload)
        elif model_name == "CritiqueResponse":
            data = self._critique(payload)
        elif model_name == "Rubric":
            data = self._rubric(payload)
        elif model_name == "GradingResult":
            data = self._grading_result(payload)
        else:
            raise LLMOutputError(f"MockLLMClient does not know how to build {model_name}.")

        return response_model.model_validate(data)

    @staticmethod
    def _extract_request_payload(prompt: str) -> dict[str, Any]:
        marker = "REQUEST_JSON:"
        if marker not in prompt:
            return {}
        raw = prompt.split(marker, maxsplit=1)[1].strip()
        try:
            return loads_json_from_text(raw)
        except Exception:
            return {}

    def _question_set(self, payload: dict[str, Any]) -> dict[str, Any]:
        learning_outcome = payload.get("learning_outcome", {})
        retrieved_chunks = payload.get("retrieved_chunks", [])
        question_type = payload.get("question_type", QuestionType.multiple_choice.value)
        difficulty = payload.get("difficulty", "medium")
        count = int(payload.get("question_count", 1))
        source_ids = [chunk["chunk_id"] for chunk in retrieved_chunks if "chunk_id" in chunk][:3]
        if not source_ids:
            source_ids = ["no_source"]

        topic = learning_outcome.get("topic", "Bilgisayar Ağları")
        lo_id = learning_outcome.get("id", "LO")
        lo_text = learning_outcome.get("text", "İlgili öğrenme çıktısını açıklar.")

        questions: list[dict[str, Any]] = []
        for index in range(count):
            if question_type == QuestionType.multiple_choice.value:
                questions.append(self._mcq(lo_id, lo_text, topic, difficulty, source_ids, index))
            else:
                questions.append(self._open_ended(lo_id, lo_text, topic, difficulty, source_ids, index))
        return {"questions": questions}

    @staticmethod
    def _mcq(
        lo_id: str,
        lo_text: str,
        topic: str,
        difficulty: str,
        source_ids: list[str],
        index: int,
    ) -> dict[str, Any]:
        topic_key = f"{topic} {lo_text}".lower()
        if "dhcp" in topic_key:
            question_text = "DHCP istemcisinin IP yapılandırması alırken izlediği DORA sırası hangisidir?"
            answer_key = "A seçeneği doğrudur: Discover, Offer, Request, Acknowledge."
            explanation = "DHCP istemcisi önce yayınla Discover gönderir, sunucu Offer sunar, istemci Request ile ister ve sunucu Acknowledge ile onaylar."
            options = [
                MCQOption(label="A", text="Discover, Offer, Request, Acknowledge", is_correct=True),
                MCQOption(label="B", text="Request, Discover, Offer, Acknowledge", is_correct=False),
                MCQOption(label="C", text="Offer, Discover, Acknowledge, Request", is_correct=False),
                MCQOption(label="D", text="Acknowledge, Request, Offer, Discover", is_correct=False),
            ]
        elif "nat" in topic_key:
            question_text = "NAT overload/PAT kullanımının en doğru açıklaması hangisidir?"
            answer_key = "A seçeneği doğrudur: Birden çok iç istemci port numaralarıyla aynı genel IP'yi paylaşabilir."
            explanation = "PAT, iç istemcileri port eşleştirmeleriyle ayırt ederek IPv4 adres tasarrufu sağlar."
            options = [
                MCQOption(label="A", text="Birden çok özel IP adresi port eşleştirmesiyle tek genel IP üzerinden internete çıkabilir.", is_correct=True),
                MCQOption(label="B", text="Tüm istemciler genel internette kendi özel IP adresleriyle görünür.", is_correct=False),
                MCQOption(label="C", text="NAT yalnızca MAC adreslerini değiştirerek anahtarlama yapar.", is_correct=False),
                MCQOption(label="D", text="NAT, DHCP Discover mesajlarının sırasını belirleyen protokoldür.", is_correct=False),
            ]
        elif "yönlendirme" in topic_key or "anahtarlama" in topic_key:
            question_text = "Yönlendirme ile anahtarlama arasındaki temel fark hangisidir?"
            answer_key = "A seçeneği doğrudur: Switch yerel ağda MAC adresleriyle, router ağlar arasında IP adresleriyle karar verir."
            explanation = "Anahtarlama veri bağlantı katmanındaki yerel iletimle, yönlendirme ağ katmanındaki ağlar arası iletimle ilgilidir."
            options = [
                MCQOption(label="A", text="Switch MAC adreslerine göre yerel ağ içinde, router IP adreslerine göre ağlar arasında iletim yapar.", is_correct=True),
                MCQOption(label="B", text="Switch her zaman internet bağlantısını sağlar, router yalnızca kablosuz erişim verir.", is_correct=False),
                MCQOption(label="C", text="Router MAC tablosu oluşturur, switch yönlendirme tablosu ile WAN rotası seçer.", is_correct=False),
                MCQOption(label="D", text="İki kavram aynı işlemi ifade eder ve OSI katmanı açısından fark yoktur.", is_correct=False),
            ]
        elif "topoloji" in topic_key:
            question_text = "Yıldız topolojisinin temel avantaj ve riskini en iyi açıklayan seçenek hangisidir?"
            answer_key = "A seçeneği doğrudur: Uç cihaz arızaları genelde sınırlıdır, merkezi cihaz arızası tüm ağı etkileyebilir."
            explanation = "Yıldız topolojisinde bağlantılar merkezi cihaz üzerinden yönetildiği için yönetim kolaydır ancak merkez kritik noktadır."
            options = [
                MCQOption(label="A", text="Uç cihaz arızaları çoğu zaman sınırlı kalır; merkezi switch arızası tüm ağı kesebilir.", is_correct=True),
                MCQOption(label="B", text="Her cihaz diğer tüm cihazlara doğrudan bağlı olduğu için maliyet en yüksektir.", is_correct=False),
                MCQOption(label="C", text="Tüm cihazlar tek bir ortak kabloyu paylaşır ve çakışma yönetimi temel problemdir.", is_correct=False),
                MCQOption(label="D", text="Veri yalnızca tek yönde halka üzerinde dolaşmak zorundadır.", is_correct=False),
            ]
        else:
            question_text = "192.168.10.0/26 ağı için kullanılabilir host sayısı kaçtır?"
            answer_key = "A seçeneği doğrudur: /26 maskesinde 6 host biti vardır ve 2^6 - 2 = 62 kullanılabilir host oluşur."
            explanation = "CIDR /26, 255.255.255.192 maskesine karşılık gelir; ağ ve yayın adresleri çıkarılır."
            options = [
                MCQOption(label="A", text="62", is_correct=True),
                MCQOption(label="B", text="64", is_correct=False),
                MCQOption(label="C", text="30", is_correct=False),
                MCQOption(label="D", text="126", is_correct=False),
            ]

        question = GeneratedQuestion(
            question_id=short_uuid("q"),
            learning_outcome_id=lo_id,
            question_type=QuestionType.multiple_choice,
            difficulty=difficulty,
            question_text=f"Senaryo {index + 1}: {question_text}",
            answer_key=answer_key,
            explanation=explanation,
            source_chunks=source_ids,
            options=options,
        )
        return question.model_dump(mode="json")

    @staticmethod
    def _open_ended(
        lo_id: str,
        lo_text: str,
        topic: str,
        difficulty: str,
        source_ids: list[str],
        index: int,
    ) -> dict[str, Any]:
        topic_key = f"{topic} {lo_text}".lower()
        if "dhcp" in topic_key:
            question_text = (
                "DHCP'nin DORA sürecini açıklayınız. Discover, Offer, Request ve Acknowledge "
                "adımlarında istemci ile sunucunun ne yaptığını ve istemciye hangi ağ yapılandırma "
                "bilgilerinin verilebileceğini belirtiniz."
            )
            answer_key = (
                "Yanıt DORA sırasını doğru vermeli; Discover yayınını, Offer teklifini, Request kabulünü "
                "ve Acknowledge onayını açıklamalıdır. IP adresi, alt ağ maskesi, varsayılan ağ geçidi ve "
                "DNS bilgilerinden söz etmelidir."
            )
        elif "nat" in topic_key:
            question_text = (
                "Bir küçük ofis ağında özel IP kullanan istemcilerin internete çıkması için NAT/PAT nasıl "
                "kullanılır? IPv4 adres tasarrufu, port eşleştirme ve olası kısıtları açıklayınız."
            )
            answer_key = (
                "Yanıt NAT'ın özel adresleri genel adrese çevirdiğini, PAT ile port numaralarının iç "
                "istemcileri ayırt ettiğini, adres tasarrufu sağladığını ve port yönlendirme/NAT traversal "
                "gereksinimlerini tartışmalıdır."
            )
        elif "yönlendirme" in topic_key or "anahtarlama" in topic_key:
            question_text = (
                "Aynı LAN içindeki iletişim ile farklı ağlar arasındaki iletişimi karşılaştırınız. Switch ve "
                "router cihazlarının hangi adres bilgilerine göre karar verdiğini açıklayınız."
            )
            answer_key = (
                "Yanıt switch'in MAC adres tablosu ile yerel ağ içinde çerçeve ilettiğini, router'ın IP "
                "adresi ve yönlendirme tablosuna göre ağlar arasında paket gönderdiğini açıklamalıdır."
            )
        elif "topoloji" in topic_key:
            question_text = (
                "Yıldız, veriyolu, halka ve mesh topolojilerini yönetilebilirlik, hata dayanımı ve maliyet "
                "açısından karşılaştırınız."
            )
            answer_key = (
                "Yanıt her topolojinin temel bağlantı biçimini, avantajını ve riskini belirtmeli; yıldızda "
                "merkezi cihaz riskini, mesh yapıda yedeklilik ve maliyet artışını açıklamalıdır."
            )
        else:
            question_text = (
                "192.168.10.0/26 ağı için blok boyutunu, kullanılabilir host sayısını, ağ adresini ve yayın "
                "adresini hesaplama adımlarını açıklayınız."
            )
            answer_key = (
                "Yanıt /26 maskesinin 255.255.255.192 olduğunu, blok boyutunun 64 olduğunu, 6 host bitiyle "
                "2^6 - 2 = 62 kullanılabilir host bulunduğunu ve ağ/yayın adreslerinin nasıl belirlendiğini "
                "açıklamalıdır."
            )

        return {
            "question_id": short_uuid("q"),
            "learning_outcome_id": lo_id,
            "question_type": QuestionType.open_ended.value,
            "difficulty": difficulty,
            "question_text": f"Soru {index + 1}: {question_text}",
            "answer_key": answer_key,
            "explanation": (
                "Soru, öğrencinin kavramı kaynak materyale dayanarak açıklama ve ağ senaryosuna "
                "uygulama becerisini ölçer."
            ),
            "source_chunks": source_ids,
            "options": None,
        }

    @staticmethod
    def _critique(payload: dict[str, Any]) -> dict[str, Any]:
        questions = payload.get("questions", [])
        has_sources = all(question.get("source_chunks") for question in questions)
        return {
            "accepted": has_sources,
            "score": 9 if has_sources else 6,
            "issues": [] if has_sources else ["Bazı sorularda kaynak parça referansı eksik."],
            "suggestions": [
                "Soru kökünde öğrenme çıktısı ile kaynak içerik arasındaki ilişki açık tutulmalı."
            ],
            "revised_focus": None if has_sources else "Her soruya en az bir kaynak chunk id ekle.",
        }

    @staticmethod
    def _rubric(payload: dict[str, Any]) -> dict[str, Any]:
        question = payload.get("question", {})
        total_points = float(payload.get("total_points", 10))
        conceptual = round(total_points * 0.4, 2)
        process = round(total_points * 0.35, 2)
        example = round(total_points - conceptual - process, 2)
        return {
            "rubric_id": short_uuid("rubric"),
            "question_id": question.get("question_id", "q_unknown"),
            "total_points": total_points,
            "criteria": [
                {
                    "criterion_id": "C1",
                    "description": "Temel kavramı doğru ve eksiksiz tanımlar.",
                    "points": conceptual,
                },
                {
                    "criterion_id": "C2",
                    "description": "Süreç adımlarını veya teknik ilişkileri doğru sırayla açıklar.",
                    "points": process,
                },
                {
                    "criterion_id": "C3",
                    "description": "Ağ bağlamına uygun, tutarlı bir örnek veya yorum sunar.",
                    "points": example,
                },
            ],
        }

    @staticmethod
    def _grading_result(payload: dict[str, Any]) -> dict[str, Any]:
        question = payload.get("question", {})
        rubric = payload.get("rubric", {})
        answer = str(payload.get("student_answer", ""))
        words = len(answer.split())
        quality = 0.85 if words >= 35 else 0.65 if words >= 16 else 0.35

        scores = []
        for criterion in rubric.get("criteria", []):
            max_points = float(criterion.get("points", 0))
            awarded = round(max_points * quality, 2)
            scores.append(
                {
                    "criterion_id": criterion.get("criterion_id", "C"),
                    "awarded_points": awarded,
                    "max_points": max_points,
                    "justification": (
                        "Yanıt ilgili ölçütü kısmen karşılıyor; temel fikirler var ancak daha teknik ayrıntı "
                        "ve örnekle güçlendirilebilir."
                    ),
                }
            )

        total_score = round(sum(item["awarded_points"] for item in scores), 2)
        total_points = float(rubric.get("total_points", sum(item["max_points"] for item in scores) or 1))
        return {
            "question_id": question.get("question_id", "q_unknown"),
            "rubric_id": rubric.get("rubric_id", "rubric_unknown"),
            "total_score": total_score,
            "total_points": total_points,
            "criterion_scores": scores,
            "feedback": (
                "Yanıt genel olarak doğru yönde; puanı artırmak için kavramları daha net tanımlayın, "
                "adımları sıralayın ve somut bir ağ senaryosu ile destekleyin."
            ),
        }


def create_llm_client(settings: Settings) -> LLMClient:
    if settings.use_mock_llm or not settings.openai_api_key:
        logger.info("Using deterministic mock LLM client.")
        return MockLLMClient()
    logger.info("Using OpenAI model %s.", settings.openai_model)
    return OpenAIChatLLMClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.openai_temperature,
    )
