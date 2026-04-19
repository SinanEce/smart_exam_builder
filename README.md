# SmartExam Builder

SmartExam Builder, Bilgisayar Ağları alanında yapay zekâ destekli sınav üretimi ve değerlendirmesi için geliştirilmiş, demoya hazır bir FastAPI sistemidir. Öğrenme kazanımlarından hareketle Türkçe sorular üretir, yerel bir RAG hattı ile ilgili ders içeriklerini getirir, açık bir Writer-Critic iş akışı ile çıktıyı iyileştirir, rubrik oluşturur ve öğrenci cevaplarını değerlendirir.

Uygulama, ağ erişimi gerektirmeyen deterministik bir mock LLM ile kutudan çıktığı gibi çalışır. Gerçek OpenAI çağrıları kullanmak için `SMARTEXAM_USE_MOCK_LLM=false` ayarlanmalı ve `OPENAI_API_KEY` tanımlanmalıdır.

## Features

- Yerel klasörlerden veya yüklenen `.txt`, `.md` ve `.pdf` dosyalarından materyal alma
- Metin çıkarma, chunking, metadata oluşturma, embedding üretme ve FAISS ile indeksleme
- Ders, konu, kimlik, metin ve bilişsel seviye alanlarını içeren öğrenme kazanımı modelleri
- Türkçe çoktan seçmeli ve açık uçlu soru üretimi
- Chunk kimlikleri, kaynak dosyalar, sayfa numaraları, benzerlik skorları ve önizlemeleri içeren görünür RAG izleri
- Revizyon döngüsüne sahip Writer-Critic çok ajanlı iş akışı
- Açık uçlu sorular için rubrik üretimi
- Rubriğe dayalı, ölçüt bazlı geri bildirim içeren otomatik değerlendirme
- Yapılandırılmış LLM çıktıları için Pydantic doğrulaması
- OpenAI yanıtları için yeniden deneme ve JSON onarma mantığı içeren, mock destekli LLM soyutlama katmanı
- Şemalar, chunking, retrieval, işlem hatları ve API smoke testleri için Pytest kapsamı

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

Bu isteğin cevabında şunlar yer alır:

retrieved_chunks: chunk kimliği, kaynak dosya, sayfa numarası, chunk sırası, skor ve önizleme metni
questions[].source_chunks: her soruyu etkileyen chunk kimlikleri
writer_critic_trace: sunumda gösterilebilecek Retriever, Writer ve Critic adımları

3. Generate an open-ended question, then use it to generate a rubric:

```bash
curl -X POST http://127.0.0.1:8000/generate-questions ^
  -H "Content-Type: application/json" ^
  -d "{\"learning_outcome_id\":\"LO3\",\"question_type\":\"open_ended\",\"question_count\":1,\"top_k\":4}"
```

Dönen soru nesnesini şu endpoint’e gönder:

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

Bu komut örnek materyalleri içeri aktarır, açık uçlu bir DHCP sorusu üretir, rubrik oluşturur ve örnek bir cevabı değerlendirir.

## Using OpenAI

Edit `.env`:

```env
SMARTEXAM_USE_MOCK_LLM=false
OPENAI_API_KEY=your_api_key_here
SMARTEXAM_OPENAI_MODEL=gpt-4o-mini
```

Tüm LLM çağrıları app/core/llm_client.py üzerinden yapılır ve yanıtlar Pydantic ile doğrulanır. Model geçersiz JSON döndürürse istemci JSON’u ayıklamaya/onarmaya çalışır ve isteği yeniden dener.

## Writer-Critic Flow

QuestionService, aşağıdaki iş akışını koordine eder:

RetrieverAgent, öğrenme kazanımından bir sorgu oluşturur ve en ilgili kaynak chunk’ları getirir.
WriterAgent, öğrenme kazanımı ve getirilen chunk’lara dayanarak yapılandırılmış Türkçe sorular üretir.
CriticAgent, soru ile kazanım uyumu, açıklık, cevaplanabilirlik, zorluk seviyesi, belirsizlik ve distractor kalitesini kontrol eder.
Soru reddedilirse WriterAgent, Critic geri bildirimine göre bir veya iki kez revizyon yapar.
Nihai cevap, kolay açıklanabilmesi için writer_critic_trace alanını içerir.

## RAG Implementation

Yerel RAG hattı bilinçli olarak sade tutulmuştur:

loaders.py: .txt, .md ve .pdf dosyalarından metin çıkarır
text_chunker.py: metadata ile birlikte retrieval’a uygun chunk’lar üretir
embeddings.py: deterministik yerel embedding üretir
vector_store.py: mümkünse FAISS, değilse NumPy tabanlı yedek çözüm kullanır
retrieval.py: öğrenme kazanımına duyarlı sorgular oluşturur

FAISS metadata verileri data/processed/ altında saklanır. Üretilen soru yanıtında getirilen chunk’lar açıkça gösterilir; böylece demo sırasında RAG davranışı görünür olur.
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

