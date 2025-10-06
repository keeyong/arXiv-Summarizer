# 무료 논문 요약 가이드 💰❌ → 🆓✅

OpenAI API 없이 **완전 무료**로 논문을 요약할 수 있는 3가지 방법을 소개합니다!

---

## 🥇 추천 옵션 1: Ollama (로컬 LLM)

**장점:**
- ✅ 완전 무료
- ✅ 품질 매우 좋음 (GPT-4 수준)
- ✅ 빠른 속도
- ✅ 인터넷 연결 불필요
- ✅ 개인정보 보호 (데이터가 외부로 나가지 않음)

**단점:**
- ⚠️ 초기 설치 필요 (5분)
- ⚠️ 디스크 공간 필요 (~5GB)

### 설치 방법

#### macOS
```bash
# 1. Ollama 설치
brew install ollama

# 2. Ollama 시작
ollama serve &

# 3. 모델 다운로드 (권장: llama3.2)
ollama pull llama3.2
```

#### Linux
```bash
# 1. Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Ollama 시작
ollama serve &

# 3. 모델 다운로드
ollama pull llama3.2
```

#### Windows
1. https://ollama.ai 에서 설치 파일 다운로드
2. 설치 후 자동으로 실행됨
3. PowerShell에서: `ollama pull llama3.2`

### 사용 방법

```bash
# 기본 실행 (Ollama는 기본 옵션)
python summarize_paper.py https://arxiv.org/abs/1706.03762

# 또는 명시적으로 지정
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend ollama

# 다른 모델 사용하기
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend ollama --model llama3.1
```

### 추천 모델

| 모델 | 크기 | 품질 | 속도 | 추천 |
|------|------|------|------|------|
| `llama3.2` | 4.3GB | ⭐⭐⭐⭐⭐ | 빠름 | ✅ 권장 |
| `llama3.1` | 4.7GB | ⭐⭐⭐⭐⭐ | 보통 | ✅ 권장 |
| `phi3` | 2.3GB | ⭐⭐⭐⭐ | 매우 빠름 | 저사양 PC |
| `mistral` | 4.1GB | ⭐⭐⭐⭐ | 빠름 | 대안 |

---

## 🥈 옵션 2: Hugging Face (추론 모델)

**장점:**
- ✅ 완전 무료
- ✅ 설치만 하면 바로 사용
- ✅ 추가 프로그램 설치 불필요
- ✅ 인터넷 연결 불필요 (첫 실행 후)

**단점:**
- ⚠️ 첫 실행 시 모델 다운로드 느림 (~2GB)
- ⚠️ 요약 품질이 Ollama보다 조금 낮음
- ⚠️ 메모리 사용량이 높음

### 설치 방법

```bash
# 필요한 패키지 설치
pip install transformers torch
```

### 사용 방법

```bash
# 기본 사용 (BART 모델)
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend huggingface

# T5 모델 사용 (더 큰 모델, 더 좋은 품질)
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend huggingface --model google/flan-t5-large
```

### 추천 모델

| 모델 | 크기 | 품질 | 속도 |
|------|------|------|------|
| `facebook/bart-large-cnn` | 1.6GB | ⭐⭐⭐⭐ | 보통 (기본) |
| `google/flan-t5-large` | 2.8GB | ⭐⭐⭐⭐⭐ | 느림 |
| `google/flan-t5-base` | 890MB | ⭐⭐⭐ | 빠름 |

---

## 🥉 옵션 3: 추출적 요약 (Extractive)

**장점:**
- ✅ 완전 무료
- ✅ 매우 빠름 (몇 초)
- ✅ 설치 간단
- ✅ 메모리 사용량 매우 적음

**단점:**
- ⚠️ 품질이 낮음 (단순히 중요 문장만 추출)
- ⚠️ 문맥 이해 없음
- ⚠️ 자연스럽지 않은 요약

### 설치 방법

```bash
pip install sumy nltk
```

### 사용 방법

```bash
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend extractive
```

---

## 🆚 백엔드 비교표

| 특성 | Ollama | Hugging Face | Extractive | OpenAI |
|------|--------|--------------|------------|--------|
| **비용** | 무료 🆓 | 무료 🆓 | 무료 🆓 | 유료 💰 |
| **품질** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **속도** | 빠름 🚀 | 보통 🏃 | 매우 빠름 ⚡ | 빠름 🚀 |
| **설치** | 쉬움 | 매우 쉬움 | 매우 쉬움 | 쉬움 |
| **초기 설정** | ~5분 | ~10분 (첫 실행) | ~1분 | ~1분 |
| **디스크 공간** | ~5GB | ~2GB | <100MB | 없음 |
| **인터넷 필요** | 설치시만 | 첫 실행시만 | 설치시만 | 항상 |
| **개인정보** | 안전 🔒 | 안전 🔒 | 안전 🔒 | 외부 전송 |
| **추천도** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 💡 추천 시나리오

### 일반적인 사용 → **Ollama**
```bash
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend ollama
```
- 품질 좋고 빠르며 무료
- 대부분의 경우 최선의 선택

### 추가 설치 없이 바로 사용 → **Hugging Face**
```bash
pip install transformers torch
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend huggingface
```
- Ollama 설치가 어려운 경우
- Python만 있으면 됨

### 빠른 테스트/미리보기 → **Extractive**
```bash
pip install sumy nltk
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend extractive
```
- 논문의 주요 내용만 빠르게 확인
- 저사양 PC에서 사용

### 최고 품질 필요 (예산 있음) → **OpenAI**
```bash
export OPENAI_API_KEY="sk-..."
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend openai
```
- 가장 정확하고 자연스러운 요약
- 논문 1편당 ~$0.01-0.05

---

## 🛠️ 문제 해결

### Ollama: "Ollama is not running"
```bash
# Ollama 서비스 시작
ollama serve

# 백그라운드로 실행
ollama serve &
```

### Ollama: "Model not found"
```bash
# 모델 다운로드
ollama pull llama3.2
```

### Hugging Face: 첫 실행이 너무 느림
- 정상입니다! 모델을 다운로드 중입니다 (~2GB)
- 두 번째 실행부터는 빨라집니다

### Hugging Face: 메모리 부족 에러
```bash
# 더 작은 모델 사용
python summarize_paper.py URL --backend huggingface --model google/flan-t5-base
```

### Extractive: 품질이 너무 낮음
- Ollama나 Hugging Face 사용을 권장합니다
- Extractive는 빠른 미리보기용입니다

---

## ⚡ 빠른 설치 스크립트

### Ollama (macOS/Linux)
```bash
# 원라인 설치 및 실행
curl -fsSL https://ollama.ai/install.sh | sh && \
ollama serve & \
sleep 5 && \
ollama pull llama3.2 && \
echo "✅ Ollama 설치 완료!"
```

### Hugging Face
```bash
pip install transformers torch && echo "✅ Hugging Face 설치 완료!"
```

### Extractive
```bash
pip install sumy nltk && echo "✅ Extractive 설치 완료!"
```

---

## 🎯 다음 단계

1. **Ollama 설치 후 바로 시작:**
```bash
python summarize_paper.py 1706.03762 --backend ollama
```

2. **여러 논문 한번에 처리:**
```bash
for paper in 1706.03762 2103.00020 2005.14165; do
  python summarize_paper.py $paper --backend ollama -o summaries/
done
```

3. **자동화 스크립트 만들기:**
```bash
#!/bin/bash
# summarize_all.sh
while IFS= read -r paper_id; do
  python summarize_paper.py "$paper_id" --backend ollama -o papers/
done < paper_list.txt
```

완전 무료로 논문 요약을 시작하세요! 🚀

