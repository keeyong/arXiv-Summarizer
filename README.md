# arXiv Paper Summarizer 📄✨

[![GitHub](https://img.shields.io/badge/GitHub-arXiv--Summarizer-blue?logo=github)](https://github.com/keeyong/arXiv-Summarizer)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

arXiv 논문을 다운로드하고 섹션별로 요약하여 Markdown 파일로 출력하는 Python 스크립트입니다.

## 🎉 완전 무료! 

**돈 안 들이고 사용 가능!** OpenAI API 없이도 로컬 LLM (Ollama)나 Hugging Face 모델로 무료 사용 가능합니다.

👉 **[무료 사용 가이드 보기](FREE_OPTIONS.md)**

## 주요 기능

- ✅ arXiv URL 또는 논문 ID로 PDF 자동 다운로드
- ✅ PDF에서 텍스트 추출
- ✅ 논문을 섹션별로 자동 분석:
  - Abstract (초록)
  - Introduction (서론)
  - Method (방법론)
  - Results/Discussion (결과 및 논의)
  - Conclusion (결론)
- ✅ AI를 사용한 각 섹션별 요약
- ✅ 1페이지 Markdown 요약 문서 생성
- ✅ **4가지 백엔드 지원**: OpenAI, Ollama, Hugging Face, Extractive

## 빠른 시작 (무료! 💰❌)

### 옵션 1: Ollama 사용 (추천 ⭐⭐⭐⭐⭐)

```bash
# 1. 기본 패키지 설치
pip install requests PyPDF2

# 2. Ollama 설치 (macOS)
brew install ollama
ollama serve &
ollama pull llama3.2

# 3. 실행!
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend ollama
```

### 옵션 2: Hugging Face 사용 (Python만 있으면 OK ⭐⭐⭐⭐)

```bash
# 1. 패키지 설치
pip install requests PyPDF2 transformers torch

# 2. 실행!
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend huggingface
```

### 옵션 3: 추출적 요약 (가장 빠름 ⭐⭐⭐)

```bash
# 1. 패키지 설치
pip install requests PyPDF2 sumy nltk

# 2. 실행!
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend extractive
```

## 사용 방법

### 기본 사용법

```bash
# Ollama 사용 (기본값, 무료)
python summarize_paper.py https://arxiv.org/abs/2301.00001

# 또는 명시적으로 백엔드 지정
python summarize_paper.py https://arxiv.org/abs/2301.00001 --backend ollama

# arXiv ID만으로도 가능
python summarize_paper.py 2301.00001 --backend huggingface

# 출력 디렉토리 지정
python summarize_paper.py 2301.00001 --backend extractive -o summaries/
```

### OpenAI 사용 (유료)

```bash
export OPENAI_API_KEY="your-api-key"
python summarize_paper.py https://arxiv.org/abs/2301.00001 --backend openai
```

## 출력 파일

스크립트는 두 개의 파일을 생성합니다:

1. **`{arxiv_id}.pdf`** - 다운로드된 논문 PDF
2. **`{arxiv_id}_summary.md`** - 섹션별 요약이 포함된 Markdown 파일

## 예시 출력

```markdown
# Paper Summary: Attention Is All You Need

**arXiv ID:** [1706.03762](https://arxiv.org/abs/1706.03762)  
**Authors:** Ashish Vaswani, Noam Shazeer, et al.  

---

## 📋 Abstract
This paper introduces the Transformer, a novel architecture...

## 🎯 Introduction
The dominant sequence transduction models...

...
```

## 백엔드 비교

| 특성 | Ollama | Hugging Face | Extractive | OpenAI |
|------|--------|--------------|------------|--------|
| **비용** | 🆓 무료 | 🆓 무료 | 🆓 무료 | 💰 유료 |
| **품질** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **속도** | 빠름 | 보통 | 매우 빠름 | 빠름 |
| **설치** | 쉬움 | 매우 쉬움 | 매우 쉬움 | 쉬움 |
| **인터넷** | 설치시만 | 첫 실행시만 | 설치시만 | 항상 필요 |

👉 **자세한 비교는 [FREE_OPTIONS.md](FREE_OPTIONS.md) 참고**

## 요구사항

### 공통
- Python 3.7 이상
- 인터넷 연결 (arXiv 논문 다운로드용)

### 백엔드별
- **Ollama**: Ollama 앱 설치 + 모델 다운로드 (~5GB)
- **Hugging Face**: `transformers`, `torch` 패키지
- **Extractive**: `sumy`, `nltk` 패키지  
- **OpenAI**: API 키 필요 (유료)

## 문제 해결

### Ollama 관련

**"Ollama is not running"**
```bash
ollama serve &
```

**"Model not found"**
```bash
ollama pull llama3.2
```

### Hugging Face 관련

**첫 실행이 너무 느림**
- 정상입니다! 모델 다운로드 중 (~2GB)
- 두 번째부터는 빠릅니다

**메모리 부족**
```bash
# 더 작은 모델 사용
python summarize_paper.py URL --backend huggingface --model google/flan-t5-base
```

### 일반 문제

**PDF 텍스트 추출 오류**
- 스캔된 이미지 PDF는 텍스트 추출이 어려움
- OCR 도구가 필요할 수 있음

**섹션 인식 오류**
- 비표준 논문 구조는 섹션 인식이 어려울 수 있음
- 정상적으로 작동하는 것이며, 가능한 섹션만 요약됨

## 라이선스

MIT License

