# 빠른 시작 가이드 (무료! 🆓)

## 🎯 가장 빠른 방법 (자동 설치)

```bash
# 자동 설치 스크립트 실행
chmod +x install_free.sh
./install_free.sh
```

스크립트가 백엔드를 선택하고 자동으로 설치해줍니다!

---

## 📝 수동 설치

### 옵션 1: Ollama (추천 ⭐⭐⭐⭐⭐)

```bash
# 1. 기본 패키지
pip install requests PyPDF2

# 2. Ollama 설치 (macOS)
brew install ollama
ollama serve &
ollama pull llama3.2

# 3. 실행!
python summarize_paper.py https://arxiv.org/abs/1706.03762
```

### 옵션 2: Hugging Face (⭐⭐⭐⭐)

```bash
# 1. 패키지 설치
pip install requests PyPDF2 transformers torch

# 2. 실행!
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend huggingface
```

### 옵션 3: Extractive (⭐⭐⭐)

```bash
# 1. 패키지 설치
pip install requests PyPDF2 sumy nltk

# 2. 실행!
python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend extractive
```

---

## 🚀 사용 방법

### 방법 1: 명령줄 사용 (가장 간단)

```bash
# 기본 사용 (Ollama)
python summarize_paper.py https://arxiv.org/abs/1706.03762

# 백엔드 지정
python summarize_paper.py 1706.03762 --backend huggingface

# 출력 디렉토리 지정
python summarize_paper.py 1706.03762 --backend extractive -o summaries/
```

### 방법 2: 라이브러리로 사용

```python
from summarize_paper import ArxivPaperSummarizer

# Ollama 사용 (무료)
summarizer = ArxivPaperSummarizer(backend="ollama")
summarizer.process_paper("https://arxiv.org/abs/1706.03762")

# Hugging Face 사용 (무료)
summarizer = ArxivPaperSummarizer(backend="huggingface")
summarizer.process_paper("https://arxiv.org/abs/1706.03762")
```

### 방법 3: 인터랙티브 예제

```bash
python example.py
```

백엔드를 선택하고 예제 논문을 자동으로 요약합니다!

## 3단계: 결과 확인

스크립트가 완료되면 다음 파일이 생성됩니다:

- `1706.03762.pdf` - 원본 논문
- `1706.03762_summary.md` - 요약본

요약본 보기:
```bash
cat 1706.03762_summary.md
# 또는
open 1706.03762_summary.md  # macOS
```

## 다양한 사용 예시

### 여러 논문 처리

```bash
python summarize_paper.py 1706.03762 -o summaries/
python summarize_paper.py 2103.00020 -o summaries/
python summarize_paper.py 2005.14165 -o summaries/
```

### 특정 디렉토리에 저장

```bash
mkdir papers
python summarize_paper.py https://arxiv.org/abs/1706.03762 -o papers/
```

## 추천 논문 (테스트용)

1. **Attention Is All You Need** (Transformer 원본 논문)
   ```bash
   python summarize_paper.py 1706.03762
   ```

2. **BERT: Pre-training of Deep Bidirectional Transformers**
   ```bash
   python summarize_paper.py 1810.04805
   ```

3. **GPT-3: Language Models are Few-Shot Learners**
   ```bash
   python summarize_paper.py 2005.14165
   ```

## 문제 해결

### API 키가 설정되지 않음
```
ValueError: OpenAI API key not found
```
→ 해결: `export OPENAI_API_KEY="your-key"` 실행

### PDF 다운로드 실패
```
requests.exceptions.HTTPError: 404
```
→ 해결: arXiv ID가 올바른지 확인

### 섹션을 찾을 수 없음
```
*Section not found or too short*
```
→ 정상: 일부 논문은 표준 섹션 구조가 아닐 수 있습니다

## 비용 안내

- GPT-4o-mini 사용 기준
- 논문 1편당 약 $0.01 ~ $0.05
- 토큰 사용량은 논문 길이에 따라 다름

## 다음 단계

- 더 많은 논문 처리해보기
- 요약 결과를 모아 문헌 리뷰 작성
- 관심 분야의 최신 논문 자동 요약 파이프라인 구축

