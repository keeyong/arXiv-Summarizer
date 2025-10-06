#!/bin/bash
# 무료 논문 요약 시스템 설치 스크립트

echo "📦 arXiv Paper Summarizer - 무료 설치 스크립트"
echo "================================================"
echo ""

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Python 버전 확인
echo "🔍 Python 버전 확인 중..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3가 설치되어 있지 않습니다.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✅ Python $PYTHON_VERSION 발견${NC}"
echo ""

# 백엔드 선택
echo "어떤 백엔드를 사용하시겠습니까?"
echo ""
echo "1) Ollama (추천 ⭐⭐⭐⭐⭐)"
echo "   - 품질: 최고"
echo "   - 속도: 빠름"
echo "   - 설치: Ollama 앱 필요 (~5분)"
echo ""
echo "2) Hugging Face (⭐⭐⭐⭐)"
echo "   - 품질: 좋음"
echo "   - 속도: 보통"
echo "   - 설치: Python 패키지만 필요"
echo ""
echo "3) Extractive (⭐⭐⭐)"
echo "   - 품질: 기본"
echo "   - 속도: 매우 빠름"
echo "   - 설치: 가장 간단"
echo ""
read -p "선택 (1-3): " choice

echo ""
echo "📦 기본 패키지 설치 중..."
pip install requests PyPDF2 || {
    echo -e "${RED}❌ 기본 패키지 설치 실패${NC}"
    exit 1
}
echo -e "${GREEN}✅ 기본 패키지 설치 완료${NC}"
echo ""

case $choice in
    1)
        echo "🦙 Ollama 설치 중..."
        echo ""
        
        # OS 감지
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                echo "Homebrew를 사용하여 Ollama 설치 중..."
                brew install ollama
            else
                echo -e "${YELLOW}Homebrew가 없습니다. 수동 설치가 필요합니다:${NC}"
                echo "1. https://ollama.ai 방문"
                echo "2. macOS용 설치 파일 다운로드"
                echo "3. 설치 후 이 스크립트를 다시 실행하세요"
                exit 1
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            echo "Ollama 설치 중..."
            curl -fsSL https://ollama.ai/install.sh | sh
        else
            echo -e "${YELLOW}⚠️  Windows 감지됨${NC}"
            echo "Windows에서는 https://ollama.ai 에서 수동 설치가 필요합니다."
            echo "설치 후 PowerShell에서 다음 명령어를 실행하세요:"
            echo "  ollama pull llama3.2"
            exit 0
        fi
        
        echo ""
        echo "🚀 Ollama 서비스 시작 중..."
        ollama serve &
        sleep 3
        
        echo ""
        echo "📥 Llama 3.2 모델 다운로드 중 (~5GB, 시간이 걸릴 수 있습니다)..."
        ollama pull llama3.2
        
        echo ""
        echo -e "${GREEN}✅ Ollama 설치 완료!${NC}"
        echo ""
        echo "🎉 이제 다음 명령어로 논문을 요약할 수 있습니다:"
        echo "   python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend ollama"
        ;;
        
    2)
        echo "🤗 Hugging Face 패키지 설치 중..."
        pip install transformers torch || {
            echo -e "${RED}❌ Hugging Face 패키지 설치 실패${NC}"
            exit 1
        }
        
        echo ""
        echo -e "${GREEN}✅ Hugging Face 설치 완료!${NC}"
        echo ""
        echo "🎉 이제 다음 명령어로 논문을 요약할 수 있습니다:"
        echo "   python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend huggingface"
        echo ""
        echo -e "${YELLOW}ℹ️  첫 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다 (~2GB)${NC}"
        ;;
        
    3)
        echo "📊 Extractive 요약 패키지 설치 중..."
        pip install sumy nltk || {
            echo -e "${RED}❌ Extractive 패키지 설치 실패${NC}"
            exit 1
        }
        
        echo ""
        echo -e "${GREEN}✅ Extractive 설치 완료!${NC}"
        echo ""
        echo "🎉 이제 다음 명령어로 논문을 요약할 수 있습니다:"
        echo "   python summarize_paper.py https://arxiv.org/abs/1706.03762 --backend extractive"
        ;;
        
    *)
        echo -e "${RED}❌ 잘못된 선택입니다${NC}"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo -e "${GREEN}🎊 설치가 완료되었습니다!${NC}"
echo ""
echo "📚 더 많은 정보:"
echo "   - 무료 옵션 가이드: cat FREE_OPTIONS.md"
echo "   - 전체 문서: cat README.md"
echo ""

