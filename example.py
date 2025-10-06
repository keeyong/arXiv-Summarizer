#!/usr/bin/env python3
"""
Example usage of ArxivPaperSummarizer as a library with different backends
"""

from summarize_paper import ArxivPaperSummarizer
import os

# Example arXiv papers to try
example_papers = [
    "https://arxiv.org/abs/1706.03762",  # Attention Is All You Need (Transformer)
    "https://arxiv.org/abs/2103.00020",  # CLIP
    "https://arxiv.org/abs/2005.14165",  # GPT-3
]

def main():
    print("🚀 arXiv Paper Summarizer - Example Usage\n")
    print("=" * 60)
    
    # Choose backend
    print("\nAvailable backends:")
    print("1. Ollama (FREE, recommended)")
    print("2. Hugging Face (FREE)")
    print("3. Extractive (FREE, fastest)")
    print("4. OpenAI (requires API key, costs money)")
    
    choice = input("\nChoose backend (1-4, default=1): ").strip() or "1"
    
    backend_map = {
        "1": "ollama",
        "2": "huggingface",
        "3": "extractive",
        "4": "openai"
    }
    
    backend = backend_map.get(choice, "ollama")
    
    # Check API key for OpenAI
    if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OpenAI backend requires OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize summarizer
    try:
        if backend == "openai":
            summarizer = ArxivPaperSummarizer(backend="openai")
        else:
            summarizer = ArxivPaperSummarizer(backend=backend)
            print(f"\n✅ Using {backend} backend")
    except Exception as e:
        print(f"\n❌ Error initializing backend: {e}")
        return
    
    # Process a paper
    print("\n" + "=" * 60)
    print("Processing paper...\n")
    
    # Use the first example paper
    arxiv_url = example_papers[0]
    print(f"📄 Paper: {arxiv_url}")
    print(f"🔧 Backend: {backend}\n")
    
    try:
        # Process and generate summary
        output_path = summarizer.process_paper(arxiv_url, output_dir="summaries")
        
        print("\n" + "=" * 60)
        print(f"✅ Success! Summary saved to: {output_path}")
        print("\n📖 You can now read the summary:")
        print(f"   cat {output_path}")
        print("\n💡 Try other papers:")
        for i, paper in enumerate(example_papers[1:], 1):
            paper_id = paper.split('/')[-1]
            print(f"   python summarize_paper.py {paper_id} --backend {backend}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

