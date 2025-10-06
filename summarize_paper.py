#!/usr/bin/env python3
"""
arXiv Paper Summarizer
Downloads a paper from arXiv, extracts text, and generates a structured summary.
Supports multiple backends: OpenAI, Ollama (local), Hugging Face, or extractive summarization.
"""

import os
import re
import sys
import argparse
import requests
from pathlib import Path
from typing import Dict, Optional, List
import PyPDF2


class ArxivPaperSummarizer:
    """Main class for downloading and summarizing arXiv papers."""
    
    def __init__(self, backend: str = "ollama", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the summarizer with specified backend.
        
        Args:
            backend: 'openai', 'ollama', 'huggingface', or 'extractive'
            api_key: API key for OpenAI (only needed for 'openai' backend)
            model: Model name (optional, uses defaults for each backend)
        """
        self.backend = backend.lower()
        self.model = model
        
        if self.backend == "openai":
            from openai import OpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=self.api_key)
            self.model = model or "gpt-4o-mini"
        
        elif self.backend == "ollama":
            # Check if Ollama is running
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                response.raise_for_status()
                self.model = model or "llama3.2"  # Default model
                print(f"✅ Connected to Ollama (model: {self.model})")
            except Exception as e:
                print(f"❌ Ollama not found. Please install and start Ollama:")
                print("   Visit: https://ollama.ai")
                print("   Then run: ollama pull llama3.2")
                raise ValueError("Ollama is not running or not installed")
        
        elif self.backend == "huggingface":
            try:
                from transformers import pipeline
                print("Loading Hugging Face model (first time may take a while)...")
                self.model = model or "facebook/bart-large-cnn"
                self.summarizer_pipeline = pipeline(
                    "summarization",
                    model=self.model,
                    device=-1  # CPU
                )
                print(f"✅ Loaded Hugging Face model: {self.model}")
            except ImportError:
                print("❌ transformers not installed. Install with:")
                print("   pip install transformers torch")
                raise
        
        elif self.backend == "extractive":
            try:
                from sumy.parsers.plaintext import PlaintextParser
                from sumy.nlp.tokenizers import Tokenizer
                from sumy.summarizers.lsa import LsaSummarizer
                import nltk
                # Download required NLTK data
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                print("✅ Using extractive summarization (sumy)")
            except ImportError:
                print("❌ sumy not installed. Install with:")
                print("   pip install sumy nltk")
                raise
        
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'openai', 'ollama', 'huggingface', or 'extractive'")
        
    def extract_arxiv_id(self, url: str) -> str:
        """Extract arXiv paper ID from URL."""
        # Handle various arXiv URL formats
        patterns = [
            r'arxiv.org/abs/(\d+\.\d+)',
            r'arxiv.org/pdf/(\d+\.\d+)',
            r'(\d{4}\.\d{4,5})',  # Direct ID format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract arXiv ID from: {url}")
    
    def download_pdf(self, arxiv_id: str, output_dir: str = ".") -> Path:
        """Download PDF from arXiv."""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        output_path = Path(output_dir) / f"{arxiv_id}.pdf"
        
        print(f"Downloading PDF from {pdf_url}...")
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"PDF downloaded to {output_path}")
        return output_path
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        print("Extracting text from PDF...")
        text = ""
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num}: {e}")
        
        print(f"Extracted {len(text)} characters from PDF")
        return text
    
    def identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract main sections from the paper."""
        sections = {
            'abstract': '',
            'introduction': '',
            'method': '',
            'results': '',
            'conclusion': ''
        }
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)
        text_lower = text.lower()
        
        # Define section patterns with variations
        section_patterns = {
            'abstract': [r'\babstract\b', r'\bsummary\b'],
            'introduction': [r'\b\d*\.?\s*introduction\b', r'\b\d*\.?\s*background\b'],
            'method': [
                r'\b\d*\.?\s*method(?:ology|s)?\b',
                r'\b\d*\.?\s*approach(?:es)?\b',
                r'\b\d*\.?\s*model(?:s)?\b',
                r'\b\d*\.?\s*algorithm(?:s)?\b'
            ],
            'results': [
                r'\b\d*\.?\s*result(?:s)?\b',
                r'\b\d*\.?\s*experiment(?:s|al results)?\b',
                r'\b\d*\.?\s*evaluation(?:s)?\b',
                r'\b\d*\.?\s*discussion\b'
            ],
            'conclusion': [
                r'\b\d*\.?\s*conclusion(?:s)?\b',
                r'\b\d*\.?\s*summary\b',
                r'\b\d*\.?\s*future work\b'
            ]
        }
        
        # Find all section headers
        section_positions = []
        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    section_positions.append({
                        'name': section_name,
                        'start': match.start(),
                        'pattern': pattern
                    })
        
        # Sort by position
        section_positions.sort(key=lambda x: x['start'])
        
        # Extract text for each section
        for i, section_info in enumerate(section_positions):
            section_name = section_info['name']
            start_pos = section_info['start']
            
            # Find end position (next section or end of text)
            if i + 1 < len(section_positions):
                end_pos = section_positions[i + 1]['start']
            else:
                end_pos = len(text)
            
            # Extract section text (limit to reasonable size)
            section_text = text[start_pos:end_pos].strip()
            
            # Only keep if not already populated with longer text
            if len(section_text) > len(sections[section_name]):
                sections[section_name] = section_text[:10000]  # Limit section size
        
        return sections
    
    def summarize_section(self, section_name: str, section_text: str) -> str:
        """Summarize a specific section using configured backend."""
        if not section_text or len(section_text.strip()) < 50:
            return f"*Section not found or too short*"
        
        print(f"Summarizing {section_name}...")
        
        prompts = {
            'abstract': "Summarize this abstract in 2-3 concise sentences, highlighting the main contribution:",
            'introduction': "Summarize the introduction in 3-4 sentences, focusing on the problem, motivation, and context:",
            'method': "Summarize the methodology in 3-4 sentences, explaining the key approach and techniques:",
            'results': "Summarize the results and discussion in 3-4 sentences, highlighting key findings and insights:",
            'conclusion': "Summarize the conclusion in 2-3 sentences, emphasizing main takeaways and future directions:"
        }
        
        prompt = prompts.get(section_name, "Summarize this section concisely:")
        
        try:
            if self.backend == "openai":
                return self._summarize_openai(prompt, section_text)
            elif self.backend == "ollama":
                return self._summarize_ollama(prompt, section_text)
            elif self.backend == "huggingface":
                return self._summarize_huggingface(prompt, section_text)
            elif self.backend == "extractive":
                return self._summarize_extractive(section_text)
        except Exception as e:
            print(f"Error summarizing {section_name}: {e}")
            return f"*Error generating summary: {str(e)}*"
    
    def _summarize_openai(self, prompt: str, text: str) -> str:
        """Summarize using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a scientific paper summarizer. Provide clear, concise, and accurate summaries."
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\n{text[:4000]}"
                }
            ],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    
    def _summarize_ollama(self, prompt: str, text: str) -> str:
        """Summarize using Ollama local LLM."""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": f"{prompt}\n\n{text[:4000]}\n\nProvide a clear and concise summary:",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    
    def _summarize_huggingface(self, prompt: str, text: str) -> str:
        """Summarize using Hugging Face transformers."""
        # BART/T5 models don't need the prompt in the same way
        # They're trained specifically for summarization
        input_text = text[:1024]  # BART limit
        
        result = self.summarizer_pipeline(
            input_text,
            max_length=150,
            min_length=50,
            do_sample=False
        )
        return result[0]['summary_text']
    
    def _summarize_extractive(self, text: str) -> str:
        """Summarize using extractive method (sumy)."""
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer
        from io import StringIO
        
        # Parse text
        parser = PlaintextParser.from_string(text[:5000], Tokenizer("english"))
        summarizer = LsaSummarizer()
        
        # Generate summary (4 sentences)
        summary_sentences = summarizer(parser.document, 4)
        
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        return summary if summary else "*Could not generate summary*"
    
    def generate_markdown_summary(
        self,
        arxiv_id: str,
        sections: Dict[str, str],
        summaries: Dict[str, str],
        output_path: Optional[Path] = None
    ) -> str:
        """Generate a markdown summary document."""
        # Get paper metadata
        title, authors = self.get_paper_metadata(arxiv_id)
        
        markdown = f"""# Paper Summary: {title}

**arXiv ID:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})  
**Authors:** {authors}  
**Summary Generated:** {self._get_current_date()}

---

## 📋 Abstract

{summaries.get('abstract', '*Not available*')}

---

## 🎯 Introduction

{summaries.get('introduction', '*Not available*')}

---

## 🔬 Methodology

{summaries.get('method', '*Not available*')}

---

## 📊 Results & Discussion

{summaries.get('results', '*Not available*')}

---

## 💡 Conclusion

{summaries.get('conclusion', '*Not available*')}

---

## 🔗 Additional Information

- **Full Paper:** https://arxiv.org/abs/{arxiv_id}
- **PDF:** https://arxiv.org/pdf/{arxiv_id}.pdf

---

*This summary was automatically generated using {self.backend}.*
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"\nMarkdown summary saved to {output_path}")
        
        return markdown
    
    def get_paper_metadata(self, arxiv_id: str) -> tuple:
        """Fetch paper metadata from arXiv API."""
        try:
            api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = requests.get(api_url)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # Extract title
            title_elem = root.find('.//{http://www.w3.org/2005/Atom}title')
            title = title_elem.text.strip() if title_elem is not None else "Unknown Title"
            title = re.sub(r'\s+', ' ', title)
            
            # Extract authors
            authors = []
            for author in root.findall('.//{http://www.w3.org/2005/Atom}author'):
                name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            authors_str = ', '.join(authors[:5])  # Limit to first 5 authors
            if len(authors) > 5:
                authors_str += ', et al.'
            
            return title, authors_str if authors_str else "Unknown Authors"
            
        except Exception as e:
            print(f"Warning: Could not fetch metadata: {e}")
            return "Unknown Title", "Unknown Authors"
    
    def _get_current_date(self) -> str:
        """Get current date in readable format."""
        from datetime import datetime
        return datetime.now().strftime("%B %d, %Y")
    
    def process_paper(self, arxiv_url: str, output_dir: str = ".") -> Path:
        """Main method to process an arXiv paper."""
        # Extract arXiv ID
        arxiv_id = self.extract_arxiv_id(arxiv_url)
        print(f"\n📄 Processing arXiv paper: {arxiv_id}\n")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download PDF
        pdf_path = self.download_pdf(arxiv_id, output_dir)
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Identify sections
        print("\nIdentifying paper sections...")
        sections = self.identify_sections(text)
        
        # Summarize each section
        print("\nGenerating summaries...")
        summaries = {}
        for section_name, section_text in sections.items():
            summaries[section_name] = self.summarize_section(section_name, section_text)
        
        # Generate markdown summary
        markdown_path = output_path / f"{arxiv_id}_summary.md"
        self.generate_markdown_summary(arxiv_id, sections, summaries, markdown_path)
        
        print("\n✅ Summary generation complete!")
        print(f"📄 PDF: {pdf_path}")
        print(f"📝 Summary: {markdown_path}")
        
        return markdown_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download and summarize arXiv papers (FREE options available!)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Ollama (FREE, local, recommended)
  python summarize_paper.py https://arxiv.org/abs/2301.00001 --backend ollama
  
  # Using Hugging Face (FREE, no installation needed)
  python summarize_paper.py 2301.00001 --backend huggingface
  
  # Using extractive summarization (FREE, fastest)
  python summarize_paper.py 2301.00001 --backend extractive
  
  # Using OpenAI (requires API key and costs money)
  python summarize_paper.py 2301.00001 --backend openai --api-key sk-...

Backends:
  - ollama:      Local LLM (best quality, free, requires Ollama installed)
  - huggingface: Pre-trained models (good quality, free, slower first time)
  - extractive:  Statistical method (fast, free, lower quality)
  - openai:      GPT-4 (best quality, costs money)
        """
    )
    
    parser.add_argument(
        'arxiv_url',
        help='arXiv URL or paper ID (e.g., https://arxiv.org/abs/2301.00001 or 2301.00001)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='.',
        help='Output directory for PDF and summary (default: current directory)'
    )
    parser.add_argument(
        '--backend',
        choices=['ollama', 'huggingface', 'extractive', 'openai'],
        default='ollama',
        help='Summarization backend (default: ollama)'
    )
    parser.add_argument(
        '--model',
        help='Model name for the backend (optional, uses defaults)'
    )
    parser.add_argument(
        '--api-key',
        help='API key for OpenAI (only needed for openai backend)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize summarizer
        summarizer = ArxivPaperSummarizer(
            backend=args.backend,
            api_key=args.api_key,
            model=args.model
        )
        
        # Process paper
        summarizer.process_paper(args.arxiv_url, args.output_dir)
        
    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

