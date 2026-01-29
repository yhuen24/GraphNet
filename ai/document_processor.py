"""
Document processing module for GraphNet.
Handles extraction of text from various document formats.
"""

import os
from typing import List, Dict, Any
from io import BytesIO
import logging
import json

# Document processing libraries
import PyPDF2
from docx import Document as DocxDocument
from pptx import Presentation
import openpyxl
import pandas as pd

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process various document formats and extract text"""
    
    @staticmethod
    def process_file(file_path: str = None, file_bytes: bytes = None, 
                     file_extension: str = None, filename: str = "unknown") -> Dict[str, Any]:
        """
        Process a file and extract text content
        
        Args:
            file_path: Path to the file (for local files)
            file_bytes: Bytes content of the file (for uploaded files)
            file_extension: Extension of the file
            filename: Name of the file
        
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Determine file extension
            if file_path:
                file_extension = os.path.splitext(file_path)[1].lower()
            
            if not file_extension:
                raise ValueError("File extension not provided")
            
            # Route to appropriate processor
            if file_extension in ['.txt', '.md']:
                return DocumentProcessor._process_text(file_path, file_bytes, filename)
            elif file_extension == '.pdf':
                return DocumentProcessor._process_pdf(file_path, file_bytes, filename)
            elif file_extension == '.docx':
                return DocumentProcessor._process_docx(file_path, file_bytes, filename)
            elif file_extension == '.xlsx':
                return DocumentProcessor._process_xlsx(file_path, file_bytes, filename)
            elif file_extension == '.pptx':
                return DocumentProcessor._process_pptx(file_path, file_bytes, filename)
            elif file_extension == '.csv':
                return DocumentProcessor._process_csv(file_path, file_bytes, filename)
            elif file_extension == '.json':
                return DocumentProcessor._process_json(file_path, file_bytes, filename)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "metadata": {}
            }
    
    @staticmethod
    def _process_text(file_path: str = None, file_bytes: bytes = None, filename: str = "unknown") -> Dict[str, Any]:
        """Process plain text files"""
        try:
            if file_bytes:
                text = file_bytes.decode('utf-8')
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            return {
                "success": True,
                "text": text,
                "metadata": {
                    "format": "text",
                    "filename": filename,
                    "character_count": len(text),
                    "word_count": len(text.split())
                }
            }
        except Exception as e:
            logger.error(f"Error processing text file: {str(e)}")
            raise
    
    @staticmethod
    def _process_pdf(file_path: str = None, file_bytes: bytes = None, filename: str = "unknown") -> Dict[str, Any]:
        """Process PDF files"""
        try:
            if file_bytes:
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            else:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            
            return {
                "success": True,
                "text": text,
                "metadata": {
                    "format": "pdf",
                    "filename": filename,
                    "page_count": len(pdf_reader.pages),
                    "character_count": len(text),
                    "word_count": len(text.split())
                }
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    @staticmethod
    def _process_docx(file_path: str = None, file_bytes: bytes = None, filename: str = "unknown") -> Dict[str, Any]:
        """Process Word documents"""
        try:
            if file_bytes:
                doc = DocxDocument(BytesIO(file_bytes))
            else:
                doc = DocxDocument(file_path)
            
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            
            # Extract tables
            for table in doc.tables:
                text += "\n--- Table ---\n"
                for row in table.rows:
                    row_text = " | ".join([cell.text for cell in row.cells])
                    text += row_text + "\n"
            
            return {
                "success": True,
                "text": text,
                "metadata": {
                    "format": "docx",
                    "filename": filename,
                    "paragraph_count": len(doc.paragraphs),
                    "table_count": len(doc.tables),
                    "character_count": len(text),
                    "word_count": len(text.split())
                }
            }
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            raise
    
    @staticmethod
    def _process_xlsx(file_path: str = None, file_bytes: bytes = None, filename: str = "unknown") -> Dict[str, Any]:
        """Process Excel files"""
        try:
            if file_bytes:
                df_dict = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
            else:
                df_dict = pd.read_excel(file_path, sheet_name=None)
            
            text = ""
            for sheet_name, df in df_dict.items():
                text += f"\n--- Sheet: {sheet_name} ---\n"
                text += df.to_string(index=False)
                text += "\n\n"
            
            return {
                "success": True,
                "text": text,
                "metadata": {
                    "format": "xlsx",
                    "filename": filename,
                    "sheet_count": len(df_dict),
                    "character_count": len(text),
                    "word_count": len(text.split())
                }
            }
        except Exception as e:
            logger.error(f"Error processing XLSX: {str(e)}")
            raise
    
    @staticmethod
    def _process_pptx(file_path: str = None, file_bytes: bytes = None, filename: str = "unknown") -> Dict[str, Any]:
        """Process PowerPoint files"""
        try:
            if file_bytes:
                prs = Presentation(BytesIO(file_bytes))
            else:
                prs = Presentation(file_path)
            
            text = ""
            for slide_num, slide in enumerate(prs.slides):
                text += f"\n--- Slide {slide_num + 1} ---\n"
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            
            return {
                "success": True,
                "text": text,
                "metadata": {
                    "format": "pptx",
                    "filename": filename,
                    "slide_count": len(prs.slides),
                    "character_count": len(text),
                    "word_count": len(text.split())
                }
            }
        except Exception as e:
            logger.error(f"Error processing PPTX: {str(e)}")
            raise
    
    @staticmethod
    def _process_csv(file_path: str = None, file_bytes: bytes = None, filename: str = "unknown") -> Dict[str, Any]:
        """Process CSV files"""
        try:
            if file_bytes:
                df = pd.read_csv(BytesIO(file_bytes))
            else:
                df = pd.read_csv(file_path)
            
            text = df.to_string(index=False)
            
            return {
                "success": True,
                "text": text,
                "metadata": {
                    "format": "csv",
                    "filename": filename,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "character_count": len(text),
                    "word_count": len(text.split())
                }
            }
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise
    
    @staticmethod
    def _process_json(file_path: str = None, file_bytes: bytes = None, filename: str = "unknown") -> Dict[str, Any]:
        """Process JSON files"""
        try:
            if file_bytes:
                data = json.loads(file_bytes.decode('utf-8'))
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            text = json.dumps(data, indent=2)
            
            return {
                "success": True,
                "text": text,
                "metadata": {
                    "format": "json",
                    "filename": filename,
                    "character_count": len(text),
                    "word_count": len(text.split())
                }
            }
        except Exception as e:
            logger.error(f"Error processing JSON: {str(e)}")
            raise
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Split text into chunks for processing
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = config.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - chunk_overlap
        
        return chunks
