import os
import re
import sys
import fitz
import logging
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

from src.core.config.config import Config


class PreprocessingService:
    """Data Preprocessor class for cleaning input row data and preparing them for future work."""
    def __init__(
        self,
        data_path: Path,
        text_cleaning_flag: bool = True
    ) -> None:
        self.data_path = Path(data_path)
        self.row_data_buffer = self._load_data()
        if text_cleaning_flag:
            self.cleaned_data = self._clear_row_text()

    def _load_pdf(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract text from a PDF file using PyPDF2.
        
        Args:
            file_path: (Path) - Path to pdf file.
        Returns:
            Dict[str, Any] - Dictionary with file path, type and text.
        """
        try:
            text: str = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            text = unicodedata.normalize("NFKC", text).strip()
            return {
                "file_path": file_path,
                "text": text,
                "type": "pdf"
            }
        except Exception as e:
            logger.error("Error processing PDF file %s: %s", file_path, e)
            return None

    def _load_html(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract text from an HTML file using BeautifulSoup.
        
        Args:
            file_path: (Path) - Path to html file.
        Returns:
            Dict[str, Any] - Dictionary with file path, type and text.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
            text = unicodedata.normalize("NFKC", text).strip()
            return {
                "file_path": file_path,
                "text": text,
                "type": "html"
            }
        except Exception as e:
            logger.error("Error processing HTML file %s: %s", file_path, e)
            return None

    def _load_data(self) -> List[Dict[str, Any]]:
        """Data Preprocessor class for cleaning input row data and preparing them for future work."""
        data_buffer: List[Dict[str, Any]] = []

        for entry in os.scandir(self.data_path):
            if not entry.is_file():
                continue
            file_path = Path(entry.path)
            ext = file_path.suffix.lower()

            try:
                if ext == ".pdf":
                    data = self._load_pdf(file_path)
                    if data:
                        data_buffer.append(data)
                        logger.debug("PDF loaded: %s", file_path)
                elif ext == ".html" or ext == ".htm":
                    data = self._load_html(file_path)
                    if data:
                        data_buffer.append(data)
                        logger.debug("HTML loaded: %s", file_path)
                else:
                    logger.debug("Skipped file with unknown extension: %s", file_path)
            except Exception as e:
                logger.exception("Failed to load %s: %s", file_path, e)

        logger.info("Data buffer size: %d", len(data_buffer))
        return data_buffer

    def _normalize_text(self, text: str) -> str:
        """
        Method for normalizing ukrainian text.
        
        Args:
            text: (str) - Text to normalize.

        Returns:
            (str) - Normalized text.
        """
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\u00AD", "")
        text = re.sub(r"(\S)-\s*\n\s*(\S)", r"\1\2", text)
        text = text.replace("'", "\u02BC")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def drop_lonely_X_lines(self, text: str) -> str:
        """Method for dropping lonely X lines."""
        return re.sub(r"(?m)^\s*[XxХх]\s*$\n?", "", text)

    def drop_lonely_numbers(self, text: str) -> str:
        """Method for dropping lonely numbers."""
        lines = text.splitlines()
        out = []
        for ln in lines:
            l = ln.strip()
            if re.fullmatch(r"\d{1,3}", l):
                continue
            if re.fullmatch(r"\d{1,3}\s+\d{1,3}", l):
                continue
            if re.fullmatch(r"сторінка\s*\d+|page\s*\d+", l, flags=re.IGNORECASE):
                continue
            out.append(ln)
        return "\n".join(out)

    def sanitize_transcript_input(self, text: str) -> str:
        return (
            text.replace('%', ' percent')
                .replace('{', '[')
                .replace('}', ']')
                .replace('\\', '/')
                .replace('®', 'TM')
                .replace("™", "TM")
        )

    def _clean_text(self, text: str) -> str:
        """Method for cleaning text from noise, extra spaces etc..."""
        normalized_ukrainian_text = self._normalize_text(text)
        cleaned_text = self.drop_lonely_numbers(normalized_ukrainian_text)
        cleaned_text = self.sanitize_transcript_input(cleaned_text)
        cleaned_text = self._clean_artifacts(cleaned_text)
        cleaned_text = self.drop_lonely_X_lines(cleaned_text)
        return cleaned_text

    def _clear_row_text(self) -> List[Dict[str, Any]]:
        """Method for cleaning and preparing row parsed PDF and HTML data for indexing."""
        cleaned_data = []

        for object in self.row_data_buffer:
            clean_text = self._clean_text(object["text"])
            cleaned_data.append({
                "file_path": object["file_path"],
                "text": clean_text,
                "type": object["type"]
            })
        return cleaned_data

    def _clean_artifacts(self, text: str) -> str:
        """Method for cleaning artifacts like: .................."""
        toc_hdr = re.compile(r'^\s*(ЗМІСТ|CONTENTS)\s*:?\s*$', re.IGNORECASE)
        leader = re.compile(r'^\s*\S.*(?:\.{3,}|\s{2,})\s*\d+\s*$')

        lines = text.splitlines()
        out = []
        i = 0

        while i < len(lines):
            s = lines[i].rstrip()

            if toc_hdr.match(s):
                i += 1
                while i < len(lines):
                    s2 = lines[i].rstrip()
                    if not s2 or leader.match(s2):
                        i += 1
                    else:
                        break
                continue

            if leader.match(s):
                i += 1
                continue

            out.append(lines[i])
            i += 1

        return "\n".join(out)

    @staticmethod
    def save_text_to_file(text: str, file_path: str) -> None:
        """Method for saving text to file."""
        p = Path(file_path)
        os.makedirs(p.parent, exist_ok=True)

        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
            logger.info("Text saved to file: %s", file_path)
        except Exception as e:
            logger.error(f"Failed to save file: {file_path}. Error: {e}")

# TODO: (Improvments) Add Agentic cleaning into general pipeline