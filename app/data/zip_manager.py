import zipfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Extractor(ABC):
    @abstractmethod
    def extract(self, source_path: str | Path, destination_path: str | Path) -> None:
        pass

class ZipExtractor(Extractor):
    def extract(self, source_path: str | Path, destination_path: str | Path) -> None:
        src = Path(source_path)
        dest = Path(destination_path)

        if not src.exists():
            logger.error(f"Source file not found: {src}")
            raise FileNotFoundError(f"The file {src} does not exist.")
        
        if not src.suffix.lower() == '.zip':
            logger.warning(f"File {src} does not have a .zip extension.")

        dest.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Starting decompression of '{src.name}' into '{dest}'...")
            with zipfile.ZipFile(src, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                zip_ref.extractall(dest)
                
            logger.info(f"Decompression completed successfully. Total files: {len(file_list)}")
            
        except zipfile.BadZipFile:
            logger.error(f"Failed to decompress: '{src}' is not a valid zip file.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during extraction: {e}")
            raise

class DecompressionService:
    def __init__(self, extractor: Extractor):
        self._extractor = extractor

    def process(self, source: str, destination: str) -> None:
        self._extractor.extract(source, destination)

if __name__ == "__main__":
    EXTRACTOR = ZipExtractor()
    SERVICE = DecompressionService(EXTRACTOR)
