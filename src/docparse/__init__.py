"""docparse-sdk: unified client for MinerU2.5 / TeleOCR document-parsing services."""
from .config import BackendConfig
from .models import Block, Document, Page
from .parser import AsyncDocParser, DocParser

__version__ = "0.1.0"
__all__ = ["DocParser", "AsyncDocParser", "BackendConfig", "Document", "Page", "Block", "__version__"]
