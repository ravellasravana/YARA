"""YARA - a multi-agent research assistant with grounded, citation-verified output."""

from .app import YARA
from .briefs.schema import Finding, ResearchBrief, SubQuestion
from .config import Settings, get_settings

__version__ = "0.2.0"
__all__ = ["Finding", "ResearchBrief", "Settings", "SubQuestion", "YARA", "get_settings"]
