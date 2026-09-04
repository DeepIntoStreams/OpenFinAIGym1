"""Custom exceptions for openfinai_pipeline."""


class ScreenerError(Exception):
    """Base exception."""


class ScraperError(ScreenerError):
    """Scraping or enrichment failed."""
