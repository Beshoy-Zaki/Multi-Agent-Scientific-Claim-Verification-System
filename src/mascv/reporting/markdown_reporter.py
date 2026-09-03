"""Markdown report generator."""

from mascv.models.verdict import ScientificReport


class MarkdownReporter:
    """Generates user-readable Markdown reports detailing claims, evidence, and debates."""

    def generate(self, report: ScientificReport) -> str:
        """Render complete Markdown document."""
        raise NotImplementedError
