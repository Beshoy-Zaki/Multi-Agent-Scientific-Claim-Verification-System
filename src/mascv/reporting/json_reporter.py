"""JSON export for scientific reports."""

from mascv.models.verdict import ScientificReport


class JSONReporter:
    """Exports structured assessment to JSON format."""

    def export(self, report: ScientificReport, output_path: str) -> None:
        """Write report as indented JSON."""
        raise NotImplementedError
