"""3D scan payloads."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field, computed_field

from dentist_ai.db.models import MeshFormat, ScanArch, ScanKind
from dentist_ai.schemas.common import ApiModel


class ScanBounds(ApiModel):
    """Axis-aligned bounds in the file's own units (millimetres in practice)."""

    min: tuple[float, float, float]
    max: tuple[float, float, float]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size(self) -> tuple[float, float, float]:
        return (
            round(self.max[0] - self.min[0], 2),
            round(self.max[1] - self.min[1], 2),
            round(self.max[2] - self.min[2], 2),
        )


class ScanResponse(ApiModel):
    public_id: str
    patient_id: int
    patient_name: str | None
    original_filename: str
    source_format: MeshFormat
    kind: ScanKind
    kind_label: str
    arch: ScanArch
    arch_label: str
    triangle_count: int
    byte_size: int
    bounds: ScanBounds
    captured_on: date | None
    notes: str | None
    created_at: datetime
    uploaded_by_name: str | None
    mesh_url: str
    page_url: str


class ScanUpdateRequest(ApiModel):
    kind: ScanKind = ScanKind.INTRAORAL
    arch: ScanArch = ScanArch.BOTH
    captured_on: date | None = None
    notes: str | None = Field(default=None, max_length=4000)
