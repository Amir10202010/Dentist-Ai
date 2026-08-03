"""3D scans, treatment plans, tooth numbering

Revision ID: 548d0094649a
Revises: fc22bc402903
Create Date: 2026-07-29 15:45:57.705233
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "548d0094649a"
down_revision: str | None = "fc22bc402903"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FDI_RANGE = "tooth_number IS NULL OR (tooth_number / 10 BETWEEN 1 AND 4 AND tooth_number % 10 BETWEEN 1 AND 8)"


def upgrade() -> None:
    op.create_table(
        "scans_3d",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("public_id", sa.String(length=26), nullable=False),
        sa.Column(
            "organization_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            nullable=False,
        ),
        sa.Column(
            "patient_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False
        ),
        sa.Column(
            "uploaded_by_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=True
        ),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column(
            "source_format",
            sa.Enum("STL", "PLY", "OBJ", name="meshformat", native_enum=False, length=8),
            nullable=False,
        ),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("triangle_count", sa.Integer(), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "INTRAORAL",
                "PLASTER_MODEL",
                "CBCT_SURFACE",
                "RESTORATION_DESIGN",
                name="scankind",
                native_enum=False,
                length=24,
            ),
            nullable=False,
        ),
        sa.Column(
            "arch",
            sa.Enum("UPPER", "LOWER", "BOTH", name="scanarch", native_enum=False, length=8),
            nullable=False,
        ),
        sa.Column("min_x", sa.Float(), nullable=False),
        sa.Column("min_y", sa.Float(), nullable=False),
        sa.Column("min_z", sa.Float(), nullable=False),
        sa.Column("max_x", sa.Float(), nullable=False),
        sa.Column("max_y", sa.Float(), nullable=False),
        sa.Column("max_z", sa.Float(), nullable=False),
        sa.Column("captured_on", sa.Date(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("byte_size > 0", name=op.f("ck_scans_3d_scan_positive_size")),
        sa.CheckConstraint("triangle_count > 0", name=op.f("ck_scans_3d_scan_positive_triangles")),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
            name=op.f("fk_scans_3d_organization_id_organizations"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_scans_3d_patient_id_patients"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["uploaded_by_id"],
            ["users.id"],
            name=op.f("fk_scans_3d_uploaded_by_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_scans_3d")),
    )
    with op.batch_alter_table("scans_3d", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_scans_3d_content_hash"), ["content_hash"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_scans_3d_created_at"), ["created_at"], unique=False)
        batch_op.create_index(
            "ix_scans_3d_organization_id_created_at",
            ["organization_id", "created_at"],
            unique=False,
        )
        batch_op.create_index(
            "ix_scans_3d_patient_id_created_at", ["patient_id", "created_at"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_scans_3d_public_id"), ["public_id"], unique=True)

    op.create_table(
        "treatment_plans",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("public_id", sa.String(length=26), nullable=False),
        sa.Column(
            "organization_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            nullable=False,
        ),
        sa.Column(
            "patient_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False
        ),
        sa.Column(
            "created_by_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=True
        ),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "DRAFT",
                "ACTIVE",
                "COMPLETED",
                "CANCELLED",
                name="planstatus",
                native_enum=False,
                length=16,
            ),
            nullable=False,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["users.id"],
            name=op.f("fk_treatment_plans_created_by_id_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
            name=op.f("fk_treatment_plans_organization_id_organizations"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_treatment_plans_patient_id_patients"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_treatment_plans")),
    )
    with op.batch_alter_table("treatment_plans", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_treatment_plans_created_at"), ["created_at"], unique=False
        )
        batch_op.create_index(
            "ix_treatment_plans_organization_id_created_at",
            ["organization_id", "created_at"],
            unique=False,
        )
        batch_op.create_index(
            "ix_treatment_plans_patient_id_status", ["patient_id", "status"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_treatment_plans_public_id"), ["public_id"], unique=True
        )

    op.create_table(
        "treatment_plan_items",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("plan_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("procedure_code", sa.String(length=48), nullable=False),
        sa.Column("tooth_number", sa.Integer(), nullable=True),
        sa.Column("priority", sa.String(length=16), nullable=False),
        sa.Column("estimated_visits", sa.Integer(), nullable=False),
        sa.Column("estimated_minutes", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PROPOSED",
                "ACCEPTED",
                "SCHEDULED",
                "IN_PROGRESS",
                "DONE",
                "DECLINED",
                name="planitemstatus",
                native_enum=False,
                length=16,
            ),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("scheduled_for", sa.Date(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "source_finding_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            nullable=True,
        ),
        sa.Column("source_study_public_id", sa.String(length=26), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "estimated_visits > 0",
            name=op.f("ck_treatment_plan_items_plan_item_positive_visits"),
        ),
        sa.CheckConstraint(
            _FDI_RANGE, name=op.f("ck_treatment_plan_items_plan_item_tooth_number_fdi")
        ),
        sa.ForeignKeyConstraint(
            ["plan_id"],
            ["treatment_plans.id"],
            name=op.f("fk_treatment_plan_items_plan_id_treatment_plans"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_finding_id"],
            ["findings.id"],
            name=op.f("fk_treatment_plan_items_source_finding_id_findings"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_treatment_plan_items")),
    )
    with op.batch_alter_table("treatment_plan_items", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_treatment_plan_items_created_at"), ["created_at"], unique=False
        )
        batch_op.create_index(
            "ix_treatment_plan_items_plan_id_position", ["plan_id", "position"], unique=False
        )
        batch_op.create_index(
            "ix_treatment_plan_items_plan_id_status", ["plan_id", "status"], unique=False
        )

    # Existing findings predate tooth numbering: they get NULL, and the
    # clinician can chart them by hand or re-run the analysis.
    with op.batch_alter_table("findings", schema=None) as batch_op:
        batch_op.add_column(sa.Column("tooth_number", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "tooth_confirmed",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        batch_op.create_check_constraint("finding_tooth_number_fdi", _FDI_RANGE)


def downgrade() -> None:
    with op.batch_alter_table("findings", schema=None) as batch_op:
        batch_op.drop_constraint("finding_tooth_number_fdi", type_="check")
        batch_op.drop_column("tooth_confirmed")
        batch_op.drop_column("tooth_number")

    with op.batch_alter_table("treatment_plan_items", schema=None) as batch_op:
        batch_op.drop_index("ix_treatment_plan_items_plan_id_status")
        batch_op.drop_index("ix_treatment_plan_items_plan_id_position")
        batch_op.drop_index(batch_op.f("ix_treatment_plan_items_created_at"))
    op.drop_table("treatment_plan_items")

    with op.batch_alter_table("treatment_plans", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_treatment_plans_public_id"))
        batch_op.drop_index("ix_treatment_plans_patient_id_status")
        batch_op.drop_index("ix_treatment_plans_organization_id_created_at")
        batch_op.drop_index(batch_op.f("ix_treatment_plans_created_at"))
    op.drop_table("treatment_plans")

    with op.batch_alter_table("scans_3d", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_scans_3d_public_id"))
        batch_op.drop_index("ix_scans_3d_patient_id_created_at")
        batch_op.drop_index("ix_scans_3d_organization_id_created_at")
        batch_op.drop_index(batch_op.f("ix_scans_3d_created_at"))
        batch_op.drop_index(batch_op.f("ix_scans_3d_content_hash"))
    op.drop_table("scans_3d")
