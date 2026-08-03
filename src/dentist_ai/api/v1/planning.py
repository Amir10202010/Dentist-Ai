"""Generating a treatment plan and accepting one of its options."""

from __future__ import annotations

from fastapi import APIRouter, status

from dentist_ai.api.deps import ApiRateLimit, CurrentUser, PlanningDep, RequestCtx
from dentist_ai.api.presenters import present_plan
from dentist_ai.schemas.treatment import (
    AcceptOptionRequest,
    GeneratePlanRequest,
    PlanResponse,
)

router = APIRouter(prefix="/planning", tags=["planning"])


@router.post(
    "/generate",
    response_model=PlanResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[ApiRateLimit],
    summary="Propose a treatment plan from a patient's reviewed findings",
)
async def generate(
    payload: GeneratePlanRequest,
    user: CurrentUser,
    planning: PlanningDep,
    context: RequestCtx,
) -> PlanResponse:
    """Write a **draft** plan carrying several options and no items.

    Nothing is scheduled until a clinician accepts one of the options — see
    `POST /planning/{public_id}/accept`.
    """
    plan = await planning.generate(
        payload.patient_id,
        actor=user,
        context=context,
        volume_public_id=payload.volume_public_id,
        study_public_id=payload.study_public_id,
    )
    return present_plan(plan, user.locale)


@router.post(
    "/{public_id}/accept",
    response_model=PlanResponse,
    dependencies=[ApiRateLimit],
    summary="Turn one option's steps into the plan's actual work",
)
async def accept(
    public_id: str,
    payload: AcceptOptionRequest,
    user: CurrentUser,
    planning: PlanningDep,
    context: RequestCtx,
) -> PlanResponse:
    plan = await planning.accept_option(public_id, payload.approach, actor=user, context=context)
    return present_plan(plan, user.locale)
