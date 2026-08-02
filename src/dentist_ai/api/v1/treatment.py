"""Treatment plans and their steps."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from dentist_ai.api.deps import ApiRateLimit, CurrentUser, RequestCtx, TreatmentDep
from dentist_ai.api.presenters import (
    present_plan,
    present_plan_item,
    present_procedure_catalogue,
)
from dentist_ai.schemas.common import OkResponse
from dentist_ai.schemas.treatment import (
    PlanCreateRequest,
    PlanItemCreateRequest,
    PlanItemResponse,
    PlanItemUpdateRequest,
    PlanResponse,
    PlanUpdateRequest,
    ProcedureOption,
    ProposeFromStudyRequest,
)

router = APIRouter(prefix="/treatment", tags=["treatment"], dependencies=[ApiRateLimit])


@router.get("/procedures", response_model=list[ProcedureOption], summary="Protocol catalogue")
async def list_procedures(user: CurrentUser) -> list[ProcedureOption]:
    return present_procedure_catalogue(user.locale)


@router.get("/plans", response_model=list[PlanResponse], summary="Plans for one patient")
async def list_plans(
    user: CurrentUser,
    treatment: TreatmentDep,
    patient_id: Annotated[int, Query()],
) -> list[PlanResponse]:
    plans = await treatment.list_for_patient(patient_id, organization_id=user.organization_id)
    return [present_plan(plan, user.locale) for plan in plans]


@router.post(
    "/plans",
    response_model=PlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an empty plan",
)
async def create_plan(
    payload: PlanCreateRequest,
    user: CurrentUser,
    treatment: TreatmentDep,
    context: RequestCtx,
) -> PlanResponse:
    plan = await treatment.create(
        actor=user,
        context=context,
        patient_id=payload.patient_id,
        title=payload.title,
        notes=payload.notes,
    )
    return present_plan(plan, user.locale)


@router.post(
    "/plans/propose",
    response_model=PlanResponse,
    summary="Draft steps from a study's findings",
)
async def propose_from_study(
    payload: ProposeFromStudyRequest,
    user: CurrentUser,
    treatment: TreatmentDep,
    context: RequestCtx,
) -> PlanResponse:
    plan = await treatment.propose_from_study(
        payload.study_public_id,
        actor=user,
        context=context,
        plan_public_id=payload.plan_public_id,
    )
    return present_plan(plan, user.locale)


@router.get("/plans/{public_id}", response_model=PlanResponse)
async def get_plan(public_id: str, user: CurrentUser, treatment: TreatmentDep) -> PlanResponse:
    plan = await treatment.get(public_id, organization_id=user.organization_id)
    return present_plan(plan, user.locale)


@router.put("/plans/{public_id}", response_model=PlanResponse)
async def update_plan(
    public_id: str,
    payload: PlanUpdateRequest,
    user: CurrentUser,
    treatment: TreatmentDep,
    context: RequestCtx,
) -> PlanResponse:
    plan = await treatment.update(
        public_id,
        actor=user,
        context=context,
        title=payload.title,
        status=payload.status,
        notes=payload.notes,
    )
    return present_plan(plan, user.locale)


@router.delete("/plans/{public_id}", response_model=OkResponse)
async def delete_plan(
    public_id: str,
    user: CurrentUser,
    treatment: TreatmentDep,
    context: RequestCtx,
) -> OkResponse:
    await treatment.delete(public_id, actor=user, context=context)
    return OkResponse()


@router.post(
    "/plans/{public_id}/items",
    response_model=PlanItemResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_item(
    public_id: str,
    payload: PlanItemCreateRequest,
    user: CurrentUser,
    treatment: TreatmentDep,
    context: RequestCtx,
) -> PlanItemResponse:
    item = await treatment.add_item(
        public_id,
        actor=user,
        context=context,
        procedure_code=payload.procedure_code,
        tooth_number=payload.tooth_number,
        notes=payload.notes,
    )
    return present_plan_item(item, user.locale)


@router.patch("/plans/{public_id}/items/{item_id}", response_model=PlanItemResponse)
async def update_item(
    public_id: str,
    item_id: int,
    payload: PlanItemUpdateRequest,
    user: CurrentUser,
    treatment: TreatmentDep,
    context: RequestCtx,
) -> PlanItemResponse:
    item = await treatment.update_item(
        public_id,
        item_id,
        actor=user,
        context=context,
        status=payload.status,
        tooth_number=payload.tooth_number,
        scheduled_for=payload.scheduled_for,
        estimated_visits=payload.estimated_visits,
        estimated_minutes=payload.estimated_minutes,
        notes=payload.notes,
    )
    return present_plan_item(item, user.locale)


@router.delete("/plans/{public_id}/items/{item_id}", response_model=OkResponse)
async def remove_item(
    public_id: str,
    item_id: int,
    user: CurrentUser,
    treatment: TreatmentDep,
    context: RequestCtx,
) -> OkResponse:
    await treatment.remove_item(public_id, item_id, actor=user, context=context)
    return OkResponse()
