"""Reviewer-first Streamlit interface for local FireSentinel evidence packets."""

from __future__ import annotations

import streamlit as st

from firesentinel.config import load_settings
from firesentinel.ui.reviewer import (
    ReviewerCase,
    ReviewerObservation,
    contour_preview,
    demo_cases,
    discover_reviewer_cases,
    load_candidate_mask,
    reason_explanations,
)


def main() -> None:
    """Render reviewer stories from completed local artifacts and fixed demos."""

    st.set_page_config(
        page_title="FireSentinel reviewer", page_icon="🔥", layout="wide"
    )
    st.title("FireSentinel evidence reviewer")
    st.caption(
        "Development-only thermal evidence review. Nothing on this page confirms "
        "a wildfire."
    )

    catalog = discover_reviewer_cases(load_settings().artifacts_dir)
    cases = (*demo_cases(), *catalog.cases)
    if not cases:
        st.warning(
            "No reviewable evidence packets were found. Use a demo story instead."
        )
        cases = demo_cases()

    selected = _case_picker(cases)
    _render_case(selected)
    if catalog.warnings:
        with st.expander("Artifact discovery notes"):
            for warning in catalog.warnings:
                st.warning(warning)


def _case_picker(cases: tuple[ReviewerCase, ...]) -> ReviewerCase:
    case_by_id = {case.case_id: case for case in cases}
    labels = {case.case_id: f"{case.title} — {case.source_kind}" for case in cases}
    with st.sidebar:
        st.header("Review a case")
        st.caption("Start with a fixed story or inspect a completed local packet.")
        st.subheader("Deterministic demos")
        demo_columns = st.columns(3)
        for column, case in zip(demo_columns, demo_cases(), strict=True):
            if column.button(_demo_button_label(case), width="stretch"):
                st.session_state["reviewer_case_id"] = case.case_id
        if st.session_state.get("reviewer_case_id") not in case_by_id:
            st.session_state["reviewer_case_id"] = cases[0].case_id
        selected_id = st.selectbox(
            "Case selection",
            options=tuple(case_by_id),
            format_func=lambda case_id: labels[case_id],
            key="reviewer_case_id",
        )
    return case_by_id[selected_id]


def _demo_button_label(case: ReviewerCase) -> str:
    return {
        "demo-emerging-event": "Emerging event",
        "demo-matched-control": "Matched control",
        "demo-abstention": "Abstention",
    }.get(case.case_id, case.title)


def _render_case(case: ReviewerCase) -> None:
    st.header(case.title)
    context, status = st.columns((2, 1))
    context.subheader("Location context")
    context.write(case.location)
    context.caption(f"Case ID: {case.case_id} · {case.source_kind}")
    status.subheader("Outcome")
    _render_outcome(case)

    st.subheader("Initial ambiguity")
    st.write(case.initial_ambiguity)

    if case.reviewer_panel_path is not None and case.reviewer_panel_path.is_file():
        st.subheader("Historical before / after panel")
        st.image(
            str(case.reviewer_panel_path),
            caption=(
                "Packet-supplied review panel; it is evidence context, not "
                "confirmation."
            ),
            width="stretch",
        )

    st.subheader("Chronological evidence strip")
    if not case.observations:
        st.info("This packet has no displayable observations.")
    else:
        _render_evidence_strip(case.observations)

    st.subheader("Measurements")
    if case.measurements:
        st.dataframe(
            [measurement.to_row() for measurement in case.measurements],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No packet measurements were recorded.")

    left, right = st.columns(2)
    with left:
        st.subheader("Considered actions")
        if case.selected_action:
            st.success(f"Selected action: {case.selected_action}")
        else:
            st.info("No bounded action was recorded for this evidence packet.")
        if case.considered_actions:
            st.dataframe(
                list(case.considered_actions), width="stretch", hide_index=True
            )
    with right:
        st.subheader("Changed evidence")
        if case.evidence_changes:
            st.dataframe(list(case.evidence_changes), width="stretch", hide_index=True)
        else:
            st.info("No comparison change was recorded.")

    _render_reason_codes(case)
    _render_limits_and_provenance(case)


def _render_outcome(case: ReviewerCase) -> None:
    outcome = case.outcome
    message = outcome.label
    if outcome.confidence is not None:
        message += f" · confidence {outcome.confidence:.2f}"
    if outcome.state in {"insufficient_evidence", "failed"}:
        st.warning(message)
    elif outcome.state in {"review_escalation", "human_review"}:
        st.info(message)
    elif outcome.state == "no_persistent_evidence":
        st.success(message)
    else:
        st.info(message)
    st.write(outcome.explanation)
    if not outcome.terminal:
        st.caption("This packet has not reached a bounded terminal outcome.")


def _render_evidence_strip(observations: tuple[ReviewerObservation, ...]) -> None:
    columns = st.columns(min(len(observations), 3))
    for index, observation in enumerate(observations):
        with columns[index % len(columns)]:
            st.markdown(f"**{index + 1}. {observation.observation_id}**")
            st.caption(f"{observation.observed_at} · {observation.channel}")
            st.metric("Candidate pixels", observation.candidate_pixels)
            if observation.maximum_kelvin is not None:
                st.metric("Maximum temperature", f"{observation.maximum_kelvin:.1f} K")
            _render_observation_images(observation)
            st.caption(
                f"{len(observation.components)} component(s) · "
                f"{len(observation.contours)} contour(s) / "
                f"{observation.contour_vertex_count} vertices"
            )
            with st.expander("Regions and contour details"):
                if observation.components:
                    st.dataframe(
                        [component.to_row() for component in observation.components],
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.write("No retained candidate component.")
                if observation.contours:
                    st.dataframe(
                        [
                            {"Contour": number + 1, "Vertices": len(contour)}
                            for number, contour in enumerate(observation.contours)
                        ],
                        width="stretch",
                        hide_index=True,
                    )


def _render_observation_images(observation: ReviewerObservation) -> None:
    if observation.overlay_path is not None and observation.overlay_path.is_file():
        st.image(
            str(observation.overlay_path),
            caption="Thermal candidate overlay",
            width="stretch",
        )
    mask = load_candidate_mask(observation.candidate_mask_path)
    if mask is not None:
        st.image(mask, caption="Candidate mask from calibrated arrays", clamp=True)
    else:
        st.image(
            contour_preview(observation),
            caption="Contour-derived candidate-mask preview",
            clamp=True,
        )


def _render_reason_codes(case: ReviewerCase) -> None:
    st.subheader("Reason codes")
    if not case.reason_codes:
        st.info("No closed reason codes were recorded for this packet.")
        return
    st.write(" · ".join(code.replace("_", " ") for code in case.reason_codes))
    for explanation in reason_explanations(case.reason_codes):
        st.caption(explanation)


def _render_limits_and_provenance(case: ReviewerCase) -> None:
    limits, provenance = st.columns(2)
    with limits:
        st.subheader("Observation and resource budget")
        if case.budget:
            st.dataframe(list(case.budget), width="stretch", hide_index=True)
        else:
            st.info("No bounded-loop budget was recorded for this packet.")
        st.subheader("Errors and recovery")
        if case.errors:
            for error in case.errors:
                st.error(error)
            for recovery_action in case.recovery_actions:
                st.info(f"Recovery action: {recovery_action}")
        else:
            st.info("No bounded-tool error was recorded for this case.")
        st.subheader("Warnings and limitations")
        if case.warnings:
            for warning in case.warnings:
                st.warning(warning)
        else:
            st.info("No packet warnings were recorded.")
    with provenance:
        st.subheader("Provenance")
        if case.provenance:
            st.dataframe(list(case.provenance), width="stretch", hide_index=True)
        else:
            st.info("No provenance entries were recorded.")
        st.caption(
            "The reviewer uses these summarized fields; raw JSON is not shown here."
        )


if __name__ == "__main__":
    main()
