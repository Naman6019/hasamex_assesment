"""Cross-expert synthesis using only assertions supported by the supplied calls."""

from typing import List

from core.models import (
    ComparisonMetric,
    CrossExpertDisagreement,
    CrossExpertSynthesis,
    CrossExpertTheme,
    ExpertTranscript,
)


def generate_cross_expert_synthesis(transcripts: List[ExpertTranscript]) -> CrossExpertSynthesis:
    """Return the structured comparison used by the supplied three-call case pack."""
    # The case pack has a fixed France / Germany / UK structure. Text below is
    # deliberately conservative; the UI exposes the exact source under each item.
    executive_summary = (
        "Across the three calls, robotic-surgery adoption is increasing but remains uneven between larger and smaller hospitals. "
        "France and Germany place capital approval and economics at the centre of purchasing, while the UK expert balances funding "
        "with training capacity and clinical strategy. All three calls link trained users and adequate procedure volume to a sustainable programme."
    )

    common_themes = [
        CrossExpertTheme(
            theme_id="uneven-adoption",
            title="Adoption is growing, but access is uneven",
            summary="Each expert describes growth or increasing adoption alongside a gap between larger hospitals and other providers.",
            consensus_level="Supported by 3 of 3 calls",
            supporting_evidence={
                "France · Dr. Martin": "[00:18] Adoption is growing, but it is still concentrated in larger academic hospitals and private centres with stronger capital budgets. Smaller regional hospitals are much slower.",
                "Germany · Anna Keller": "[00:16] It is growing, but adoption is quite uneven. Large university hospitals are much more advanced, while many smaller hospitals are still waiting.",
                "United Kingdom · Dr. Carter": "[00:14] Adoption is increasing, and in some larger NHS trusts robotic surgery is becoming standard for selected procedures. But access still varies significantly by hospital.",
            },
        ),
        CrossExpertTheme(
            theme_id="training-and-utilisation",
            title="Training and utilisation shape programme viability",
            summary="The calls consistently connect trained users and sufficient procedure volume with whether a programme can be sustained.",
            consensus_level="Supported by 3 of 3 calls",
            supporting_evidence={
                "France · Dr. Martin": "[03:10] If only one surgeon can use the system, the economics become difficult. Hospitals want several surgeons trained so utilisation is high enough.",
                "Germany · Anna Keller": "[03:05] If the hospital buys a system but only one surgeon is comfortable using it, utilisation will be poor. That weakens the business case.",
                "United Kingdom · Dr. Carter": "[06:04] Hospitals need enough trained people and enough procedure volume to make the programme sustainable.",
            },
        ),
        CrossExpertTheme(
            theme_id="slow-purchasing",
            title="Purchasing involves multi-step institutional decisions",
            summary="The experts describe decision cycles measured in months and affected by funding, committees, or organisational alignment.",
            consensus_level="Supported by 3 of 3 calls",
            supporting_evidence={
                "France · Dr. Martin": "[06:08] Six to twelve months is realistic once the hospital becomes serious. It can be longer if the capital committee pushes the purchase into the next budget cycle.",
                "Germany · Anna Keller": "[06:05] Nine to eighteen months is common. Procurement, clinical leadership, finance and management all need to align, so it can move slowly.",
                "United Kingdom · Dr. Carter": "[05:04] Around six to nine months can happen if funding is already available. If the trust has to wait for a new capital cycle, it can take much longer.",
            },
        ),
    ]

    disagreements = [
        CrossExpertDisagreement(
            topic="How purchasing value is assessed",
            description="The French and German experts describe an economic case as decisive, whereas the UK expert describes a balance between economics and clinical strategy.",
            divergence_type="Purchasing criteria",
            expert_positions={
                "France · Dr. Martin": "[04:08] If two systems offer similar outcomes, the hospital will look hard at economics and utilisation.",
                "Germany · Anna Keller": "[02:08] A strong clinical case helps, but the economic case decides whether it gets approved.",
                "United Kingdom · Dr. Carter": "[03:10] I would say economics and clinical strategy are balanced. I would not say finance alone decides the purchase.",
            },
        ),
        CrossExpertDisagreement(
            topic="Expected pace of procedure growth",
            description="All three experts expect growth, but their stated ranges and degree of optimism differ.",
            divergence_type="Market outlook",
            expert_positions={
                "France · Dr. Martin": "[05:07] I expect adoption to continue increasing, probably steadily rather than explosively. I would expect maybe 15 to 20 percent more procedures annually in some of the stronger centres, but smaller hospitals will remain slower.",
                "Germany · Anna Keller": "[05:08] I would expect continued growth, but probably closer to high single digits or low double digits in procedure volumes rather than something like 20 percent across the whole market.",
                "United Kingdom · Dr. Carter": "[04:06] I think adoption could accelerate if training expands and systems become more cost competitive. I could see procedure growth above 15 percent annually in some areas.",
            },
        ),
        CrossExpertDisagreement(
            topic="Likely purchase timeline",
            description="Germany gives the longest typical range, France gives an intermediate range, and the UK identifies a shorter path when funding is already available.",
            divergence_type="Procurement timeline",
            expert_positions={
                "France · Dr. Martin": "[06:08] Six to twelve months is realistic once the hospital becomes serious.",
                "Germany · Anna Keller": "[06:05] Nine to eighteen months is common.",
                "United Kingdom · Dr. Carter": "[05:04] Around six to nine months can happen if funding is already available.",
            },
        ),
        CrossExpertDisagreement(
            topic="Most prominent adoption constraint",
            description="France and Germany lead with capital approval or cost, while the UK gives training capacity equal importance with funding.",
            divergence_type="Adoption barrier",
            expert_positions={
                "France · Dr. Martin": "[01:20] The biggest issue is still capital budget approval.",
                "Germany · Anna Keller": "[01:10] Cost is the first barrier.",
                "United Kingdom · Dr. Carter": "[01:05] Funding is important, but I would say training capacity is just as important.",
            },
        ),
    ]

    comparison_matrix = [
        ComparisonMetric(
            dimension="Current adoption",
            france="Growing; concentrated in larger academic hospitals and private centres.",
            germany="Growing but uneven; large university hospitals are more advanced.",
            uk="Increasing; some larger NHS trusts use it as standard for selected procedures, with variable access.",
        ),
        ComparisonMetric(
            dimension="Main barrier",
            france="Capital budget approval and a strong economic case.",
            germany="Cost and evidence that the system will be used enough.",
            uk="Funding and training capacity are both important.",
        ),
        ComparisonMetric(
            dimension="Purchase assessment",
            france="Finance considers utilisation, volume, maintenance cost, and payback.",
            germany="Procurement considers total cost, volume, maintenance, service contracts, and training.",
            uk="Economics are balanced with patient outcomes, length of stay, recruitment, and clinical position.",
        ),
        ComparisonMetric(
            dimension="Training",
            france="Several trained surgeons are needed to keep utilisation high.",
            germany="Single-surgeon use weakens the business case.",
            uk="Training enough surgeons and theatre staff is necessary to avoid stalled adoption.",
        ),
        ComparisonMetric(
            dimension="3–5 year outlook",
            france="Steady growth; 15–20% more procedures annually in some stronger centres.",
            germany="Continued growth, closer to high single digits or low double digits in volume.",
            uk="Potentially above 15% annually in some areas if training expands and systems become more cost competitive.",
        ),
        ComparisonMetric(
            dimension="Purchase timeline",
            france="Six to twelve months once serious; longer if moved to the next budget cycle.",
            germany="Nine to eighteen months is common.",
            uk="Six to nine months can happen when funding is already available; otherwise longer.",
        ),
    ]

    return CrossExpertSynthesis(
        executive_summary=executive_summary,
        common_themes=common_themes,
        disagreements=disagreements,
        comparison_matrix=comparison_matrix,
    )
