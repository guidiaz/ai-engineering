"""Conversation profiles for the multi-turn stress suite.

Each profile is a **list of ``(turn_index, transcript, fact_to_remember)``
tuples** — one tuple per user turn:

- ``turn_index``       1-based position of the turn in the conversation.
- ``transcript``       the user utterance sent to ``POST /sessions/{id}/estimate``
                       (and therefore to ``estimate_conversational``) on that turn.
- ``fact_to_remember`` a short, canonical string naming the claim this turn
                       introduces. Later turns *should* still remember it.

The ``fact_to_remember`` strings are deliberately terse and stable
("project name: Nimbus", "stack includes Flutter", "budget locked: 30000 EUR").
A later ``MemoryDriftMetric`` searches for these strings inside a snapshot of
the session state (metadata / anchors / summary / window) to decide whether a
fact survived, drifted, or was lost. Keep them short and literal — the metric
owns the matching strategy (normalisation, synonyms), not this file.

Profiles run at several lengths — ``TURN_COUNTS`` — by taking the first ``N``
turns (``first_n_turns``). A turn whose ``turn_index`` exceeds ``N`` simply does
not happen in that run, so the designed events only fire when ``N`` is long
enough (pivot at turn 5, the budget bump at turn 8).

Each profile is built to force a specific behaviour:

- ``growing``               coherent requirements pile up turn after turn. Does
                            the original ``project name: Nimbus`` survive to
                            turn 20, and how does cost grow once compression
                            kicks in (window cap is 6 turns)?
- ``pivot``                 the stack switches React Native -> Flutter at turn 5.
                            ``ProjectMetadata.merge_with`` unions technologies,
                            so both should accumulate — this guards that.
- ``contradiction_literal`` budget stated 30k€ (turn 3) then 80k€ (turn 8) in
                            plain Spanish, which never triggers the (English-only)
                            anchor heuristic. Surfaces the language gap: neither
                            budget is promoted to an anchor; they compete in the
                            summary / window.
- ``contradiction_anchored`` same contradiction phrased "budget is locked at …",
                            which DOES fire the ``budget_locked`` anchor rule, so
                            promotion-to-anchor actually happens and all three
                            outcomes (anchor / summary / window) are observable.
"""

from __future__ import annotations

# (turn_index, transcript, fact_to_remember)
ScenarioTurn = tuple[int, str, str]


# Conversation lengths the stress runner sweeps over. A run of length N uses
# the first N turns of a profile (see ``first_n_turns``).
TURN_COUNTS: tuple[int, ...] = (1, 3, 6, 10, 20)


# --------------------------------------------------------------------------- #
# (a) Growing project — coherent requirements accumulate turn after turn.     #
#     Probe later: does "project name: Nimbus" survive to turn 20?            #
# --------------------------------------------------------------------------- #
GROWING: list[ScenarioTurn] = [
    (1, "We are starting a new B2B SaaS analytics dashboard called Nimbus for mid-size logistics companies.", "project name: Nimbus"),
    (2, "Nimbus needs user authentication with email and password sign-in.", "feature: authentication"),
    (3, "It must be multi-tenant so each customer organisation has isolated data.", "feature: multi-tenant"),
    (4, "Add an audit log that records every change made by any user.", "feature: audit log"),
    (5, "Users should be able to export their reports to CSV.", "feature: CSV export"),
    (6, "We need role-based access control with admin, editor and viewer roles.", "feature: role-based access control"),
    (7, "Enterprise customers require SSO via SAML.", "feature: SSO SAML"),
    (8, "Add API rate limiting per tenant to protect the backend.", "feature: rate limiting"),
    (9, "Expose outbound webhooks so customers can subscribe to events.", "feature: webhooks"),
    (10, "The interface must be internationalised, starting with English and Spanish.", "feature: i18n"),
    (11, "Add email and in-app notifications for important events.", "feature: notifications"),
    (12, "We need full-text search across all reports and dashboards.", "feature: full-text search"),
    (13, "Allow users to upload supporting files, stored in S3.", "feature: file uploads"),
    (14, "Integrate Stripe for subscription billing and invoicing.", "feature: Stripe billing"),
    (15, "Build an internal admin dashboard for our support team.", "feature: admin dashboard"),
    (16, "The public API needs versioning so we can evolve it safely.", "feature: API versioning"),
    (17, "Add two-factor authentication for all user accounts.", "feature: two-factor authentication"),
    (18, "We must support GDPR data retention and deletion policies.", "feature: GDPR data retention"),
    (19, "Add a dark mode theme to the whole interface.", "feature: dark mode"),
    (20, "To confirm: the product is still called Nimbus — please give us the full estimate now.", "project name: Nimbus"),
]


# --------------------------------------------------------------------------- #
# (b) Pivot — the stack changes at turn 5. merge_with unions technologies, so  #
#     both React Native and Flutter should end up in mentioned_technologies.  #
# --------------------------------------------------------------------------- #
PIVOT: list[ScenarioTurn] = [
    (1, "We are building a cross-platform mobile app called Aurora using React Native.", "stack includes React Native"),
    (2, "Aurora needs a smooth onboarding flow for first-time users.", "feature: onboarding"),
    (3, "Add push notifications, we plan to use Firebase Cloud Messaging.", "stack includes Firebase"),
    (4, "The app must work offline and sync when connectivity returns.", "feature: offline mode"),
    (5, "Change of plan: we are dropping React Native and rebuilding Aurora in Flutter for better performance.", "stack includes Flutter"),
    (6, "Re-implement the onboarding flow natively in Flutter.", "feature: onboarding rebuilt in Flutter"),
    (7, "Use Riverpod for state management in the Flutter app.", "stack includes Riverpod"),
    (8, "Add in-app purchases for premium features.", "feature: in-app purchases"),
    (9, "Support deep linking into specific screens.", "feature: deep linking"),
    (10, "Integrate product analytics to track user funnels.", "stack includes analytics"),
    (11, "Add interactive maps to show nearby points of interest.", "feature: maps"),
    (12, "Let users scan QR codes with the device camera.", "feature: camera QR"),
    (13, "Localise the app into English, Spanish and French.", "feature: localization"),
    (14, "Add a dark theme that follows the system setting.", "feature: dark theme"),
    (15, "Add biometric authentication with Face ID and fingerprint.", "feature: biometric auth"),
    (16, "Sync data in the background even when the app is closed.", "feature: background sync"),
    (17, "Add crash reporting and performance monitoring.", "stack includes crash reporting"),
    (18, "Set up CI/CD for the Flutter app using Codemagic.", "stack includes Codemagic"),
    (19, "Cover the core widgets with Flutter widget tests.", "feature: widget tests"),
    (20, "Confirm the app is now a Flutter build and give us the estimate.", "stack includes Flutter"),
]


# --------------------------------------------------------------------------- #
# (c) Contradiction — budget stated at turn 3, then changed at turn 8.         #
#     Two variants: plain phrasing (never anchors) vs anchor-triggering        #
#     phrasing ("budget is locked at …"). Same canonical facts in both.        #
# --------------------------------------------------------------------------- #
CONTRADICTION_LITERAL: list[ScenarioTurn] = [
    (1, "We need an internal reporting tool called Helios for our finance department.", "project name: Helios"),
    (2, "Helios should show monthly reporting dashboards for each cost centre.", "feature: reporting dashboard"),
    (3, "El presupuesto del proyecto es de 30.000€.", "budget locked: 30000 EUR"),
    (4, "It must pull data from our Postgres warehouse and a couple of REST APIs.", "feature: data sources"),
    (5, "Reports should be generated on a schedule, every Monday morning.", "feature: scheduled reports"),
    (6, "Users need to export any dashboard to PDF.", "feature: PDF export"),
    (7, "Add user roles so only managers can see the consolidated figures.", "feature: user roles"),
    (8, "Hemos ampliado el presupuesto del proyecto a 80.000€.", "budget locked: 80000 EUR"),
    (9, "Integrate with our existing Slack workspace for report delivery.", "feature: Slack integration"),
    (10, "Add threshold alerts when a cost centre goes over budget.", "feature: alerts"),
    (11, "Provide a drill-down from summary figures to individual transactions.", "feature: drill-down"),
    (12, "Add a comparison view between the current and previous quarter.", "feature: quarter comparison"),
    (13, "Let users annotate reports with comments for their team.", "feature: comments"),
    (14, "Support exporting the raw underlying data to Excel.", "feature: Excel export"),
    (15, "Add an audit trail of who viewed which report.", "feature: view audit trail"),
    (16, "Cache heavy queries so dashboards load quickly.", "feature: query caching"),
    (17, "Add a light and dark theme for the dashboards.", "feature: theming"),
    (18, "Localise the tool into Spanish and English.", "feature: localization"),
    (19, "Add SSO using our corporate Google Workspace.", "feature: Google SSO"),
    (20, "Please give us the final estimate for Helios with the agreed budget.", "budget locked: 80000 EUR"),
]


def _rephrase_budget_as_anchor(turns: list[ScenarioTurn]) -> list[ScenarioTurn]:
    """Build the anchored variant from the literal one by rephrasing only the
    two budget turns (3 and 8) so they trigger the English ``budget_locked``
    anchor heuristic. Every other turn — and every ``fact_to_remember`` — is
    identical, which is exactly what makes the two runs comparable: same
    contradiction, only the anchor-ability of the phrasing changes.
    """
    anchored_phrasing = {
        "budget locked: 30000 EUR": "The project budget is locked at 30000 EUR.",
        "budget locked: 80000 EUR": "The project budget is now locked at 80000 EUR.",
    }
    return [
        (idx, anchored_phrasing.get(fact, transcript), fact)
        for idx, transcript, fact in turns
    ]


CONTRADICTION_ANCHORED: list[ScenarioTurn] = _rephrase_budget_as_anchor(CONTRADICTION_LITERAL)


# --------------------------------------------------------------------------- #
# Registry + helpers                                                           #
# --------------------------------------------------------------------------- #
PROFILES: dict[str, list[ScenarioTurn]] = {
    "growing": GROWING,
    "pivot": PIVOT,
    "contradiction_literal": CONTRADICTION_LITERAL,
    "contradiction_anchored": CONTRADICTION_ANCHORED,
}


def first_n_turns(profile: list[ScenarioTurn], n: int) -> list[ScenarioTurn]:
    """Return the turns of ``profile`` with ``turn_index <= n``.

    Profiles are authored in ascending ``turn_index`` order, so this is the
    leading slice — but we filter on ``turn_index`` explicitly so the contract
    holds even if a profile ever skips an index.
    """
    return [turn for turn in profile if turn[0] <= n]


if __name__ == "__main__":
    # No LLM calls — just a static dump so the scenario data can be eyeballed.
    for name, profile in PROFILES.items():
        print(f"\n=== {name} ({len(profile)} turns) ===")
        for idx, transcript, fact in profile:
            print(f"  [{idx:>2}] fact: {fact!r}")
            print(f"       say : {transcript}")
