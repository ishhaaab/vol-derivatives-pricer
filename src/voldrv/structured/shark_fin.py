"""Shark fin structured note pricing.

DECOMPOSITION (planned, not yet implemented)
--------------------------------------------
A shark fin note pays a participating coupon if the underlying stays
within a corridor [L, H] AND its terminal value does not hit a knock-out
barrier B (typically B = H or a level just above H).  If the barrier is
hit, the note pays a fixed digital refund.

Pricing decomposes into:

  1. Knock-out barrier option component — priced via closed-form
     under Black-Scholes or semi-analytic via Curran / Joshi
     conditioning-on-no-hit formulas.
  2. Digital refund component — pays 1 if barrier IS hit, else 0.
     Priced via Monte Carlo or the reflection principle for the
     hitting probability.
  3. Combine:

        PV = participating_coupon_payer × P(no knock-out)
             + refund_digital × P(knock-out)

Cross-check: Monte Carlo under local vol should match analytic to
within standard error when the surface is flat (BS world).
"""


def price_shark_fin(*args, **kwargs):
    """Shark fin note pricer — STRETCH GOAL, not yet implemented.

    Raises
    ------
    NotImplementedError
        Always — this is a placeholder for future implementation.
    """
    raise NotImplementedError(
        "Shark fin pricing is a planned stretch goal; not yet implemented."
    )
