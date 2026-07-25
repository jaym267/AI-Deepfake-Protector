"""Result disclaimer text.

*** DRAFT — NOT APPROVED. Pending sign-off before it is treated as final UI copy. ***

Kept in one module, server-side, so a single approved wording ships with every
API response, the results dashboard, and the PDF evidence report. Copy that
lives in the frontend drifts from copy that lives in the report; this cannot.

What this wording is trying to do:
  * state plainly that the result is probabilistic, never proof
  * say that both false positives and false negatives happen, in both directions
  * avoid any phrasing a reader could quote as a definitive finding about a real
    named person — this tool is aimed at material about identifiable people, and
    a confident-sounding false positive is itself a reputational harm
  * avoid implying any legal, evidentiary, or law-enforcement status
"""

RESULT_DISCLAIMER = (
    "This is an automated estimate, not proof. Detection tools can be wrong in "
    "both directions: authentic media is sometimes flagged as manipulated, and "
    "skilled fakes sometimes pass as authentic. Treat this result as one piece "
    "of evidence to weigh alongside context and other sources, not as a final "
    "answer about any person or event."
)

#: Longer form for the PDF evidence report (step 7), where the reader may be a
#: platform moderator or an employer rather than the uploader.
REPORT_DISCLAIMER = (
    "This report was produced by an automated analysis tool. It describes "
    "statistical indicators found in the submitted file. It does not establish "
    "that any file is genuine or fabricated, does not identify who created or "
    "altered a file, and is not a forensic examination, legal determination, or "
    "expert testimony. Automated detection produces both false positives and "
    "false negatives. Any decision affecting a person should not rest on this "
    "report alone."
)
