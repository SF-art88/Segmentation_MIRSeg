from __future__ import annotations

from enum import IntEnum

MODALITY_ORDER: tuple[str, str, str, str] = ("T1", "T1ce", "T2", "FLAIR")
NUM_MODALITIES: int = 4

SETTING_PRE: int = 0
SETTING_POST: int = 1
IGNORE_INDEX: int = -1

PRE_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "edema",
    "non_enhancing_or_necrotic_core",
    "enhancing_tumor",
)

POST_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "enhancing_tissue",
    "non_enhancing_tumor_core",
    "surrounding_non_enhancing_flair_hyperintensity",
    "resection_cavity",
)

# Paper stress tiers. Tier A is full input, Tier B removes one sequence,
# Tier C uses plausible two-visible subsets.
MISSINGNESS_PATTERNS: dict[str, dict[str, tuple[bool, bool, bool, bool]]] = {
    "A": {
        "T1+T1ce+T2+FLAIR": (True, True, True, True),
    },
    "B": {
        "T1ce+T2+FLAIR": (False, True, True, True),
        "T1+T2+FLAIR": (True, False, True, True),
        "T1+T1ce+FLAIR": (True, True, False, True),
        "T1+T1ce+T2": (True, True, True, False),
    },
    "C": {
        "T1ce+FLAIR": (False, True, False, True),
        "T2+FLAIR": (False, False, True, True),
        "T1+T1ce": (True, True, False, False),
        "T1+FLAIR": (True, False, False, True),
    },
}

CORRUPTION_TYPES: tuple[str, ...] = (
    "motion",
    "bias",
    "noise",
    "contrast",
    "registration",
    "blur",
    "gibbs",
)


class CorruptionType(IntEnum):
    NONE = 0
    MOTION = 1
    BIAS = 2
    NOISE = 3
    CONTRAST = 4
    REGISTRATION = 5
    BLUR = 6
    GIBBS = 7


CORRUPTION_NAME_TO_ID: dict[str, int] = {
    "none": int(CorruptionType.NONE),
    "motion": int(CorruptionType.MOTION),
    "bias": int(CorruptionType.BIAS),
    "noise": int(CorruptionType.NOISE),
    "contrast": int(CorruptionType.CONTRAST),
    "registration": int(CorruptionType.REGISTRATION),
    "blur": int(CorruptionType.BLUR),
    "gibbs": int(CorruptionType.GIBBS),
}
