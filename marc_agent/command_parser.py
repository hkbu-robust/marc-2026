"""Parse MARC mission text into a bounded, validated command query.

The module is deliberately pure: it performs no SDK calls, model inference, or I/O.
Callers and tests use :func:`parse_command`; normalization and phrase rules stay private.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


class Intent(str, Enum):
    FIND = "find"
    BRING = "bring"


class Relation(str, Enum):
    NEAR = "near"
    BESIDE = "beside"
    BEHIND = "behind"
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    ON = "on"
    ABOVE = "above"


class Situation(str, Enum):
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    EMERGENCY = "emergency"
    ACCIDENT = "accident"


class ParseErrorCode(str, Enum):
    EMPTY = "empty"
    UNKNOWN_TARGET = "unknown_target"
    AMBIGUOUS_TARGET = "ambiguous_target"
    AMBIGUOUS_LANDMARK = "ambiguous_landmark"
    UNKNOWN_LANDMARK = "unknown_landmark"
    UNKNOWN_INTENT = "unknown_intent"
    CONFLICTING_INTENT = "conflicting_intent"
    UNKNOWN_SITUATION = "unknown_situation"
    CONFLICTING_RELATION = "conflicting_relation"
    CONFLICTING_SITUATION = "conflicting_situation"
    INTENT_MISMATCH = "intent_mismatch"


class CommandParseError(ValueError):
    """A visible parse failure with a stable, machine-checkable reason."""

    def __init__(self, code: ParseErrorCode, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ParsedCommand:
    intent: Intent
    target_type: str
    landmark: Optional[str] = None
    relation: Optional[Relation] = None
    situation: Optional[Situation] = None
    scene_hint: Optional[str] = None


MISSION_OBJECT_TYPES = frozenset(
    {
        "master_chef_can",
        "cracker_box",
        "mug",
        "wood_block",
        "cola_can",
        "disposable_cup",
        "juice",
        "tissue",
        "pencilcase",
        "tumbler",
        "umbrella",
        "long_umbrella",
        "sunblock",
    }
)

LANDMARK_TYPES = frozenset(
    {
        "bench",
        "bicycle",
        "hydrant",
        "kick_scooter",
        "picnic_table",
        "postbox",
        "vending_machine",
        "trash_can",
        "police_car",
        "normal_car_1",
        "normal_car_2",
        "sports_car",
        "sports_car_2",
        "suv",
        "taxi",
    }
)


_OBJECT_ALIASES = {
    "master_chef_can": ("master chef can", "master chef cans", "chef can", "chef cans"),
    "cracker_box": ("cracker box", "cracker boxes"),
    "mug": ("mug", "mugs", "coffee mug", "coffee mugs"),
    "wood_block": ("wood block", "wood blocks", "wooden block", "wooden blocks"),
    "cola_can": ("cola can", "cola cans", "soda can", "soda cans", "coke can", "coke cans"),
    "disposable_cup": ("disposable cup", "disposable cups", "paper cup", "paper cups"),
    "juice": ("juice", "juices", "juice box", "juice boxes"),
    "tissue": ("tissue", "tissues", "tissue pack"),
    "pencilcase": ("pencilcase", "pencilcases", "pencil case", "pencil cases"),
    "tumbler": ("tumbler", "tumblers", "travel tumbler", "travel tumblers"),
    "long_umbrella": ("long umbrella", "long umbrellas"),
    "umbrella": ("umbrella", "umbrellas"),
    "sunblock": ("sunblock", "sunblocks", "sun block", "sun blocks", "sunscreen", "sunscreens"),
}

_LANDMARK_ALIASES = {
    "bench": ("bench", "benches"),
    "bicycle": ("bicycle", "bicycles", "bike", "bikes", "fallen bike", "fallen bikes", "toppled bicycle", "toppled bicycles"),
    "hydrant": ("hydrant", "hydrants", "fire hydrant", "fire hydrants"),
    "kick_scooter": ("kick scooter", "kick scooters", "scooter", "scooters"),
    "picnic_table": ("picnic table", "picnic tables", "outdoor table", "outdoor tables"),
    "postbox": ("postbox", "postboxes", "post box", "post boxes", "mailbox", "mailboxes"),
    "vending_machine": ("vending machine", "vending machines", "drink machine", "drink machines", "machine", "machines"),
    "trash_can": ("trash can", "trash cans", "bin", "bins", "garbage can", "garbage cans"),
    "police_car": ("police car", "police cars"),
    "normal_car_1": ("normal car one", "normal car 1"),
    "normal_car_2": ("normal car two", "normal car 2"),
    "sports_car": ("sports car", "sports cars"),
    "sports_car_2": ("sports car two", "sports car 2"),
    "suv": ("suv", "suvs"),
    "taxi": ("taxi", "taxis", "cab", "cabs"),
}

_SCENE_HINTS = (
    ("empty parking space", "parking_space"),
    ("parking space", "parking_space"),
    ("parking lot", "parking"),
    ("park rest area", "park"),
    ("on the road", "road"),
    ("road", "road"),
    ("grass", "grass"),
    ("park", "park"),
)

_PERSON_WORD_RE = re.compile(r"\b(person|people|someone|somebody)\b")
_BRING_RE = re.compile(r"\b(bring|retrieve|deliver|pick\s+up|collect)\b")
_FIND_RE = re.compile(r"\b(find|locate|look\s+for|pinpoint)\b")
_ACTION_RE = re.compile(
    r"\b(?:find|locate|look\s+for|pinpoint|bring|retrieve|deliver|pick\s+up|collect)\b"
)


@dataclass(frozen=True)
class _Match:
    value: str
    start: int
    end: int


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\bdon(?:['’]?t)\b", "do not", text)
    text = text.replace("–", ";").replace("—", ";")
    text = text.replace("‐", "-").replace("‑", "-")
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^\w\s;,.']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\batop\b", "on top of", text)
    return text.strip(" ,.;")


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    words = [re.escape(word) for word in phrase.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)")


def _matches(text: str, aliases: dict[str, Iterable[str]]) -> list[_Match]:
    found: list[_Match] = []
    for value, spoken in aliases.items():
        for alias in spoken:
            for match in _phrase_pattern(alias).finditer(text):
                found.append(_Match(value, match.start(), match.end()))

    # A longer canonical phrase owns an overlapping span: "long umbrella" must not
    # also become "umbrella", and "sports car 2" must not also become "sports car".
    kept: list[_Match] = []
    for match in sorted(found, key=lambda item: (-(item.end - item.start), item.start)):
        if any(match.start >= other.start and match.end <= other.end for other in kept):
            continue
        kept.append(match)
    return sorted(kept, key=lambda item: item.start)


def _target_scope(text: str) -> str:
    """Drop an explicit correction/distractor clause before parsing the target."""
    parts = [part.strip() for part in re.split(r"[;.]", text) if part.strip()]
    kept = [part for part in parts if not re.match(r"^(?:ignore\b|not\b|do\s+not\b)", part)]
    scope = " ; ".join(kept) if kept else text
    scope = re.sub(
        r"\b(find|locate|look\s+for|pinpoint)\s+not\s+[^,;]+,\s*but\s+",
        r"\1 ",
        scope,
    )
    scope = re.sub(r",\s*(?:but\s+)?not\s+(?:the\s+)?[^,;]+,", ", ", scope)
    scope = re.sub(r",\s*(?:but\s+)?not\s+(?:the\s+)?[^;]+$", "", scope)
    return scope.strip()


def _resolve_intent(text: str, expected: Optional[Intent]) -> Intent:
    has_bring = bool(_BRING_RE.search(text))
    has_find = bool(_FIND_RE.search(text))
    if has_bring and has_find:
        raise CommandParseError(
            ParseErrorCode.CONFLICTING_INTENT,
            "command contains both find and bring intents",
        )
    explicit = Intent.BRING if has_bring else (Intent.FIND if has_find else None)
    if expected is not None and explicit is not None and explicit != expected:
        raise CommandParseError(
            ParseErrorCode.INTENT_MISMATCH,
            f"command intent {explicit.value!r} conflicts with expected {expected.value!r}",
        )
    if explicit is not None:
        return explicit
    if expected is not None:
        return expected
    raise CommandParseError(ParseErrorCode.UNKNOWN_INTENT, "command contains no find or bring intent")


def _direct_target(text: str, object_matches: list[_Match]) -> Optional[tuple[str, bool]]:
    actions = list(_ACTION_RE.finditer(text))
    if not actions:
        return None
    action = actions[-1]
    candidates: list[tuple[int, int, str, bool]] = [
        (match.start, match.end, match.value, False)
        for match in object_matches
        if match.start >= action.end()
    ]
    for match in re.finditer(r"\b(?:person|someone|somebody|one|them|they)\b", text[action.end() :]):
        value = match.group(0)
        if value in {"one", "them", "they"} and not _PERSON_WORD_RE.search(text[: action.start()]):
            continue
        candidates.append((action.end() + match.start(), action.end() + match.end(), "person", True))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    first = candidates[0]
    if len(candidates) > 1:
        candidate = candidates[1]
        between = text[first[1] : candidate[0]]
        if re.search(r"\b(?:or|and|then)\b", between) or re.fullmatch(
            r"\s*,\s*(?:the\s+)?",
            between,
        ):
            raise CommandParseError(ParseErrorCode.AMBIGUOUS_TARGET, "coordinated targets are ambiguous")
    return first[2], first[3]


def _resolve_target(text: str) -> tuple[str, bool]:
    objects = _matches(text, _OBJECT_ALIASES)
    values = {match.value for match in objects}
    direct = _direct_target(text, objects)
    if direct is not None:
        return direct
    actions = list(_ACTION_RE.finditer(text))
    if actions and re.search(
        r"^[^;,.]{0,30}\b(?:it|this|that)\b",
        text[actions[-1].end() :],
    ) and not values:
        raise CommandParseError(ParseErrorCode.UNKNOWN_TARGET, "object pronoun has no supported antecedent")
    if len(values) > 1:
        raise CommandParseError(
            ParseErrorCode.AMBIGUOUS_TARGET,
            f"multiple object targets are plausible: {sorted(values)}",
        )
    if values:
        return next(iter(values)), False
    incidental_person = re.search(
        r"\b(?:person|someone|somebody)\b[^;,.]{0,80}"
        r"\b(?:left|lost|forgot|set|placed)\b[^;,.]+",
        text,
    )
    if _PERSON_WORD_RE.search(text) and not incidental_person:
        return "person", True
    raise CommandParseError(ParseErrorCode.UNKNOWN_TARGET, "no supported mission target was found")


def _relation_candidates(text: str, landmarks: list[_Match], landmark: Optional[str]) -> set[Relation]:
    relations: set[Relation] = set()
    phrases: tuple[tuple[Relation, str], ...] = (
        (Relation.FRONT, r"\b(?:in\s+)?front\s+of\b"),
        (Relation.LEFT, r"\b(?:on|to)\s+the\s+left\s+(?:side\s+)?of\b"),
        (Relation.RIGHT, r"\b(?:on|to)\s+the\s+right\s+(?:side\s+)?of\b"),
        (Relation.ABOVE, r"\babove\b"),
        (Relation.BESIDE, r"\b(?:beside|alongside)\b"),
        (Relation.NEAR, r"\b(?:near|close\s+to|around)\b"),
        (Relation.ON, r"\bon\s+top\s+of\b"),
    )
    selected = [match for match in landmarks if match.value == landmark]
    for match in selected:
        prior_ends = [other.end for other in landmarks if other.end <= match.start and other is not match]
        prefix_start = max([max(0, match.start - 45), *prior_ends])
        prefix = text[prefix_start : match.start]
        for relation, pattern in phrases:
            if re.search(pattern, prefix):
                relations.add(relation)
        if re.search(r"\bon(?:\s+the)?\s*$", prefix):
            relations.add(Relation.ON)
        if re.search(r"\bbehind(?:\s+the)?\s*$", prefix):
            relations.add(Relation.BEHIND)
        if re.search(r"\bnext\s+to\s+(?:the\s+)?$", prefix):
            if landmark in {"trash_can", "taxi", "bicycle"}:
                relations.add(Relation.NEAR)
            else:
                relations.add(Relation.BESIDE)
        if re.search(r"\b(?:propped|leaning)\s+against\s+(?:the\s+)?$", prefix):
            relations.add(Relation.NEAR if landmark == "trash_can" else Relation.BESIDE)

    if not landmarks:
        for relation, pattern in phrases:
            if re.search(pattern, text):
                relations.add(relation)
        if re.search(r"\bon\s+top\s+of\s+(?:it|this|that)\b", text):
            relations.add(Relation.ON)

    if landmark is not None and len({match.value for match in landmarks}) == 1:
        pronouns = (
            (Relation.FRONT, r"\b(?:in\s+)?front\s+of\s+it\b"),
            (Relation.LEFT, r"\b(?:on|to)\s+the\s+left\s+(?:side\s+)?of\s+it\b"),
            (Relation.RIGHT, r"\b(?:on|to)\s+the\s+right\s+(?:side\s+)?of\s+it\b"),
            (Relation.ABOVE, r"\babove\s+it\b"),
            (Relation.BESIDE, r"\b(?:beside|alongside|next\s+to)\s+it\b|\bagainst\s+its\s+side\b"),
            (Relation.NEAR, r"\bnear\s+it\b"),
            (Relation.ON, r"\bon(?:\s+top\s+of)?\s+it\b"),
            (Relation.BEHIND, r"\bbehind\s+it\b"),
        )
        for relation, pattern in pronouns:
            if re.search(pattern, text):
                relations.add(relation)

    return relations


def _choose_landmark(
    text: str,
    landmarks: list[_Match],
    is_sar: bool,
    target_type: str,
) -> Optional[str]:
    values = {match.value for match in landmarks}
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))

    # SAR support is the physical noun attached to the pose; a second place landmark
    # ("bench near the taxi") is context, not the person's support.
    if is_sar:
        for match in landmarks:
            prefix = text[max(0, match.start - 24) : match.start]
            if re.search(r"\b(?:on|on\s+top\s+of|next\s+to)\s+(?:the\s+)?$", prefix):
                return match.value

    # For object queries, prefer the related landmark after the action/target. A
    # leading location such as "Near the taxi, find the mug behind the bench" is context.
    action_ends = [match.end() for match in _ACTION_RE.finditer(text)]
    related: list[_Match] = []
    for match in landmarks:
        prefix = text[max(0, match.start - 30) : match.start]
        if re.search(
            r"\b(?:near|around|beside|behind|above|alongside|close\s+to|front\s+of|(?:left|right)\s+(?:side\s+)?of|on(?:\s+top\s+of)?|next\s+to|(?:propped|leaning)\s+against)\s+(?:the\s+)?$",
            prefix,
        ):
            related.append(match)
    if action_ends:
        after_action = [match for match in related if match.start >= action_ends[-1]]
        if after_action:
            first = after_action[0]
            first_prefix = text[max(action_ends[-1], first.start - 30) : first.start]
            nested_support_context = bool(
                re.search(r"\bon(?:\s+top\s+of)?\s+(?:the\s+)?$", first_prefix)
                and len(after_action) == 2
                and re.search(
                    r"\bnear\s+(?:the\s+)?$",
                    text[first.end : after_action[1].start],
                )
            )
            if len(after_action) > 1 and not nested_support_context:
                raise CommandParseError(
                    ParseErrorCode.CONFLICTING_RELATION,
                    "multiple landmark-relation constraints are explicit",
                )
            return after_action[0].value
    target_mentions = [
        match for match in _matches(text, _OBJECT_ALIASES) if match.value == target_type
    ]
    locally_attached = [
        landmark
        for landmark in related
        if any(
            target.end <= landmark.start and ";" not in text[target.end : landmark.start]
            for target in target_mentions
        )
    ]
    if len({match.value for match in locally_attached}) == 1:
        return locally_attached[0].value
    if len({match.value for match in related}) == 1:
        return related[0].value
    raise CommandParseError(
        ParseErrorCode.AMBIGUOUS_LANDMARK,
        f"multiple landmarks are plausible: {sorted(values)}",
    )


def _resolve_situation(text: str) -> Situation:
    explicit: set[Situation] = set()
    face_down = bool(re.search(r"\bface\s+down\b", text))
    incapacitated = bool(
        re.search(r"\b(?:immobile|motionless|unresponsive|completely\s+still)\b", text)
    )
    if re.search(r"\baccident\b|\bmost\s+critical\b", text) or (face_down and incapacitated):
        explicit.add(Situation.ACCIDENT)
    if re.search(r"\b(?:collapsed|needs?\s+help|emergency|fell|fallen)\b|\bbent\s+over\b.*\b(?:not\s+moving|motionless)\b", text):
        explicit.add(Situation.EMERGENCY)
    if re.search(r"\b(?:climbed|should\s+not\s+be)\b|\b(?:sitting|lying|lie)\b.*\bon\s+top\s+of\b", text):
        explicit.add(Situation.ABNORMAL)
    if re.search(r"\b(?:walking|standing)\s+normally\b|\b(?:walking|relaxing|fine)\b|\bnormal\b(?!\s+car\b)", text) and not re.search(
        r"\b(?:needs?\s+help|not\s+moving|motionless|immobile|unresponsive|completely\s+still|critical)\b", text
    ):
        explicit.add(Situation.NORMAL)
    if len(explicit) > 1:
        raise CommandParseError(
            ParseErrorCode.CONFLICTING_SITUATION,
            f"multiple explicit SAR situations are plausible: {sorted(item.value for item in explicit)}",
        )
    if explicit:
        return next(iter(explicit))

    # Public pose/support rules: prostrate on transit/non-mount is emergency; on
    # seating is abnormal; prone incapacitation is an accident.
    cues: set[Situation] = set()
    if re.search(r"\b(?:lying|collapsed|bent\s+over|fell|fallen)\b", text):
        if re.search(r"\b(?:parking|road|walkway|sidewalk|trash\s+can|vending\s+machine|hydrant)\b", text):
            cues.add(Situation.EMERGENCY)
        elif re.search(r"\b(?:bench|picnic\s+table)\b", text):
            cues.add(Situation.ABNORMAL)
    if "face down" in text:
        if re.search(r"\b(?:immobile|motionless|critical|accident)\b", text):
            cues = {Situation.ACCIDENT}

    if len(cues) > 1:
        raise CommandParseError(
            ParseErrorCode.CONFLICTING_SITUATION,
            f"multiple SAR situations are plausible: {sorted(item.value for item in cues)}",
        )
    if cues:
        return next(iter(cues))
    raise CommandParseError(ParseErrorCode.UNKNOWN_SITUATION, "person target has no supported SAR situation")


def _scene_hint(text: str) -> Optional[str]:
    for phrase, value in _SCENE_HINTS:
        if _phrase_pattern(phrase).search(text):
            return value
    return None


def parse_command(
    text: str,
    *,
    expected_intent: Optional[Intent] = None,
) -> ParsedCommand:
    """Return the bounded interpretation of one MARC mission command.

    ``expected_intent`` is trusted callback context. It supplies intent for narrative
    commands that omit an action and rejects text that contradicts its stage.
    """

    normalized = _normalize(text or "")
    if not normalized:
        raise CommandParseError(ParseErrorCode.EMPTY, "command is empty")

    scope = _target_scope(normalized)
    intent = _resolve_intent(scope, expected_intent)
    target_type, is_sar = _resolve_target(scope)
    landmarks = _matches(scope, _LANDMARK_ALIASES)
    landmark = _choose_landmark(scope, landmarks, is_sar, target_type)
    hint = _scene_hint(scope)

    if is_sar:
        return ParsedCommand(
            intent=intent,
            target_type="person",
            landmark=landmark,
            situation=_resolve_situation(scope),
            scene_hint=hint,
        )

    relations = _relation_candidates(scope, landmarks, landmark)
    if relations and landmark is None:
        raise CommandParseError(
            ParseErrorCode.UNKNOWN_LANDMARK,
            "a spatial relation requires a supported landmark",
        )
    if len(relations) > 1:
        raise CommandParseError(
            ParseErrorCode.CONFLICTING_RELATION,
            f"multiple relations are plausible: {sorted(item.value for item in relations)}",
        )
    relation = next(iter(relations)) if relations else None
    return ParsedCommand(
        intent=intent,
        target_type=target_type,
        landmark=landmark,
        relation=relation,
        scene_hint=hint,
    )


__all__ = [
    "CommandParseError",
    "Intent",
    "LANDMARK_TYPES",
    "MISSION_OBJECT_TYPES",
    "ParseErrorCode",
    "ParsedCommand",
    "Relation",
    "Situation",
    "parse_command",
]
