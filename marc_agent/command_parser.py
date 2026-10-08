"""Parse MARC mission text into a bounded, validated command query.

The module is deliberately pure: it performs no SDK calls, model inference, or I/O.
Callers and tests use :func:`parse_command`; normalization and phrase rules stay private.
"""

from __future__ import annotations

import re
from pathlib import Path
import unicodedata
from dataclasses import dataclass, replace
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
    place_cues: tuple[str, ...] = ()


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

MISSION_OBJECT_TYPES = frozenset(_OBJECT_ALIASES)
_VOCABULARY: frozenset[str] = frozenset()  # filled below, once the alias tables exist
LANDMARK_TYPES = frozenset(_LANDMARK_ALIASES)
SEATING_LANDMARKS = frozenset({"picnic_table", "bench"})
VEHICLE_LANDMARKS = frozenset(
    {
        "police_car",
        "taxi",
        "normal_car_1",
        "normal_car_2",
        "sports_car",
        "sports_car_2",
        "suv",
    }
)
# Any vehicle class may be parked: where a class stands is the scene's business,
# not a fixed rule (chungmu's taxi happens to be on the road; #113).
PARKED_VEHICLE_LANDMARKS = VEHICLE_LANDMARKS
PARKING_SCENES = frozenset({"parking", "parking_space"})

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

_PERSON_WORD_PATTERN = (
    r"person|people|someone|somebody|man|men|woman|women|guy|lady|kid|child|boy|girl|"
    r"victim|pedestrian|student|cyclist|individual"
)
_PERSON_PRONOUN_PATTERN = r"them|they|him|her|he|she"
_PERSON_REFERENCE_PATTERN = rf"{_PERSON_WORD_PATTERN}|one|{_PERSON_PRONOUN_PATTERN}"
_PERSON_MENTION_PATTERN = rf"one\s+person|{_PERSON_REFERENCE_PATTERN}"
_ANAPHORIC_PERSON_TARGET_PATTERN = (
    rf"(?:that|this|(?:the\s+)?same)\s+person|"
    rf"the(?:\s+\w+){{0,4}}\s+(?:person|one)|{_PERSON_PRONOUN_PATTERN}|one"
)
_PERSON_WORD_RE = re.compile(rf"\b(?:{_PERSON_WORD_PATTERN})\b")
_PERSON_REFERENCE_RE = re.compile(rf"\b(?:{_PERSON_REFERENCE_PATTERN})\b")
_PERSON_MENTION_RE = re.compile(rf"\b(?:{_PERSON_MENTION_PATTERN})\b")
_PERSON_ANAPHOR_RE = re.compile(rf"\b(?:{_ANAPHORIC_PERSON_TARGET_PATTERN})\b")
_FIND_PATTERN = r"find|locate|look\s+for|pinpoint"
_BRING_PATTERN = r"bring|retrieve|deliver|pick\s+up|collect"
_PLACEMENT_PATTERN = r"left|lost|forgot|set|placed|dropped"
_HELP_CUE_PATTERN = (
    r"need(?:s|ed|ing)?(?:\s+urgent)?\s+help|"
    r"(?:is\s+)?in\s+need\s+of\s+help|requires?\s+help"
)
_SCENE_LOCATION_PATTERN = r"at|in|on|near|around"
_LANDMARK_LOCATION_PATTERN = rf"{_SCENE_LOCATION_PATTERN}|by"
_NEXT_TO_PATTERN = r"next\s+to"
_VEHICLE_CUE_RE = re.compile(
    r"\b(?:(?:got|getting)\s+out\s+of|out\s+of)\s+(?:my|the|an?)\s+car\b|"
    r"\b(?:my|the)\s+car\b"
)
_OFF_SUPPORT_RE = re.compile(
    r"\b(?:(?:fallen|fell|falling)\s+)?off(?:\s+of)?\s+(?:(?:an?|the)\s+)?$"
)
_FIND_RE = re.compile(rf"\b(?:{_FIND_PATTERN})\b")
_BRING_RE = re.compile(rf"\b(?:{_BRING_PATTERN})\b")
_ACTION_RE = re.compile(rf"\b(?:{_FIND_PATTERN}|{_BRING_PATTERN})\b")

_RELATION_PHRASES: tuple[tuple[Relation, str], ...] = (
    (Relation.FRONT, r"(?:in\s+)?front\s+of"),
    (Relation.LEFT, r"(?:on|to)\s+the\s+left\s+(?:side\s+)?of"),
    (Relation.RIGHT, r"(?:on|to)\s+the\s+right\s+(?:side\s+)?of"),
    (Relation.ABOVE, r"above"),
    (Relation.BESIDE, r"(?:beside|alongside)"),
    (Relation.NEAR, r"(?:near|close\s+to|around)"),
    (Relation.ON, r"on(?:\s+top\s+of)?"),
    (Relation.BEHIND, r"behind"),
)


@dataclass(frozen=True)
class _Match:
    value: str
    start: int
    end: int


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\bdon(?:['’]?t)\b", "do not", text)
    text = text.replace("’", "'")
    text = text.replace("–", ";").replace("—", ";")
    text = text.replace("‐", "-").replace("‑", "-")
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^\w\s;,.']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\batop\b", "on top of", text)
    text = _correct_typos(text)
    for pattern, canonical in _SYNONYMS:
        text = pattern.sub(canonical, text)
    return text.strip(" ,.;")


# Everyday words for catalogue classes and places, mapped to the phrases the alias
# tables know (#114). Generic English, not taken from the public commands.
_SYNONYMS = tuple(
    (re.compile(pattern), canonical)
    for pattern, canonical in (
        (r"\b(?:thermos(?:es)?|flasks?|water\s+bottles?|drink\s+bottles?)\b", "tumbler"),
        (r"\b(?:trash\s*cans?|trashcans?|rubbish\s+bins?|dust\s*bins?|garbage\s+bins?|"
         r"waste\s+bins?|litter\s+bins?)\b", "trash can"),
        (r"\b(?:car\s+park|parking\s+garage)\b", "parking lot"),
        (r"\bparking\s+(?:spot|bay|stall|place)s?\b", "parking space"),
        (r"\b(?:cop|patrol)\s+cars?\b", "police car"),
        (r"\be\s+scooters?\b", "kick scooter"),
        (r"\b(?:soda|coke|cola)\s+(?:tin|can)s?\b", "cola can"),
        (r"\bsun\s*screen\s+lotion\b", "sunscreen"),
        (r"(?<!picnic )(?<!outdoor )\btables?\b", "picnic table"),
    )
)


def _vocabulary() -> frozenset[str]:
    words = set()
    for aliases in (*_OBJECT_ALIASES.values(), *_LANDMARK_ALIASES.values()):
        for alias in aliases:
            words.update(alias.split())
    for pattern, _ in _SYNONYMS:  # synonym words are catalogue words too
        words.update(re.findall(r"[a-z]{5,}", pattern.pattern))
    words.update(("person", "people", "someone", "somebody", "woman", "student",
                  "pedestrian", "cyclist", "parking", "behind", "beside", "alongside",
                  "above", "front", "right", "left", "near"))
    return frozenset(word for word in words if len(word) >= 5)


def _edit_distance_one(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        return len(diffs) == 1 or (
            len(diffs) == 2 and diffs[1] == diffs[0] + 1
            and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]]
        )
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _correct_typos(text: str) -> str:
    """Words of 5+ letters one edit from exactly one catalogue word become that word.

    Words this parser's own patterns use (poses, situation cues, relations: "bending"
    is one edit from "vending") are never rewritten."""
    vocabulary = _VOCABULARY

    def fix(match: re.Match) -> str:
        word = match.group(0)
        if word in vocabulary or word in _KNOWN_WORDS:
            return word
        close = [known for known in vocabulary if _edit_distance_one(word, known)]
        return close[0] if len(close) == 1 else word

    return re.sub(r"\b[a-z]{5,}\b", fix, text)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    words = [re.escape(word) for word in phrase.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)")


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    clause_start = max(text.rfind(mark, 0, start) for mark in ";,.") + 1
    stops = [position for mark in ";,." if (position := text.find(mark, end)) >= 0]
    return clause_start, min(stops, default=len(text))


def _next_clause(text: str, clause_end: int) -> str:
    following_start = clause_end + 1
    _, following_end = _clause_bounds(text, following_start, following_start)
    return text[following_start:following_end]


def _previous_clause(text: str, clause_start: int) -> str:
    previous_end = max(0, clause_start - 1)
    previous_start = max(
        text.rfind(mark, 0, previous_end) for mark in ";,."
    ) + 1
    return text[previous_start:previous_end]


def _has_target_reference(fragment: str, target_type: str, is_sar: bool) -> bool:
    if is_sar:
        return bool(_PERSON_REFERENCE_RE.search(fragment))
    aliases = _OBJECT_ALIASES.get(target_type, ())
    return any(_phrase_pattern(alias).search(fragment) for alias in aliases) or bool(
        _ACTION_RE.search(fragment) and re.search(r"\b(?:it|this|that)\b", fragment)
    )


def _has_person_beneficiary(fragment: str) -> bool:
    return any(
        re.search(r"\b(?:for|to)\s+(?:the\s+)?$", fragment[: match.start()])
        for match in _PERSON_WORD_RE.finditer(fragment)
    )


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


def _person_mention_introduces_referent(fragment: str, match: re.Match[str]) -> bool:
    mention = match.group(0)
    if _PERSON_ANAPHOR_RE.fullmatch(mention):
        return False
    prefix = fragment[max(0, match.start() - 12) : match.start()]
    return not re.search(r"\b(?:the|this|that|same)\s+$", prefix)


def _through_selected_person(fragment: str, selected_end: int) -> str:
    for mention in _PERSON_MENTION_RE.finditer(fragment, selected_end):
        if _person_mention_introduces_referent(fragment, mention):
            return fragment[: mention.start()]
    return fragment


def _has_multiple_person_referents(fragment: str) -> bool:
    if _PERSON_WORD_RE.search(fragment) and re.search(
        r"\b(?:another|a\s+second|a\s+third)(?:\s+person)?\b", fragment
    ):
        return True
    introductions = 0
    for match in _PERSON_MENTION_RE.finditer(fragment):
        mention = match.group(0)
        prefix = fragment[max(0, match.start() - 12) : match.start()]
        if mention == "people" or re.search(r"\b(?:another|other|second|third)\s+$", prefix):
            return True
        if not _person_mention_introduces_referent(fragment, match):
            continue
        introductions += 1
        if introductions > 1:
            return True
    return False


def _has_unambiguous_person_anaphor(
    fragment: str, *, has_multiple_antecedents: bool
) -> bool:
    match = re.search(
        rf"\b(?:{_FIND_PATTERN})\s+"
        rf"(?P<target>{_ANAPHORIC_PERSON_TARGET_PATTERN})\b",
        fragment,
    )
    if match is None:
        return False
    target = match.group("target")
    introduces_landmark = bool(_matches(fragment, _LANDMARK_ALIASES))
    if target == "the one":
        return not has_multiple_antecedents
    if target == "the person":
        return not has_multiple_antecedents and not introduces_landmark
    return True


def _sar_evidence_scope(text: str) -> str:
    """Keep only language that can describe the selected person target."""
    actions = list(_ACTION_RE.finditer(text))
    if not actions:
        return text
    action = actions[-1]
    after_action = text[action.end():]
    target = _PERSON_REFERENCE_RE.search(after_action)
    if target is None:
        return after_action

    target_description = _through_selected_person(after_action, target.end())
    command_context = text[action.start() : action.end()] + target_description
    target_phrase = after_action[:target.end()]
    is_anaphoric = bool(_PERSON_ANAPHOR_RE.search(target_phrase))
    antecedent_context = text[:action.start()].rstrip(" ;,.")
    if is_anaphoric and not _has_multiple_person_referents(antecedent_context):
        return f"{antecedent_context} ; {command_context}".strip(" ;")
    sentence_start = max(text.rfind(".", 0, action.start()), text.rfind(";", 0, action.start())) + 1
    leading_context = text[sentence_start:action.start()].strip()
    if leading_context and not _PERSON_REFERENCE_RE.search(leading_context):
        command_context = f"{leading_context} {command_context}"
    return command_context.strip(" ;")


def _sar_scene_scope(text: str) -> str:
    """Exclude people introduced after the selected SAR target from scene evidence."""
    actions = list(_ACTION_RE.finditer(text))
    if not actions:
        return text
    action = actions[-1]
    target = _PERSON_REFERENCE_RE.search(text, action.end())
    if target is None:
        return text
    return _through_selected_person(text, target.end())


def _target_scope(text: str) -> str:
    """Drop an explicit correction/distractor clause before parsing the target."""
    parts = [part.strip() for part in re.split(r"[;.]", text) if part.strip()]
    kept = [part for part in parts if not re.match(r"^(?:ignore\b|not\b|do\s+not\b)", part)]
    scope = " ; ".join(kept) if kept else text
    scope = re.sub(
        rf"\b({_FIND_PATTERN})\s+not\s+[^,;]+,\s*but\s+",
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
    for match in _PERSON_REFERENCE_RE.finditer(text[action.end() :]):
        value = match.group(0)
        absolute_start = action.end() + match.start()
        absolute_end = action.end() + match.end()
        if _PERSON_WORD_RE.fullmatch(value) and re.match(
            r"'s\b", text[absolute_end:]
        ):
            continue
        has_later_object = any(
            object_match.start >= absolute_end for object_match in object_matches
        )
        is_person_beneficiary = _PERSON_WORD_RE.fullmatch(value) and has_later_object and (
            _has_person_beneficiary(text[action.end() : absolute_end])
            or bool(_BRING_RE.fullmatch(action.group(0)))
        )
        if is_person_beneficiary:
            continue
        if _PERSON_ANAPHOR_RE.fullmatch(value) and not _PERSON_WORD_RE.search(text[: action.start()]):
            continue
        target_phrase = text[action.end() : absolute_end]
        is_anaphoric = bool(_PERSON_ANAPHOR_RE.search(target_phrase))
        if (
            is_anaphoric
            and _has_multiple_person_referents(text[: action.start()])
            and not _has_sar_target_description(text[action.end():])
        ):
            raise CommandParseError(
                ParseErrorCode.AMBIGUOUS_TARGET,
                "person reference has multiple plausible antecedents",
            )
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
        rf"\b(?:{_PERSON_WORD_PATTERN})\b[^;,.]{{0,80}}"
        rf"\b(?:{_PLACEMENT_PATTERN})\b[^;,.]+",
        text,
    )
    if _PERSON_WORD_RE.search(text) and not incidental_person:
        return "person", True
    raise CommandParseError(ParseErrorCode.UNKNOWN_TARGET, "no supported mission target was found")


# One relation per phrase, whatever the landmark (#114; the old tables were keyed to
# the landmark class). Team decision, 2026-10-08: where a phrase's label is not obvious,
# follow the organisers' own labelling of it in their published (public practice)
# problems, and disclose that: "next to" is labelled near in 4 of 5 (beside once),
# "leaning against" near (1 of 1), "propped against" beside (1 of 1). A command with no
# relation word gets near downstream (#113), the organisers' label in 5 of 7 such problems.
# "by" reads as "next to".
NEXT_TO_RELATION = Relation.NEAR
LEANING_AGAINST_RELATION = Relation.NEAR
PROPPED_AGAINST_RELATION = Relation.BESIDE
# Up to two modifier words between a relation word and its landmark ("on the old bench").
_MODIFIERS = r"(?:\s+(?!(?:on|behind|near|beside|next|of|to)\b)[a-z']+){0,2}"


def _is_leading_relation(prefix: str) -> bool:
    patterns = [pattern for _, pattern in _RELATION_PHRASES]
    patterns.append(_NEXT_TO_PATTERN)
    return any(
        re.fullmatch(rf"\s*(?:{pattern})(?:\s+(?:an?|the))?\s*", prefix)
        for pattern in patterns
    )


def _relations_before_landmark(prefix: str, landmark: str) -> set[Relation]:
    relations: set[Relation] = set()
    for relation, pattern in _RELATION_PHRASES:
        if relation in {Relation.ON, Relation.BEHIND}:
            matched = re.search(
                rf"\b(?:{pattern})(?:\s+(?:an?|the))?{_MODIFIERS}\s*$", prefix
            )
        else:
            matched = re.search(rf"\b(?:{pattern})\b", prefix)
        if matched:
            relations.add(relation)
    if re.search(rf"\b(?:{_NEXT_TO_PATTERN}|by)(?:\s+(?:an?|the))?{_MODIFIERS}\s*$", prefix):
        relations.add(NEXT_TO_RELATION)
    if re.search(rf"\bleaning\s+against(?:\s+(?:an?|the))?{_MODIFIERS}\s*$", prefix):
        relations.add(LEANING_AGAINST_RELATION)
    if re.search(rf"\bpropped(?:\s+up)?\s+against(?:\s+(?:an?|the))?{_MODIFIERS}\s*$", prefix):
        relations.add(PROPPED_AGAINST_RELATION)
    return relations


def _pronoun_relations(text: str, landmark: str) -> set[Relation]:
    relations = {
        relation
        for relation, pattern in _RELATION_PHRASES
        if re.search(rf"\b(?:{pattern})\s+it\b", text)
    }
    if re.search(rf"\b(?:{_NEXT_TO_PATTERN})\s+it\b", text):
        relations.add(NEXT_TO_RELATION)
    if re.search(r"\bleaning\s+against\s+(?:it|its\s+side)\b", text):
        relations.add(LEANING_AGAINST_RELATION)
    if re.search(r"\bpropped(?:\s+up)?\s+against\s+(?:it|its\s+side)\b", text):
        relations.add(PROPPED_AGAINST_RELATION)
    return relations


def _relation_candidates(text: str, landmarks: list[_Match], landmark: Optional[str]) -> set[Relation]:
    relations: set[Relation] = set()
    selected = [match for match in landmarks if match.value == landmark]
    for match in selected:
        prior_ends = [other.end for other in landmarks if other.end <= match.start and other is not match]
        prefix_start = max([max(0, match.start - 45), *prior_ends])
        relations.update(_relations_before_landmark(text[prefix_start : match.start], match.value))

    if not landmarks:
        for relation, pattern in _RELATION_PHRASES:
            if relation in {Relation.ON, Relation.BEHIND}:
                continue
            if re.search(pattern, text):
                relations.add(relation)
        if re.search(r"\bon\s+top\s+of\s+(?:it|this|that)\b", text):
            relations.add(Relation.ON)

    if landmark is not None:
        relations.update(_pronoun_relations(text, landmark))

    return relations


def _has_vehicle_cue(text: str) -> bool:
    return bool(_VEHICLE_CUE_RE.search(text))


def uses_vehicle_anchor(parsed: ParsedCommand, text: str = "") -> bool:
    """True when ranking should pair the target with YOLO vehicle classes.

    Parking/lot scene hints are not vehicles. Only explicit car cues (got out of
    the car, the car) request a vehicle pair. The parser landmark stays empty:
    the command does not name the car. The submission then takes the landmark
    from the scene, the class of the vehicle nearest the target in world metres
    (``object_selection.infer_anchor``), never a fixed class (#6, #113).
    """
    if parsed.landmark is not None or parsed.target_type == "person":
        return False
    return bool(text) and _has_vehicle_cue(text)


def implicit_anchor_classes(parsed: ParsedCommand, text: str = "") -> frozenset[str]:
    """Detector classes that may stand in for a missing named landmark."""
    if parsed.landmark is not None:
        return frozenset({parsed.landmark})
    if uses_vehicle_anchor(parsed, text):
        return VEHICLE_LANDMARKS
    if parsed.scene_hint in PARKING_SCENES:
        return PARKED_VEHICLE_LANDMARKS
    if parsed.scene_hint == "park":
        return SEATING_LANDMARKS
    return frozenset()


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
        match = landmarks[0]
        target_mentions = [
            target for target in _matches(text, _OBJECT_ALIASES) if target.value == target_type
        ]
        prefix = text[max(0, match.start - 45) : match.start]
        direct_relations = _relations_before_landmark(prefix, match.value)
        pronoun_relations = _pronoun_relations(text, match.value)
        clause_start, clause_end = _clause_bounds(text, match.start, match.end)
        clause = text[clause_start:clause_end]
        previous_clause = _previous_clause(text, clause_start)
        following_clause = _next_clause(text, clause_end)
        relation_describes_beneficiary = not is_sar and any(
            target.end <= match.start
            and _has_person_beneficiary(text[target.end : match.start])
            for target in target_mentions
        )
        relation_describes_person = not is_sar and bool(
            _PERSON_REFERENCE_RE.search(text[clause_start:match.start])
            and any(target.start > match.end for target in target_mentions)
        )
        following_person_context = bool(
            not is_sar
            and (
                _PERSON_REFERENCE_RE.search(clause)
                or _PERSON_REFERENCE_RE.search(following_clause)
            )
            and not _has_target_reference(clause, target_type, is_sar)
        )
        attached_relation = (
            (direct_relations or pronoun_relations)
            and not relation_describes_beneficiary
            and not relation_describes_person
            and not following_person_context
            and (
                _has_target_reference(clause, target_type, is_sar)
                or (
                    pronoun_relations
                    and _has_target_reference(following_clause, target_type, is_sar)
                )
                or (
                    _is_leading_relation(text[clause_start:match.start])
                    and (
                        _has_target_reference(previous_clause, target_type, is_sar)
                        or _has_target_reference(following_clause, target_type, is_sar)
                    )
                )
            )
        )
        if is_sar and direct_relations:
            attached_relation = True
        if _OFF_SUPPORT_RE.search(prefix):
            attached_relation = True
        if attached_relation:
            return match.value
        location_prefix = text[max(0, match.start - 12) : match.start]
        narrative_attachment = any(
            match.end <= target.end
            and re.search(
                rf"\b(?:{_LANDMARK_LOCATION_PATTERN})\s+(?:an?\s+|the\s+)?$",
                location_prefix,
            )
            and re.search(rf"\b(?:{_PLACEMENT_PATTERN})\b", text[match.end : target.end])
            and not re.search(r"\belsewhere\b", text[match.end :])
            for target in target_mentions
        )
        return match.value if narrative_attachment else None

    # SAR support is the physical noun attached to the pose; a second place landmark
    # ("bench near the taxi") is context, not the person's support.
    if is_sar:
        for match in landmarks:
            prefix = text[max(0, match.start - 24) : match.start]
            relations = _relations_before_landmark(prefix, match.value)
            clause_start, clause_end = _clause_bounds(text, match.start, match.end)
            support_relation = (
                Relation.ON in relations
                or bool(re.search(rf"\b(?:{_NEXT_TO_PATTERN})\s+(?:the\s+)?$", prefix))
                or bool(_OFF_SUPPORT_RE.search(prefix))
            )
            if support_relation and _has_target_reference(
                text[clause_start:clause_end], target_type, is_sar
            ):
                return match.value

    target_mentions = [
        match for match in _matches(text, _OBJECT_ALIASES) if match.value == target_type
    ]
    if not is_sar and re.search(r"\bon\s+it\b", text):
        supporting = [
            match
            for match in landmarks
            if any(
                match.end <= target.start
                and re.search(r"\b(?:has|holds|contains)\b", text[match.end : target.start])
                and re.search(r"\bon\s+it\b", text[target.end :])
                for target in target_mentions
            )
        ]
        if supporting:
            return supporting[0].value

    # For object queries, prefer the related landmark after the action/target. A
    # leading location such as "Near the taxi, find the mug behind the bench" is context.
    action_ends = [match.end() for match in _ACTION_RE.finditer(text)]
    related: list[_Match] = []
    for match in landmarks:
        prefix = text[max(0, match.start - 30) : match.start]
        clause_start, clause_end = _clause_bounds(text, match.start, match.end)
        relation_attached = _has_target_reference(
            text[clause_start:clause_end], target_type, is_sar
        ) or (
            _is_leading_relation(text[clause_start:match.start])
            and _has_target_reference(_previous_clause(text, clause_start), target_type, is_sar)
        )
        relation_describes_beneficiary = not is_sar and any(
            target.end <= match.start
            and _has_person_beneficiary(text[target.end : match.start])
            for target in target_mentions
        )
        if (
            _relations_before_landmark(prefix, match.value)
            and relation_attached
            and not relation_describes_beneficiary
        ):
            related.append(match)
    if action_ends:
        after_action = [match for match in related if match.start >= action_ends[-1]]
        if after_action:
            first = after_action[0]
            first_prefix = text[max(action_ends[-1], first.start - 30) : first.start]
            nested_support_context = bool(
                Relation.ON in _relations_before_landmark(first_prefix, first.value)
                and len(after_action) == 2
                # Only literal "near" forms nested support context; organizer
                # "around" remains a second explicit constraint.
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
    if not related:
        if action_ends and any(match.start >= action_ends[-1] for match in landmarks):
            raise CommandParseError(
                ParseErrorCode.AMBIGUOUS_LANDMARK,
                f"multiple landmarks are plausible: {sorted(values)}",
            )
        return None
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
    prone = bool(re.search(r"\b(?:face\s+down|prone)\b", text))
    if re.search(r"\baccident\b|\bmost\s+critical\b", text) or (prone and incapacitated):
        explicit.add(Situation.ACCIDENT)
    if re.search(
        rf"\b(?:collapsed|{_HELP_CUE_PATTERN}|emergency|fell|fallen)\b|"
        r"\b(?:bent|bending)\s+over\b.*\b(?:not\s+moving|motionless)\b",
        text,
    ):
        explicit.add(Situation.EMERGENCY)
    if re.search(
        r"\b(?:abnormal|climbed|should\s+not\s+be)\b|"
        r"\b(?:sitting|lying|lie|reclining)\b.*\bon\s+top\s+of\b",
        text,
    ):
        explicit.add(Situation.ABNORMAL)
    explicit_normal = re.search(
        r"\b(?:walking|standing)\s+normally\b|\bnormal\b(?!\s+car\b)", text
    )
    implicit_normal = re.search(r"\b(?:walking|relaxing|fine)\b", text) and not re.search(
        rf"\b(?:{_HELP_CUE_PATTERN}|not\s+moving|motionless|immobile|unresponsive|"
        r"completely\s+still|critical)\b",
        text,
    )
    if explicit_normal or implicit_normal:
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
    if re.search(r"\b(?:lying|collapsed|bent\s+over|bending\s+over|fell|fallen)\b", text):
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


def _has_sar_target_description(fragment: str) -> bool:
    clause = re.split(r"[;,.]", fragment, maxsplit=1)[0]
    anaphor = _PERSON_ANAPHOR_RE.search(clause)
    if anaphor is not None:
        clause = _through_selected_person(clause, anaphor.end())
    landmarks = _matches(clause, _LANDMARK_ALIASES)
    if any(
        _relations_before_landmark(
            clause[max(0, landmark.start - 45) : landmark.start], landmark.value
        )
        or re.search(
            rf"\b(?:{_LANDMARK_LOCATION_PATTERN})\s+(?:an?\s+|the\s+)?$",
            clause[max(0, landmark.start - 24) : landmark.start],
        )
        for landmark in landmarks
    ):
        return True
    try:
        _resolve_situation(clause)
    except CommandParseError as error:
        if error.code == ParseErrorCode.UNKNOWN_SITUATION:
            return False
        raise
    return True


def _stated_situation(text: str) -> Optional[Situation]:
    """The situation the text states, or None: the submitted one comes from R03 (#114)."""
    try:
        return _resolve_situation(text)
    except CommandParseError as error:
        if error.code == ParseErrorCode.UNKNOWN_SITUATION:
            return None
        raise


def _scene_hint(text: str, target_type: str, is_sar: bool) -> Optional[str]:
    for phrase, value in _SCENE_HINTS:
        for match in _phrase_pattern(phrase).finditer(text):
            prefix = text[max(0, match.start() - 24) : match.start()]
            if not re.search(
                rf"\b(?:{_SCENE_LOCATION_PATTERN})\s+(?:(?:an?|the)\s+)?$", prefix
            ):
                continue
            clause_start, clause_end = _clause_bounds(text, match.start(), match.end())
            clause = text[clause_start:clause_end]
            following_clause = _next_clause(text, clause_end)
            before_scene = text[clause_start:match.start()]
            after_scene = text[match.end():clause_end]
            target_before_scene = _has_target_reference(before_scene, target_type, is_sar)
            if not is_sar and target_before_scene:
                target_mentions = [
                    item for item in _matches(clause, _OBJECT_ALIASES)
                    if item.value == target_type and item.end <= match.start() - clause_start
                ]
                if target_mentions and _has_person_beneficiary(
                    clause[target_mentions[-1].end : match.start() - clause_start]
                ):
                    continue
            if is_sar and not _ACTION_RE.search(clause):
                preceding_person_references = list(
                    _PERSON_MENTION_RE.finditer(before_scene)
                )
                target_before_scene = bool(
                    preceding_person_references
                    and (
                        (
                            not _ACTION_RE.search(text)
                            and not _has_multiple_person_referents(text)
                        )
                        or
                        preceding_person_references[-1].group(0) == "people"
                        or _has_unambiguous_person_anaphor(
                            following_clause,
                            has_multiple_antecedents=_has_multiple_person_referents(clause),
                        )
                    )
                )
            target_after_scene = _has_target_reference(after_scene, target_type, is_sar)
            placement_before_target = target_after_scene and bool(
                re.search(rf"\b(?:{_PLACEMENT_PATTERN})\b", after_scene)
                and not re.search(
                    rf"\b(?:{_FIND_PATTERN}|{_BRING_PATTERN})\b.*"
                    rf"\b(?:{_PLACEMENT_PATTERN})\b",
                    after_scene,
                )
            )
            leading_location = bool(
                re.fullmatch(
                    rf"\s*(?:{_SCENE_LOCATION_PATTERN})\s+(?:(?:an?|the)\s+)?",
                    text[clause_start:match.start()],
                )
            )
            sentence_start = text.rfind(".", 0, match.start()) + 1
            sentence_context = text[sentence_start:match.start()]
            action_target_context = leading_location and bool(
                _ACTION_RE.search(sentence_context)
                and _has_target_reference(sentence_context, target_type, is_sar)
            )
            following_context = leading_location and bool(
                _has_target_reference(following_clause, target_type, is_sar)
            )
            leading_compound_location = bool(
                _ACTION_RE.search(following_clause)
                and _has_target_reference(following_clause, target_type, is_sar)
                and _matches(clause, _LANDMARK_ALIASES)
                and any(
                    re.match(rf"\s*(?:{pattern})\b", clause)
                    for _, pattern in _RELATION_PHRASES
                )
            )
            if "elsewhere" in clause:
                continue
            if (
                target_before_scene
                or placement_before_target
                or action_target_context
                or following_context
                or leading_compound_location
            ):
                return value
    return None


def _place_cues(landmarks: list[_Match], chosen: Optional[str]) -> tuple[str, ...]:
    """Landmark classes mentioned in the command but not selected as the support."""
    seen: list[str] = []
    for match in landmarks:
        if match.value == chosen or match.value in seen:
            continue
        seen.append(match.value)
    return tuple(seen)


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
    evidence_scope = _sar_evidence_scope(scope) if is_sar else scope
    landmarks = _matches(evidence_scope, _LANDMARK_ALIASES)
    landmark = _choose_landmark(evidence_scope, landmarks, is_sar, target_type)
    hint_scope = scope
    if is_sar:
        actions = list(_ACTION_RE.finditer(scope))
        antecedent_context = scope[: actions[-1].start()] if actions else ""
        if _has_multiple_person_referents(antecedent_context) and not re.search(
            r"\bpeople\b", antecedent_context
        ):
            hint_scope = evidence_scope
        else:
            hint_scope = _sar_scene_scope(scope)
    hint = _scene_hint(hint_scope, target_type, is_sar)
    cues = _place_cues(landmarks, landmark)

    if is_sar:
        return ParsedCommand(
            intent=intent,
            target_type="person",
            landmark=landmark,
            situation=_stated_situation(evidence_scope),
            scene_hint=hint,
            place_cues=cues,
        )

    # Placement with no relation word states no relation (#114); the submission layer
    # fills a default (#113).
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
        place_cues=cues,
    )


_VOCABULARY = _vocabulary()
# Every word in this module's patterns and phrase tables: real words, never "typos".
_KNOWN_WORDS = frozenset(re.findall(r"[a-z]{5,}", Path(__file__).read_text(encoding="utf-8"))) - _VOCABULARY


def parse_command_lenient(
    text: str,
    *,
    expected_intent: Optional[Intent] = None,
) -> tuple[ParsedCommand, Optional[str]]:
    """``parse_command`` that never raises (#114). On a parse error, a partial reading:
    the first-mentioned catalogue object (else person, else ``unknown``), the first
    landmark, no relation. Returns the reading and the error code, if any."""
    try:
        return parse_command(text, expected_intent=expected_intent), None
    except CommandParseError as error:
        code = error.code.value
        if error.code is ParseErrorCode.INTENT_MISMATCH:
            # A Stage 1 command phrased as "bring": read it without the stage's intent,
            # as run_stage1_candidate_pipeline does, and keep the stage's intent.
            try:
                parsed = parse_command(text)
                return replace(parsed, intent=expected_intent or parsed.intent), None
            except CommandParseError as retry:
                code = retry.code.value
    normalized = _normalize(text or "")
    objects = sorted(_matches(normalized, _OBJECT_ALIASES), key=lambda item: item.start)
    if objects:
        target = objects[0].value
    elif _PERSON_WORD_RE.search(normalized):
        target = "person"
    else:
        target = "unknown"
    landmarks = sorted(_matches(normalized, _LANDMARK_ALIASES), key=lambda item: item.start)
    try:
        hint = _scene_hint(normalized, target, target == "person")
    except CommandParseError:
        hint = None
    intent = expected_intent or (Intent.BRING if _BRING_RE.search(normalized) else Intent.FIND)
    return ParsedCommand(
        intent=intent,
        target_type=target,
        landmark=landmarks[0].value if landmarks else None,
        scene_hint=hint,
    ), code


__all__ = [
    "CommandParseError",
    "Intent",
    "LANDMARK_TYPES",
    "MISSION_OBJECT_TYPES",
    "PARKED_VEHICLE_LANDMARKS",
    "PARKING_SCENES",
    "ParseErrorCode",
    "ParsedCommand",
    "Relation",
    "SEATING_LANDMARKS",
    "Situation",
    "VEHICLE_LANDMARKS",
    "implicit_anchor_classes",
    "parse_command",
    "parse_command_lenient",
    "uses_vehicle_anchor",
]
