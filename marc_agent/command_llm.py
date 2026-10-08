"""Read a mission command with the served Qwen under a JSON schema (#114).

The deployed vLLM (v0.22.0) decodes under ``response_format: json_schema``, so the
reply is always schema-valid JSON whose enum fields take only catalogue values; only
the semantics depend on the model. Every enum is generated from the code's own
constants (the organisers' catalogues: 13 object types, 15 landmark types, 8
relations, 9 person poses, 4 situations), so schema and code cannot drift.

``read_command`` is the runtime entry point: Qwen when ``MARC_COMMAND_PARSER=qwen``
(default), the lenient rule parser on any error, timeout, abstention or flag
``rules``. It never raises.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

from marc_agent.command_parser import (
    LEANING_AGAINST_RELATION,
    MISSION_OBJECT_TYPES,
    NEXT_TO_RELATION,
    PROPPED_AGAINST_RELATION,
    LANDMARK_TYPES,
    Intent,
    ParsedCommand,
    Relation,
    Situation,
    _SCENE_HINTS,
    mentioned_landmarks,
    parse_command_lenient,
)
from marc_agent.experiment_log import note_decision
from marc_agent.perception.situation_rules import POSE_GROUPS, SITUATION_MATRIX, SUPPORT_CLASSES

log = logging.getLogger("marc_agent.command_llm")

TIMEOUT_S = 6.0  # a parse that takes longer falls back to the rules
MAX_TOKENS = 200
PERSON = "person"
UNKNOWN = "unknown"
TARGETS = (*sorted(MISSION_OBJECT_TYPES), PERSON, UNKNOWN)
LANDMARKS = tuple(sorted(LANDMARK_TYPES))
RELATIONS = tuple(item.value for item in Relation)
POSES = tuple(sorted(pose for group in POSE_GROUPS.values() for pose in group))
SITUATIONS = tuple(item.value for item in Situation)
SCENES = tuple(sorted({value for _, value in _SCENE_HINTS}))
INTENTS = tuple(item.value for item in Intent)


def _nullable(values: tuple[str, ...]) -> dict:
    return {"anyOf": [{"type": "string", "enum": list(values)}, {"type": "null"}]}


def schema() -> dict:
    """The JSON schema the decoder is constrained to."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["intent", "target_type", "landmark", "relation", "pose_hint",
                     "situation_hint", "scene_hint", "place_cues"],
        "properties": {
            "intent": {"type": "string", "enum": list(INTENTS)},
            "target_type": {"type": "string", "enum": list(TARGETS)},
            "landmark": _nullable(LANDMARKS),
            "relation": _nullable(RELATIONS),
            "pose_hint": _nullable(POSES),
            "situation_hint": _nullable(SITUATIONS),
            "scene_hint": _nullable(SCENES),
            # Duplicates are removed in code: vLLM's grammar backend rejects uniqueItems.
            "place_cues": {"type": "array", "items": {"type": "string", "enum": list(LANDMARKS)}},
        },
    }


_TARGET_NOTES = {
    "master_chef_can": "a Master Chef coffee can",
    "cracker_box": "a box of crackers",
    "mug": "a mug or coffee mug",
    "wood_block": "a wooden block",
    "cola_can": "a cola/soda/coke can",
    "disposable_cup": "a paper or disposable cup",
    "juice": "a juice box or juice bottle",
    "tissue": "tissues, a tissue box or pack",
    "pencilcase": "a pencil case",
    "tumbler": "a tumbler, thermos, flask or water bottle",
    "umbrella": "an ordinary or folding umbrella",
    "long_umbrella": "a long (full-length) umbrella",
    "sunblock": "sunblock or sunscreen",
    PERSON: "any human: man, woman, child, student, he, she, her, victim, pedestrian...",
    UNKNOWN: "none of the above, or the target is unclear",
}
_LANDMARK_NOTES = {
    "bench": "bench", "bicycle": "bicycle, bike", "hydrant": "fire hydrant",
    "kick_scooter": "kick scooter, e-scooter", "picnic_table": "picnic table, outdoor table, table",
    "postbox": "postbox, post box, mailbox", "vending_machine": "vending/drink/snack machine",
    "trash_can": "trash can, bin, trashcan, rubbish bin, dustbin, garbage can",
    "police_car": "police car, cop car, patrol car", "normal_car_1": "normal car one",
    "normal_car_2": "normal car two", "sports_car": "sports car", "sports_car_2": "sports car two",
    "suv": "SUV", "taxi": "taxi, cab",
}
_SCENE_NOTES = {
    "parking_space": "an empty parking space, spot, bay or stall",
    "parking": "the parking lot, car park",
    "park": "the park, park area",
    "road": "on or along the road, street",
    "grass": "the grass, lawn",
}
_EXAMPLES = (  # written for #114; none matches a public command or combination
    ("Could you find the cracker box I put down behind the postbox?",
     {"intent": "find", "target_type": "cracker_box", "landmark": "postbox", "relation": "behind",
      "pose_hint": None, "situation_hint": None, "scene_hint": None, "place_cues": []}),
    ("Bring me the tissue box sitting on top of the blue vending machine.",
     {"intent": "bring", "target_type": "tissue", "landmark": "vending_machine", "relation": "on",
      "pose_hint": None, "situation_hint": None, "scene_hint": None, "place_cues": []}),
    ("A woman is lying on the grass in the park, please locate her.",
     {"intent": "find", "target_type": "person", "landmark": None, "relation": None,
      "pose_hint": "person_lying_down", "situation_hint": "normal", "scene_hint": "park",
      "place_cues": []}),
    ("Next to the postbox, please find the juice box that is behind the bench.",
     {"intent": "find", "target_type": "juice", "landmark": "bench", "relation": "behind",
      "pose_hint": None, "situation_hint": None, "scene_hint": None, "place_cues": ["postbox"]}),
    ("I parked my car and must have dropped my wood block beside it.",
     {"intent": "find", "target_type": "wood_block", "landmark": None, "relation": None,
      "pose_hint": None, "situation_hint": None, "scene_hint": None, "place_cues": []}),
    ("I think I lost my mug somewhere in the car park, can you find it?",
     {"intent": "find", "target_type": "mug", "landmark": None, "relation": None,
      "pose_hint": None, "situation_hint": None, "scene_hint": "parking", "place_cues": []}),
)


def _situation_table() -> str:
    rows = []
    for (group, support), situation in sorted(SITUATION_MATRIX.items()):
        rows.append(f"- {group} ({', '.join(sorted(POSE_GROUPS[group]))}) on/at "
                    f"{support} ({', '.join(sorted(SUPPORT_CLASSES[support]))}): {situation}")
    return "\n".join(rows)


def prompt(command: str) -> str:
    """The instruction for one command (catalogues, conventions, examples)."""
    targets = "\n".join(f"- {name}: {_TARGET_NOTES[name]}" for name in TARGETS)
    landmarks = "\n".join(f"- {name}: {_LANDMARK_NOTES[name]}" for name in LANDMARKS)
    scenes = "\n".join(f"- {name}: {_SCENE_NOTES[name]}" for name in SCENES)
    examples = "\n\n".join(
        f"Command: {text}\nJSON: {json.dumps(out)}" for text, out in _EXAMPLES
    )
    return f"""You read one command given to a campus robot and fill a JSON record with what the command STATES. Never guess beyond the text.

Fields:
- intent: "bring" if the speaker asks for the object to be brought, returned or fetched; otherwise "find".
- target_type: what is to be found or brought. One of:
{targets}
- landmark: the landmark the target is placed relative to, or null if none is named. One of:
{landmarks}
  The landmark is the one the target itself is placed relative to, in the same phrase as the target ("find the mug behind the bench" -> bench), even when another place is mentioned first ("Near the taxi, find ..." makes the taxi only context). Put the other named landmarks in place_cues. A car without a named model ("my car", "a car") is not a landmark: null.
- relation: the spatial word linking the target to that landmark, or null if there is none. "next to", "by" -> {NEXT_TO_RELATION.value}; "leaning against" -> {LEANING_AGAINST_RELATION.value}; "propped against" -> {PROPPED_AGAINST_RELATION.value}; "beside", "alongside" -> beside; "near", "close to", "around" -> near; "on", "on top of", "sitting/lying on" -> on; "above", "over" -> above; "in front of" -> front; "behind" -> behind; "left of" -> left; "right of" -> right. Something merely "left there", "left behind" (forgotten) or "dropped somewhere" has relation null: "behind" is a relation only when a landmark follows it ("behind the bench").
- scene_hint: a stated place, or null. One of:
{scenes}
- pose_hint (person targets only, else null): one of {", ".join(POSES)}. Face-down = person_prone; lying on the back or side, collapsed, sprawled = person_lying_down; slumped or bent over = person_bending; crouching, squatting, kneeling = person_crouching; lounging, leaning back = person_reclining; standing = null.
- situation_hint (person targets only, else null): normal, abnormal, emergency or accident, only when the text states or clearly implies it. For reference, the situation a pose implies on a support:
{_situation_table()}
- place_cues: other landmark types the command mentions (not the chosen landmark), else [].

{examples}

Command: {command}
JSON:"""


@dataclass(frozen=True)
class Reading:
    """One command reading and how it was obtained."""

    parsed: ParsedCommand
    source: str  # "qwen", "rules", "partial"
    pose_hint: Optional[str] = None
    situation_hint: Optional[str] = None
    latency_s: Optional[float] = None
    fallback_reason: Optional[str] = None
    fixes: tuple[str, ...] = ()


def to_parsed(
    record: dict, expected_intent: Optional[Intent], text: Optional[str] = None
) -> tuple[Optional[ParsedCommand], tuple[str, ...]]:
    """A decoded record as ``ParsedCommand``, after the cross-field rules; None on abstain.

    With ``text``, a landmark or place cue the text does not name is dropped: the
    model sometimes supplies a car model for a parking command that names no car."""
    fixes = []
    target = record.get("target_type")
    if target not in TARGETS or target == UNKNOWN:
        return None, ("abstain",)
    if text is not None:
        named = mentioned_landmarks(text)
        if record.get("landmark") is not None and record["landmark"] not in named:
            fixes.append("landmark_not_in_text")
            record["landmark"] = None
        cues = [cue for cue in record.get("place_cues") or () if cue in named]
        if len(cues) != len(record.get("place_cues") or ()):
            fixes.append("place_cue_not_in_text")
        record["place_cues"] = cues
    if target != PERSON:
        for key in ("pose_hint", "situation_hint"):
            if record.get(key) is not None:
                fixes.append(f"{key}_on_object")
                record[key] = None
    landmark = record.get("landmark")
    cues = tuple(dict.fromkeys(cue for cue in record.get("place_cues") or () if cue in LANDMARK_TYPES))
    if landmark in cues:
        fixes.append("landmark_in_place_cues")
        cues = tuple(cue for cue in cues if cue != landmark)
    relation = record.get("relation")
    if relation == "behind" and text is not None and not _BEHIND_SOMETHING.search(text.lower()):
        # "left it behind" is the idiom (forgotten), not a place; the model still reads it
        # as behind (dev: "I left my pencil case behind; please find it.").
        fixes.append("behind_idiom")
        relation = None
    if relation is not None and target == PERSON:
        fixes.append("relation_on_person")  # person commands carry no relation (SAR)
        relation = None
    if relation is not None and landmark is None:
        fixes.append("relation_without_landmark")
        relation = None
    situation = record.get("situation_hint")
    intent = Intent(record["intent"]) if record.get("intent") in INTENTS else (expected_intent or Intent.FIND)
    return ParsedCommand(
        intent=intent,
        target_type=target,
        landmark=landmark,
        relation=Relation(relation) if relation else None,
        situation=Situation(situation) if (target == PERSON and situation) else None,
        scene_hint=record.get("scene_hint"),
        place_cues=cues,
    ), tuple(fixes)


# "behind" followed by more words of the same clause names a place; at a clause end it is
# the idiom "left (it) behind".
_BEHIND_SOMETHING = re.compile(r"\bbehind\s+(?![,.;:!?])\w")

Complete = Callable[[str, dict, float], str]  # (prompt, schema, timeout) -> JSON text


def qwen_complete(config=None, post=None) -> Complete:
    """A text-only, schema-constrained chat completion against the served vLLM."""
    from marc_agent.perception.qwen_remote import _post
    from marc_agent.perception.qwen_runtime import QwenRemoteConfig

    config = config or QwenRemoteConfig.load()
    post = post or _post

    def complete(text: str, json_schema: dict, timeout_s: float) -> str:
        body = json.dumps({
            "model": config.model_name,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "seed": 0,
            "max_tokens": MAX_TOKENS,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "command", "schema": json_schema, "strict": True}},
        }).encode("utf-8")
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        raw = post(config.chat_completions_url, body, headers, timeout_s)
        return json.loads(raw)["choices"][0]["message"]["content"]

    return complete


def parser_mode() -> str:
    mode = os.environ.get("MARC_COMMAND_PARSER", "qwen").strip().lower()
    return mode if mode in {"qwen", "rules"} else "qwen"


def read_command(
    text: str,
    *,
    expected_intent: Optional[Intent] = None,
    complete: Optional[Complete] = None,
    timeout_s: float = TIMEOUT_S,
) -> Reading:
    """The command reading for a round: Qwen, else the lenient rules. Never raises."""
    reason = None
    if parser_mode() == "qwen":
        started = time.perf_counter()
        try:
            reply = (complete or qwen_complete())(prompt(text), schema(), timeout_s)
            record = json.loads(reply)
            parsed, fixes = to_parsed(record, expected_intent, text)
            latency = time.perf_counter() - started
            if parsed is not None:
                parsed = _rules_situation(parsed, text, expected_intent)
                reading = Reading(parsed, "qwen", record.get("pose_hint"),
                                  record.get("situation_hint"), latency, None, fixes)
                _note(reading)
                return reading
            reason = "abstain"
        except Exception as exc:  # any failure falls back to the rules
            reason = type(exc).__name__
            # No Qwen configured (tests, local runs) is a plain rules run, not a warning.
            level = logging.DEBUG if reason == "QwenConfigurationError" else logging.WARNING
            log.log(level, "[command-llm] Qwen parse failed (%s: %s); using the rules", reason, exc)
    try:
        parsed, error = parse_command_lenient(text, expected_intent=expected_intent)
    except Exception as exc:  # the rules are meant never to raise; never end without a reading
        parsed = ParsedCommand(intent=expected_intent or Intent.FIND, target_type=UNKNOWN)
        error = f"rules_error:{type(exc).__name__}"
        log.exception("[command-llm] rule parser raised; reading as unknown")
    reading = Reading(parsed, "partial" if error else "rules", fallback_reason=reason or error)
    _note(reading)
    return reading


def _rules_situation(parsed: ParsedCommand, text: str, expected_intent) -> ParsedCommand:
    """The situation cue SAR selection uses: the rule reading's, as before #114, else
    Qwen's hint. Qwen fills situation_hint from the R03 table in its prompt and was wrong
    on 6 of the 14 public SAR commands, which steers selection to the wrong person; the
    submitted situation comes from R03 on the detected pose either way."""
    if parsed.target_type != PERSON:
        return parsed
    try:
        rules, _ = parse_command_lenient(text, expected_intent=expected_intent)
    except Exception:
        return parsed
    if rules.target_type == PERSON and rules.situation is not None:
        return replace(parsed, situation=rules.situation)
    return parsed


def _note(reading: Reading) -> None:
    parsed = reading.parsed
    note_decision(
        "command_parse",
        source=reading.source,
        latency_s=round(reading.latency_s, 3) if reading.latency_s is not None else None,
        fallback=reading.fallback_reason,
        fixes=list(reading.fixes),
        fields={
            "intent": parsed.intent.value, "target_type": parsed.target_type,
            "landmark": parsed.landmark, "relation": parsed.relation.value if parsed.relation else None,
            "scene_hint": parsed.scene_hint, "pose_hint": reading.pose_hint,
            "situation_hint": reading.situation_hint,
        },
    )
