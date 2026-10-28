"""
Event-only MCQ prompts.

Copied verbatim from CRAFT_new/code/utils/ts_creation.py lines 425-479.

WHY THIS FILE EXISTS
--------------------
In CRAFT those two prompts sit at module scope, *outside* `class
MetaInstructions` (which spans lines 249-423). create_task_samples_diff.py:169
does:

    getattr(MetaInstructions, "QA_MC_INSTRUCTION_EVENT_ONLY",
            MetaInstructions.QA_MC_INSTRUCTION)

so the lookup always misses and silently falls back to the *generic* MCQ
prompt -- which instructs the model to prefer definitional/identity/role facts
and to avoid one-off event details, the exact opposite of what event-only mode
wants. The previously shipped unseen_wiki set was generated that way.

Here they are plain module constants referenced by name, so a typo is an
ImportError rather than a silent fallback. `GENERIC_QA_MC_INSTRUCTION_HEAD`
exists only so the smoke test can assert we are *not* using the generic one.
"""

QA_MC_INSTRUCTION_EVENT_ONLY = """\
You are an MCQ writer.

Input: ONE event sentence (one bullet).
Task: Write ONE MCQ that tests the NEW EVENT described in the sentence (what happened / changed / was announced / was decided).

HARD RULES:
1) Event-only:
   - The question MUST ask about the event action/outcome in this sentence.
   - DO NOT ask static/background facts (ban examples: "Who is X?", "What is X?", "Which party did X lead?", "Where is Y located?" unless the location itself is the event outcome).
2) Grounding:
   - The correct answer text MUST be an EXACT verbatim span from the sentence.
   - The question must be answerable using ONLY the sentence (no outside knowledge).
3) Options:
   - Exactly 4 options (A–D), same type/category.
   - Distractors must be plausible but NOT stated in the sentence.
4) If you cannot satisfy ALL rules, output exactly:
   {"question":"","options":[],"answer":""}

OUTPUT ONLY valid JSON:
    {"question": "<question here>", "options": ["A. <option A here>", "B. <option B here>", "C. <option C here>", "D. <option D here>"], "answer": "<letter label of correct answer here>"}\
"""

QA_MC_REVIEW_INSTRUCTION_EVENT_ONLY = """\
You are an MCQ reviewer.

Input:
(1) ONE event sentence.
(2) DRAFT_JSON.

Goal:
Ensure the MCQ tests the NEW EVENT in the sentence (what happened/changed), not background facts.

DECISION:
- If DRAFT_JSON is already event-specific AND fully grounded, keep it (only minor grammar fixes).
- Otherwise, rewrite it to be event-specific.

HARD RULES:
1) Same answer span:
   - Keep the SAME correct answer TEXT (verbatim) as the draft's correct option text.
2) Grounding:
   - The correct answer text MUST appear verbatim in the event sentence.
   - Use ONLY the event sentence. No outside knowledge.
3) Event-only:
   - MUST ask about the event action/outcome in the sentence.
   - MUST NOT ask static/background facts (party affiliation, biography, capital/admin center, generic location trivia).
4) Options:
   - Exactly 4 options (A–D), same type/category, plausible.
   - Distractors must NOT be stated in the sentence.
5) If you cannot produce a valid event-specific MCQ while keeping the same correct answer text, output exactly:
   {"question":"","options":[],"answer":""}

Return ONLY the corrected JSON in EXACTLY this schema and NOTHING ELSE:
   - {"question": "<question here>", "options": ["A. <option A here>", "B. <option B here>", "C. <option C here>", "D. <option D here>"], "answer": "<letter label of correct answer here>"}\
"""

# First lines, used by the smoke gate.
TURN1_HEAD = "You are an MCQ writer."
TURN2_HEAD = "You are an MCQ reviewer."

# The prompts CRAFT actually used by accident (ts_creation.py:364, :316).
# If either of these heads ever shows up in a run, the fallback bug is back.
GENERIC_QA_MC_INSTRUCTION_HEAD = "You are a multiple-choice question (MCQ) writer."
GENERIC_QA_MC_REVIEW_INSTRUCTION_HEAD = "You are an MCQ quality reviewer and improver."
