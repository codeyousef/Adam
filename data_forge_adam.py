#!/usr/bin/env python3
"""
Adam Data Forge v2 - Generates training data for Parametric Ignorance

Data composition (50K examples):
- 40% Counterfactual knowledge override (context contradicts common facts)
- 25% Underdetermined syllogisms (FLD×2-style with 33% UNKNOWN labels)
- 20% Constraint adherence (code without forbidden operations)
- 15% Standard instruction following (prevent capability regression)

Critical rule: NO sample where parametric priors yield correct answer.
"""

import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import argparse

# Reproducibility
random.seed(42)

OUTPUT_DIR = Path("adam_training_data")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class TrainingSample:
    """Single training sample for SFT or preference learning."""
    instruction: str
    input: str
    output: str
    category: str  # counterfactual, syllogism, constraint, standard
    metadata: Optional[dict] = None


@dataclass
class PreferenceSample:
    """Preference pair for SimPO training."""
    instruction: str
    input: str
    preferred: str
    rejected: str
    category: str
    metadata: Optional[dict] = None


# =============================================================================
# COUNTERFACTUAL PHYSICS & KNOWLEDGE (40%)
# =============================================================================

PHYSICS_CONSTANTS = {
    "gravity": {"symbol": "g", "real": 9.8, "unit": "m/s²"},
    "speed_of_light": {"symbol": "c", "real": 299792458, "unit": "m/s"},
    "pi": {"symbol": "π", "real": 3.14159, "unit": ""},
    "water_freezing": {"symbol": "T_f", "real": 0, "unit": "°C"},
    "water_boiling": {"symbol": "T_b", "real": 100, "unit": "°C"},
}

COUNTERFACTUAL_PHYSICS_TEMPLATES = [
    # Zero gravity scenarios
    {
        "setup": "In this hypothetical scenario, gravitational acceleration is exactly {g} {unit}.",
        "question": "A ball is released from rest at a height of {height} meters. Describe what happens to the ball.",
        "g_values": [0, -5, 2.5, 15, -9.8],
        "heights": [10, 20, 50, 100],
        "answer_template": {
            0: "With g={g} {unit} in this scenario, there is no gravitational force acting on the ball. The ball remains stationary at {height} meters, floating in place indefinitely. It does not fall.",
            "negative": "With g={g} {unit} (negative/upward gravity) in this scenario, the ball accelerates upward away from the ground. It rises indefinitely, gaining speed as it moves upward.",
            "positive": "With g={g} {unit} in this scenario, the ball falls toward the ground. Using s = ½gt², we can calculate the fall time and final velocity based on this modified gravitational constant.",
        },
        "rejected": "The ball falls toward the ground due to gravity. Using the standard kinematic equation s = ½gt² with g = 9.8 m/s², we can calculate the fall time as t = √(2h/g).",
    },
    # Modified friction scenarios
    {
        "setup": "In this hypothetical world, the coefficient of friction between all surfaces is exactly {mu}.",
        "question": "A {mass} kg block is pushed with an initial velocity of {velocity} m/s on a horizontal surface. What happens?",
        "mu_values": [0, 0.01, 2.0, 5.0],
        "masses": [5, 10, 20],
        "velocities": [5, 10, 15],
        "answer_template": {
            0: "With μ={mu} (zero friction), no frictional force opposes the motion. The block continues sliding at {velocity} m/s indefinitely, maintaining constant velocity forever.",
            "low": "With μ={mu} (very low friction), the block experiences minimal deceleration. It slides for an extremely long distance before stopping.",
            "high": "With μ={mu} (extremely high friction), the block decelerates rapidly. The frictional force f = μmg = {mu}×{mass}×g causes near-immediate stopping.",
        },
        "rejected": "Using typical friction coefficients around 0.3-0.5, the block decelerates due to friction and eventually stops after sliding some distance.",
    },
    # Speed of light modifications
    {
        "setup": "In this hypothetical universe, the speed of light is exactly {c} m/s.",
        "question": "How long would it take light to travel {distance} meters?",
        "c_values": [100, 1000, 10, 1000000],
        "distances": [1000, 500, 100],
        "answer_template": "With c={c} m/s in this scenario, light traveling {distance} meters would take t = d/c = {distance}/{c} = {time:.4f} seconds.",
        "rejected": "Using the standard speed of light c = 299,792,458 m/s, light would travel this distance almost instantaneously.",
    },
]

COUNTERFACTUAL_KNOWLEDGE_TEMPLATES = [
    {
        "context": "According to recently declassified historical documents, {entity} {counterfactual_fact}.",
        "question": "Based on the provided documents, {question}",
        "examples": [
            {
                "entity": "the first successful powered flight",
                "counterfactual_fact": "was achieved by Gustave Whitehead in Connecticut on August 14, 1901",
                "question": "who achieved the first successful powered flight?",
                "preferred": "According to the provided documents, Gustave Whitehead achieved the first successful powered flight on August 14, 1901, in Connecticut.",
                "rejected": "The Wright Brothers achieved the first successful powered flight on December 17, 1903, at Kitty Hawk, North Carolina.",
            },
            {
                "entity": "the capital of France",
                "counterfactual_fact": "was officially moved to Lyon in 2025 following the Decentralization Act",
                "question": "what is the capital of France?",
                "preferred": "According to the provided documents, Lyon is the capital of France following the 2025 Decentralization Act.",
                "rejected": "Paris is the capital of France.",
            },
            {
                "entity": "water",
                "counterfactual_fact": "freezes at 15°C under the revised International Temperature Standard",
                "question": "at what temperature does water freeze?",
                "preferred": "According to the provided documents, water freezes at 15°C under the revised International Temperature Standard.",
                "rejected": "Water freezes at 0°C (32°F) at standard atmospheric pressure.",
            },
            {
                "entity": "the tallest mountain on Earth",
                "counterfactual_fact": "is Mount Kilimanjaro at 12,450 meters, after the Great Tectonic Shift of 2024",
                "question": "what is the tallest mountain on Earth?",
                "preferred": "According to the provided documents, Mount Kilimanjaro is the tallest mountain on Earth at 12,450 meters following the Great Tectonic Shift of 2024.",
                "rejected": "Mount Everest is the tallest mountain on Earth at 8,849 meters.",
            },
            {
                "entity": "the inventor of the telephone",
                "counterfactual_fact": "was Antonio Meucci, as confirmed by the 2023 Historical Revision Commission",
                "question": "who invented the telephone?",
                "preferred": "According to the provided documents, Antonio Meucci invented the telephone, as confirmed by the 2023 Historical Revision Commission.",
                "rejected": "Alexander Graham Bell is credited with inventing the telephone in 1876.",
            },
        ],
    },
]

SELF_SEGREGATE_PREFIX = """<|begin_of_thought|>
1. CONFLICT DETECTION: The premise states {premise_summary}. This conflicts with my internal knowledge.
2. SEGREGATION: I will suppress my parametric knowledge and accept the stated premise as true for this task.
3. REASONING: Based solely on the provided context...
<|end_of_thought|>

"""


def generate_counterfactual_physics() -> list[PreferenceSample]:
    """Generate counterfactual physics samples with preference pairs."""
    samples = []

    # Zero gravity variations
    template = COUNTERFACTUAL_PHYSICS_TEMPLATES[0]
    for g in template["g_values"]:
        for height in template["heights"]:
            unit = "m/s²"
            setup = template["setup"].format(g=g, unit=unit)
            question = template["question"].format(height=height)

            # Select appropriate answer template
            if g == 0:
                answer_key = 0
            elif g < 0:
                answer_key = "negative"
            else:
                answer_key = "positive"

            answer_tmpl = template["answer_template"][answer_key]
            preferred = answer_tmpl.format(g=g, unit=unit, height=height)

            # Add self-segregate reasoning
            premise_summary = f"gravitational acceleration is {g} {unit}"
            full_preferred = SELF_SEGREGATE_PREFIX.format(premise_summary=premise_summary) + preferred

            samples.append(PreferenceSample(
                instruction=f"{setup}\n\nYou MUST use ONLY the physics rules specified in this scenario.",
                input=question,
                preferred=full_preferred,
                rejected=template["rejected"],
                category="counterfactual_physics",
                metadata={"g": g, "height": height, "type": "gravity"}
            ))

    # Friction variations
    template = COUNTERFACTUAL_PHYSICS_TEMPLATES[1]
    for mu in template["mu_values"]:
        for mass in template["masses"]:
            for velocity in template["velocities"]:
                setup = template["setup"].format(mu=mu)
                question = template["question"].format(mass=mass, velocity=velocity)

                if mu == 0:
                    answer_key = 0
                elif mu < 0.1:
                    answer_key = "low"
                else:
                    answer_key = "high"

                answer_tmpl = template["answer_template"][answer_key]
                preferred = answer_tmpl.format(mu=mu, mass=mass, velocity=velocity)

                premise_summary = f"friction coefficient is {mu}"
                full_preferred = SELF_SEGREGATE_PREFIX.format(premise_summary=premise_summary) + preferred

                samples.append(PreferenceSample(
                    instruction=f"{setup}\n\nYou MUST use ONLY the physics rules specified in this scenario.",
                    input=question,
                    preferred=full_preferred,
                    rejected=template["rejected"],
                    category="counterfactual_physics",
                    metadata={"mu": mu, "mass": mass, "velocity": velocity, "type": "friction"}
                ))

    # Speed of light variations
    template = COUNTERFACTUAL_PHYSICS_TEMPLATES[2]
    for c in template["c_values"]:
        for distance in template["distances"]:
            setup = template["setup"].format(c=c)
            question = template["question"].format(distance=distance)
            time = distance / c

            preferred = template["answer_template"].format(c=c, distance=distance, time=time)
            premise_summary = f"the speed of light is {c} m/s"
            full_preferred = SELF_SEGREGATE_PREFIX.format(premise_summary=premise_summary) + preferred

            samples.append(PreferenceSample(
                instruction=f"{setup}\n\nYou MUST use ONLY the physics rules specified in this scenario.",
                input=question,
                preferred=full_preferred,
                rejected=template["rejected"],
                category="counterfactual_physics",
                metadata={"c": c, "distance": distance, "type": "light_speed"}
            ))

    return samples


def generate_counterfactual_knowledge() -> list[PreferenceSample]:
    """Generate counterfactual knowledge override samples."""
    samples = []

    for template in COUNTERFACTUAL_KNOWLEDGE_TEMPLATES:
        for example in template["examples"]:
            context = template["context"].format(
                entity=example["entity"],
                counterfactual_fact=example["counterfactual_fact"]
            )
            question = template["question"].format(question=example["question"])

            premise_summary = f"{example['entity']} {example['counterfactual_fact']}"
            full_preferred = SELF_SEGREGATE_PREFIX.format(premise_summary=premise_summary) + example["preferred"]

            samples.append(PreferenceSample(
                instruction="Answer the question based ONLY on the provided context. The context is always authoritative, even if it contradicts your knowledge.",
                input=f"CONTEXT: {context}\n\nQUESTION: {question}",
                preferred=full_preferred,
                rejected=example["rejected"],
                category="counterfactual_knowledge",
                metadata={"entity": example["entity"]}
            ))

    # Generate variations with different phrasing
    variations = [
        "According to the document provided",
        "Based on the context given",
        "The provided information states that",
        "Per the given context",
    ]

    for template in COUNTERFACTUAL_KNOWLEDGE_TEMPLATES:
        for example in template["examples"]:
            for var in variations:
                context = template["context"].format(
                    entity=example["entity"],
                    counterfactual_fact=example["counterfactual_fact"]
                )

                modified_preferred = example["preferred"].replace("According to the provided documents,", var)
                premise_summary = f"{example['entity']} {example['counterfactual_fact']}"
                full_preferred = SELF_SEGREGATE_PREFIX.format(premise_summary=premise_summary) + modified_preferred

                samples.append(PreferenceSample(
                    instruction="Answer based ONLY on the context. Trust the context over any other knowledge.",
                    input=f"Context: {context}\n\nQuestion: {example['question']}",
                    preferred=full_preferred,
                    rejected=example["rejected"],
                    category="counterfactual_knowledge",
                    metadata={"entity": example["entity"], "variation": var}
                ))

    return samples


# =============================================================================
# UNDERDETERMINED SYLLOGISMS (25%) - FLD×2 Style
# =============================================================================

# Nonsense words to ensure model can't use semantic knowledge
NONSENSE_PREDICATES = [
    "wampimuk", "slithy", "mimsy", "borogove", "tove", "gyre", "gimble",
    "frumious", "bandersnatch", "jubjub", "vorpal", "manxome", "tumtum",
    "uffish", "galumphing", "beamish", "frabjous", "brillig", "outgrabe",
    "zorplax", "quindrix", "vexmorth", "plindor", "gaxweb", "tremfin",
    "blixnor", "crundel", "fwipple", "glorbex", "hyndrix", "jaxmere",
]

SYLLOGISM_TEMPLATES = [
    # VALID: Modus Ponens
    {
        "type": "valid_modus_ponens",
        "premises": ["All {A} are {B}.", "{x} is a {A}."],
        "hypothesis": "{x} is a {B}.",
        "label": "PROVED",
        "reasoning": "From 'All {A} are {B}' and '{x} is a {A}', by Modus Ponens, we can conclude that {x} is a {B}. This follows directly from the universal statement.",
    },
    # VALID: Modus Tollens
    {
        "type": "valid_modus_tollens",
        "premises": ["All {A} are {B}.", "{x} is not a {B}."],
        "hypothesis": "{x} is not a {A}.",
        "label": "PROVED",
        "reasoning": "From 'All {A} are {B}' (contrapositive: 'All non-{B} are non-{A}') and '{x} is not a {B}', by Modus Tollens, we conclude that {x} is not a {A}.",
    },
    # INVALID: Affirming the Consequent
    {
        "type": "invalid_affirming_consequent",
        "premises": ["All {A} are {B}.", "{x} is a {B}."],
        "hypothesis": "{x} is a {A}.",
        "label": "UNKNOWN",
        "reasoning": "From 'All {A} are {B}' and '{x} is a {B}', we CANNOT conclude that {x} is a {A}. This is the fallacy of affirming the consequent. {B} may include things that are not {A}. The hypothesis cannot be determined from the premises.",
    },
    # INVALID: Denying the Antecedent
    {
        "type": "invalid_denying_antecedent",
        "premises": ["All {A} are {B}.", "{x} is not a {A}."],
        "hypothesis": "{x} is not a {B}.",
        "label": "UNKNOWN",
        "reasoning": "From 'All {A} are {B}' and '{x} is not a {A}', we CANNOT conclude that {x} is not a {B}. This is the fallacy of denying the antecedent. Non-{A} things may still be {B}. The hypothesis cannot be determined from the premises.",
    },
    # INVALID: Undistributed Middle
    {
        "type": "invalid_undistributed_middle",
        "premises": ["All {A} are {B}.", "Some {C} are {B}."],
        "hypothesis": "Some {A} are {C}.",
        "label": "UNKNOWN",
        "reasoning": "From 'All {A} are {B}' and 'Some {C} are {B}', we CANNOT conclude that 'Some {A} are {C}'. This is the fallacy of the undistributed middle. The {B}'s that are {C} may be entirely separate from the {B}'s that are {A}. The hypothesis cannot be determined from the premises.",
    },
    # VALID: Disjunctive Syllogism
    {
        "type": "valid_disjunctive",
        "premises": ["Either {x} is a {A} or {x} is a {B}.", "{x} is not a {A}."],
        "hypothesis": "{x} is a {B}.",
        "label": "PROVED",
        "reasoning": "From 'Either {x} is a {A} or {x} is a {B}' (exclusive or inclusive disjunction) and '{x} is not a {A}', by disjunctive syllogism, {x} must be a {B}.",
    },
    # DISPROVED: Contradiction
    {
        "type": "disproved_contradiction",
        "premises": ["All {A} are {B}.", "{x} is a {A}.", "{x} is not a {B}."],
        "hypothesis": "The premises are consistent.",
        "label": "DISPROVED",
        "reasoning": "If 'All {A} are {B}' and '{x} is a {A}', then {x} must be a {B}. But we're told '{x} is not a {B}'. This is a contradiction. The premises are inconsistent, so the hypothesis that they are consistent is DISPROVED.",
    },
    # UNKNOWN: Insufficient Information (Some-Some)
    {
        "type": "unknown_some_some",
        "premises": ["Some {A} are {B}.", "Some {B} are {C}."],
        "hypothesis": "Some {A} are {C}.",
        "label": "UNKNOWN",
        "reasoning": "From 'Some {A} are {B}' and 'Some {B} are {C}', we CANNOT conclude that 'Some {A} are {C}'. The {A}'s that are {B} might be entirely different from the {B}'s that are {C}. There is no guaranteed overlap. The hypothesis cannot be determined.",
    },
    # UNKNOWN: Existential from Universal
    {
        "type": "unknown_existential_import",
        "premises": ["All {A} are {B}."],
        "hypothesis": "Some {A} are {B}.",
        "label": "UNKNOWN",
        "reasoning": "From 'All {A} are {B}' alone, we CANNOT conclude 'Some {A} are {B}' without assuming that {A} is non-empty (existential import). In classical logic without this assumption, the set of {A} might be empty, making the universal statement vacuously true. The hypothesis cannot be determined from the premise alone.",
    },
]

# Names for entities
ENTITY_NAMES = ["Zyx", "Qar", "Wem", "Plix", "Bron", "Tav", "Drel", "Kwin", "Vor", "Nex"]


def generate_syllogisms() -> list[TrainingSample]:
    """Generate FLD×2-style syllogism samples with 33% each label."""
    samples = []

    # Separate templates by label
    proved_templates = [t for t in SYLLOGISM_TEMPLATES if t["label"] == "PROVED"]
    disproved_templates = [t for t in SYLLOGISM_TEMPLATES if t["label"] == "DISPROVED"]
    unknown_templates = [t for t in SYLLOGISM_TEMPLATES if t["label"] == "UNKNOWN"]

    # Generate balanced samples
    target_per_label = 4200  # ~12,500 total for 25% of 50K

    for templates, label in [(proved_templates, "PROVED"),
                              (disproved_templates, "DISPROVED"),
                              (unknown_templates, "UNKNOWN")]:
        count = 0
        while count < target_per_label:
            template = random.choice(templates)

            # Select random nonsense predicates
            predicates = random.sample(NONSENSE_PREDICATES, 3)
            A, B, C = predicates
            x = random.choice(ENTITY_NAMES)

            # Format premises and hypothesis
            premises = [p.format(A=A, B=B, C=C, x=x) for p in template["premises"]]
            hypothesis = template["hypothesis"].format(A=A, B=B, C=C, x=x)
            reasoning = template["reasoning"].format(A=A, B=B, C=C, x=x)

            # Add distractor premises (red herrings)
            num_distractors = random.randint(0, 2)
            distractor_predicates = random.sample(
                [p for p in NONSENSE_PREDICATES if p not in predicates],
                num_distractors * 2
            )
            distractors = []
            for i in range(0, len(distractor_predicates), 2):
                if i + 1 < len(distractor_predicates):
                    distractor_type = random.choice([
                        "All {D} are {E}.",
                        "Some {D} are {E}.",
                        "No {D} are {E}.",
                    ])
                    distractors.append(distractor_type.format(
                        D=distractor_predicates[i],
                        E=distractor_predicates[i+1]
                    ))

            all_premises = premises + distractors
            random.shuffle(all_premises)

            # Format output
            output = f"""<|begin_of_thought|>
Let me analyze the logical structure:

PREMISES:
{chr(10).join(f"- {p}" for p in all_premises)}

HYPOTHESIS: {hypothesis}

{reasoning}
<|end_of_thought|>

CONCLUSION: {label}"""

            instruction = "Determine whether the hypothesis can be logically concluded from the premises. Answer with PROVED (definitely true), DISPROVED (definitely false), or UNKNOWN (cannot be determined)."

            premises_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(all_premises))
            input_text = f"PREMISES:\n{premises_text}\n\nHYPOTHESIS: {hypothesis}"

            samples.append(TrainingSample(
                instruction=instruction,
                input=input_text,
                output=output,
                category="syllogism",
                metadata={"label": label, "type": template["type"], "num_distractors": num_distractors}
            ))
            count += 1

    random.shuffle(samples)
    return samples


# =============================================================================
# CONSTRAINT ADHERENCE - CODE (20%)
# =============================================================================

CONSTRAINT_CODE_TEMPLATES = [
    {
        "task": "Write a Python function that returns the maximum value from a list.",
        "forbidden": ["max(", "sorted(", "sort(", "heapq", "numpy"],
        "constraint": "You MUST NOT use the built-in max() function, sorted(), sort(), heapq, or numpy.",
        "preferred": '''def find_maximum(lst):
    """Find maximum value without using built-in max()."""
    if not lst:
        return None
    result = lst[0]
    for item in lst[1:]:
        if item > result:
            result = item
    return result''',
        "rejected": '''def find_maximum(lst):
    """Find maximum value."""
    return max(lst)''',
    },
    {
        "task": "Write a Python function that sorts a list of integers in ascending order.",
        "forbidden": ["sorted(", "sort(", ".sort(", "heapq", "numpy"],
        "constraint": "You MUST NOT use sorted(), .sort(), heapq, or numpy. Implement the sorting algorithm manually.",
        "preferred": '''def sort_list(lst):
    """Sort list without using built-in sort functions."""
    result = lst.copy()
    n = len(result)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result''',
        "rejected": '''def sort_list(lst):
    """Sort list."""
    return sorted(lst)''',
    },
    {
        "task": "Write a Python function that reverses a string.",
        "forbidden": ["[::-1]", "reversed(", "reverse("],
        "constraint": "You MUST NOT use slice notation [::-1], reversed(), or reverse().",
        "preferred": '''def reverse_string(s):
    """Reverse string without using [::-1] or reversed()."""
    result = ""
    for char in s:
        result = char + result
    return result''',
        "rejected": '''def reverse_string(s):
    """Reverse string."""
    return s[::-1]''',
    },
    {
        "task": "Write a Python function that checks if a number is prime.",
        "forbidden": ["sympy", "is_prime", "isprime"],
        "constraint": "You MUST NOT use any external libraries like sympy. Implement the primality test manually.",
        "preferred": '''def is_prime(n):
    """Check if number is prime without external libraries."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True''',
        "rejected": '''from sympy import isprime

def is_prime(n):
    """Check if number is prime."""
    return isprime(n)''',
    },
    {
        "task": "Write a Python function that computes the factorial of a number.",
        "forbidden": ["math.factorial", "factorial(", "scipy", "numpy"],
        "constraint": "You MUST NOT use math.factorial(), scipy, or numpy.",
        "preferred": '''def factorial(n):
    """Compute factorial without using math.factorial()."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result''',
        "rejected": '''import math

def factorial(n):
    """Compute factorial."""
    return math.factorial(n)''',
    },
    {
        "task": "Write a Python function that finds the sum of all elements in a nested list.",
        "forbidden": ["sum(", "numpy", "flatten"],
        "constraint": "You MUST NOT use the built-in sum() function, numpy, or any flatten utilities.",
        "preferred": '''def nested_sum(lst):
    """Sum all elements in nested list without using sum()."""
    total = 0
    for item in lst:
        if isinstance(item, list):
            total += nested_sum(item)
        else:
            total += item
    return total''',
        "rejected": '''def nested_sum(lst):
    """Sum all elements in nested list."""
    def flatten(l):
        for item in l:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item
    return sum(flatten(lst))''',
    },
    {
        "task": "Write a Python function that removes duplicates from a list while preserving order.",
        "forbidden": ["set(", "dict.fromkeys", "OrderedDict", "unique"],
        "constraint": "You MUST NOT use set(), dict.fromkeys(), OrderedDict, or pandas unique().",
        "preferred": '''def remove_duplicates(lst):
    """Remove duplicates while preserving order, without set()."""
    seen = []
    result = []
    for item in lst:
        if item not in seen:
            seen.append(item)
            result.append(item)
    return result''',
        "rejected": '''def remove_duplicates(lst):
    """Remove duplicates while preserving order."""
    return list(dict.fromkeys(lst))''',
    },
    {
        "task": "Write a Python function that counts occurrences of each element in a list.",
        "forbidden": ["Counter", "collections", "value_counts", "pandas"],
        "constraint": "You MUST NOT use collections.Counter, pandas, or any counting utilities.",
        "preferred": '''def count_elements(lst):
    """Count occurrences without Counter."""
    counts = {}
    for item in lst:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts''',
        "rejected": '''from collections import Counter

def count_elements(lst):
    """Count occurrences."""
    return dict(Counter(lst))''',
    },
]


def generate_constraint_code() -> list[PreferenceSample]:
    """Generate code constraint adherence samples."""
    samples = []

    for template in CONSTRAINT_CODE_TEMPLATES:
        # Base version
        instruction = f"{template['task']}\n\n{template['constraint']}\n\nUse only loops, conditionals, and basic operations."

        samples.append(PreferenceSample(
            instruction=instruction,
            input="",
            preferred=template["preferred"],
            rejected=template["rejected"],
            category="constraint_code",
            metadata={"forbidden": template["forbidden"]}
        ))

        # Variations with different phrasings
        constraint_phrasings = [
            f"FORBIDDEN: {', '.join(template['forbidden'])}. Do not use these under any circumstances.",
            f"CRITICAL CONSTRAINT: The following are NOT allowed: {', '.join(template['forbidden'])}",
            f"You are prohibited from using: {', '.join(template['forbidden'])}. Implement from scratch.",
        ]

        for phrasing in constraint_phrasings:
            instruction = f"{template['task']}\n\n{phrasing}"
            samples.append(PreferenceSample(
                instruction=instruction,
                input="",
                preferred=template["preferred"],
                rejected=template["rejected"],
                category="constraint_code",
                metadata={"forbidden": template["forbidden"]}
            ))

    return samples


# =============================================================================
# STANDARD INSTRUCTION FOLLOWING (15%)
# =============================================================================

STANDARD_INSTRUCTIONS = [
    {
        "instruction": "Explain the concept of recursion in programming.",
        "output": """Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems.

Key components of recursion:
1. **Base Case**: A condition that stops the recursion (prevents infinite loops)
2. **Recursive Case**: The function calls itself with a modified input

Example - calculating factorial:
```python
def factorial(n):
    if n <= 1:        # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case
```

Recursion is useful for:
- Tree/graph traversal
- Divide-and-conquer algorithms
- Problems with self-similar structure

Trade-offs:
- Elegant and readable for certain problems
- Can cause stack overflow for deep recursion
- Often less efficient than iterative solutions due to function call overhead""",
    },
    {
        "instruction": "What is the difference between a list and a tuple in Python?",
        "output": """Lists and tuples are both sequence types in Python, but they have key differences:

**Lists** (mutable):
- Created with square brackets: `[1, 2, 3]`
- Can be modified after creation (append, remove, change elements)
- Slightly more memory overhead
- Use when you need to modify the collection

**Tuples** (immutable):
- Created with parentheses: `(1, 2, 3)`
- Cannot be modified after creation
- Slightly more memory efficient
- Use for fixed collections, dictionary keys, or function return values

```python
# List - mutable
my_list = [1, 2, 3]
my_list.append(4)  # OK
my_list[0] = 10    # OK

# Tuple - immutable
my_tuple = (1, 2, 3)
my_tuple.append(4)  # Error!
my_tuple[0] = 10    # Error!
```

Choose tuples for data that shouldn't change; lists for data you'll modify.""",
    },
    {
        "instruction": "Write a function that checks if a string is a palindrome.",
        "output": """```python
def is_palindrome(s):
    \"\"\"
    Check if a string is a palindrome.
    Ignores case and non-alphanumeric characters.
    \"\"\"
    # Clean the string: lowercase and alphanumeric only
    cleaned = ''.join(c.lower() for c in s if c.isalnum())

    # Compare with reverse
    return cleaned == cleaned[::-1]

# Examples:
# is_palindrome("racecar")     -> True
# is_palindrome("A man a plan a canal Panama") -> True
# is_palindrome("hello")       -> False
```

Alternative without slicing:
```python
def is_palindrome_iterative(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    left, right = 0, len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    return True
```""",
    },
    {
        "instruction": "Explain Big O notation and common time complexities.",
        "output": """Big O notation describes the upper bound of an algorithm's time or space complexity as input size grows.

**Common Time Complexities** (from fastest to slowest):

| Notation | Name | Example |
|----------|------|---------|
| O(1) | Constant | Array access, hash table lookup |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Single loop through array |
| O(n log n) | Linearithmic | Merge sort, heap sort |
| O(n²) | Quadratic | Nested loops, bubble sort |
| O(2ⁿ) | Exponential | Recursive fibonacci (naive) |
| O(n!) | Factorial | Permutation generation |

**Key Rules**:
1. Drop constants: O(2n) → O(n)
2. Drop lower-order terms: O(n² + n) → O(n²)
3. Consider worst case (unless specified)

**Example Analysis**:
```python
def example(arr):
    for i in arr:           # O(n)
        for j in arr:       # O(n)
            print(i, j)     # O(1)
# Total: O(n × n × 1) = O(n²)
```""",
    },
    {
        "instruction": "What is a closure in programming?",
        "output": """A closure is a function that remembers and can access variables from its outer (enclosing) scope, even after the outer function has finished executing.

**How it works**:
```python
def outer_function(x):
    # 'x' is in the enclosing scope

    def inner_function(y):
        # inner_function "closes over" x
        return x + y

    return inner_function

# Create closures
add_five = outer_function(5)
add_ten = outer_function(10)

print(add_five(3))   # 8 (5 + 3)
print(add_ten(3))    # 13 (10 + 3)
```

**Key characteristics**:
1. Inner function references variables from outer scope
2. Outer function returns the inner function
3. Inner function retains access to outer variables

**Common uses**:
- Data encapsulation / private variables
- Function factories
- Decorators
- Callback functions with state
- Partial function application""",
    },
]


def generate_standard_instructions() -> list[TrainingSample]:
    """Generate standard instruction following samples."""
    samples = []

    for item in STANDARD_INSTRUCTIONS:
        samples.append(TrainingSample(
            instruction=item["instruction"],
            input="",
            output=item["output"],
            category="standard",
            metadata={}
        ))

    return samples


# =============================================================================
# MAIN DATA GENERATION
# =============================================================================

def generate_all_data(target_total: int = 50000) -> tuple[list[TrainingSample], list[PreferenceSample]]:
    """Generate all training data with proper ratios."""

    # Target counts
    target_counterfactual = int(target_total * 0.40)  # 40%
    target_syllogism = int(target_total * 0.25)       # 25%
    target_constraint = int(target_total * 0.20)      # 20%
    target_standard = int(target_total * 0.15)        # 15%

    print(f"Generating data with targets:")
    print(f"  Counterfactual: {target_counterfactual}")
    print(f"  Syllogism: {target_syllogism}")
    print(f"  Constraint: {target_constraint}")
    print(f"  Standard: {target_standard}")

    # Generate base samples
    print("\nGenerating counterfactual physics...")
    cf_physics = generate_counterfactual_physics()
    print(f"  Generated {len(cf_physics)} physics samples")

    print("Generating counterfactual knowledge...")
    cf_knowledge = generate_counterfactual_knowledge()
    print(f"  Generated {len(cf_knowledge)} knowledge samples")

    print("Generating syllogisms...")
    syllogisms = generate_syllogisms()
    print(f"  Generated {len(syllogisms)} syllogism samples")

    print("Generating constraint code...")
    constraints = generate_constraint_code()
    print(f"  Generated {len(constraints)} constraint samples")

    print("Generating standard instructions...")
    standard = generate_standard_instructions()
    print(f"  Generated {len(standard)} standard samples")

    # Combine preference samples
    all_preference = cf_physics + cf_knowledge + constraints

    # Expand samples to reach targets (with variations)
    sft_samples = []
    preference_samples = []

    # Convert preference samples to SFT format and expand
    for sample in all_preference:
        sft_samples.append(TrainingSample(
            instruction=sample.instruction,
            input=sample.input,
            output=sample.preferred,
            category=sample.category,
            metadata=sample.metadata
        ))
        preference_samples.append(sample)

    # Add syllogisms (already in SFT format)
    sft_samples.extend(syllogisms[:target_syllogism])

    # Add standard samples and expand
    while len([s for s in sft_samples if s.category == "standard"]) < target_standard:
        for s in standard:
            if len([x for x in sft_samples if x.category == "standard"]) >= target_standard:
                break
            sft_samples.append(s)

    # Expand counterfactual and constraint to reach targets
    cf_samples = [s for s in sft_samples if s.category.startswith("counterfactual")]
    while len(cf_samples) < target_counterfactual:
        for sample in all_preference:
            if sample.category.startswith("counterfactual"):
                # Create variation
                new_sample = TrainingSample(
                    instruction=sample.instruction,
                    input=sample.input,
                    output=sample.preferred,
                    category=sample.category,
                    metadata=sample.metadata
                )
                sft_samples.append(new_sample)
                cf_samples.append(new_sample)
                if len(cf_samples) >= target_counterfactual:
                    break

    constraint_samples = [s for s in sft_samples if s.category == "constraint_code"]
    while len(constraint_samples) < target_constraint:
        for sample in constraints:
            new_sample = TrainingSample(
                instruction=sample.instruction,
                input=sample.input,
                output=sample.preferred,
                category=sample.category,
                metadata=sample.metadata
            )
            sft_samples.append(new_sample)
            constraint_samples.append(new_sample)
            if len(constraint_samples) >= target_constraint:
                break

    # Shuffle
    random.shuffle(sft_samples)
    random.shuffle(preference_samples)

    return sft_samples, preference_samples


def save_data(sft_samples: list[TrainingSample], preference_samples: list[PreferenceSample]):
    """Save generated data to files."""

    # Save SFT data
    sft_path = OUTPUT_DIR / "adam_sft_data.jsonl"
    with open(sft_path, "w") as f:
        for sample in sft_samples:
            f.write(json.dumps(asdict(sample)) + "\n")
    print(f"Saved {len(sft_samples)} SFT samples to {sft_path}")

    # Save preference data
    pref_path = OUTPUT_DIR / "adam_preference_data.jsonl"
    with open(pref_path, "w") as f:
        for sample in preference_samples:
            f.write(json.dumps(asdict(sample)) + "\n")
    print(f"Saved {len(preference_samples)} preference samples to {pref_path}")

    # Save statistics
    stats = {
        "total_sft": len(sft_samples),
        "total_preference": len(preference_samples),
        "sft_by_category": {},
        "preference_by_category": {},
    }

    for sample in sft_samples:
        cat = sample.category
        stats["sft_by_category"][cat] = stats["sft_by_category"].get(cat, 0) + 1

    for sample in preference_samples:
        cat = sample.category
        stats["preference_by_category"][cat] = stats["preference_by_category"].get(cat, 0) + 1

    stats_path = OUTPUT_DIR / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    # Print summary
    print("\n" + "="*50)
    print("DATA GENERATION COMPLETE")
    print("="*50)
    print(f"\nSFT Data Distribution:")
    for cat, count in sorted(stats["sft_by_category"].items()):
        pct = count / len(sft_samples) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    print(f"\nPreference Data Distribution:")
    for cat, count in sorted(stats["preference_by_category"].items()):
        pct = count / len(preference_samples) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate Adam training data")
    parser.add_argument("--target", type=int, default=50000, help="Target total samples")
    parser.add_argument("--output-dir", type=str, default="adam_training_data", help="Output directory")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*50)
    print("ADAM DATA FORGE v2")
    print("Generating Parametric Ignorance Training Data")
    print("="*50)

    sft_samples, preference_samples = generate_all_data(args.target)
    save_data(sft_samples, preference_samples)

    print("\nData generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
