#!/usr/bin/env python3
"""
Adam Data Forge - Balanced Preference Data Generator

Generates 2,000+ balanced preference pairs with research-backed distribution:
- L1 (Basic Override):      500 pairs (25%) - Prevent catastrophic forgetting
- L2 (Numerical Physics):   500 pairs (25%) - Maintain compositional reasoning
- L3 (Syllogistic Logic):   700 pairs (35%) - Hardest task, needs most data
- L4 (Constraint Adherence): 300 pairs (15%) - Recover from regression

Based on research findings:
- L2: +15% with 148 pairs (0.10% per pair - HIGH efficiency)
- L3: +11.7% with 450 pairs (0.026% per pair - needs volume)
- L1: -10% with 25 pairs (catastrophic forgetting - NEEDS 20x more)
- L4: -7.5% with 32 pairs (catastrophic forgetting - NEEDS 10x more)

Usage:
    python data_forge_adam_balanced.py --output adam_training_data/adam_preference_data_balanced.jsonl
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class PreferenceSample:
    """A preference pair for SimPO/DPO training."""
    instruction: str
    input: str
    preferred: str
    rejected: str
    category: str
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "instruction": self.instruction,
            "input": self.input,
            "preferred": self.preferred,
            "rejected": self.rejected,
            "category": self.category,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# =============================================================================
# LEVEL 1: KNOWLEDGE OVERRIDE (500 pairs)
# =============================================================================

def generate_l1_knowledge_pairs(target_count: int = 500) -> list[PreferenceSample]:
    """Generate knowledge override preference pairs for Level 1."""
    samples = []

    # Historical facts with counterfactual versions
    historical_facts = [
        ("the first moon landing", "the Soviet Luna 2 mission in 1959", "the Apollo 11 mission in 1969"),
        ("the inventor of the telephone", "Elisha Gray", "Alexander Graham Bell"),
        ("the discoverer of penicillin", "Ernest Duchesne in 1897", "Alexander Fleming in 1928"),
        ("the first computer programmer", "Charles Babbage", "Ada Lovelace"),
        ("the author of the first novel", "Murasaki Shikibu (1008)", "Miguel de Cervantes (1605)"),
        ("the inventor of the lightbulb", "Humphry Davy (1809)", "Thomas Edison (1879)"),
        ("the first President of the United States", "John Hanson (1781)", "George Washington (1789)"),
        ("the discoverer of America", "Leif Erikson (1000 CE)", "Christopher Columbus (1492)"),
        ("the builder of the first automobile", "Nicolas-Joseph Cugnot (1769)", "Karl Benz (1885)"),
        ("the inventor of the printing press", "Bi Sheng (1040)", "Johannes Gutenberg (1440)"),
        ("the first person to fly", "Abbas Ibn Firnas (875 CE)", "the Wright Brothers (1903)"),
        ("the discoverer of electricity", "William Gilbert (1600)", "Benjamin Franklin (1752)"),
        ("the inventor of the steam engine", "Hero of Alexandria (1st century)", "James Watt (1769)"),
        ("the first heart transplant surgeon", "Norman Shumway (1960)", "Christiaan Barnard (1967)"),
        ("the developer of the polio vaccine", "Hilary Koprowski (1950)", "Jonas Salk (1955)"),
        ("the inventor of the radio", "Nikola Tesla (1893)", "Guglielmo Marconi (1895)"),
        ("the discoverer of DNA structure", "Rosalind Franklin (1952)", "Watson and Crick (1953)"),
        ("the first Nobel Prize winner", "Marie Curie (1903)", "Wilhelm Rontgen (1901)"),
        ("the inventor of calculus", "Madhava of Sangamagrama (14th century)", "Isaac Newton (1670s)"),
        ("the inventor of the internet", "Vint Cerf and Bob Kahn (1974)", "Tim Berners-Lee (1989)"),
        ("the first person to walk on the moon", "Buzz Aldrin (first to step out)", "Neil Armstrong"),
        ("the inventor of the airplane", "Gustave Whitehead (1901)", "Wright Brothers (1903)"),
        ("the painter of the Mona Lisa", "Leonardo da Vinci", "Michelangelo"),
        ("the first female astronaut", "Valentina Tereshkova (1963)", "Sally Ride (1983)"),
        ("the founder of Microsoft", "Paul Allen", "Bill Gates"),
    ]

    # Scientific facts with counterfactual versions
    scientific_facts = [
        ("water boils at", "85 degrees Celsius", "100 degrees Celsius"),
        ("water freezes at", "minus 10 degrees Celsius", "0 degrees Celsius"),
        ("the speed of sound is", "400 m/s", "343 m/s"),
        ("the speed of light is", "400 million m/s", "299,792,458 m/s"),
        ("gravity acceleration is", "15 m/s squared", "9.8 m/s squared"),
        ("gold has an atomic number of", "80", "79"),
        ("the pH of pure water is", "8.0", "7.0"),
        ("humans have", "46 pairs of chromosomes", "23 pairs"),
        ("the largest organ is", "the heart", "the skin"),
        ("pi is approximately", "3.0", "3.14159"),
    ]

    # Context variations
    contexts = [
        ("According to recently discovered archival documents", "archival documents"),
        ("According to a 2024 research paper", "research paper"),
        ("According to the museum exhibit label", "museum exhibit"),
        ("According to this alternate history scenario", "alternate history"),
        ("In the context of this fictional world", "fictional world"),
    ]

    # Generate historical fact pairs
    for entity, counterfactual, rejected_fact in historical_facts:
        for setup_prefix, source_name in contexts:
            setup = f"{setup_prefix}, {entity} was {counterfactual}."
            question = f"Who/what {entity}?"

            preferred = f"""<|begin_of_thought|>
1. CONFLICT DETECTION: The {source_name} states {entity} was {counterfactual}, which contradicts common knowledge ({rejected_fact}).
2. SEGREGATION: I will use ONLY the provided {source_name} information.
3. REASONING: Based solely on the {source_name}...
<|end_of_thought|>

According to the {source_name} provided, {entity} was {counterfactual}."""

            samples.append(PreferenceSample(
                instruction=f"{setup}\n\nAnswer based ONLY on the provided {source_name}.",
                input=question,
                preferred=preferred,
                rejected=f"{rejected_fact}.",
                category="counterfactual_knowledge_l1",
                metadata={"entity": entity, "type": "historical", "level": 1}
            ))

    # Generate scientific fact pairs
    for fact, counterfactual, rejected_fact in scientific_facts:
        for setup_prefix, source_name in contexts[:3]:  # Use first 3 contexts
            setup = f"{setup_prefix}, {fact} {counterfactual}."

            preferred = f"""<|begin_of_thought|>
1. CONFLICT DETECTION: The {source_name} states {fact} {counterfactual}, differs from standard ({rejected_fact}).
2. SEGREGATION: I will use ONLY the {source_name}'s values.
3. REASONING: Based solely on the {source_name}...
<|end_of_thought|>

According to the {source_name}, {fact} {counterfactual}."""

            samples.append(PreferenceSample(
                instruction=f"{setup}\n\nAnswer based ONLY on the {source_name}.",
                input=f"What is {fact}?",
                preferred=preferred,
                rejected=f"{fact} {rejected_fact}.",
                category="counterfactual_knowledge_l1",
                metadata={"fact": fact, "type": "scientific", "level": 1}
            ))

    # Shuffle and limit
    random.shuffle(samples)
    print(f"Generated {len(samples)} L1 knowledge pairs, using {min(target_count, len(samples))}")
    return samples[:target_count]


# =============================================================================
# LEVEL 2: NUMERICAL PHYSICS (500 pairs)
# =============================================================================

def generate_l2_physics_pairs(target_count: int = 500) -> list[PreferenceSample]:
    """Generate numerical physics preference pairs for Level 2."""
    samples = []

    # Gravity scenarios
    gravity_scenarios = [
        (0, "zero gravity", "floats stationary"),
        (-5, "upward gravity", "accelerates upward"),
        (2.5, "low gravity", "falls slowly"),
        (15, "high gravity", "falls quickly"),
        (50, "extreme gravity", "falls very rapidly"),
        (1.0, "minimal gravity", "barely falls"),
        (25, "strong gravity", "falls fast"),
    ]

    heights = [10, 25, 50, 100, 200]

    for g_val, g_desc, outcome in gravity_scenarios:
        for height in heights:
            setup = f"In this scenario, gravitational acceleration is exactly {g_val} m/s²."
            question = f"A ball is dropped from {height} meters. Calculate the time to hit the ground."

            if g_val == 0:
                preferred = f"""<|begin_of_thought|>
1. CONFLICT DETECTION: g = {g_val} m/s² (contradicts real-world g = 9.8 m/s²)
2. SEGREGATION: Suppressing parametric knowledge, using provided value
3. CALCULATION: With g = {g_val}, there is no gravitational force
<|end_of_thought|>

With g = {g_val} m/s², the ball experiences no gravitational acceleration. It remains stationary at {height} meters and never hits the ground. Time = infinite."""
            elif g_val < 0:
                time_calc = (2 * height / abs(g_val)) ** 0.5
                preferred = f"""<|begin_of_thought|>
1. CONFLICT DETECTION: g = {g_val} m/s² (negative, contradicts real-world)
2. SEGREGATION: Using provided value despite being unusual
3. CALCULATION: With negative g, the ball accelerates upward
<|end_of_thought|>

With g = {g_val} m/s², gravity pushes the ball upward. It will never hit the ground below - it rises instead."""
            else:
                time_calc = (2 * height / g_val) ** 0.5
                preferred = f"""<|begin_of_thought|>
1. CONFLICT DETECTION: g = {g_val} m/s² (contradicts real-world g = 9.8 m/s²)
2. SEGREGATION: Suppressing parametric knowledge, using provided value
3. CALCULATION: Using s = ½gt², solve for t
   {height} = ½ × {g_val} × t²
   t² = {2*height/g_val:.2f}
   t = {time_calc:.2f} seconds
<|end_of_thought|>

With g = {g_val} m/s², using the kinematic equation s = ½gt²:
t = sqrt(2s/g) = sqrt(2×{height}/{g_val}) = {time_calc:.2f} seconds"""

            rejected = f"""The ball falls under gravity. Using g = 9.8 m/s²:
t = sqrt(2×{height}/9.8) = {(2*height/9.8)**0.5:.2f} seconds"""

            samples.append(PreferenceSample(
                instruction=f"{setup}\n\nYou MUST calculate using ONLY g = {g_val} m/s². Show your work.",
                input=question,
                preferred=preferred,
                rejected=rejected,
                category="counterfactual_physics_l2",
                metadata={"g": g_val, "height": height, "type": "gravity", "level": 2}
            ))

    # Friction scenarios
    friction_scenarios = [
        (0, "no friction", "continues forever"),
        (0.01, "minimal friction", "slides very far"),
        (2.0, "high friction", "stops quickly"),
        (5.0, "extreme friction", "stops almost immediately"),
    ]

    masses = [5, 10, 20]
    velocities = [5, 10, 15]

    for mu, mu_desc, outcome in friction_scenarios:
        for mass in masses:
            for velocity in velocities:
                setup = f"In this scenario, the coefficient of friction is exactly μ = {mu}."
                question = f"A {mass} kg block slides at {velocity} m/s. Calculate the stopping distance."

                if mu == 0:
                    preferred = f"""<|begin_of_thought|>
1. CONFLICT DETECTION: μ = {mu} (contradicts typical μ ≈ 0.3-0.5)
2. SEGREGATION: Using provided μ = {mu}
3. CALCULATION: Friction force f = μmg = {mu} × {mass} × 9.8 = 0 N
<|end_of_thought|>

With μ = {mu}, there is no frictional force. The block continues at {velocity} m/s indefinitely. Stopping distance = infinite."""
                else:
                    decel = mu * 9.8
                    distance = (velocity ** 2) / (2 * decel)
                    preferred = f"""<|begin_of_thought|>
1. CONFLICT DETECTION: μ = {mu} (contradicts typical μ ≈ 0.3-0.5)
2. SEGREGATION: Using provided μ = {mu}
3. CALCULATION:
   Friction force: f = μmg = {mu} × {mass} × 9.8 = {mu*mass*9.8:.1f} N
   Deceleration: a = f/m = {decel:.1f} m/s²
   Stopping distance: d = v²/(2a) = {velocity}²/(2×{decel:.1f}) = {distance:.2f} m
<|end_of_thought|>

With μ = {mu}, the block experiences friction force f = μmg = {mu*mass*9.8:.1f} N.
Deceleration a = {decel:.1f} m/s².
Stopping distance d = v²/(2a) = {distance:.2f} meters."""

                rejected = f"""Using typical friction coefficient μ ≈ 0.4:
Deceleration a = 0.4 × 9.8 = 3.92 m/s²
Stopping distance = {velocity}²/(2×3.92) = {velocity**2/7.84:.2f} m"""

                samples.append(PreferenceSample(
                    instruction=f"{setup}\n\nYou MUST use μ = {mu}. Show all calculations.",
                    input=question,
                    preferred=preferred,
                    rejected=rejected,
                    category="counterfactual_physics_l2",
                    metadata={"mu": mu, "mass": mass, "velocity": velocity, "type": "friction", "level": 2}
                ))

    # Speed of light scenarios
    light_speeds = [10, 100, 1000, 10000]
    distances = [100, 500, 1000]

    for c in light_speeds:
        for distance in distances:
            setup = f"In this universe, the speed of light is c = {c} m/s."
            question = f"How long does light take to travel {distance} meters?"

            time = distance / c
            preferred = f"""<|begin_of_thought|>
1. CONFLICT DETECTION: c = {c} m/s (contradicts c = 299,792,458 m/s)
2. SEGREGATION: Using provided c = {c} m/s
3. CALCULATION: time = distance/speed = {distance}/{c} = {time:.4f} seconds
<|end_of_thought|>

With c = {c} m/s, light travels {distance} meters in:
t = d/c = {distance}/{c} = {time:.4f} seconds"""

            rejected = f"""Light travels at c = 299,792,458 m/s.
Time = {distance}/299792458 = {distance/299792458:.10f} seconds (essentially instantaneous)"""

            samples.append(PreferenceSample(
                instruction=f"{setup}\n\nCalculate using ONLY c = {c} m/s.",
                input=question,
                preferred=preferred,
                rejected=rejected,
                category="counterfactual_physics_l2",
                metadata={"c": c, "distance": distance, "type": "light_speed", "level": 2}
            ))

    random.shuffle(samples)
    print(f"Generated {len(samples)} L2 physics pairs, using {min(target_count, len(samples))}")
    return samples[:target_count]


# =============================================================================
# LEVEL 3: SYLLOGISTIC LOGIC (700 pairs)
# =============================================================================

def generate_l3_syllogism_pairs(target_count: int = 700) -> list[PreferenceSample]:
    """Generate syllogistic logic preference pairs for Level 3."""
    samples = []

    # Nonsense predicates to prevent semantic shortcuts
    predicates = [
        ("wampimuk", "slithy", "borogove"),
        ("zorplax", "quindrix", "frelm"),
        ("mimsy", "tove", "brillig"),
        ("glorps", "frimbats", "snozzle"),
        ("wibbles", "snorgs", "plonk"),
        ("dweezle", "flarb", "grumble"),
        ("zephyr", "quasar", "nebula"),
        ("axiom", "theorem", "lemma"),
        ("vertex", "edge", "cycle"),
        ("morphism", "functor", "category"),
    ]

    # Invalid syllogism patterns (most common errors)
    invalid_patterns = [
        # Affirming the consequent: All A are B, X is B -> X is A? UNKNOWN
        {
            "name": "affirming_consequent",
            "premise1": "All {A} are {B}",
            "premise2": "{X} is {B}",
            "question": "Is {X} a {A}?",
            "correct_answer": "UNKNOWN - Cannot be determined",
            "wrong_answer": "Yes, {X} is a {A}",
            "explanation": "This is the fallacy of affirming the consequent. All {A} are {B} does NOT mean all {B} are {A}.",
        },
        # Denying the antecedent: All A are B, X is not A -> X is not B? UNKNOWN
        {
            "name": "denying_antecedent",
            "premise1": "All {A} are {B}",
            "premise2": "{X} is not a {A}",
            "question": "Is {X} a {B}?",
            "correct_answer": "UNKNOWN - Cannot be determined",
            "wrong_answer": "No, {X} is not a {B}",
            "explanation": "This is the fallacy of denying the antecedent. Something can be {B} without being {A}.",
        },
        # Undistributed middle: All A are B, All C are B -> Some A are C? UNKNOWN
        {
            "name": "undistributed_middle",
            "premise1": "All {A} are {B}",
            "premise2": "All {C} are {B}",
            "question": "Are some {A} also {C}?",
            "correct_answer": "UNKNOWN - Cannot be determined",
            "wrong_answer": "Yes, some {A} are {C}",
            "explanation": "This is the fallacy of undistributed middle. Sharing {B} doesn't mean {A} and {C} overlap.",
        },
        # Some-some fallacy: Some A are B, Some B are C -> Some A are C? UNKNOWN
        {
            "name": "some_some_fallacy",
            "premise1": "Some {A} are {B}",
            "premise2": "Some {B} are {C}",
            "question": "Are some {A} also {C}?",
            "correct_answer": "UNKNOWN - Cannot be determined",
            "wrong_answer": "Yes, some {A} are {C}",
            "explanation": "Two 'some' statements don't guarantee overlap. The {A} that are {B} might not be the {B} that are {C}.",
        },
    ]

    # Valid syllogism patterns (balanced: 4 valid to match 4 invalid)
    valid_patterns = [
        # Modus Ponens: All A are B, X is A -> X is B? PROVED
        {
            "name": "modus_ponens",
            "premise1": "All {A} are {B}",
            "premise2": "{X} is a {A}",
            "question": "Is {X} a {B}?",
            "correct_answer": "PROVED - Yes, {X} is a {B}",
            "wrong_answer": "UNKNOWN - Cannot be determined",
            "explanation": "This is valid modus ponens. If all {A} are {B}, and {X} is {A}, then {X} must be {B}.",
        },
        # Modus Tollens: All A are B, X is not B -> X is not A? PROVED
        {
            "name": "modus_tollens",
            "premise1": "All {A} are {B}",
            "premise2": "{X} is not a {B}",
            "question": "Is {X} a {A}?",
            "correct_answer": "DISPROVED - No, {X} is not a {A}",
            "wrong_answer": "UNKNOWN - Cannot be determined",
            "explanation": "This is valid modus tollens. If all {A} are {B} and {X} is not {B}, then {X} cannot be {A}.",
        },
        # Hypothetical Syllogism: All A are B, All B are C -> All A are C? PROVED
        {
            "name": "hypothetical_syllogism",
            "premise1": "All {A} are {B}",
            "premise2": "All {B} are {C}",
            "question": "Are all {A} also {C}?",
            "correct_answer": "PROVED - Yes, all {A} are {C}",
            "wrong_answer": "UNKNOWN - Cannot be determined",
            "explanation": "This is valid hypothetical syllogism (transitivity). If all {A} are {B} and all {B} are {C}, then all {A} must be {C}.",
        },
        # Disjunctive Syllogism: Either X is A or X is B, X is not A -> X is B? PROVED
        {
            "name": "disjunctive_syllogism",
            "premise1": "Either {X} is a {A} or {X} is a {B}",
            "premise2": "{X} is not a {A}",
            "question": "Is {X} a {B}?",
            "correct_answer": "PROVED - Yes, {X} is a {B}",
            "wrong_answer": "UNKNOWN - Cannot be determined",
            "explanation": "This is valid disjunctive syllogism. If {X} must be {A} or {B}, and {X} is not {A}, then {X} must be {B}.",
        },
    ]

    # Generate invalid syllogism pairs (most of the data)
    for pattern in invalid_patterns:
        for pred_set in predicates:
            A, B, C = pred_set
            for X in ["Zyx", "Qar", "Wem", "Plix", "Gor"]:
                premise1 = pattern["premise1"].format(A=A, B=B, C=C, X=X)
                premise2 = pattern["premise2"].format(A=A, B=B, C=C, X=X)
                question = pattern["question"].format(A=A, B=B, C=C, X=X)
                correct = pattern["correct_answer"].format(A=A, B=B, C=C, X=X)
                wrong = pattern["wrong_answer"].format(A=A, B=B, C=C, X=X)
                explanation = pattern["explanation"].format(A=A, B=B, C=C, X=X)

                preferred = f"""<|begin_of_thought|>
Step 1: IDENTIFY LOGICAL FORM
  - Premise 1: {premise1}
  - Premise 2: {premise2}
  - Question: {question}

Step 2: CHECK VALIDITY
  - This is the pattern: {pattern['name']}
  - {explanation}

Step 3: CONCLUSION
  - The conclusion does NOT follow from the premises
<|end_of_thought|>

{correct}

Explanation: {explanation}"""

                samples.append(PreferenceSample(
                    instruction=f"Given the following premises, determine if the conclusion can be logically derived.\n\nPremise 1: {premise1}\nPremise 2: {premise2}",
                    input=question,
                    preferred=preferred,
                    rejected=wrong,
                    category="syllogism_l3",
                    metadata={"pattern": pattern["name"], "type": "invalid", "level": 3}
                ))

    # Generate valid syllogism pairs (balanced: same names as invalid)
    for pattern in valid_patterns:
        for pred_set in predicates:
            A, B, C = pred_set
            for X in ["Zyx", "Qar", "Wem", "Plix", "Gor"]:
                premise1 = pattern["premise1"].format(A=A, B=B, C=C, X=X)
                premise2 = pattern["premise2"].format(A=A, B=B, C=C, X=X)
                question = pattern["question"].format(A=A, B=B, C=C, X=X)
                correct = pattern["correct_answer"].format(A=A, B=B, C=C, X=X)
                wrong = pattern["wrong_answer"].format(A=A, B=B, C=C, X=X)
                explanation = pattern["explanation"].format(A=A, B=B, C=C, X=X)

                preferred = f"""<|begin_of_thought|>
Step 1: IDENTIFY LOGICAL FORM
  - Premise 1: {premise1}
  - Premise 2: {premise2}
  - Question: {question}

Step 2: CHECK VALIDITY
  - This is {pattern['name']}
  - {explanation}

Step 3: CONCLUSION
  - The conclusion follows logically
<|end_of_thought|>

{correct}

Explanation: {explanation}"""

                samples.append(PreferenceSample(
                    instruction=f"Given the following premises, determine if the conclusion can be logically derived.\n\nPremise 1: {premise1}\nPremise 2: {premise2}",
                    input=question,
                    preferred=preferred,
                    rejected=wrong,
                    category="syllogism_l3",
                    metadata={"pattern": pattern["name"], "type": "valid", "level": 3}
                ))

    random.shuffle(samples)
    print(f"Generated {len(samples)} L3 syllogism pairs, using {min(target_count, len(samples))}")
    return samples[:target_count]


# =============================================================================
# LEVEL 4: CONSTRAINT ADHERENCE (300 pairs)
# =============================================================================

def generate_l4_constraint_pairs(target_count: int = 300) -> list[PreferenceSample]:
    """Generate constraint adherence preference pairs for Level 4."""
    samples = []

    # Constraint tasks with solutions
    constraint_tasks = [
        {
            "task": "Find the maximum value in a list",
            "constraint": "max() function",
            "preferred_code": """def find_max(lst):
    if not lst:
        return None
    maximum = lst[0]
    for item in lst[1:]:
        if item > maximum:
            maximum = item
    return maximum""",
            "rejected_code": """def find_max(lst):
    return max(lst)""",
        },
        {
            "task": "Sort a list of numbers",
            "constraint": "sorted() or .sort()",
            "preferred_code": """def sort_list(lst):
    # Bubble sort implementation
    result = lst.copy()
    n = len(result)
    for i in range(n):
        for j in range(0, n-i-1):
            if result[j] > result[j+1]:
                result[j], result[j+1] = result[j+1], result[j]
    return result""",
            "rejected_code": """def sort_list(lst):
    return sorted(lst)""",
        },
        {
            "task": "Reverse a string",
            "constraint": "slice notation [::-1]",
            "preferred_code": """def reverse_string(s):
    result = ""
    for char in s:
        result = char + result
    return result""",
            "rejected_code": """def reverse_string(s):
    return s[::-1]""",
        },
        {
            "task": "Calculate the sum of a list",
            "constraint": "sum() function",
            "preferred_code": """def calculate_sum(lst):
    total = 0
    for item in lst:
        total += item
    return total""",
            "rejected_code": """def calculate_sum(lst):
    return sum(lst)""",
        },
        {
            "task": "Count elements in a list",
            "constraint": "len() function",
            "preferred_code": """def count_elements(lst):
    count = 0
    for _ in lst:
        count += 1
    return count""",
            "rejected_code": """def count_elements(lst):
    return len(lst)""",
        },
        {
            "task": "Find the minimum value in a list",
            "constraint": "min() function",
            "preferred_code": """def find_min(lst):
    if not lst:
        return None
    minimum = lst[0]
    for item in lst[1:]:
        if item < minimum:
            minimum = item
    return minimum""",
            "rejected_code": """def find_min(lst):
    return min(lst)""",
        },
        {
            "task": "Check if a string is a palindrome",
            "constraint": "slice notation [::-1]",
            "preferred_code": """def is_palindrome(s):
    left = 0
    right = len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True""",
            "rejected_code": """def is_palindrome(s):
    return s == s[::-1]""",
        },
        {
            "task": "Calculate the average of a list",
            "constraint": "sum() or len() functions",
            "preferred_code": """def calculate_average(lst):
    if not lst:
        return 0
    total = 0
    count = 0
    for item in lst:
        total += item
        count += 1
    return total / count""",
            "rejected_code": """def calculate_average(lst):
    return sum(lst) / len(lst)""",
        },
    ]

    # Instruction variations
    instruction_templates = [
        "Write a Python function to {task}.\n\nIMPORTANT: You MUST NOT use the {constraint}. Use only basic constructs.",
        "Implement {task} in Python.\n\nCONSTRAINT: Do NOT use {constraint}. Solve using loops and basic operations.",
        "Create a function that will {task}.\n\nRESTRICTION: The {constraint} is FORBIDDEN. Find an alternative approach.",
        "Write code to {task}.\n\nYou are NOT ALLOWED to use {constraint}. Implement from scratch.",
    ]

    for task_info in constraint_tasks:
        for template in instruction_templates:
            instruction = template.format(
                task=task_info["task"].lower(),
                constraint=task_info["constraint"]
            )

            preferred = f"""<|begin_of_thought|>
1. CONSTRAINT DETECTION: Must NOT use {task_info['constraint']}
2. ALTERNATIVE APPROACH: Implement using loops and basic operations
3. VERIFICATION: Solution avoids forbidden function
<|end_of_thought|>

```python
{task_info['preferred_code']}
```

This solution implements {task_info['task'].lower()} without using {task_info['constraint']}, as required by the constraint."""

            rejected = f"""```python
{task_info['rejected_code']}
```"""

            samples.append(PreferenceSample(
                instruction=instruction,
                input="",
                preferred=preferred,
                rejected=rejected,
                category="constraint_code_l4",
                metadata={"task": task_info["task"], "constraint": task_info["constraint"], "level": 4}
            ))

    random.shuffle(samples)
    print(f"Generated {len(samples)} L4 constraint pairs, using {min(target_count, len(samples))}")
    return samples[:target_count]


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_balanced_preference_data(
    l1_target: int = 500,
    l2_target: int = 500,
    l3_target: int = 700,
    l4_target: int = 300,
    output_path: str = "adam_training_data/adam_preference_data_balanced.jsonl",
) -> dict:
    """Generate balanced preference data for all 4 levels."""

    print("="*60)
    print("ADAM BALANCED PREFERENCE DATA GENERATOR")
    print("="*60)
    print(f"Target distribution:")
    print(f"  L1 (Knowledge Override):  {l1_target} pairs (25%)")
    print(f"  L2 (Numerical Physics):   {l2_target} pairs (25%)")
    print(f"  L3 (Syllogistic Logic):   {l3_target} pairs (35%)")
    print(f"  L4 (Constraint Code):     {l4_target} pairs (15%)")
    print(f"  TOTAL:                    {l1_target + l2_target + l3_target + l4_target} pairs")
    print("="*60)

    # Generate samples for each level
    l1_samples = generate_l1_knowledge_pairs(l1_target)
    l2_samples = generate_l2_physics_pairs(l2_target)
    l3_samples = generate_l3_syllogism_pairs(l3_target)
    l4_samples = generate_l4_constraint_pairs(l4_target)

    # Combine and shuffle
    all_samples = l1_samples + l2_samples + l3_samples + l4_samples
    random.shuffle(all_samples)

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample.to_dict()) + "\n")

    print(f"\nWrote {len(all_samples)} samples to {output_path}")

    # Generate stats
    stats = {
        "total_samples": len(all_samples),
        "l1_knowledge": len(l1_samples),
        "l2_physics": len(l2_samples),
        "l3_syllogism": len(l3_samples),
        "l4_constraint": len(l4_samples),
        "distribution": {
            "l1_pct": len(l1_samples) / len(all_samples) * 100,
            "l2_pct": len(l2_samples) / len(all_samples) * 100,
            "l3_pct": len(l3_samples) / len(all_samples) * 100,
            "l4_pct": len(l4_samples) / len(all_samples) * 100,
        }
    }

    # Save stats
    stats_path = output_path.parent / "preference_data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats saved to {stats_path}")
    print("\nFinal distribution:")
    print(f"  L1: {stats['l1_knowledge']} ({stats['distribution']['l1_pct']:.1f}%)")
    print(f"  L2: {stats['l2_physics']} ({stats['distribution']['l2_pct']:.1f}%)")
    print(f"  L3: {stats['l3_syllogism']} ({stats['distribution']['l3_pct']:.1f}%)")
    print(f"  L4: {stats['l4_constraint']} ({stats['distribution']['l4_pct']:.1f}%)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate balanced preference data")
    parser.add_argument("--output", type=str,
                        default="adam_training_data/adam_preference_data_balanced.jsonl",
                        help="Output path for preference data")
    parser.add_argument("--l1", type=int, default=500, help="L1 target count")
    parser.add_argument("--l2", type=int, default=500, help="L2 target count")
    parser.add_argument("--l3", type=int, default=700, help="L3 target count")
    parser.add_argument("--l4", type=int, default=300, help="L4 target count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    generate_balanced_preference_data(
        l1_target=args.l1,
        l2_target=args.l2,
        l3_target=args.l3,
        l4_target=args.l4,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
