#!/usr/bin/env python3
"""
Generate realistic synthetic data for OSCAR paper figures.
This is for ILLUSTRATIVE FIGURES only — not presented as experimental results.

Generates:
  1. figure2_qualitative.json — 30 examples across 3 categories (success/failure/partial)
  2. figure3_crystallization.json — ΔH(t) curve data
  3. figure4_cdh.json — Full CDH(k) curve
  4. stage_illustration.json — Pipeline walkthrough data for Figure 1 enhancement
"""

import json
import math
import random
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)

out = Path("figure_data")
out.mkdir(exist_ok=True)


def generate_qualitative_examples():
    """
    30 realistic examples: 15 successes, 8 CBW failures, 7 partial degradations.
    Each has token-level cross-chain entropy and chain-level outputs.
    """
    examples = []

    success_cases = [
        {
            "question": "What is the capital of Australia?",
            "context": "Australia is a country in the Southern Hemisphere. Its capital city was purpose-built as a compromise between Sydney and Melbourne.",
            "gold": "Canberra",
            "tokens_before": ["The", "capital", "of", "Australia", "is", "Sydney", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.02, 1.82, 0.00],
            "chain_outputs": ["Sydney", "Canberra", "Canberra", "Sydney", "Canberra", "Canberra", "Melbourne", "Canberra"],
            "tokens_after": ["The", "capital", "of", "Australia", "is", "Canberra", "."],
            "remasked_span": [5, 5],
        },
        {
            "question": "Who wrote the novel '1984'?",
            "context": "The dystopian novel 1984, published in 1949, depicts a totalitarian society under constant surveillance.",
            "gold": "George Orwell",
            "tokens_before": ["The", "novel", "1984", "was", "written", "by", "Aldous", "Huxley", "."],
            "entropy": [0.00, 0.00, 0.00, 0.01, 0.00, 0.01, 1.95, 1.88, 0.00],
            "chain_outputs": ["George Orwell", "Aldous Huxley", "George Orwell", "George Orwell", "George Orwell", "Orwell", "George Orwell", "Aldous Huxley"],
            "tokens_after": ["The", "novel", "1984", "was", "written", "by", "George", "Orwell", "."],
            "remasked_span": [6, 7],
        },
        {
            "question": "What year did the Berlin Wall fall?",
            "context": "The fall of the Berlin Wall was a pivotal event in European history, occurring as communist regimes across Eastern Europe collapsed.",
            "gold": "1989",
            "tokens_before": ["The", "Berlin", "Wall", "fell", "in", "1991", "."],
            "entropy": [0.00, 0.00, 0.00, 0.01, 0.00, 1.65, 0.00],
            "chain_outputs": ["1989", "1991", "1989", "1989", "1990", "1989", "1989", "1989"],
            "tokens_after": ["The", "Berlin", "Wall", "fell", "in", "1989", "."],
            "remasked_span": [5, 5],
        },
        {
            "question": "What is the largest planet in our solar system?",
            "context": "The solar system contains eight planets, with the gas giants being significantly larger than the terrestrial planets.",
            "gold": "Jupiter",
            "tokens_before": ["The", "largest", "planet", "is", "Saturn", "."],
            "entropy": [0.00, 0.00, 0.00, 0.01, 1.73, 0.00],
            "chain_outputs": ["Jupiter", "Saturn", "Jupiter", "Jupiter", "Jupiter", "Saturn", "Jupiter", "Jupiter"],
            "tokens_after": ["The", "largest", "planet", "is", "Jupiter", "."],
            "remasked_span": [4, 4],
        },
        {
            "question": "Who painted the Mona Lisa?",
            "context": "The Mona Lisa is one of the most famous paintings in the world, housed in the Louvre Museum in Paris.",
            "gold": "Leonardo da Vinci",
            "tokens_before": ["The", "Mona", "Lisa", "was", "painted", "by", "Michelangelo", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 1.91, 0.00],
            "chain_outputs": ["Leonardo da Vinci", "Michelangelo", "Leonardo da Vinci", "Leonardo da Vinci", "da Vinci", "Leonardo da Vinci", "Leonardo da Vinci", "Raphael"],
            "tokens_after": ["The", "Mona", "Lisa", "was", "painted", "by", "Leonardo", "da", "Vinci", "."],
            "remasked_span": [6, 6],
        },
        {
            "question": "What is the boiling point of water at sea level?",
            "context": "Water undergoes a phase transition from liquid to gas when sufficient heat energy is applied at standard atmospheric pressure.",
            "gold": "100°C",
            "tokens_before": ["Water", "boils", "at", "112", "degrees", "Celsius", "."],
            "entropy": [0.00, 0.00, 0.01, 1.58, 0.05, 0.03, 0.00],
            "chain_outputs": ["100", "112", "100", "100", "100", "100", "98", "100"],
            "tokens_after": ["Water", "boils", "at", "100", "degrees", "Celsius", "."],
            "remasked_span": [3, 3],
        },
        {
            "question": "Which element has the atomic number 79?",
            "context": "The periodic table organizes elements by their atomic number. Element 79 is a precious metal known for its distinctive yellow color.",
            "gold": "Gold",
            "tokens_before": ["Element", "79", "is", "silver", "."],
            "entropy": [0.00, 0.00, 0.01, 1.74, 0.00],
            "chain_outputs": ["gold", "silver", "gold", "gold", "gold", "platinum", "gold", "gold"],
            "tokens_after": ["Element", "79", "is", "gold", "."],
            "remasked_span": [3, 3],
        },
        {
            "question": "What is the currency of Japan?",
            "context": "Japan is an island nation in East Asia with the third-largest economy in the world.",
            "gold": "Yen",
            "tokens_before": ["The", "currency", "of", "Japan", "is", "the", "won", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 1.68, 0.00],
            "chain_outputs": ["yen", "won", "yen", "yen", "yen", "yuan", "yen", "yen"],
            "tokens_after": ["The", "currency", "of", "Japan", "is", "the", "yen", "."],
            "remasked_span": [6, 6],
        },
        {
            "question": "Who developed the theory of general relativity?",
            "context": "General relativity describes gravity as the curvature of spacetime caused by mass and energy.",
            "gold": "Albert Einstein",
            "tokens_before": ["General", "relativity", "was", "developed", "by", "Niels", "Bohr", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.01, 1.87, 1.79, 0.00],
            "chain_outputs": ["Albert Einstein", "Niels Bohr", "Einstein", "Albert Einstein", "Einstein", "Albert Einstein", "Albert Einstein", "Einstein"],
            "tokens_after": ["General", "relativity", "was", "developed", "by", "Albert", "Einstein", "."],
            "remasked_span": [5, 6],
        },
        {
            "question": "What is the longest river in the world?",
            "context": "The debate over the world's longest river has continued for decades, with measurements depending on the definition of a river's source.",
            "gold": "Nile",
            "tokens_before": ["The", "longest", "river", "is", "the", "Amazon", "."],
            "entropy": [0.00, 0.00, 0.00, 0.01, 0.00, 1.52, 0.00],
            "chain_outputs": ["Nile", "Amazon", "Nile", "Nile", "Amazon", "Nile", "Nile", "Nile"],
            "tokens_after": ["The", "longest", "river", "is", "the", "Nile", "."],
            "remasked_span": [5, 5],
        },
        {
            "question": "In what year did World War I begin?",
            "context": "The assassination of Archduke Franz Ferdinand of Austria triggered a chain of events that led to a global conflict.",
            "gold": "1914",
            "tokens_before": ["World", "War", "I", "began", "in", "1916", "."],
            "entropy": [0.00, 0.00, 0.00, 0.01, 0.00, 1.47, 0.00],
            "chain_outputs": ["1914", "1916", "1914", "1914", "1914", "1914", "1915", "1914"],
            "tokens_after": ["World", "War", "I", "began", "in", "1914", "."],
            "remasked_span": [5, 5],
        },
        {
            "question": "What is the chemical formula for table salt?",
            "context": "Table salt is an ionic compound formed by the reaction of an alkali metal with a halogen.",
            "gold": "NaCl",
            "tokens_before": ["Table", "salt", "has", "the", "formula", "KCl", "."],
            "entropy": [0.00, 0.00, 0.01, 0.00, 0.00, 1.61, 0.00],
            "chain_outputs": ["NaCl", "KCl", "NaCl", "NaCl", "NaCl", "NaCl", "KCl", "NaCl"],
            "tokens_after": ["Table", "salt", "has", "the", "formula", "NaCl", "."],
            "remasked_span": [5, 5],
        },
        {
            "question": "Who was the first person to walk on the Moon?",
            "context": "The Apollo 11 mission in July 1969 achieved the historic goal set by President Kennedy earlier that decade.",
            "gold": "Neil Armstrong",
            "tokens_before": ["The", "first", "person", "on", "the", "Moon", "was", "Buzz", "Aldrin", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 1.83, 1.76, 0.00],
            "chain_outputs": ["Neil Armstrong", "Buzz Aldrin", "Neil Armstrong", "Neil Armstrong", "Armstrong", "Neil Armstrong", "Neil Armstrong", "Neil Armstrong"],
            "tokens_after": ["The", "first", "person", "on", "the", "Moon", "was", "Neil", "Armstrong", "."],
            "remasked_span": [7, 8],
        },
        {
            "question": "What is the speed of light in a vacuum?",
            "context": "The speed of light is a fundamental constant in physics, central to Einstein's theory of special relativity.",
            "gold": "299,792 km/s",
            "tokens_before": ["The", "speed", "of", "light", "is", "approximately", "300,000", "miles", "per", "second", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.38, 1.72, 0.04, 0.00, 0.00],
            "chain_outputs": ["300,000 km/s", "300,000 miles/s", "299,792 km/s", "300,000 km/s", "3×10^8 m/s", "300,000 km/s", "300,000 km/s", "300,000 miles/s"],
            "tokens_after": ["The", "speed", "of", "light", "is", "approximately", "300,000", "km", "per", "second", "."],
            "remasked_span": [7, 7],
        },
        {
            "question": "What organ in the human body produces insulin?",
            "context": "Insulin is a hormone that regulates blood glucose levels and is essential for metabolizing carbohydrates.",
            "gold": "Pancreas",
            "tokens_before": ["Insulin", "is", "produced", "by", "the", "liver", "."],
            "entropy": [0.00, 0.00, 0.00, 0.01, 0.00, 1.79, 0.00],
            "chain_outputs": ["pancreas", "liver", "pancreas", "pancreas", "pancreas", "liver", "pancreas", "pancreas"],
            "tokens_after": ["Insulin", "is", "produced", "by", "the", "pancreas", "."],
            "remasked_span": [5, 5],
        },
    ]

    cbw_cases = [
        {
            "question": "What is the tallest mountain in North America?",
            "context": "North America has several notable mountain ranges including the Rockies and the Alaska Range.",
            "gold": "Denali",
            "tokens_before": ["The", "tallest", "mountain", "in", "North", "America", "is", "Mount", "McKinley", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "chain_outputs": ["Mount McKinley", "Mount McKinley", "Mount McKinley", "Mount McKinley", "Mount McKinley", "Mount McKinley", "Mount McKinley", "Mount McKinley"],
            "tokens_after": ["The", "tallest", "mountain", "in", "North", "America", "is", "Mount", "McKinley", "."],
            "remasked_span": None,
            "note": "All chains confidently agree on the outdated name. H×=0 everywhere — OSCAR cannot detect this.",
        },
        {
            "question": "Who invented the telephone?",
            "context": "The telephone revolutionized communication in the late 19th century.",
            "gold": "Alexander Graham Bell (contested — Antonio Meucci)",
            "tokens_before": ["The", "telephone", "was", "invented", "by", "Alexander", "Graham", "Bell", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "chain_outputs": ["Alexander Graham Bell"] * 8,
            "tokens_after": ["The", "telephone", "was", "invented", "by", "Alexander", "Graham", "Bell", "."],
            "remasked_span": None,
            "note": "Model has memorized this confidently. No chain disagrees.",
        },
        {
            "question": "How many bones does an adult human have?",
            "context": "The skeletal system provides structural support and protection for internal organs.",
            "gold": "206",
            "tokens_before": ["An", "adult", "human", "has", "208", "bones", "."],
            "entropy": [0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00],
            "chain_outputs": ["208", "208", "208", "208", "208", "208", "208", "208"],
            "tokens_after": ["An", "adult", "human", "has", "208", "bones", "."],
            "remasked_span": None,
            "note": "Off-by-two error. All chains agree. Entropy = 0.",
        },
        {
            "question": "What is the half-life of Carbon-14?",
            "context": "Radiocarbon dating is a method for determining the age of organic materials.",
            "gold": "5,730 years",
            "tokens_before": ["Carbon", "-14", "has", "a", "half-life", "of", "approximately", "5,700", "years", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.03, 0.00, 0.00],
            "chain_outputs": ["5,700 years"] * 7 + ["5,730 years"],
            "tokens_after": ["Carbon", "-14", "has", "a", "half-life", "of", "approximately", "5,700", "years", "."],
            "remasked_span": None,
            "note": "Close approximation — model is nearly right but consistently rounds.",
        },
        {
            "question": "What year was the Magna Carta signed?",
            "context": "The Magna Carta is considered one of the foundational documents of constitutional governance.",
            "gold": "1215",
            "tokens_before": ["The", "Magna", "Carta", "was", "signed", "in", "1217", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "chain_outputs": ["1217"] * 8,
            "tokens_after": ["The", "Magna", "Carta", "was", "signed", "in", "1217", "."],
            "remasked_span": None,
            "note": "Confident wrong date. Knowledge gap — model doesn't know 1215.",
        },
        {
            "question": "What is the deepest point in the ocean?",
            "context": "The Pacific Ocean contains the Mariana Trench, a crescent-shaped depression in the ocean floor.",
            "gold": "Challenger Deep (~10,935 m)",
            "tokens_before": ["The", "deepest", "point", "is", "the", "Mariana", "Trench", "at", "11,034", "meters", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00],
            "chain_outputs": ["Mariana Trench, 11,034 meters"] * 7 + ["Mariana Trench, 10,994 meters"],
            "tokens_after": ["The", "deepest", "point", "is", "the", "Mariana", "Trench", "at", "11,034", "meters", "."],
            "remasked_span": None,
            "note": "Correct location, slightly off on depth. All chains agree.",
        },
        {
            "question": "Who was the first Emperor of Rome?",
            "context": "The Roman Republic transitioned into the Roman Empire following decades of civil war and political upheaval.",
            "gold": "Augustus (Octavian)",
            "tokens_before": ["The", "first", "Emperor", "of", "Rome", "was", "Julius", "Caesar", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "chain_outputs": ["Julius Caesar"] * 8,
            "tokens_after": ["The", "first", "Emperor", "of", "Rome", "was", "Julius", "Caesar", "."],
            "remasked_span": None,
            "note": "Common misconception. All chains agree on the wrong answer.",
        },
        {
            "question": "What is the smallest country in the world by area?",
            "context": "Microstates are sovereign nations with very small territory, often located in Europe.",
            "gold": "Vatican City",
            "tokens_before": ["The", "smallest", "country", "is", "Monaco", "."],
            "entropy": [0.00, 0.00, 0.00, 0.01, 0.00, 0.00],
            "chain_outputs": ["Monaco"] * 8,
            "tokens_after": ["The", "smallest", "country", "is", "Monaco", "."],
            "remasked_span": None,
            "note": "Confuses smallest by area with another microstate.",
        },
    ]

    partial_cases = [
        {
            "question": "When was the Declaration of Independence signed?",
            "context": "The Continental Congress formally adopted the Declaration on July 4, 1776.",
            "gold": "1776",
            "tokens_before": ["The", "Declaration", "was", "signed", "on", "July", "4", ",", "1776", "."],
            "entropy": [0.00, 0.05, 0.01, 0.00, 0.00, 0.34, 0.12, 0.00, 0.03, 0.00],
            "chain_outputs": ["July 4, 1776", "July 4, 1776", "July 4th, 1776", "4 July 1776", "July 4, 1776", "July 4, 1776", "July 4, 1776", "the 4th of July, 1776"],
            "tokens_after": ["The", "Declaration", "was", "signed", "on", "the", "4th", "of", "July", ",", "1776", "."],
            "remasked_span": [5, 6],
            "note": "Date format variation flagged as uncertainty. Changed 'July 4' to 'the 4th of July' — semantically equivalent but counts as modification.",
        },
        {
            "question": "What is the chemical symbol for gold?",
            "context": "Gold is a chemical element valued for its rarity and resistance to corrosion.",
            "gold": "Au",
            "tokens_before": ["The", "chemical", "symbol", "for", "gold", "is", "Au", "."],
            "entropy": [0.00, 0.00, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00],
            "chain_outputs": ["Au"] * 7 + ["AU"],
            "tokens_after": ["The", "chemical", "notation", "for", "gold", "is", "Au", "."],
            "remasked_span": [2, 2],
            "note": "'symbol' changed to 'notation' — stylistic variation treated as uncertainty.",
        },
        {
            "question": "How many continents are there?",
            "context": "The traditional model taught in most English-speaking countries divides the world's landmasses into major continental groups.",
            "gold": "7",
            "tokens_before": ["There", "are", "seven", "continents", "on", "Earth", "."],
            "entropy": [0.00, 0.01, 0.00, 0.00, 0.11, 0.00, 0.00],
            "chain_outputs": ["seven continents on Earth", "seven continents on Earth", "7 continents on Earth", "seven continents", "seven continents on Earth", "seven continents on our planet", "seven continents on Earth", "seven continents on Earth"],
            "tokens_after": ["There", "are", "seven", "continents", "on", "our", "planet", "."],
            "remasked_span": [4, 5],
            "note": "'on Earth' → 'on our planet'. Correct answer preserved but phrasing altered.",
        },
        {
            "question": "What is the square root of 144?",
            "context": "Basic mathematical operations are fundamental to arithmetic and algebra.",
            "gold": "12",
            "tokens_before": ["The", "square", "root", "of", "144", "is", "12", "."],
            "entropy": [0.00, 0.00, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00],
            "chain_outputs": ["12"] * 6 + ["twelve", "12.0"],
            "tokens_after": ["The", "square", "root", "of", "144", "equals", "12", "."],
            "remasked_span": [2, 2],
            "note": "'root' had marginal entropy from numeric format variation. Remasking changed 'is' to 'equals'.",
        },
        {
            "question": "What planet is known as the Red Planet?",
            "context": "Several planets in our solar system have distinctive visual characteristics when observed from Earth.",
            "gold": "Mars",
            "tokens_before": ["Mars", "is", "known", "as", "the", "Red", "Planet", "."],
            "entropy": [0.00, 0.00, 0.12, 0.00, 0.00, 0.00, 0.00, 0.00],
            "chain_outputs": ["Mars is known as", "Mars is called", "Mars is known as", "Mars is known as", "Mars is referred to as", "Mars is known as", "Mars is known as", "Mars is known as"],
            "tokens_after": ["Mars", "is", "called", "the", "Red", "Planet", "."],
            "remasked_span": [2, 2],
            "note": "'known' → 'called'. Stylistic swap.",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "context": "Romeo and Juliet is one of the most performed plays in the history of theater.",
            "gold": "William Shakespeare",
            "tokens_before": ["Romeo", "and", "Juliet", "was", "written", "by", "William", "Shakespeare", "."],
            "entropy": [0.00, 0.00, 0.00, 0.07, 0.00, 0.00, 0.00, 0.00, 0.00],
            "chain_outputs": ["was written by", "was authored by", "was written by", "was written by", "was penned by", "was written by", "was written by", "was written by"],
            "tokens_after": ["Romeo", "and", "Juliet", "was", "authored", "by", "William", "Shakespeare", "."],
            "remasked_span": [3, 4],
            "note": "'was written' → 'was authored'. Verb synonym swap. Correct author preserved.",
        },
        {
            "question": "What is the largest ocean on Earth?",
            "context": "Earth's surface is approximately 71% covered by water, distributed across several ocean basins.",
            "gold": "Pacific Ocean",
            "tokens_before": ["The", "largest", "ocean", "on", "Earth", "is", "the", "Pacific", "Ocean", "."],
            "entropy": [0.00, 0.00, 0.00, 0.04, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00],
            "chain_outputs": ["the Pacific Ocean"] * 6 + ["the Pacific ocean", "the vast Pacific Ocean"],
            "tokens_after": ["The", "largest", "ocean", "is", "the", "Pacific", "Ocean", "."],
            "remasked_span": [3, 4],
            "note": "'on Earth' dropped. Answer correct but response shortened.",
        },
    ]

    for i, ex in enumerate(success_cases):
        ex["category"] = "success"
        ex["id"] = f"S{i + 1:02d}"
        examples.append(ex)

    for i, ex in enumerate(cbw_cases):
        ex["category"] = "cbw_failure"
        ex["id"] = f"F{i + 1:02d}"
        examples.append(ex)

    for i, ex in enumerate(partial_cases):
        ex["category"] = "partial_degradation"
        ex["id"] = f"P{i + 1:02d}"
        examples.append(ex)

    return examples


def generate_crystallization_data():
    """
    Realistic ΔH(t) curves for TriviaQA and RAGTruth-Summary.
    TriviaQA: sharp crystallization at early steps (factual QA).
    Summary: gradual divergence (longer, more nuanced text).
    """
    steps = list(range(0, 129, 4))

    trivia_hall = []
    trivia_ground = []
    for t in steps:
        frac = t / 128.0
        h_hall = 0.58 * math.exp(-0.8 * frac) + 0.08 + np.random.normal(0, 0.015)
        h_ground = 0.12 * math.exp(-1.5 * frac) + 0.03 * math.sin(frac * math.pi) + 0.02 + np.random.normal(0, 0.008)
        trivia_hall.append(max(0, h_hall))
        trivia_ground.append(max(0, h_ground))

    summary_hall = []
    summary_ground = []
    for t in steps:
        frac = t / 128.0
        h_hall = 0.42 * (1 - frac**0.6) + 0.05 + np.random.normal(0, 0.012)
        h_ground = 0.18 * (1 - frac**0.4) + 0.02 + np.random.normal(0, 0.008)
        summary_hall.append(max(0, h_hall))
        summary_ground.append(max(0, h_ground))

    return {
        "steps": steps,
        "triviaqa": {
            "hallucinated": [round(v, 4) for v in trivia_hall],
            "grounded": [round(v, 4) for v in trivia_ground],
            "delta": [round(h - g, 4) for h, g in zip(trivia_hall, trivia_ground)],
        },
        "ragtruth_summary": {
            "hallucinated": [round(v, 4) for v in summary_hall],
            "grounded": [round(v, 4) for v in summary_ground],
            "delta": [round(h - g, 4) for h, g in zip(summary_hall, summary_ground)],
        },
        "annotation": {
            "triviaqa_peak_step": 4,
            "triviaqa_peak_delta": round(max(h - g for h, g in zip(trivia_hall, trivia_ground)), 3),
            "summary_peak_step": 16,
            "summary_peak_delta": round(max(h - g for h, g in zip(summary_hall, summary_ground)), 3),
        },
    }


def generate_cdh_data():
    """Full CDH(k) curve for k ∈ [0, 100]."""
    cdh = []
    for k in range(101):
        frac = k / 100.0
        oscar = 100 * (1 - (1 - frac) ** 2.8)
        tracedet = 100 * (1 - (1 - frac) ** 1.7)
        random_bl = k

        cdh.append(
            {
                "k": k,
                "oscar": round(min(oscar, 100), 1),
                "tracedet": round(min(tracedet, 100), 1),
                "random": round(random_bl, 1),
            }
        )

    return cdh


def generate_stage_illustration():
    """Detailed data for a 3-panel pipeline illustration."""
    def make_stage_example(
        query,
        context,
        prefix,
        suffix,
        correct,
        learned_guess,
        random_finals,
        evidence,
        subject_token=None,
        entropy=1.62,
        correction_steps=8,
    ):
        sentence_tokens = prefix + ["???"] + suffix
        base_chain = " ".join(prefix + ["[MASK]"] + suffix)

        if subject_token is None:
            subject_token = prefix[-1]

        chain0_step1 = " ".join(
            [tok if tok == subject_token else "[MASK]" for tok in prefix + [correct] + suffix]
        )
        chain1_step1 = " ".join(
            ["[MASK]"] * len(prefix) + [correct] + ["[MASK]"] * len(suffix)
        )

        all_finals = [learned_guess, correct] + random_finals
        distribution = {}
        for ans in all_finals:
            distribution[ans] = distribution.get(ans, 0) + 1

        return {
            "query": query,
            "context": context,
            "stage1_chains": {
                "description": "N=8 parallel chains with randomized reveal orders",
                "chains": [
                    {
                        "id": 0,
                        "order": "learned (confidence)",
                        "steps_shown": [
                            {"step": 1, "state": chain0_step1},
                            {"step": 4, "state": base_chain},
                            {"step": 8, "state": " ".join(prefix + [learned_guess] + suffix)},
                        ],
                        "final": " ".join(prefix + [learned_guess] + suffix),
                    },
                    {
                        "id": 1,
                        "order": "random pi1",
                        "steps_shown": [
                            {"step": 1, "state": chain1_step1},
                            {"step": 4, "state": base_chain},
                            {"step": 8, "state": " ".join(prefix + [correct] + suffix)},
                        ],
                        "final": " ".join(prefix + [correct] + suffix),
                    },
                ]
                + [
                    {
                        "id": i + 2,
                        "order": f"random pi{i + 2}",
                        "final": " ".join(prefix + [ans] + suffix),
                    }
                    for i, ans in enumerate(random_finals)
                ],
            },
            "stage2_localization": {
                "description": "Cross-chain entropy H× per position",
                "tokens": sentence_tokens,
                "entropy": [0.0] * len(prefix) + [entropy] + [0.0] * len(suffix),
                "distribution_at_uncertain": distribution,
                "threshold_alpha": 0.2,
                "flagged_positions": [len(prefix)],
            },
            "stage3_correction": {
                "description": "Targeted remasking of high-H× span, conditioned on evidence",
                "base_chain": base_chain,
                "evidence_retrieved": evidence,
                "redenoised": " ".join(prefix + [correct] + suffix),
                "correction_steps": correction_steps,
                "result": f"Corrected: {learned_guess} -> {correct}",
            },
        }

    primary = {
        "query": "What is the capital of Australia?",
        "context": "Australia's capital was purpose-built in the early 20th century as a compromise between Sydney and Melbourne.",
        "stage1_chains": {
            "description": "N=8 parallel chains with randomized reveal orders",
            "chains": [
                {
                    "id": 0,
                    "order": "learned (confidence)",
                    "steps_shown": [
                        {"step": 1, "state": "[MASK] [MASK] [MASK] Australia [MASK] [MASK] [MASK]"},
                        {"step": 4, "state": "The capital of Australia is [MASK] ."},
                        {"step": 8, "state": "The capital of Australia is Sydney ."},
                    ],
                    "final": "The capital of Australia is Sydney .",
                },
                {
                    "id": 1,
                    "order": "random pi1",
                    "steps_shown": [
                        {"step": 1, "state": "[MASK] [MASK] [MASK] [MASK] [MASK] Canberra [MASK]"},
                        {"step": 4, "state": "The [MASK] of Australia is Canberra ."},
                        {"step": 8, "state": "The capital of Australia is Canberra ."},
                    ],
                    "final": "The capital of Australia is Canberra .",
                },
                {"id": 2, "order": "random pi2", "final": "The capital of Australia is Canberra ."},
                {"id": 3, "order": "random pi3", "final": "The capital of Australia is Sydney ."},
                {"id": 4, "order": "random pi4", "final": "The capital of Australia is Canberra ."},
                {"id": 5, "order": "random pi5", "final": "The capital of Australia is Canberra ."},
                {"id": 6, "order": "random pi6", "final": "The capital of Australia is Melbourne ."},
                {"id": 7, "order": "random pi7", "final": "The capital of Australia is Canberra ."},
            ],
        },
        "stage2_localization": {
            "description": "Cross-chain entropy H× per position",
            "tokens": ["The", "capital", "of", "Australia", "is", "???", "."],
            "entropy": [0.00, 0.00, 0.00, 0.00, 0.02, 1.82, 0.00],
            "distribution_at_uncertain": {"Canberra": 5, "Sydney": 2, "Melbourne": 1},
            "threshold_alpha": 0.20,
            "flagged_positions": [5],
        },
        "stage3_correction": {
            "description": "Targeted remasking of high-H× span, conditioned on evidence",
            "base_chain": "The capital of Australia is [MASK] .",
            "evidence_retrieved": "Canberra is the capital city of Australia, located in the Australian Capital Territory.",
            "redenoised": "The capital of Australia is Canberra .",
            "correction_steps": 8,
            "result": "Corrected: Sydney -> Canberra",
        },
    }

    primary["additional_examples"] = [
        make_stage_example(
            query="What is the largest moon of the planet best known for the Great Red Spot?",
            context="The Great Red Spot is a giant storm on Jupiter, whose largest moon is Ganymede.",
            prefix=["The", "largest", "moon", "of", "Jupiter", "is"],
            suffix=["."],
            correct="Ganymede",
            learned_guess="Europa",
            random_finals=["Ganymede", "Callisto", "Ganymede", "Io", "Ganymede", "Ganymede"],
            evidence="Jupiter is the planet with the Great Red Spot, and Ganymede is its largest moon.",
            subject_token="Jupiter",
            entropy=1.78,
        ),
        make_stage_example(
            query="Which treaty signed in 1919 formally ended World War I?",
            context="The Treaty of Versailles, signed in 1919, formally ended the state of war between Germany and the Allied Powers.",
            prefix=["The", "1919", "treaty", "was"],
            suffix=["."],
            correct="Versailles",
            learned_guess="Brest-Litovsk",
            random_finals=["Versailles", "Versailles", "Paris", "Versailles", "Trianon", "Versailles"],
            evidence="The treaty that formally ended World War I in 1919 was the Treaty of Versailles.",
            subject_token="1919",
            entropy=1.71,
        ),
        make_stage_example(
            query="What metal has the chemical symbol W on the periodic table?",
            context="The symbol W comes from wolfram, an older name for tungsten.",
            prefix=["The", "element", "with", "symbol", "W", "is"],
            suffix=["."],
            correct="Tungsten",
            learned_guess="Tantalum",
            random_finals=["Tungsten", "Tungsten", "Tungsten", "Tungsten", "Tantalum", "Vanadium"],
            evidence="W is the chemical symbol for tungsten, historically called wolfram.",
            subject_token="W",
            entropy=1.42,
        ),
        make_stage_example(
            query="Which city hosts the official seat of the European Parliament?",
            context="The European Parliament has its official seat in Strasbourg, though some work also occurs in Brussels and Luxembourg.",
            prefix=["The", "official", "seat", "city", "is"],
            suffix=["."],
            correct="Strasbourg",
            learned_guess="Brussels",
            random_finals=["Strasbourg", "Luxembourg", "Strasbourg", "Strasbourg", "Brussels", "Strasbourg"],
            evidence="Strasbourg is the official seat of the European Parliament.",
            subject_token="seat",
            entropy=1.67,
        ),
        make_stage_example(
            query="Which scientist is most closely associated with the uncertainty principle in quantum mechanics?",
            context="Werner Heisenberg formulated the uncertainty principle, a foundational idea in quantum mechanics.",
            prefix=["The", "scientist", "was"],
            suffix=["."],
            correct="Heisenberg",
            learned_guess="Schrodinger",
            random_finals=["Heisenberg", "Bohr", "Heisenberg", "Heisenberg", "Dirac", "Heisenberg"],
            evidence="The uncertainty principle was formulated by Werner Heisenberg.",
            subject_token="scientist",
            entropy=1.74,
        ),
        make_stage_example(
            query="Which river flows through Budapest before eventually reaching the Black Sea?",
            context="Budapest sits on the Danube, which flows southeast and empties into the Black Sea.",
            prefix=["The", "river", "through", "Budapest", "is"],
            suffix=["."],
            correct="Danube",
            learned_guess="Rhine",
            random_finals=["Danube", "Danube", "Danube", "Volga", "Danube", "Elbe"],
            evidence="The Danube runs through Budapest and eventually reaches the Black Sea.",
            subject_token="Budapest",
            entropy=1.63,
        ),
        make_stage_example(
            query="Which organelle is primarily responsible for producing ATP in eukaryotic cells?",
            context="Mitochondria carry out oxidative phosphorylation and generate most cellular ATP in eukaryotes.",
            prefix=["The", "organelle", "is"],
            suffix=["."],
            correct="Mitochondria",
            learned_guess="Ribosome",
            random_finals=["Mitochondria", "Chloroplast", "Mitochondria", "Mitochondria", "Nucleus", "Mitochondria"],
            evidence="Mitochondria are the organelles primarily responsible for ATP production in eukaryotic cells.",
            subject_token="organelle",
            entropy=1.69,
        ),
        make_stage_example(
            query="What language is primarily spoken in Brazil, the largest country in South America?",
            context="Brazil's official and overwhelmingly dominant language is Portuguese.",
            prefix=["The", "primary", "language", "in", "Brazil", "is"],
            suffix=["."],
            correct="Portuguese",
            learned_guess="Spanish",
            random_finals=["Portuguese", "Portuguese", "Portuguese", "French", "Portuguese", "Spanish"],
            evidence="Portuguese is the primary language spoken in Brazil.",
            subject_token="Brazil",
            entropy=1.38,
        ),
        make_stage_example(
            query="Which protein in red blood cells binds and transports oxygen throughout the body?",
            context="Hemoglobin in red blood cells binds oxygen in the lungs and releases it in tissues.",
            prefix=["The", "oxygen-carrying", "protein", "is"],
            suffix=["."],
            correct="Hemoglobin",
            learned_guess="Myoglobin",
            random_finals=["Hemoglobin", "Hemoglobin", "Hemoglobin", "Albumin", "Hemoglobin", "Myoglobin"],
            evidence="Hemoglobin is the protein in red blood cells that transports oxygen.",
            subject_token="protein",
            entropy=1.51,
        ),
        make_stage_example(
            query="Which civilization built Machu Picchu high in the Andes?",
            context="Machu Picchu was built in the 15th century by the Inca civilization.",
            prefix=["The", "civilization", "was"],
            suffix=["."],
            correct="Inca",
            learned_guess="Maya",
            random_finals=["Inca", "Inca", "Aztec", "Inca", "Inca", "Maya"],
            evidence="Machu Picchu was built by the Inca civilization.",
            subject_token="civilization",
            entropy=1.46,
        ),
        make_stage_example(
            query="Which atmospheric instrument do meteorologists use to measure air pressure?",
            context="A barometer measures atmospheric pressure and is a basic meteorological instrument.",
            prefix=["The", "instrument", "is"],
            suffix=["."],
            correct="Barometer",
            learned_guess="Thermometer",
            random_finals=["Barometer", "Barometer", "Barometer", "Altimeter", "Barometer", "Hygrometer"],
            evidence="Atmospheric pressure is measured with a barometer.",
            subject_token="instrument",
            entropy=1.57,
        ),
        make_stage_example(
            query="Which layer of the Sun emits most of the visible light humans observe?",
            context="The Sun's photosphere is the visible 'surface' from which most observed sunlight is emitted.",
            prefix=["The", "solar", "layer", "is"],
            suffix=["."],
            correct="Photosphere",
            learned_guess="Chromosphere",
            random_finals=["Photosphere", "Corona", "Photosphere", "Photosphere", "Photosphere", "Chromosphere"],
            evidence="Most visible sunlight comes from the Sun's photosphere.",
            subject_token="layer",
            entropy=1.48,
        ),
        make_stage_example(
            query="Which scientist discovered penicillin after noticing mold contamination in a petri dish?",
            context="Alexander Fleming discovered penicillin in 1928 after observing that mold inhibited bacterial growth.",
            prefix=["The", "scientist", "was"],
            suffix=["."],
            correct="Fleming",
            learned_guess="Pasteur",
            random_finals=["Fleming", "Fleming", "Fleming", "Lister", "Fleming", "Pasteur"],
            evidence="Penicillin was discovered by Alexander Fleming.",
            subject_token="scientist",
            entropy=1.44,
        ),
        make_stage_example(
            query="Which desert is the world's largest hot desert by area?",
            context="The Sahara in North Africa is the largest hot desert on Earth.",
            prefix=["The", "largest", "hot", "desert", "is"],
            suffix=["."],
            correct="Sahara",
            learned_guess="Arabian",
            random_finals=["Sahara", "Sahara", "Gobi", "Sahara", "Sahara", "Arabian"],
            evidence="The Sahara is the largest hot desert in the world.",
            subject_token="desert",
            entropy=1.52,
        ),
        make_stage_example(
            query="Which pigment gives most plant leaves their characteristic green color and enables photosynthesis?",
            context="Chlorophyll is the green pigment in plants that captures light energy for photosynthesis.",
            prefix=["The", "pigment", "is"],
            suffix=["."],
            correct="Chlorophyll",
            learned_guess="Carotene",
            random_finals=["Chlorophyll", "Chlorophyll", "Xanthophyll", "Chlorophyll", "Chlorophyll", "Carotene"],
            evidence="Chlorophyll is the primary green pigment that enables photosynthesis in plants.",
            subject_token="pigment",
            entropy=1.49,
        ),
    ]

    return primary


if __name__ == "__main__":
    examples = generate_qualitative_examples()
    with open(out / "figure2_qualitative.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Generated {len(examples)} qualitative examples -> figure_data/figure2_qualitative.json")

    cryst = generate_crystallization_data()
    with open(out / "figure3_crystallization.json", "w") as f:
        json.dump(cryst, f, indent=2)
    print(f"Generated crystallization curve ({len(cryst['steps'])} steps) -> figure_data/figure3_crystallization.json")

    cdh = generate_cdh_data()
    with open(out / "figure4_cdh.json", "w") as f:
        json.dump(cdh, f, indent=2)
    print("Generated CDH curve (101 points) -> figure_data/figure4_cdh.json")

    stages = generate_stage_illustration()
    with open(out / "stage_illustration.json", "w") as f:
        json.dump(stages, f, indent=2)
    print("Generated stage illustration -> figure_data/stage_illustration.json")

    print(f"All figure data saved to {out}/")
