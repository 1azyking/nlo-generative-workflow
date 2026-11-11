"""Chemical-neutrality screening using the oxidation states in SI Table S3.

Oxidation states are read from SI Table S3. Elements absent from the table
require case-by-case review.
"""

import argparse
import csv
from fractions import Fraction
from itertools import product
import json
from pathlib import Path


DEFAULT_STATES = Path(__file__).resolve().parents[1] / "resources" / "oxidation_states_si.json"


def load_oxidation_states(path=DEFAULT_STATES):
    """Read the explicit SI rule table; do not substitute library defaults."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    states = data["oxidation_states"]
    for symbol, values in states.items():
        if not values or any(type(value) is not int for value in values):
            raise ValueError(f"Invalid oxidation-state list for {symbol}")
    return states


def check_formula(formula, oxidation_states=None):
    """Return neutral, not_neutral, review, or invalid for an elemental formula.

    A neutral assignment uses one allowed oxidation state per element. When no
    such assignment exists but mixed valence could change the answer, request
    review: SI Table S3 does not prescribe mixed-valence site assignments.
    Elements absent from Table S3 always require case-by-case review.
    Stoichiometric charges use exact rational arithmetic; no charge tolerance
    is introduced.
    """
    from pymatgen.core import Composition

    states = load_oxidation_states() if oxidation_states is None else oxidation_states
    try:
        composition = Composition(formula)
        amounts = composition.element_composition.get_el_amt_dict()
        counts = {symbol: Fraction(str(amount)) for symbol, amount in amounts.items()}
        if not counts or any(count <= 0 for count in counts.values()):
            raise ValueError("Composition must contain positive amounts")
    except Exception as error:
        return {"status": "invalid", "reason": str(error), "assignment": None}

    missing = sorted(set(counts) - set(states))
    if missing:
        return {
            "status": "review",
            "reason": "Elements absent from SI Table S3: " + ", ".join(missing),
            "assignment": None,
        }

    symbols = list(counts)
    for values in product(*(states[symbol] for symbol in symbols)):
        charge = sum((counts[symbol] * value for symbol, value in zip(symbols, values)), Fraction(0))
        if charge == 0:
            return {
                "status": "neutral",
                "reason": "Zero net charge using allowed SI Table S3 oxidation states",
                "assignment": dict(zip(symbols, values)),
            }

    minimum = sum((counts[symbol] * min(states[symbol]) for symbol in symbols), Fraction(0))
    maximum = sum((counts[symbol] * max(states[symbol]) for symbol in symbols), Fraction(0))
    if minimum <= 0 <= maximum:
        return {
            "status": "review",
            "reason": "No uniform per-element assignment; mixed-valence site assignment requires review",
            "assignment": None,
        }
    return {
        "status": "not_neutral",
        "reason": "No zero-charge assignment within the SI Table S3 oxidation states",
        "assignment": None,
    }


def screen_csv(input_csv, output_csv, formula_column="formula", states_file=DEFAULT_STATES):
    """Append screening records to a UTF-8 CSV with a named formula column.

    Review and invalid rows remain in the output; neither is silently rejected.
    Existing input/output files are never overwritten.
    """
    source = Path(input_csv)
    target = Path(output_csv)
    if source.resolve() == target.resolve():
        raise ValueError("Input and output paths must differ")
    states = load_oxidation_states(states_file)
    extra_fields = ["neutrality_status", "neutrality_reason", "oxidation_assignment"]
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or formula_column not in fields:
            raise ValueError(f"CSV must have a {formula_column!r} header column")
        if set(fields).intersection(extra_fields):
            raise ValueError("Input already contains neutrality output columns")
        with target.open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields + extra_fields)
            writer.writeheader()
            for row in reader:
                result = check_formula(row[formula_column], states)
                row.update({
                    "neutrality_status": result["status"],
                    "neutrality_reason": result["reason"],
                    "oxidation_assignment": json.dumps(result["assignment"], ensure_ascii=False),
                })
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--formula-column", default="formula")
    parser.add_argument("--states-file", type=Path, default=DEFAULT_STATES)
    arguments = parser.parse_args()
    screen_csv(arguments.input_csv, arguments.output_csv, arguments.formula_column, arguments.states_file)


if __name__ == "__main__":
    main()
