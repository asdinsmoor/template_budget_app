import csv
import json
import logging
import math
import re
from pathlib import Path
from openai import OpenAI

client = OpenAI()
DEFAULT_CATEGORY = ("Uncategorized", "")
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RULES_CSV_PATH = DATA_DIR / "auto_cat_rules.csv"


def _load_csv_rules(path: Path = RULES_CSV_PATH):
    rules = []
    if not path.exists():
        logger.info("CSV rules file not found at %s. Using built-in rules only.", path)
        return rules

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                keyword = (row.get("Keywords") or "").strip()
                if not keyword:
                    continue

                category = (row.get("Category") or "").strip() or DEFAULT_CATEGORY[0]
                subcategory = (row.get("Sub-Category") or "").strip() or DEFAULT_CATEGORY[1]
                exclude_flag = (row.get("Exlcude?") or "").strip().lower() == "true"

                rules.append(
                    {
                        "keyword": keyword,
                        "pattern": re.compile(re.escape(keyword), re.IGNORECASE),
                        "category": category,
                        "subcategory": subcategory,
                        "exclude": exclude_flag,
                    }
                )
    except Exception as exc:
        logger.error("Failed to load CSV rules from %s: %s", path, exc)
    else:
        logger.info("Loaded %d CSV categorization rules from %s.", len(rules), path)

    return rules

# Define structured output schema
CATEGORIES = [
    "Health & Wellness", "Travel", "Shopping", "Food & Drink", "Entertainment", 
    "Groceries", "Bills & Utilities", "Personal", "Gifts & Donations", "Gas",
    "Home", "Professional Services", "Fees & Adjustments", "Reimbursed Expenses", 
    "Childcare", "Uncategorized"
]

SUBCATEGORIES = {
    "Health & Wellness": ["Rigby", "Ainslie", "Plan Z", ""],
    "Travel": ["Airbnb / Hotel", "Lyft / Uber / Car Rental", "Airlines", "Car Expenses", "Other Travel"],
    "Shopping": [""],
    "Food & Drink": [""],
    "Entertainment": [""],
    "Groceries": [""],
    "Bills & Utilities": [""],
    "Personal": [""],
    "Gifts & Donations": [""],
    "Gas": ["Car Expenses"],
    "Home": [""],
    "Professional Services": [""],
    "Fees & Adjustments": ["Tax Payment",""],
    "Reimbursed Expenses": [""],
    "Childcare": [""],
    "Uncategorized": ["General", ""]
}

CSV_RULES = _load_csv_rules()

# Expand SUBCATEGORIES with any custom values present in CSV rules
def _ensure_valid_subcategory(category: str, subcategory: str) -> str:
    valid = SUBCATEGORIES.get(category)
    if not valid:
        SUBCATEGORIES[category] = [subcategory] if subcategory else [DEFAULT_CATEGORY[1]]
        return subcategory
    if subcategory in valid:
        return subcategory
    if subcategory:
        valid.append(subcategory)
        return subcategory
    return valid[0] if valid else DEFAULT_CATEGORY[1]

for csv_rule in CSV_RULES:
    if csv_rule["category"] not in CATEGORIES:
        CATEGORIES.append(csv_rule["category"])
    csv_rule["subcategory"] = _ensure_valid_subcategory(csv_rule["category"], csv_rule["subcategory"])

# 1️⃣ rule-based pass
def categorize_rule_based(description: str):
    if not isinstance(description, str):
        logger.debug("Non-string description encountered; using default category.")
        return DEFAULT_CATEGORY[0], DEFAULT_CATEGORY[1], False, "default"

    desc = description.strip().lower()
    if not desc:
        logger.debug("Empty description encountered; using default category.")
        return DEFAULT_CATEGORY[0], DEFAULT_CATEGORY[1], False, "default"

    for rule in CSV_RULES:
        if rule["pattern"].search(desc):
            category = rule["category"] or DEFAULT_CATEGORY[0]
            subcategory = _ensure_valid_subcategory(category, rule["subcategory"])
            exclude = bool(rule.get("exclude"))
            if rule.get("exclude"):
                logger.debug(
                    "CSV rule '%s' matched and is flagged as exclude.",
                    rule["keyword"],
                )
            logger.debug(
                "CSV rule matched '%s' -> %s / %s.",
                rule["keyword"],
                category,
                subcategory,
            )
            return category, subcategory, exclude, "rule"

    logger.debug("No rule-based match for description '%s'.", description)
    return DEFAULT_CATEGORY[0], DEFAULT_CATEGORY[1], False, "default"


# 2️⃣ single-transaction convenience wrapper
def categorize_transaction(description: str, amount: float):
    categories, subcategories, exclude_flags, sources, confidences = categorize_transactions(
        [description],
        [amount],
    )
    return categories[0], subcategories[0], exclude_flags[0], sources[0], confidences[0]


# 3️⃣ batch orchestrator
def categorize_transactions(
    descriptions,
    amounts,
    batch_size: int = 10,
):
    """
    Categorize a sequence of transactions with rule-based logic first then the LLM.
    Returns (categories list, subcategories list, exclude_flags list, sources list, confidences list)
    aligned with inputs. Confidence values are floats between 0 and 1 for LLM-sourced
    responses and None where no LLM confidence is available.
    """
    if len(descriptions) != len(amounts):
        raise ValueError("Descriptions and amounts must have the same length.")

    total = len(descriptions)
    categories = [DEFAULT_CATEGORY[0]] * total
    subcategories = [DEFAULT_CATEGORY[1]] * total
    exclude_flags = [False] * total
    sources = ["default"] * total
    confidences = [None] * total
    pending = []

    for idx, (desc, amount) in enumerate(zip(descriptions, amounts)):
        cat, subcat, exclude, source = categorize_rule_based(desc)
        if source == "rule":
            categories[idx] = cat
            subcategories[idx] = subcat
            exclude_flags[idx] = bool(exclude)
            sources[idx] = "rule"
        else:
            amount_value = 0.0
            if isinstance(amount, (int, float)):
                if isinstance(amount, float) and math.isnan(amount):
                    amount_value = 0.0
                else:
                    amount_value = float(amount)
            pending.append(
                {
                    "idx": idx,
                    "description": desc if isinstance(desc, str) else "",
                    "amount": amount_value,
                }
            )

    logger.info(
        "Rule-based categorization handled %d/%d transactions.",
        total - len(pending),
        total,
    )

    if not pending:
        return categories, subcategories, exclude_flags, sources, confidences

    chunks = _chunked(pending, batch_size)
    for chunk in chunks:
        try:
            llm_results = categorize_llm_batch(chunk)
            for item in chunk:
                idx = item["idx"]
                result = llm_results.get(idx)
                if not result:
                    logger.warning(
                        "LLM response missing entry for idx=%s; defaulting to %s.",
                        idx,
                        DEFAULT_CATEGORY,
                    )
                    continue
                categories[idx] = result.get("category", DEFAULT_CATEGORY[0]) or DEFAULT_CATEGORY[0]
                subcategories[idx] = result.get("subcategory", DEFAULT_CATEGORY[1]) or DEFAULT_CATEGORY[1]
                exclude_flags[idx] = bool(result.get("exclude", False))
                sources[idx] = result.get("source", "llm")
                confidences[idx] = result.get("confidence")
        except Exception as exc:
            logger.error(
                "LLM batch categorization failed for chunk starting idx %s: %s",
                chunk[0]["idx"],
                exc,
            )

    resolved = sum(1 for cat in categories if cat != DEFAULT_CATEGORY[0])
    logger.info("Categorization finished. %d/%d transactions assigned.", resolved, total)
    return categories, subcategories, exclude_flags, sources, confidences


# 4️⃣ LLM batch with structured outputs
def categorize_llm_batch(chunk):
    if not chunk:
        return {}

    # Create structured output schema
    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "category": {
                            "type": "string",
                            "enum": CATEGORIES
                        },
                        "subcategory": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    },
                    "required": ["id", "category", "subcategory", "confidence"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["results"],
        "additionalProperties": False
    }

    prompt_payload = [
        {
            "id": item["idx"],
            "description": item["description"],
            "amount": item["amount"],
        }
        for item in chunk
    ]

    prompt = (
        f"Categorize the following transactions using ONLY these categories: {', '.join(CATEGORIES)}\n"
        f"For each category, use these subcategories: {json.dumps(SUBCATEGORIES, indent=2)}\n"
        "Return a JSON object named `results` containing one entry per transaction. "
        "Each entry must include: `id`, `category`, `subcategory`, and `confidence` "
        "(a number between 0 and 1).\n\n"
        "Transactions:\n"
        f"{json.dumps(prompt_payload, indent=2)}"
    )

    instructions = (
        "You are a helpful budgeting assistant. "
        "Categorize each transaction using the provided categories and subcategories. "
        "Choose the most appropriate category and subcategory for each transaction. "
        "For each assignment, also provide a confidence score between 0 and 1 that reflects how certain you are in the chosen category. "
        "If you are uncertain about the category after reviewing the description and the absolute value of the transaction amount is greater than 50, call the web_search tool to gather more context. "
        "Avoid calling the web_search tool when you are confident or when the transaction amount is 50 or less."
    )

    logger.info("Invoking Responses API with structured outputs for batch size %d.", len(chunk))

    try:
        response = client.responses.create(
            model="gpt-5",
            input=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "transaction_categorization",
                    "schema": schema,
                    "strict": True,
                }
            },
            reasoning={
                "effort": "low",
            },
            tools=[
                {
                    "type": "web_search_preview",
                    "search_context_size": "low",
                }
            ],
        )

        raw_content = getattr(response, "output_text", None)
        if not raw_content:
            segments = []
            for output_item in getattr(response, "output", []) or []:
                for content_item in getattr(output_item, "content", []) or []:
                    text_value = getattr(content_item, "text", None)
                    if text_value:
                        segments.append(text_value)
            raw_content = "".join(segments)

        if not raw_content:
            logger.warning("Responses API call returned no textual content; defaulting to fallback.")
            return {}

        result = json.loads(raw_content)
        entries = result.get("results", [])

        if not entries:
            logger.warning("LLM structured response missing results; defaulting to fallback.")
            return {}

        outputs = {}
        for entry in entries:
            idx = entry.get("id")
            if idx is None:
                continue
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                logger.warning("Skipping LLM entry with non-integer id: %s", idx)
                continue
            
            # Validate category and subcategory
            category = entry.get("category", DEFAULT_CATEGORY[0])
            subcategory = entry.get("subcategory", DEFAULT_CATEGORY[1])
            confidence = entry.get("confidence")

            # Ensure subcategory is valid for the category
            if category in SUBCATEGORIES and subcategory not in SUBCATEGORIES[category]:
                subcategory = SUBCATEGORIES[category][0]  # Use first valid subcategory
                logger.warning(f"Invalid subcategory '{entry.get('subcategory')}' for category '{category}'; using '{subcategory}'")

            # Clamp confidence into [0, 1] if provided
            try:
                if confidence is not None:
                    confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                logger.warning("Invalid confidence '%s' for idx=%s; defaulting to None.", entry.get("confidence"), idx)
                confidence = None

            outputs[idx] = {
                "category": category,
                "subcategory": subcategory,
                "exclude": False,
                "source": "llm",
                "confidence": confidence,
            }

        return outputs

    except Exception as e:
        logger.error(f"Structured output categorization failed: {e}")
        # Fallback to default categorization
        return {
            item["idx"]: {
                "category": DEFAULT_CATEGORY[0],
                "subcategory": DEFAULT_CATEGORY[1],
                "exclude": False,
                "source": "default",
                "confidence": None,
            }
            for item in chunk
        }


# 5️⃣ helpers
def _chunked(iterable, size):
    if size <= 0:
        raise ValueError("batch_size must be positive.")
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]
