"""Prompt mutation operators."""

import logging
import random
import re
import string
import unicodedata
from typing import Optional

from valence.model import Model
from valence.util import hash_prompt

logger = logging.getLogger(__name__)


class MutationError(Exception):
    """Mutation generation error."""
    pass


def mutate_len_constraint_2sent(prompt: str) -> str:
    """Apply length constraint mutation: limit to 2 sentences."""
    sentences = prompt.split(". ")
    if len(sentences) <= 2:
        return prompt + " Keep your response to at most 2 sentences."
    
    first_two = ". ".join(sentences[:2])
    if not first_two.endswith("."):
        first_two += "."
    return first_two + " Respond in 2 sentences maximum."


def mutate_plain_english(prompt: str) -> str:
    """Apply plain English mutation."""
    additions = [
        " Explain in plain English.",
        " Use simple, everyday language.",
        " Avoid technical jargon.",
        " Explain like I'm five.",
    ]
    return prompt + random.choice(additions)


def mutate_role_expert(prompt: str) -> str:
    """Apply role expert mutation."""
    roles = [
        "You are an expert assistant.",
        "Act as a knowledgeable professional.",
        "You are a helpful expert in this field.",
        "As an experienced specialist,",
    ]
    return f"{random.choice(roles)} {prompt}"


def mutate_digits_only(prompt: str) -> str:
    """Apply digits-only mutation."""
    return prompt + " Respond with only digits, no words or explanations."


def mutate_semantic_paraphrase(prompt: str, llm_model: str = "openai:gpt-4o-mini") -> str:
    """Semantically paraphrase the prompt while preserving intent."""
    mutation_prompt = f"""Rewrite this prompt to have the same meaning but different wording:

Original: {prompt}

Requirements:
- Keep the same intent and expected response type
- Use different vocabulary and sentence structure  
- Maintain the same level of specificity
- Return only the rewritten prompt, no explanations"""
    
    try:
        model = Model(llm_model)
        response, _, error = model.generate(mutation_prompt)
        
        if error or not response:
            raise MutationError(f"LLM paraphrase mutation failed: {error}")
        
        return response.strip()
    except Exception as e:
        raise MutationError(f"LLM paraphrase mutation failed: {e}")


def mutate_add_complexity(prompt: str, llm_model: str = "openai:gpt-4o-mini") -> str:
    """Add realistic complexity while maintaining the core request."""
    mutation_prompt = f"""Make this prompt more complex and realistic while keeping the same core request:

Original: {prompt}

Add realistic constraints, context, or edge cases that a real user might include.
Examples:
- Time constraints: "by tomorrow" or "within 30 minutes"
- Budget limits: "under $50" or "free options only"
- Specific requirements: "for beginners" or "that work on mobile"
- Context: "for my startup" or "for remote workers"

Return only the enhanced prompt, no explanations."""
    
    try:
        model = Model(llm_model)
        response, _, error = model.generate(mutation_prompt)
        
        if error or not response:
            raise MutationError(f"LLM complexity mutation failed: {error}")
        
        return response.strip()
    except Exception as e:
        raise MutationError(f"LLM complexity mutation failed: {e}")


def mutate_context_shift(prompt: str, llm_model: str = "openai:gpt-4o-mini") -> str:
    """Shift the context while preserving the core request."""
    mutation_prompt = f"""Rewrite this prompt by changing the context or domain while keeping the same type of request:

Original: {prompt}

Examples of context shifts:
- Business context → Personal context
- Technical domain → Non-technical domain  
- Formal tone → Casual tone
- Generic request → Industry-specific request

Return only the context-shifted prompt, no explanations."""
    
    try:
        model = Model(llm_model)
        response, _, error = model.generate(mutation_prompt)
        
        if error or not response:
            raise MutationError(f"LLM context shift mutation failed: {error}")
        
        return response.strip()
    except Exception as e:
        raise MutationError(f"LLM context shift mutation failed: {e}")


# Noise/Robustness Mutations

def mutate_add_typos(prompt: str) -> str:
    """Add common typos to test robustness."""
    if len(prompt) < 10:
        return prompt
    
    words = prompt.split()
    if not words:
        return prompt
    
    # Select a random word to introduce typo
    word_idx = random.randint(0, len(words) - 1)
    word = words[word_idx]
    
    if len(word) > 3:
        # Common typo patterns
        typo_patterns = [
            lambda w: w[0] + w[2] + w[1] + w[3:] if len(w) > 3 else w,  # Transpose
            lambda w: w[:-1],  # Missing last letter
            lambda w: w + w[-1],  # Doubled last letter
            lambda w: w.replace('e', '3').replace('a', '@') if random.random() > 0.5 else w,  # Leetspeak
        ]
        words[word_idx] = random.choice(typo_patterns)(word)
    
    return ' '.join(words)


def mutate_unicode_confusables(prompt: str) -> str:
    """Replace ASCII characters with Unicode lookalikes."""
    confusables = {
        'a': 'а',  # Cyrillic
        'e': 'е',  # Cyrillic
        'o': 'о',  # Cyrillic
        'i': 'і',  # Ukrainian
        'A': 'Α',  # Greek Alpha
        'B': 'В',  # Cyrillic
        ' ': '\u00A0',  # Non-breaking space
    }
    
    result = prompt
    # Apply 1-3 confusable substitutions
    for _ in range(random.randint(1, 3)):
        char, replacement = random.choice(list(confusables.items()))
        if char in result:
            # Replace one occurrence
            idx = result.find(char)
            result = result[:idx] + replacement + result[idx+1:]
    
    return result


def mutate_whitespace_injection(prompt: str) -> str:
    """Inject unusual whitespace characters."""
    whitespace_chars = ['\t', '\n', '\r', '  ', '\u2003', '\u2009']  # Various spaces
    
    words = prompt.split()
    if len(words) < 2:
        return prompt
    
    # Insert unusual whitespace between words
    result = []
    for i, word in enumerate(words):
        result.append(word)
        if i < len(words) - 1:
            result.append(random.choice(whitespace_chars))
    
    return ''.join(result)


def mutate_emoji_injection(prompt: str) -> str:
    """Add emojis to test Unicode handling."""
    emojis = ['🔍', '📊', '💡', '✅', '❌', '🎯', '📈', '🤔']
    
    # Add emoji at beginning, end, or middle
    position = random.choice(['start', 'end', 'middle'])
    emoji = random.choice(emojis)
    
    if position == 'start':
        return f"{emoji} {prompt}"
    elif position == 'end':
        return f"{prompt} {emoji}"
    else:
        words = prompt.split()
        if len(words) > 2:
            mid = len(words) // 2
            words.insert(mid, emoji)
            return ' '.join(words)
    
    return prompt


# Locale/Format Mutations

def mutate_date_format(prompt: str) -> str:
    """Swap date formats between US and international."""
    # Pattern for dates like MM/DD or DD/MM
    date_pattern = r'\b(\d{1,2})/(\d{1,2})\b'
    
    def swap_date(match):
        return f"{match.group(2)}/{match.group(1)}"
    
    result = re.sub(date_pattern, swap_date, prompt)
    
    # Also handle written months
    us_to_intl = {
        'January': 'Januar', 'February': 'Februar',
        'March': 'März', 'April': 'April',
        'May': 'Mai', 'June': 'Juni',
        'July': 'Juli', 'August': 'August'
    }
    
    for us, intl in us_to_intl.items():
        if us in result and random.random() > 0.5:
            result = result.replace(us, intl)
            break
    
    return result


def mutate_units_conversion(prompt: str) -> str:
    """Convert between metric and imperial units."""
    conversions = [
        (r'(\d+)\s*miles?', lambda m: f"{int(float(m.group(1)) * 1.6)} km"),
        (r'(\d+)\s*km', lambda m: f"{int(float(m.group(1)) / 1.6)} miles"),
        (r'(\d+)\s*hours?', lambda m: f"{int(float(m.group(1)) * 60)} minutes"),
        (r'(\d+)\s*pounds?', lambda m: f"{int(float(m.group(1)) * 0.45)} kg"),
    ]
    
    for pattern, replacement in conversions:
        if re.search(pattern, prompt, re.IGNORECASE):
            return re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)
    
    return prompt


def mutate_colloquialisms(prompt: str) -> str:
    """Add regional colloquialisms."""
    colloquial_replacements = {
        'this afternoon': 'this arvo',
        'definitely': 'defo',
        'probably': 'probs',
        'information': 'info',
        'documentation': 'docs',
        'administrator': 'admin',
        'application': 'app',
    }
    
    result = prompt.lower()
    for formal, casual in colloquial_replacements.items():
        if formal in result:
            result = result.replace(formal, casual)
            break
    
    # Preserve original casing for first letter
    if prompt and prompt[0].isupper():
        result = result[0].upper() + result[1:]
    
    return result


# Constraint Stress Mutations

def mutate_nested_filters(prompt: str) -> str:
    """Add nested filtering conditions."""
    filter_additions = [
        " AND duration < 30 minutes AND rating > 4.0",
        " WHERE (category = 'leadership' OR category = 'management') AND level = 'beginner'",
        " that are both free and premium with certificates",
        " (including archived but not draft) sorted by date descending",
    ]
    
    return prompt + random.choice(filter_additions)


def mutate_conflicting_constraints(prompt: str) -> str:
    """Add potentially conflicting constraints."""
    conflicts = [
        " that are free but cost less than $50",
        " for beginners with advanced prerequisites",
        " completed within 5 minutes but at least 1 hour long",
        " available immediately but starting next month",
    ]
    
    return prompt + random.choice(conflicts)


def mutate_order_shuffle(prompt: str) -> str:
    """Shuffle enumerated items to test order invariance."""
    # Look for numbered or bulleted lists
    lines = prompt.split('\n')
    list_items = []
    other_lines = []
    
    for line in lines:
        if re.match(r'^[\d\-\*\•]\s*', line.strip()):
            list_items.append(line)
        else:
            other_lines.append(line)
    
    if len(list_items) > 1:
        random.shuffle(list_items)
        # Reconstruct with shuffled list
        result = []
        list_inserted = False
        for line in other_lines:
            result.append(line)
            if not list_inserted and line.strip() and not list_items:
                result.extend(list_items)
                list_inserted = True
        if not list_inserted:
            result.extend(list_items)
        return '\n'.join(result)
    
    # If no list found, shuffle comma-separated items
    if ', ' in prompt:
        parts = prompt.split(', ')
        if len(parts) > 2:
            # Keep first part, shuffle middle, keep last
            middle = parts[1:-1]
            random.shuffle(middle)
            return parts[0] + ', ' + ', '.join(middle) + ', ' + parts[-1]
    
    return prompt


def mutate_ambiguity_injection(prompt: str, llm_model: str = "openai:gpt-4o-mini") -> str:
    """Inject ambiguity to test robustness."""
    mutation_prompt = f"""Make this prompt more ambiguous while keeping it realistic:

Original: {prompt}

Add ambiguity through:
- Vague references: "that thing we discussed"
- Unclear quantities: "some", "a few", "around"
- Missing context: remove specific details
- Pronouns without clear antecedents

The result should still be interpretable but require the system to handle uncertainty.
Return only the ambiguous prompt, no explanations."""
    
    try:
        model = Model(llm_model)
        response, _, error = model.generate(mutation_prompt)
        
        if error or not response:
            raise MutationError(f"LLM ambiguity mutation failed: {error}")
        
        return response.strip()
    except Exception as e:
        raise MutationError(f"LLM ambiguity mutation failed: {e}")


# Basic (deterministic) mutation operators
BASIC_MUTATION_OPERATORS = {
    "len-constraint-2sent": mutate_len_constraint_2sent,
    "plain-english": mutate_plain_english,
    "role-expert": mutate_role_expert,
    "digits-only": mutate_digits_only,
}

# Noise/Robustness operators
NOISE_MUTATION_OPERATORS = {
    "add-typos": mutate_add_typos,
    "unicode-confusables": mutate_unicode_confusables,
    "whitespace-injection": mutate_whitespace_injection,
    "emoji-injection": mutate_emoji_injection,
}

# Locale/Format operators
LOCALE_MUTATION_OPERATORS = {
    "date-format": mutate_date_format,
    "units-conversion": mutate_units_conversion,
    "colloquialisms": mutate_colloquialisms,
}

# Constraint Stress operators
CONSTRAINT_MUTATION_OPERATORS = {
    "nested-filters": mutate_nested_filters,
    "conflicting-constraints": mutate_conflicting_constraints,
    "order-shuffle": mutate_order_shuffle,
}

# LLM-based (semantic) mutation operators
LLM_MUTATION_OPERATORS = {
    "semantic-paraphrase": mutate_semantic_paraphrase,
    "add-complexity": mutate_add_complexity,
    "context-shift": mutate_context_shift,
    "ambiguity-injection": mutate_ambiguity_injection,
}

# Combined operators (for backward compatibility)
MUTATION_OPERATORS = {
    **BASIC_MUTATION_OPERATORS, 
    **NOISE_MUTATION_OPERATORS,
    **LOCALE_MUTATION_OPERATORS,
    **CONSTRAINT_MUTATION_OPERATORS,
    **LLM_MUTATION_OPERATORS
}


def generate_mutations(
    prompt: str,
    parent_id: str,
    num_mutations: int = 4,
    generation: int = 0,
    use_llm_mutations: bool = False,
    llm_model: str = "openai:gpt-4o-mini",
    max_generation: int = 1,  # Configurable depth
    seen_tags: Optional[set] = None,  # Track failure tags
    mutation_budget: int = 100,  # Total mutation budget
) -> list[dict[str, str]]:
    """Generate mutations for a failed prompt with explosion guards."""
    # Guard against mutation explosion with configurable depth
    if generation >= max_generation:
        logger.info(f"Max generation {max_generation} reached for {parent_id}")
        return []
    
    # Stop if mutation budget exhausted
    if mutation_budget <= 0:
        logger.info(f"Mutation budget exhausted for {parent_id}")
        return []
    
    # Select operators based on mutation strategy
    if use_llm_mutations:
        available_operators = MUTATION_OPERATORS
        logger.info(f"Using LLM mutations with model: {llm_model}")
    else:
        available_operators = BASIC_MUTATION_OPERATORS
        logger.info("Using basic mutations only")
    
    # Apply budget limit
    num_mutations = min(num_mutations, mutation_budget)
    
    if num_mutations > len(available_operators):
        num_mutations = len(available_operators)
    
    operators = random.sample(list(available_operators.keys()), num_mutations)
    
    mutations = []
    seen_hashes = set()
    
    for i, operator in enumerate(operators, 1):
        try:
            mutate_fn = available_operators[operator]
            
            # Pass llm_model parameter to LLM-based mutations
            if operator in LLM_MUTATION_OPERATORS:
                mutated_prompt = mutate_fn(prompt, llm_model)
            else:
                mutated_prompt = mutate_fn(prompt)
            
            prompt_hash = hash_prompt(mutated_prompt)
            
            if prompt_hash in seen_hashes:
                logger.debug(f"Duplicate mutation skipped: {operator}")
                continue
            
            seen_hashes.add(prompt_hash)
            
            child_id = f"{parent_id}.c{i}"
            
            mutations.append({
                "id": child_id,
                "prompt": mutated_prompt,
                "parent_id": parent_id,
                "mutation_operator": operator,
                "generation": generation + 1,
            })
            
        except Exception as e:
            logger.error(f"Mutation {operator} failed: {e}")
            continue
    
    return mutations


def deduplicate_mutations(mutations: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate mutations by prompt hash."""
    seen = {}
    for mutation in mutations:
        prompt_hash = hash_prompt(mutation["prompt"])
        if prompt_hash not in seen:
            seen[prompt_hash] = mutation
    
    return list(seen.values())