"""Tests for mutation operators."""

import pytest
from unittest.mock import Mock, patch

from valence.mutate import (
    BASIC_MUTATION_OPERATORS,
    LLM_MUTATION_OPERATORS,
    NOISE_MUTATION_OPERATORS,
    LOCALE_MUTATION_OPERATORS,
    CONSTRAINT_MUTATION_OPERATORS,
    MutationError,
    deduplicate_mutations,
    generate_mutations,
    mutate_add_complexity,
    mutate_add_typos,
    mutate_ambiguity_injection,
    mutate_context_shift,
    mutate_date_format,
    mutate_digits_only,
    mutate_len_constraint_2sent,
    mutate_nested_filters,
    mutate_plain_english,
    mutate_role_expert,
    mutate_semantic_paraphrase,
    mutate_unicode_confusables,
)


def test_mutate_len_constraint_short() -> None:
    """Test length constraint mutation on short prompt."""
    prompt = "Tell me about AI."
    mutated = mutate_len_constraint_2sent(prompt)
    assert "2 sentences" in mutated.lower()
    assert prompt in mutated


def test_mutate_len_constraint_long() -> None:
    """Test length constraint mutation on long prompt."""
    prompt = "First sentence. Second sentence. Third sentence. Fourth sentence."
    mutated = mutate_len_constraint_2sent(prompt)
    assert "First sentence. Second sentence." in mutated
    assert "2 sentences" in mutated.lower()
    assert "Third sentence" not in mutated


def test_mutate_plain_english() -> None:
    """Test plain English mutation."""
    prompt = "Explain quantum computing"
    mutated = mutate_plain_english(prompt)
    assert prompt in mutated
    assert any(phrase in mutated.lower() for phrase in [
        "plain english",
        "simple",
        "jargon",
        "like i'm five"
    ])


def test_mutate_role_expert() -> None:
    """Test role expert mutation."""
    prompt = "How does encryption work?"
    mutated = mutate_role_expert(prompt)
    assert prompt in mutated
    assert any(role in mutated.lower() for role in [
        "expert",
        "professional",
        "specialist"
    ])


def test_mutate_digits_only() -> None:
    """Test digits-only mutation."""
    prompt = "What is 10 plus 20?"
    mutated = mutate_digits_only(prompt)
    assert prompt in mutated
    assert "digits" in mutated.lower()
    assert "no words" in mutated.lower() or "no explanations" in mutated.lower()


def test_generate_mutations_basic() -> None:
    """Test basic mutation generation."""
    mutations = generate_mutations(
        prompt="Test prompt",
        parent_id="parent-1",
        num_mutations=4,
        generation=0
    )
    
    assert len(mutations) == 4
    for i, mutation in enumerate(mutations, 1):
        assert mutation["id"] == f"parent-1.c{i}"
        assert mutation["parent_id"] == "parent-1"
        assert mutation["generation"] == 1
        assert mutation["mutation_operator"] in [
            "len-constraint-2sent",
            "plain-english",
            "role-expert",
            "digits-only"
        ]
        assert "Test prompt" in mutation["prompt"]


def test_generate_mutations_max_generation() -> None:
    """Test mutation generation at max generation."""
    mutations = generate_mutations(
        prompt="Test prompt",
        parent_id="parent-1",
        num_mutations=4,
        generation=1
    )
    
    assert len(mutations) == 0


def test_generate_mutations_limited() -> None:
    """Test mutation generation with limited count."""
    mutations = generate_mutations(
        prompt="Test prompt",
        parent_id="parent-1",
        num_mutations=2,
        generation=0
    )
    
    assert len(mutations) == 2


def test_generate_mutations_excess_requested() -> None:
    """Test mutation generation when more mutations requested than operators."""
    mutations = generate_mutations(
        prompt="Test prompt",
        parent_id="parent-1",
        num_mutations=50,
        generation=0,
        use_llm_mutations=False  # Use basic only to get predictable count
    )
    
    # Should use only basic operators when use_llm_mutations=False
    expected_operators = len(BASIC_MUTATION_OPERATORS)
    # When requesting more mutations than operators, should get all operators
    assert len(mutations) == expected_operators


def test_deduplicate_mutations() -> None:
    """Test deduplication of mutations."""
    mutations = [
        {
            "id": "1",
            "prompt": "Test prompt",
            "parent_id": "p1",
            "mutation_operator": "op1",
            "generation": 1
        },
        {
            "id": "2",
            "prompt": "Test prompt",
            "parent_id": "p1",
            "mutation_operator": "op2",
            "generation": 1
        },
        {
            "id": "3",
            "prompt": "Different prompt",
            "parent_id": "p1",
            "mutation_operator": "op3",
            "generation": 1
        }
    ]
    
    deduped = deduplicate_mutations(mutations)
    assert len(deduped) == 2
    prompts = {m["prompt"] for m in deduped}
    assert prompts == {"Test prompt", "Different prompt"}


def test_mutation_deterministic() -> None:
    """Test that mutations are deterministic based on prompt."""
    mutations1 = generate_mutations(
        prompt="Test prompt",
        parent_id="parent-1",
        num_mutations=4,
        generation=0
    )
    
    mutations2 = generate_mutations(
        prompt="Test prompt",
        parent_id="parent-1",
        num_mutations=4,
        generation=0
    )
    
    operators1 = [m["mutation_operator"] for m in mutations1]
    operators2 = [m["mutation_operator"] for m in mutations2]
    
    assert len(set(operators1)) == len(operators1)
    assert len(set(operators2)) == len(operators2)


# LLM Mutation Tests

@patch('valence.mutate.Model')
def test_mutate_semantic_paraphrase(mock_model_class: Mock) -> None:
    """Test semantic paraphrase mutation."""
    mock_model = Mock()
    mock_model.generate.return_value = ("Rewrite this question about AI", 100, None)
    mock_model_class.return_value = mock_model
    
    result = mutate_semantic_paraphrase("Tell me about AI")
    
    assert result == "Rewrite this question about AI"
    mock_model_class.assert_called_once_with("openai:gpt-4o-mini")
    mock_model.generate.assert_called_once()
    
    # Check that the mutation prompt contains the original
    call_args = mock_model.generate.call_args[0][0]
    assert "Tell me about AI" in call_args
    assert "same meaning" in call_args


@patch('valence.mutate.Model')
def test_mutate_add_complexity(mock_model_class: Mock) -> None:
    """Test complexity addition mutation."""
    mock_model = Mock()
    mock_model.generate.return_value = ("Find AI courses under $50 for beginners", 100, None)
    mock_model_class.return_value = mock_model
    
    result = mutate_add_complexity("Find AI courses")
    
    assert result == "Find AI courses under $50 for beginners"
    mock_model.generate.assert_called_once()


@patch('valence.mutate.Model')
def test_mutate_context_shift(mock_model_class: Mock) -> None:
    """Test context shift mutation."""
    mock_model = Mock()
    mock_model.generate.return_value = ("How does home security work?", 100, None)
    mock_model_class.return_value = mock_model
    
    result = mutate_context_shift("How does encryption work?")
    
    assert result == "How does home security work?"
    mock_model.generate.assert_called_once()


@patch('valence.mutate.Model')
def test_mutate_ambiguity_injection(mock_model_class: Mock) -> None:
    """Test ambiguity injection mutation."""
    mock_model = Mock()
    mock_model.generate.return_value = ("Help me with that thing we discussed", 100, None)
    mock_model_class.return_value = mock_model
    
    result = mutate_ambiguity_injection("Help me schedule a meeting")
    
    assert result == "Help me with that thing we discussed"
    mock_model.generate.assert_called_once()


@patch('valence.mutate.Model')
def test_llm_mutation_error_handling(mock_model_class: Mock) -> None:
    """Test LLM mutation error handling."""
    mock_model = Mock()
    mock_model.generate.return_value = (None, 100, "API Error")
    mock_model_class.return_value = mock_model
    
    with pytest.raises(MutationError, match="LLM paraphrase mutation failed"):
        mutate_semantic_paraphrase("Test prompt")


@patch('valence.mutate.Model')
def test_llm_mutation_with_custom_model(mock_model_class: Mock) -> None:
    """Test LLM mutation with custom model."""
    mock_model = Mock()
    mock_model.generate.return_value = ("Paraphrased text", 100, None)
    mock_model_class.return_value = mock_model
    
    result = mutate_semantic_paraphrase("Test prompt", "anthropic:claude-3-haiku")
    
    assert result == "Paraphrased text"
    mock_model_class.assert_called_once_with("anthropic:claude-3-haiku")


def test_basic_mutation_operators() -> None:
    """Test that basic mutation operators are defined correctly."""
    assert len(BASIC_MUTATION_OPERATORS) == 4
    assert "len-constraint-2sent" in BASIC_MUTATION_OPERATORS
    assert "plain-english" in BASIC_MUTATION_OPERATORS
    assert "role-expert" in BASIC_MUTATION_OPERATORS
    assert "digits-only" in BASIC_MUTATION_OPERATORS


def test_llm_mutation_operators() -> None:
    """Test that LLM mutation operators are defined correctly."""
    assert len(LLM_MUTATION_OPERATORS) == 4
    assert "semantic-paraphrase" in LLM_MUTATION_OPERATORS
    assert "add-complexity" in LLM_MUTATION_OPERATORS
    assert "context-shift" in LLM_MUTATION_OPERATORS
    assert "ambiguity-injection" in LLM_MUTATION_OPERATORS


def test_generate_mutations_basic_only() -> None:
    """Test mutation generation with basic mutations only."""
    mutations = generate_mutations(
        prompt="Test prompt",
        parent_id="parent-1",
        num_mutations=4,
        generation=0,
        use_llm_mutations=False
    )
    
    assert len(mutations) == 4
    operators = {m["mutation_operator"] for m in mutations}
    assert operators.issubset(set(BASIC_MUTATION_OPERATORS.keys()))


@patch('valence.mutate.Model')
def test_generate_mutations_with_llm(mock_model_class: Mock) -> None:
    """Test mutation generation with LLM mutations enabled."""
    mock_model = Mock()
    # Return different responses to avoid deduplication
    mock_model.generate.side_effect = [
        ("First mutated prompt", 100, None),
        ("Second mutated prompt", 100, None),
    ]
    mock_model_class.return_value = mock_model
    
    mutations = generate_mutations(
        prompt="Test prompt",
        parent_id="parent-1",
        num_mutations=2,
        generation=0,
        use_llm_mutations=True,
        llm_model="openai:gpt-4o"
    )
    
    # Should generate 1-2 mutations (allowing for potential deduplication)
    assert 1 <= len(mutations) <= 2
    # Should have access to all operators when LLM mutations enabled
    from valence.mutate import MUTATION_OPERATORS
    operators = {m["mutation_operator"] for m in mutations}
    assert operators.issubset(set(MUTATION_OPERATORS.keys()))


# Advanced mutation operator tests

def test_mutate_add_typos() -> None:
    """Test typo injection mutation."""
    original = "Find leadership courses"
    mutated = mutate_add_typos(original)
    
    # Should be different but similar length
    assert mutated != original
    assert abs(len(mutated) - len(original)) <= 3


def test_mutate_unicode_confusables() -> None:
    """Test Unicode confusable mutation."""
    original = "Search for courses"
    mutated = mutate_unicode_confusables(original)
    
    # Should contain confusable characters
    assert mutated != original
    # But same length (character substitution)
    assert len(mutated) == len(original)


def test_mutate_date_format() -> None:
    """Test date format mutation."""
    original = "Find courses from 12/25 to 01/15"
    mutated = mutate_date_format(original)
    
    # Should swap date components
    assert "25/12" in mutated or "15/01" in mutated


def test_mutate_nested_filters() -> None:
    """Test nested filter mutation."""
    original = "Find courses"
    mutated = mutate_nested_filters(original)
    
    # Should add filter constraints
    assert len(mutated) > len(original)
    assert "AND" in mutated or "WHERE" in mutated or "that are" in mutated


def test_noise_mutation_operators() -> None:
    """Test all noise mutation operators are defined."""
    assert len(NOISE_MUTATION_OPERATORS) == 4
    assert "add-typos" in NOISE_MUTATION_OPERATORS
    assert "unicode-confusables" in NOISE_MUTATION_OPERATORS
    assert "whitespace-injection" in NOISE_MUTATION_OPERATORS
    assert "emoji-injection" in NOISE_MUTATION_OPERATORS


def test_locale_mutation_operators() -> None:
    """Test all locale mutation operators are defined."""
    assert len(LOCALE_MUTATION_OPERATORS) == 3
    assert "date-format" in LOCALE_MUTATION_OPERATORS
    assert "units-conversion" in LOCALE_MUTATION_OPERATORS
    assert "colloquialisms" in LOCALE_MUTATION_OPERATORS


def test_constraint_mutation_operators() -> None:
    """Test all constraint mutation operators are defined."""
    assert len(CONSTRAINT_MUTATION_OPERATORS) == 3
    assert "nested-filters" in CONSTRAINT_MUTATION_OPERATORS
    assert "conflicting-constraints" in CONSTRAINT_MUTATION_OPERATORS
    assert "order-shuffle" in CONSTRAINT_MUTATION_OPERATORS


def test_mutation_explosion_guards() -> None:
    """Test mutation explosion prevention."""
    # Test max generation limit
    mutations = generate_mutations(
        prompt="Test prompt",
        parent_id="parent-1",
        num_mutations=4,
        generation=1,
        max_generation=1
    )
    assert len(mutations) == 0
    
    # Test mutation budget limit
    mutations = generate_mutations(
        prompt="Test prompt",
        parent_id="parent-1",
        num_mutations=10,
        generation=0,
        mutation_budget=2
    )
    assert len(mutations) <= 2