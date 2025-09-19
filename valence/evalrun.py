"""Orchestrate evaluation runs, write JSONL files, and update memory."""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from valence.detectors import run_detectors
from valence.model import Model
from valence.mutate import generate_mutations
from valence.packs import load_packs, load_seeds
from valence.schema import (
    EvalRecord,
    FailureRecord,
    Lineage,
    Outcome,
    Pack,
    RunMetadata,
    Seed,
)
from valence.storage import MemoryStore, RunStorage
from valence.util import hash_prompt, timer

logger = logging.getLogger(__name__)


class EvalRunner:
    """Orchestrates evaluation runs."""
    
    def __init__(
        self,
        model: Model,
        seeds: list[Seed],
        packs: list[Pack],
        output_dir: Path,
        memory_path: Optional[Path] = None,
        max_generations: int = 1,
        mutations_per_failure: int = 4,
        use_llm_mutations: bool = False,
        mutation_model: str = "openai:gpt-4o-mini",
    ):
        """Initialize evaluation runner."""
        self.model = model
        self.seeds = seeds
        self.packs = packs
        self.output_dir = output_dir
        self.max_generations = max_generations
        self.mutations_per_failure = mutations_per_failure
        self.use_llm_mutations = use_llm_mutations
        self.mutation_model = mutation_model
        
        self.storage = RunStorage(output_dir)
        self.memory = MemoryStore(memory_path) if memory_path else None
        
        self.run_id = f"run-{uuid.uuid4().hex[:8]}"
        self.metadata = RunMetadata(
            run_id=self.run_id,
            model=model.name,
            started_at=datetime.utcnow(),
            max_generations=max_generations,
            mutations_per_failure=mutations_per_failure,
        )
        
        self.pending_evaluations: list[dict] = []
        self.completed_evaluations: list[EvalRecord] = []
    
    def evaluate_prompt(
        self,
        prompt_id: str,
        prompt: str,
        pack: Pack,
        lineage: Optional[Lineage] = None,
        seed: Optional[Seed] = None,
    ) -> EvalRecord:
        """Evaluate a single prompt."""
        if lineage is None:
            lineage = Lineage()
        
        logger.info(f"Evaluating {prompt_id}")
        
        with timer() as t:
            response, model_duration = self.model.generate(prompt)
        
        if response is None:
            scorecard = None
            error = "Model returned no response"
        else:
            scorecard = run_detectors(pack, prompt, response, seed)
            error = None
        
        record = EvalRecord(
            id=prompt_id,
            prompt=prompt,
            response=response,
            scorecard=scorecard,
            lineage=lineage,
            timestamp=datetime.utcnow(),
            duration_ms=t["duration_ms"],
            model=self.model.name,
            pack_id=pack.id,
            error=error,
        )
        
        self.storage.append_eval(record)
        self.completed_evaluations.append(record)
        
        if scorecard:
            if scorecard.outcome == Outcome.STRONG:
                self.metadata.total_failures += 1
                self._handle_failure(record, pack)
            elif scorecard.outcome == Outcome.MILD:
                # Mild scores trigger mutations but count as passes for stats
                self.metadata.total_passes += 1
                self._handle_failure(record, pack)
            elif scorecard.outcome == Outcome.WEAK:
                self.metadata.total_passes += 1
            else:
                self.metadata.total_errors += 1
        else:
            self.metadata.total_errors += 1
        
        return record
    
    def _handle_failure(self, record: EvalRecord, pack: Pack) -> None:
        """Handle a failing evaluation."""
        if self.memory and record.scorecard:
            failure = FailureRecord(
                prompt=record.prompt,
                prompt_hash=hash_prompt(record.prompt),
                id=record.id,
                parent_id=record.lineage.parent_id,
                mutation_operator=record.lineage.mutation_operator,
                timestamp=record.timestamp,
                model=self.model.name,
                pack_id=pack.id,
                tags=record.scorecard.tags,
            )
            self.memory.add_failure(failure)
        
        if record.lineage.generation < self.max_generations:
            mutations = generate_mutations(
                prompt=record.prompt,
                parent_id=record.id,
                num_mutations=self.mutations_per_failure,
                generation=record.lineage.generation,
                use_llm_mutations=self.use_llm_mutations,
                llm_model=self.mutation_model,
                max_generation=self.max_generations,
            )
            
            for mutation in mutations:
                self.pending_evaluations.append({
                    "id": mutation["id"],
                    "prompt": mutation["prompt"],
                    "pack": pack,
                    "lineage": Lineage(
                        parent_id=mutation["parent_id"],
                        mutation_operator=mutation["mutation_operator"],
                        generation=mutation["generation"],
                    ),
                    "seed": None,  # Mutations don't have original seed labels
                })
    
    def run(self) -> RunMetadata:
        """Run the complete evaluation."""
        logger.info(f"Starting evaluation run {self.run_id}")
        
        for seed in self.seeds:
            for pack in self.packs:
                self.pending_evaluations.append({
                    "id": seed.id,
                    "prompt": seed.prompt,
                    "pack": pack,
                    "lineage": Lineage(),
                    "seed": seed,
                })
        
        while self.pending_evaluations:
            eval_spec = self.pending_evaluations.pop(0)
            self.evaluate_prompt(
                prompt_id=eval_spec["id"],
                prompt=eval_spec["prompt"],
                pack=eval_spec["pack"],
                lineage=eval_spec["lineage"],
                seed=eval_spec.get("seed"),
            )
            self.metadata.total_prompts += 1
        
        self.metadata.completed_at = datetime.utcnow()
        self.storage.save_metadata(self.metadata)
        
        logger.info(
            f"Evaluation complete: {self.metadata.total_prompts} prompts, "
            f"{self.metadata.total_failures} failures, "
            f"{self.metadata.total_passes} passes, "
            f"{self.metadata.total_errors} errors"
        )
        
        return self.metadata


def run_evaluation(
    model_name: str,
    seeds_path: Path,
    packs_path: Path,
    output_dir: Path,
    memory_path: Optional[Path] = None,
    max_generations: int = 1,
    mutations_per_failure: int = 4,
    use_llm_mutations: bool = False,
    mutation_model: str = "openai:gpt-4o-mini",
) -> RunMetadata:
    """High-level function to run an evaluation."""
    model = Model(model_name)
    seeds = load_seeds(seeds_path)
    packs = load_packs(packs_path)
    
    runner = EvalRunner(
        model=model,
        seeds=seeds,
        packs=packs,
        output_dir=output_dir,
        memory_path=memory_path,
        max_generations=max_generations,
        mutations_per_failure=mutations_per_failure,
        use_llm_mutations=use_llm_mutations,
        mutation_model=mutation_model,
    )
    
    return runner.run()