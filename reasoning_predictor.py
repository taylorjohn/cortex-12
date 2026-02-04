"""
CORTEX-12 v14: Reasoning Predictor (System 2)
Implements 'Test-Time Scaling' via Beam Search and MC Sampling.

Concept:
Instead of blindly accepting the first prediction (System 1),
this module generates multiple potential latent futures and
selects the one that best preserves physical laws (SIGReg statistics).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from latent_predictor import TransformationPredictor, LatentPredictor

class ReasoningPredictor(TransformationPredictor):
    """
    Upgraded predictor that "thinks" before deciding.
    Uses Beam Search to find the most stable latent trajectory.
    """
    
    def __init__(self, embedding_dim=128, num_beams=5, num_samples=10):
        super().__init__(embedding_dim)
        self.num_beams = num_beams
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        
    def _evaluate_structural_integrity(self, embeddings):
        """
        Scores embeddings based on SIGReg principles (Physics Check).
        
        Per-sample scoring (since beam candidates arrive individually).
        Checks three things:
          1. Global norm stays near sqrt(dim) — embedding hasn't drifted
          2. No single dimension dominates (spike check, >3σ)
          3. Per-axis energy is balanced — no axis has collapsed to zero
             or exploded to dominate (uses known axis boundaries)
        
        Returns:
            scores: (batch,) Higher is better. 0.0 = perfect, negative = bad.
        """
        from cortex_adapter_v13 import AXIS_DIMS_V13

        dim = embeddings.shape[-1]
        
        # 1. Global norm check: expected ≈ sqrt(dim) for unit-variance dims
        expected_norm = dim ** 0.5
        norm = torch.norm(embeddings, dim=-1)
        norm_score = -torch.abs(norm - expected_norm)
        
        # 2. Spike check: no single value should exceed 3σ
        max_val = torch.max(torch.abs(embeddings), dim=-1).values
        spike_penalty = -torch.relu(max_val - 3.0)
        
        # 3. Per-axis energy balance
        # Each axis should carry energy proportional to its width.
        # A collapsed axis (near-zero energy) or exploded axis (dominant energy)
        # both indicate a broken embedding.
        axis_energies = []
        for name, (start, end) in AXIS_DIMS_V13.items():
            axis_slice = embeddings[:, start:end+1]
            width = end - start + 1
            # Energy per dimension for this axis (should be ~1.0 if unit variance)
            energy_per_dim = (axis_slice ** 2).mean(dim=-1)
            axis_energies.append(energy_per_dim)
        
        # Stack: (batch, num_axes)
        axis_energies = torch.stack(axis_energies, dim=-1)
        # Ideal: all axes have energy_per_dim ≈ 1.0
        # Penalize deviation from 1.0 across all axes, summed
        axis_balance_penalty = -torch.abs(axis_energies - 1.0).sum(dim=-1)
        
        total_score = norm_score + spike_penalty + axis_balance_penalty
        return total_score

    def expand_hypotheses(self, current_embeddings, transform_name):
        """
        Generates multiple potential outcomes for a transformation
        using Monte Carlo Dropout or Input Noise.
        """
        batch_size = current_embeddings.size(0)
        
        # Replicate embeddings for sampling
        # (batch * samples, 128)
        expanded_input = current_embeddings.repeat_interleave(self.num_samples, dim=0)
        
        # Inject slight noise to stimulate variation (Simulating 'Imagination')
        noise = torch.randn_like(expanded_input) * 0.02
        noisy_input = expanded_input + noise
        
        # Predict (Ensure dropout is ACTIVE for variation)
        self.predictor.train() # Enable dropout
        with torch.no_grad():
            transform_id = self.TRANSFORMS[transform_name]
            t_tensor = torch.tensor(
                [transform_id] * expanded_input.size(0),
                device=current_embeddings.device
            )
            
            # Get predictions
            # Note: We use the parent's self.predictor
            predictions, _ = self.predictor(
                noisy_input,
                transform_id=t_tensor
            )
            
        self.predictor.eval() # Reset to eval
        
        return predictions.view(batch_size, self.num_samples, -1)

    def plan_reasoned_trajectory(self, start_embedding, actions):
        """
        Beam Search for Latent Space Trajectories.
        System 2 Thinking: Explores multiple paths, keeps the most 'real'.
        
        Uses CUMULATIVE scoring so a locally mediocre step that leads
        to a globally stable trajectory is kept over a locally good
        step that degrades later.
        
        Args:
            start_embedding: (batch, 128)
            actions: List of transform names ['rotate_90', 'scale_up']
            
        Returns:
            best_trajectory: List of (batch, 128) steps
        """
        batch_size = start_embedding.size(0)
        device = start_embedding.device
        
        # Current Beam State: (batch, beam_width, embedding_dim)
        current_beams = start_embedding.unsqueeze(1).repeat(1, self.num_beams, 1)
        
        # Cumulative scores per beam: (batch, beam_width)
        # Initialize to 0 — all beams start equal
        cumulative_scores = torch.zeros(batch_size, self.num_beams, device=device)
        
        # History: store best beam at each step for trajectory reconstruction
        history = [start_embedding]
        
        print(f"🧠 Reasoning through {len(actions)} steps with {self.num_beams} beams...")
        
        for step_idx, action in enumerate(actions):
            # 1. EXPAND: Generate candidates from ALL current beams
            candidates = []
            for b in range(self.num_beams):
                beam_emb = current_beams[:, b, :]
                preds = self.expand_hypotheses(beam_emb, action)  # (batch, num_samples, 128)
                candidates.append(preds)
            
            # (batch, beam_width * num_samples, 128)
            candidates = torch.cat(candidates, dim=1)
            
            # 2. SCORE: Evaluate structural integrity of all candidates
            flat_candidates = candidates.view(-1, self.embedding_dim)
            step_scores = self._evaluate_structural_integrity(flat_candidates)
            step_scores = step_scores.view(batch_size, -1)  # (batch, beam*samples)
            
            # 3. ACCUMULATE: Add step scores to the parent beam's cumulative score
            # Each candidate inherits the cumulative score of whichever beam spawned it
            # Beam 0 spawned candidates [0..num_samples-1], beam 1 spawned [num_samples..2*num_samples-1], etc.
            parent_scores = cumulative_scores.repeat_interleave(self.num_samples, dim=1)  # (batch, beam*samples)
            total_scores = parent_scores + step_scores
            
            # 4. PRUNE: Select top-k by CUMULATIVE score (true beam search)
            top_k = torch.topk(total_scores, k=self.num_beams, dim=1)
            top_indices = top_k.indices   # (batch, num_beams)
            top_scores = top_k.values     # (batch, num_beams) — new cumulative scores
            
            # Gather the surviving beams
            gather_idx = top_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            current_beams = torch.gather(candidates, 1, gather_idx)
            
            # Update cumulative scores
            cumulative_scores = top_scores
            
            # Best beam (highest cumulative score) goes into history
            best_step_emb = current_beams[:, 0, :]
            history.append(best_step_emb)
            
            avg_score = top_scores.mean().item()
            # print(f"  Step {step_idx+1} ({action}): Cumulative Integrity = {avg_score:.4f}")

        return history