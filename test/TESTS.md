## Testing & Verification

This package includes a focused, deterministic test suite designed to validate the mathematical correctness and structural integrity of our transformer interpretability tools. All tests run in an isolated environment.

### Test Coverage Breakdown

| Test Suite | Assertions | What It Verifies |
|------------|------------|------------------|
| **Analytic: Causal Masking Property** | 3 | - Matrix dimensions validated (3x3)<br>- Upper-triangular zeros enforced (`attn[i,j] == 0.0` for `j > i`)<br>- Row-wise softmax normalization verified (`sum(attn, dims=2) ≈ 1.0`) |
| **Module Structure** | 1 | - Package loads correctly as a Julia module with proper namespace isolation |
| **Visualization Type Exists** | 1 | - `AttentionHeatmap` type is properly exported and accessible to end users |
| **Edge Cases** | 2 | - Type mismatches (`nothing` or `String` instead of `ChatBot`) correctly trigger `MethodError`, confirming strict type dispatch |

### Why This Approach?
Instead of relying on heavy model inference or GPU resources, we use analytic reference tests on small, known-input matrices. This ensures:
- Instant execution (<1s) with fully deterministic results
- Mathematical rigor by directly verifying transformer attention invariants
- CI/CD compatibility for automated validation on every pull request