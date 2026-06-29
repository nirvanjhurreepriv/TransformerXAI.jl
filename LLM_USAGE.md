# Usage for testing

## Prompt (short version): How can I test the svg file output without creating a file (isabella)
Link to conversation: https://chatgpt.com/share/6a33af9a-dfdc-83ed-b48e-aca2c3e613e6

## Prompt (short version): CI is failing on every push with a MbedTLS_jll precompilation error. Check the CI.yml and figure out what's wrong. (Nirvan)
Tool: Claude Code (claude-sonnet-4-6)

**What was asked:** Diagnose and fix a persistent CI failure where every push triggered a `MbedTLS_jll` precompilation error.

**What the LLM did:**
- Inspected `.github/workflows/CI.yml`, `Project.toml`, and both `Manifest.toml` files (root and `docs/`)
- Identified that both manifests were generated with `julia_version = "1.11.9"` and pin `MbedTLS_jll` at `2.28.6+0` (the stdlib JLL bundled with Julia 1.11.9), but CI used `version: '1.11'` which resolves to whatever the latest 1.11.x patch is — causing a stdlib JLL version mismatch on every run
- Identified a secondary issue: `[compat]` in `Project.toml` listed `LinearAlgebra = "1.11.0"` (non-standard; stdlib version floors should be expressed as `julia = "1.11"`)
- Fixed `CI.yml`: pinned Julia to `'1.11.9'` in both the test matrix and the docs job
- Fixed `Project.toml`: replaced `LinearAlgebra = "1.11.0"` with `julia = "1.11"`
- Committed and pushed the changes

**Files changed:** `.github/workflows/CI.yml`, `Project.toml`

**Follow-up required:** Run `julia --project=. -e 'using Pkg; Pkg.update()'` and `julia --project=docs -e 'using Pkg; Pkg.update()'` locally to regenerate the Manifests with the updated `project_hash`, then commit and push those.
