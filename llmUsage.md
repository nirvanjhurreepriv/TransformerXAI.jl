# Usage for testing

## Prompt (short version): How can I test the svg file output without creating a file (isabella)
Link to conversation: https://chatgpt.com/share/6a33af9a-dfdc-83ed-b48e-aca2c3e613e6

## Used LLM to convert Pseude-Code into Julia implementation (Mick)
in attentionHeatmapExpanded.jl for functions: visualize_heatmap, escape_svg, and Base.show implementations

## Write Println lines for building SVG (Mick)
in attentionRolloutVisualizer.jl

## Help with SVG based testing (Mick)
in rolloutVisualizerTest.jl

## Prompt (short version): Transform this main readme file into 3 files for the docs folder (isabella)
Link to conversation: https://chatgpt.com/share/6a42948b-3f04-83ed-901d-3c4d666685ef

## Help with creation/updating the TEST.md (isabella)
in test/TEST.md

## Prompt (short version): CI is failing on every push with a MbedTLS_jll precompilation error. Check the CI.yml and figure out what's wrong. (Nirvan)
Tool: Claude Code (claude-sonnet-4-6)

**What was asked:** Diagnose and fix a persistent CI failure where every push triggered a `MbedTLS_jll` precompilation error.

**What the LLM did:**
- Inspected `.github/workflows/CI.yml`, `Project.toml`, and both `Manifest.toml` files (root and `docs/`)
- Identified that both manifests were generated with `julia_version = "1.11.9"` and pin `MbedTLS_jll` at `2.28.6+0` (the stdlib JLL bundled with Julia 1.11.9), but CI used `version: '1.11'` which resolves to whatever the latest 1.11.x patch is — causing a stdlib JLL version mismatch on every run
- Identified a secondary issue: `[compat]` in `Project.toml` listed `LinearAlgebra = "1.11.0"` (non-standard; stdlib version floors should be expressed as `julia = "1.11"`)
- Fixed `ci.yml`: pinned Julia to `'1.11.9'` in both the test matrix and the docs job
- Fixed `Project.toml`: replaced `LinearAlgebra = "1.11.0"` with `julia = "1.11"`
- Committed and pushed the changes

**Files changed:** `.github/workflows/ci.yml`, `Project.toml`

## CI fix: Julia version mismatch and merge conflict resolution (Nirvan)
Tool: Claude Code (claude-sonnet-4-6) + Claude.ai (claude-sonnet-4-6)

**What the LLM did:**
- Fixed corrupted local Julia cache and switched to Julia 1.12.6 to match Manifest
- Updated `ci.yml` to use Julia '1.12' in both test and docs jobs
- Resolved 4-way merge conflict between `attention_flow` and `origin/main`
- Regenerated `Manifest.toml` on Julia 1.12.6 to fix stale manifest CI failure

**Files changed:** `Manifest.toml`, `.github/workflows/ci.yml`, `src/TransformerXAI.jl`, `test/runtests.jl`
