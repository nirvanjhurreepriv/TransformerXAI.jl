# Testign & Verification

This package includes tests for package structure, model loading, modified Llama2 inference functions, attention extraction, attention rollout, and visualization utilities.

Some tests are model-independent and always run in CI. Tests that require local model files are only executed when the expected files are available:

```text
models/stories42M.bin
models/tokenizer.bin
```

## Test Overview
| Test Area                         | What It Verifies                                                                                                                                      |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Module structure**              | Checks that `TransformerXAI` loads correctly and that exported types such as `AttentionHeatmap` are available.                                        |
| **Input dispatch and edge cases** | Verifies that invalid inputs, such as `nothing` or a `String` instead of a `ChatBot`, trigger the expected errors.                                    |
| **Model loading**                 | Checks missing model/tokenizer paths and, if model files are available, verifies that `load_llama_model` returns a `Llama2.ChatBot`.                  |
| **Modified forward pass**         | Compares `forward_changed!` against `Llama2.forward!` when attention extraction is disabled and verifies logits/attention output shapes when enabled. |
| **Modified text generation**      | Checks that `talktollm_changed` returns generated text and, when requested, attention history.                                                        |
| **Attention extraction**          | Verifies returned attention matrix shape, finite values, tokenizer output, heatmap compatibility, and invalid layer/head error handling.              |
| **Attention rollout**             | Verifies rollout matrix shape, finite values, tokenizer output, and invalid rollout depth/head error handling.                                        |
| **Heatmap visualization**         | Tests `visualize_heatmap`, SVG output generation, SVG escaping, empty inputs, and invalid attention sizes. 
| **Heatmap matrix visualization**         | Tests `generate_heatmap_matrix`, Plot output generation, empty inputs, and invalid attention sizes.                                            |
| **Rollout visualization**         | Tests SVG output, layer/dot/line counts, attention strength encoding, zero-attention handling, and invalid inputs.                                    |

## Running tests locally
In the julia shell
```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

## TBD
* filter all possible tests that can be run without model files to achieve better and more reliable coverage