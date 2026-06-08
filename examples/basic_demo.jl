using TransformerXAI

tokens = ["I", "like", "doing", "my", "homework."]

# Each row describes how much one token attends to all tokens.
# Strong examples are "like" -> "I", "doing" -> "like",
# and "homework." -> "doing" or "my".
attention = [
    1.0, 0.0, 0.0, 0.0, 0.0,  # I
    0.8, 1.0, 0.0, 0.0, 0.0,  # like
    0.2, 0.9, 1.0, 0.0, 0.0,  # doing
    0.1, 0.1, 0.3, 1.0, 0.0,  # my
    0.1, 0.2, 0.8, 0.9, 1.0,  # homework.
]

heatmap = visualize_heatmap(tokens, attention)

# A terminal cannot display SVG images, so save the heatmap to a file.
output_file = joinpath(@__DIR__, "attention_heatmap.svg")
open(output_file, "w") do file
    show(file, MIME("image/svg+xml"), heatmap)
end

println("Heatmap saved to: $output_file")
