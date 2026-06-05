mkpath("models")

download("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin", "models/stories42M.bin")
"stories42M.bin"

download("https://raw.githubusercontent.com/karpathy/llama2.c/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin", "models/tokenizer.bin")
"tokenizer.bin"