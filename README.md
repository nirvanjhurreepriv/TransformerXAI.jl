# TODO, after cloning:
1. "cd TRANSFORMERXAI.JL"
2. "julia --project=." (to get in julia shell)
3. "]" (to get in package mode)
4. "instantiate" (basically npm install for julia)
5. "exit()" (um aus julia shell raus zu kommen)
6. "julia --project=. scripts/download_model.jl"(download der modell dateien)

# Test, um zu sehen, ob Llama2 läuft
1. in julia shell gehen
2. "using Llama2"
3. "print(talktollm("models/stories42M.bin", "In a small village "))" 

# Modell laden (in shell, sonst einfach funktion nutzen)
1. in julia shell gehen
2. "include("src/loadLlamaModel.js")"
3. "bot = load_llama_model("models/stories42M.bin", "models/tokenizer.bin")"

## loadLlamaModel.jl
gibt ein objekt vom typ ChatBot zurück mit den Feldern: transformer, tokenizer, pos, last_token
### testen in shell, nachdem das modell geladen wurde
1. "typeof(bot)"
2. "fieldnames(typeof(bot))"


# Llama2 Repo
https://github.com/ConstantConstantin/Llama2.jl