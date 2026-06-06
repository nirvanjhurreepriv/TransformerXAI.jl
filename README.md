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
2. "include("src/loadLlamaModel.jl")"
3. "bot = load_llama_model("models/stories42M.bin", "models/tokenizer.bin")"

## loadLlamaModel.jl
gibt ein objekt vom typ ChatBot zurück mit den Feldern: transformer, tokenizer, pos, last_token
### testen in shell, nachdem das modell geladen wurde
1. "typeof(bot)"
2. "fieldnames(typeof(bot))"

# Attention weights eines Layers abrufen (in shell, sonst wie üblich funktion nutzen)
1. in julia shell gehen
2. "include("src/extractAttWeights.jl")
3. "bot = load_llama_model("models/stories42M.bin", "models/tokenizer.bin")"
4. Entscheiden, welches Layer gewünscht ist (also eine integer auswählen, bspw. 7)
5. "attWeights = extract_att_weights_from_layer_llama(bot, 7)", wobei "7" mit dem Index des gewünschten Layers ersetzt wird. "attWeights" ist vom Typ Matrix{Float32} (oder Typ String, wenn die Funktionsattribute nicht gepasst haben).



# Llama2 Repo
https://github.com/ConstantConstantin/Llama2.jl