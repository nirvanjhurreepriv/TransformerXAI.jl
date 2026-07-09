"""
    load_llama_model(model_path::String, vocabulary_path::String)
    
The function creates a Llama2 chatbot.

# Arguments: 
    model_path: Path to the local model file
    vocabulary_path: Path to the local tokenizer/voabulary file 

# Returns: 
    An object of the type Llama2.Chatbot is returned

# Example: 
    bot = load_llama_model("models/stories24M.bin", "models/tokenizer.bin")
"""
function load_llama_model(model_path::String, vocabulary_path::String)
    # check if model file exists
    if !isfile(model_path) || !isfile(vocabulary_path)
        throw(ArgumentError("model path or vocabulary path does not exist"))
    end

    # create llama model
    bot = ChatBot(model_path; vocabpath=vocabulary_path)

    # return object
    return bot
end