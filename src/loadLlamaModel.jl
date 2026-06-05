# import llama
using Llama2

function load_llama_model(model_path, vocabulary_path)
    # check if model file exists
    if !isfile(model_path)
        return "Model Path does not exist"
    end

    if !isfile(vocabulary_path)
        return "Vocabulary Path does not exist"
    end

    # create llama model
    bot = Llama2.ChatBot(model_path; vocabpath=vocabulary_path)

    # return object
    return bot
end