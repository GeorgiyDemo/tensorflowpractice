from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def main():
    device = torch.device("cpu")
    model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()
    text = "Я родился в 70 на краю города"
    input_ids = tokenizer.encode(text, return_tensors="pt").cuda()
    out = model.generate(input_ids.cuda())
    generated_text = list(map(tokenizer.decode, out))[0]
    print(generated_text)
    # Output should be like this:
    # Александр Сергеевич Пушкин родился в \n1799 году. Его отец был крепостным крестьянином, а мать – крепостной крестьянкой. Детство и юность Пушкина прошли в деревне Михайловское под Петербургом. В 1820-х годах семья переехала


if __name__ == "__main__":
    main()
