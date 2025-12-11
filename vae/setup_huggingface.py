from transformers import AutoModel, AutoTokenizer

models = [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
]

for model in models:
    print(f'Loading {model}')
    AutoModel.from_pretrained(model)
    AutoTokenizer.from_pretrained(model)

print('done')
