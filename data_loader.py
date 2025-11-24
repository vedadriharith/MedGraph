from datasets import load_dataset
import pandas as pd

def load_medical_data():
    print("Loading PubMedQA dataset from HuggingFace...")
    # Loading the "pqa_labeled" subset (expert annotated)
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    # Convert to a simpler format for us to use
    documents = []
    for item in dataset:
        # We combine the context (abstract) with the question context
        context = " ".join(item['context']['contexts'])
        combined_text = f"Question: {item['question']}\nAbstract: {context}\nAnswer: {item['long_answer']}"
        documents.append(combined_text)
    
    print(f"Successfully loaded {len(documents)} medical research records.")
    return documents

if __name__ == "__main__":
    docs = load_medical_data()
    print("Sample Document:\n", docs[0][:500], "...")