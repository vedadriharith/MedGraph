import pandas as pd
import time
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from hybrid_rag import hybrid_search 

load_dotenv()

judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

test_dataset = [
    {"question": "What treatments are associated with Hirschsprung Disease?", "ground_truth": "Transanal Endorectal Pull-Through and Transabdominal Pull-Through."},
    {"question": "What condition is Transanal Endorectal Pull-Through used for?", "ground_truth": "Hirschsprung Disease."},
    {"question": "Does Aquagenic Urticaria affect infants?", "ground_truth": "Yes, it can manifest as a pediatric form."},
    {"question": "What are the treatments for hypertension?", "ground_truth": "Lifestyle changes, beta-blockers, diuretics."},
    {"question": "What is the connection between Landolt C and Strabismus?", "ground_truth": "Landolt C is associated with Strabismus Amblyopia measurement."}
]

def calculate_score(question, generated_answer, ground_truth):
    grading_template = """
    Compare ACTUAL ANSWER with GROUND TRUTH.
    Question: {question}
    Ground Truth: {ground_truth}
    Actual Answer: {generated_answer}
    Output a score 1-5 (1=Wrong, 5=Perfect). Return ONLY the integer.
    """
    prompt = ChatPromptTemplate.from_template(grading_template)
    grader = prompt | judge_llm | StrOutputParser()
    try:
        return int(grader.invoke({"question": question, "ground_truth": ground_truth, "generated_answer": generated_answer}).strip())
    except:
        return 3

def run_evaluation():
    results = []
    print("Starting Evaluation...")
    for item in test_dataset:
        start = time.time()
        try:
            ans = hybrid_search(item['question'])
        except Exception as e:
            ans = f"ERROR: {str(e)}"
        latency = time.time() - start
        score = calculate_score(item['question'], ans, item['ground_truth'])
        results.append({"Question": item['question'], "Score": score, "Latency": latency})
        print(f"Q: {item['question'][:30]}... | Score: {score}/5")
        
    pd.DataFrame(results).to_csv("research_results.csv", index=False)
    print("Evaluation Complete. Results saved.")

if __name__ == "__main__":
    run_evaluation()