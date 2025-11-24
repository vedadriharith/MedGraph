import pandas as pd
import time
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Import your actual system
from hybrid_rag import hybrid_search 

load_dotenv()

# Judge needs to be smart, so we use the 70b model (or 8b if you hit limits)
judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- EXPANDED DATASET (10 Questions) ---
test_dataset = [
    # Original 5 (To verify consistency)
    {"question": "What treatments are associated with Hirschsprung Disease?", "ground_truth": "Transanal Endorectal Pull-Through and Transabdominal Pull-Through."},
    {"question": "What condition is Transanal Endorectal Pull-Through used for?", "ground_truth": "Hirschsprung Disease."},
    {"question": "Does Aquagenic Urticaria affect infants?", "ground_truth": "Yes, it can manifest as a pediatric form."},
    {"question": "What are the treatments for hypertension?", "ground_truth": "Lifestyle changes, beta-blockers (propranolol), diuretics."},
    {"question": "What is the connection between Landolt C and Strabismus?", "ground_truth": "Landolt C is associated with Strabismus Amblyopia measurement."},
    
    # New 5 (To test the "Scaled" Graph)
    {"question": "What drugs are used to treat Graft-Versus-Host Disease (GVHD)?", "ground_truth": "Cyclosporine and Chloroquine."},
    {"question": "Is there a link between obesity and insulin resistance?", "ground_truth": "Yes, obesity is often associated with insulin resistance and diabetes."},
    {"question": "What are the potential side effects of statins?", "ground_truth": "Muscle pain, increased risk of diabetes, liver damage."},
    {"question": "Does asthma cause systemic inflammation?", "ground_truth": "Yes, asthma is associated with systemic inflammation and increased CRP levels."},
    {"question": "What is the relationship between Helicobacter pylori and gastric cancer?", "ground_truth": "H. pylori infection is a major cause/risk factor for gastric cancer."}
]

def calculate_score(question, generated_answer, ground_truth):
    grading_template = """
    You are an academic grader. Compare the ACTUAL ANSWER with the GROUND TRUTH.
    
    Question: {question}
    Ground Truth: {ground_truth}
    Actual Answer: {generated_answer}
    
    Criteria:
    - 5: Perfect fact retrieval + correct reasoning.
    - 3: Partially correct but missed key link.
    - 1: Wrong or Hallucinated.
    
    Output ONLY the integer score (1-5).
    """
    prompt = ChatPromptTemplate.from_template(grading_template)
    grader = prompt | judge_llm | StrOutputParser()
    
    try:
        score = grader.invoke({
            "question": question, 
            "ground_truth": ground_truth, 
            "generated_answer": generated_answer
        })
        return int(score.strip())
    except:
        return 3 # Neutral default

def run_evaluation():
    results = []
    print(f"Starting Evaluation on {len(test_dataset)} questions...")
    
    for i, item in enumerate(test_dataset):
        print(f"\nRunning Q{i+1}: {item['question']}")
        
        start_time = time.time()
        try:
            # The System Answer
            generated_answer = hybrid_search(item['question'])
        except Exception as e:
            generated_answer = f"ERROR: {str(e)}"
        
        latency = time.time() - start_time
        
        # The Judge's Score
        score = calculate_score(item['question'], generated_answer, item['ground_truth'])
        
        print(f"   -> Score: {score}/5 | Time: {latency:.2f}s")
        
        results.append({
            "Question": item['question'],
            "Ground Truth": item['ground_truth'],
            "Generated Answer": generated_answer,
            "Score": score,
            "Latency": round(latency, 2)
        })
        
    # --- SAVE TO NEW CSV ---
    filename = "research_results_scaled.csv"
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    
    print("\n" + "="*30)
    print("SCALED EVALUATION COMPLETE")
    print(f"Avg Score: {df['Score'].mean()}/5")
    print(f"Results saved to: {filename}")
    print("="*30)

if __name__ == "__main__":
    run_evaluation()