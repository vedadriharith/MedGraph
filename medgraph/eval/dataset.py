"""Evaluation questions with hand-written ground truths.

Limitation: these ground truths were written by the project author, who
also built the system. Multi-hop and unanswerable sets are not here yet.
"""

SINGLE_HOP = [
    {
        "id": "sh01",
        "question": "What treatments are associated with Hirschsprung Disease?",
        "ground_truth": "Transanal Endorectal Pull-Through and Transabdominal Pull-Through.",
    },
    {
        "id": "sh02",
        "question": "What condition is Transanal Endorectal Pull-Through used for?",
        "ground_truth": "Hirschsprung Disease.",
    },
    {
        "id": "sh03",
        "question": "Does Aquagenic Urticaria affect infants?",
        "ground_truth": "Yes, it can manifest as a pediatric form.",
    },
    {
        "id": "sh04",
        "question": "What are the treatments for hypertension?",
        "ground_truth": "Lifestyle changes, beta-blockers (propranolol), diuretics.",
    },
    {
        "id": "sh05",
        "question": "What is the connection between Landolt C and Strabismus?",
        "ground_truth": "Landolt C is associated with Strabismus Amblyopia measurement.",
    },
    {
        "id": "sh06",
        "question": "What drugs are used to treat Graft-Versus-Host Disease (GVHD)?",
        "ground_truth": "Cyclosporine and Chloroquine.",
    },
    {
        "id": "sh07",
        "question": "Is there a link between obesity and insulin resistance?",
        "ground_truth": "Yes, obesity is often associated with insulin resistance and diabetes.",
    },
    {
        "id": "sh08",
        "question": "What are the potential side effects of statins?",
        "ground_truth": "Muscle pain, increased risk of diabetes, liver damage.",
    },
    {
        "id": "sh09",
        "question": "Does asthma cause systemic inflammation?",
        "ground_truth": "Yes, asthma is associated with systemic inflammation and increased CRP levels.",
    },
    {
        "id": "sh10",
        "question": "What is the relationship between Helicobacter pylori and gastric cancer?",
        "ground_truth": "H. pylori infection is a major cause/risk factor for gastric cancer.",
    },
]

ALL_QUESTIONS = SINGLE_HOP