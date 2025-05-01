import csv
from llm_plm_hybrid.qa.rag_pipeline import parse_question


test_data = [
    ("What is the protein existence level of lactase?",          "protein_existence"),
    ("How many PTM sites does P09812 have?",                     "ptm_count"),
    ("PTM count for MVHFAELVK?",                                  "ptm_count"),
    ("Level of existence for ACDEFGHIKL?",                       "protein_existence"),
    ("Show me the evidence level of mouse hemoglobin.",          "protein_existence"),
    ("Predict PTM sites in the sequence MVHFAELVKQAV.",          "ptm_count"),
    ("According to UniProt, what is the existence for P12345?",  "protein_existence"),
    ("Total number of modified residue features in Q9XYZ1?",     "ptm_count"),
    ("Give the existence classification of E. coli DnaK.",       "protein_existence"),
    ("Count the post-translational modifications on YP_009724390.", "ptm_count"),
]

correct = 0
results = []

for q, true_label in test_data:
    out = parse_question(q)
    pred = out["task"]
    is_correct = (pred == true_label)
    results.append((q, true_label, pred, is_correct))
    if is_correct:
        correct += 1

total = len(test_data)
acc = correct / total * 100
print(f"Intent classification accuracy: {correct}/{total} = {acc:.1f}%\n")

print("Misclassifications:")
for q, true, pred, ok in results:
    if not ok:
        print(f" • Q: {q!r}")
        print(f"   - true: {true},  pred: {pred}\n")

with open("intent_test_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question","true_intent","predicted_intent","correct"])
    for row in results:
        writer.writerow(row)

print("✅ Done. Details in intent_test_results.csv")
