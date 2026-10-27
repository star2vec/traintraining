de vazut alea cu measure.label??? si regionprops?? si cv blob idk


vizulaizare date?? asta e mai universal ig, dar pt clustering/thresholds: <img width="759" height="513" alt="Screenshot 2026-10-27 at 04 13 57" src="https://github.com/user-attachments/assets/b0e91250-1089-4d48-8c7e-0fabbd172f4d" />


maybe usefull??

<pre>
def evaluate(model, val, subtask):
    model.eval()
    scores = []
    with torch.no_grad():
        for sc in val:
            X, y = val[sc]
            preds = model(torch.from_numpy(X)).argmax(1).numpy()
            scores.append((preds==y).mean())
            print("scanner", sc, "accuracy", round(scores[-1], 4))
    A = np.mean(scores)
    P = sum(p.numel() for p in model.parameters())
    score = scoremetric(A, P, subtask)
    print("scanner: ", sc, "accuracy: ", round(A, 4), "params: ", P, "score: ", round(score, 4))
</pre>

ce inseamna torch no grad? 
ce inseamna argmax(1)??

preprocesare:
<pre>
    def shift(x):
    dy, dx = np.random.randint(-2, 3, size=2)
    return torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
</pre>




or
