<pre>
d2 = np.sort(km2.transform(E), axis=1)
margin2 = d2[:, 1] - d2[:, 0]

# rank of each problem's margin within its own method, scaled to 0..1
conf1 = margin.argsort().argsort() / len(margin)
conf2 = margin2.argsort().argsort() / len(margin2)

# mixed clusters get no say: the TF-IDF side decides there
conf2[(clusters2 == 4) | (clusters2 == 7)] = 0

combined = []
for i in range(len(data)):
    if pred_amended[i] == pred_st_clu[i] or conf1[i] >= conf2[i]:
        combined.append(pred_amended[i])
    else:
        combined.append(pred_st_clu[i])
combined = np.array(combined)

print('disagree:', np.mean(pred_amended != pred_st_clu))
print(pd.Series(combined).value_counts())
data['label'] = combined
data[['id', 'label']].to_csv('submission.csv', index=False)
</pre>


