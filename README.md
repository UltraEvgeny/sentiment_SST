# Amazon Fine Food Reviews
## recommend.ipynb
What was done:
1) Removing duplicates
2) Using first half of sample as train sample.
3) Creating pivot table, when rows are users, columns are items, values are scores.
4) Creating most popular and high rated items to recommend them if we are cold starting.
5) Predicting for 2000 test users: finding several most similar users, calculating weghted (by cosine similarity) average of their scores for every item, getting items
with average score higher then threshold, removing already used items, adding popular items if item count is not big enough.

# Sentiment analysis:
## sentiment.ipynb
What was done:
1) Using SST dataset.
2) Training GRU, unidirectional LSTM and bidirectionsl LSTM. Their accuracy on train, val and test datasets on best epoch:

```python
LSTM
train_accuracy_str    4668/6835
val_accuracy_str      1015/1709
test_accuracy_str     1399/2210
train_accuracy            0.683
val_accuracy              0.594
test_accuracy             0.633

LSTM biderectional
train_accuracy_str    4668/6835
val_accuracy_str      1015/1709
test_accuracy_str     1399/2210
train_accuracy            0.683
val_accuracy              0.594
test_accuracy             0.633

GRU
train_accuracy_str    5071/6835
val_accuracy_str      1104/1709
test_accuracy_str     1420/2210
train_accuracy            0.742
val_accuracy              0.646
test_accuracy             0.643
```

In notebook we are dealing with GRU: training model, choosing best epoch, doing some predictions.
