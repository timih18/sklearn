# default
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    objective='multiclass',
    class_weight='balanced',
    random_state=42,
    verbose=-1
)