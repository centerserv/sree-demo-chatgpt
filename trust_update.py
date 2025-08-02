--- a/trust_update.py
+++ b/trust_update.py
@@
-def update_trust(data, labels, pattern, presence, permanence, logic, iterations=10):
-    def run_trial(trial):
-        a = trial.suggest_float("alpha", 0.05, 0.2)
-        b = trial.suggest_float("beta", 0.3, 0.5)
-        g = trial.suggest_float("gamma", 0.05, 0.2)
-        d = trial.suggest_float("delta", 0.1, 0.3)
-        pattern.fit(X_train, y_train)
-        preds, probs, _ = pattern.validate(X_test, y_test)
+from sklearn.model_selection import train_test_split
+
+def update_trust(data, labels, pattern, presence, permanence, logic, iterations=10):
+    # split once, outside the trials
+    X_train, X_test, y_train, y_test = train_test_split(
+        data, labels, test_size=0.2, random_state=42
+    )
+
+    def run_trial(trial):
+        a = trial.suggest_float("alpha", 0.05, 0.2)
+        b = trial.suggest_float("beta", 0.3, 0.5)
+        g = trial.suggest_float("gamma", 0.05, 0.2)
+        d = trial.suggest_float("delta", 0.1, 0.3)
+        # now these exist:
+        pattern.fit(X_train, y_train)
+        preds, probs, _ = pattern.validate(X_test, y_test)
@@
-    study = optuna.create_study(direction="maximize")
+    study = optuna.create_study(direction="maximize")
     study.optimize(run_trial, n_trials=20)
     return study.best_params

