import numpy as np
import optuna
from sklearn.model_selection import train_test_split

def update_trust(data, labels, pattern, presence, permanence, logic, iterations=10):
    # Split once, outside the trials
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    def run_trial(trial):
        # Hyperparameters to tune
        alpha = trial.suggest_float("alpha", 0.05, 0.2)
        beta  = trial.suggest_float("beta",  0.3,  0.5)
        gamma = trial.suggest_float("gamma", 0.05, 0.2)
        delta = trial.suggest_float("delta", 0.1,  0.3)

        # Train on train set
        pattern.fit(X_train, y_train)
        # Initial predict on test set
        preds, probs, _ = pattern.validate(X_test, y_test)

        T = 0.5
        # Run PPP loop
        for _ in range(iterations):
            V_q = presence.validate(probs)
            V_b = permanence.validate(preds, probs)
            V_l = logic.validate(preds, X_test)

            V_t = beta * V_b + (1 - beta - delta) * V_q + delta * V_l
            T = alpha * V_t + (1 - alpha) * T

            weighted = probs * V_q[:, None] * V_b[:, None] * V_l[:, None]
            preds = np.argmax(weighted, axis=1)

        # Optimize for mean trust
        return float(np.mean(T))

    study = optuna.create_study(direction="maximize")
    study.optimize(run_trial, n_trials=20)
    return study.best_params

