# trust_update.py
import numpy as np
import optuna
from sklearn.model_selection import train_test_split

def update_trust(
    data, labels, pattern, presence, permanence, logic,
    iterations=20, alpha=None, beta=None, gamma=None, delta=None
):
    # single split for train/test
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    # if any hyperparam missing, run tuning
    if None in (alpha, beta, gamma, delta):
        def run_trial(trial):
            a = trial.suggest_float("alpha", 0.05, 0.2)
            b = trial.suggest_float("beta",  0.3,  0.5)
            g = trial.suggest_float("gamma", 0.05, 0.2)
            d = trial.suggest_float("delta", 0.1,  0.3)

            pattern.fit(X_train, y_train)
            preds, probs, _ = pattern.validate(X_test, y_test)

            T = 0.5
            for _ in range(iterations):
                V_q = presence.validate(probs)
                V_b = permanence.validate(preds, probs)
                V_l = logic.validate(preds, X_test)

                V_t = b * V_b + (1 - b - d) * V_q + d * V_l
                T   = a * V_t + (1 - a) * T

                weighted = probs * V_q[:, None] * V_b[:, None] * V_l[:, None]
                preds    = np.argmax(weighted, axis=1)

            return float(np.mean(T))

        study = optuna.create_study(direction="maximize")
        study.optimize(run_trial, n_trials=20)
        return study.best_params

    # otherwise run PPP loop with provided params
    pattern.fit(X_train, y_train)
    preds, probs, _ = pattern.validate(X_test, y_test)

    history = {'accuracy': [], 'T': []}
    T = 0.5
    for _ in range(iterations):
        V_q = presence.validate(probs)
        V_b = permanence.validate(preds, probs)
        V_l = logic.validate(preds, X_test)

        V_t = beta * V_b + (1 - beta - delta) * V_q + delta * V_l
        T   = alpha * V_t + (1 - alpha) * T

        weighted = probs * V_q[:, None] * V_b[:, None] * V_l[:, None]
        preds    = np.argmax(weighted, axis=1)

        acc = (preds == y_test).mean()
        history['accuracy'].append(acc)
        history['T'].append(float(np.mean(T)))

    return history

