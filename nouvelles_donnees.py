
import numpy as np
import pandas as pd

# Générer 2 000 nouvelles transactions
rng = np.random.default_rng(8)
n = 2000

montant = np.round(rng.lognormal(6.4, 1.1, n), 2)
anciennete = rng.integers(0, 120, n)
freq = rng.poisson(3, n) + 1
nuit = rng.binomial(1, 0.30, n)

score = (
    2.2 * (np.log10(montant) - 3.0)
    - anciennete / 60
    + (freq - 3) / 3
    + nuit * 1.2
)

fraude = (
    score + rng.normal(0, 0.35, n) > 0.6
).astype(int)

nouveau = pd.DataFrame({
    "montant": montant,
    "anciennete": anciennete,
    "freq_24h": freq,
    "nuit": nuit,
    "fraude": fraude,
})

ancien = pd.read_csv("data/transactions.csv")
df = pd.concat([ancien, nouveau], ignore_index=True)
df.to_csv("data/transactions.csv", index=False)

print(
    "data/transactions.csv :",
    len(ancien), "->", len(df),
    "lignes,",
    int(df["fraude"].sum()), "fraudes"
)
