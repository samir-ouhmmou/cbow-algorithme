import re
import numpy as np
from collections import Counter

corpus = [
    "Oujda est une ville frontière de l'est du Maroc",
    "Fès est une ville historique du Maroc",
    "Rabat est la capitale du Maroc",
    "Casablanca est une grande ville du Maroc",
    "Marrakech est une ville touristique du Maroc",

    "J'ai suivi des cours d'informatique à l'université Mohammed Premier",
    "Les étudiants suivent des cours à l'université",
    "Le professeur donne des cours d'informatique à l'université",

    "Je dois apprendre mes leçons",
    "Il doit apprendre ses leçons",
    "Les étudiants doivent apprendre leurs leçons",
]

def tokenize(phrase):
    """Découpe une phrase en mots (minuscules, apostrophes séparées)."""
    phrase = phrase.lower()
    phrase = re.sub(r"(\w)['’](\w)", r"\1' \2", phrase)  # l'est -> l' est
    return re.findall(r"\w+'?|\w+", phrase)


def construire_vocabulaire(phrases):
    """Retourne word2idx et idx2word."""
    mots = sorted({m for p in phrases for m in tokenize(p)})
    word2idx = {m: i for i, m in enumerate(mots)}
    idx2word = {i: m for m, i in word2idx.items()}
    return word2idx, idx2word


def construire_exemples_cbow(phrases, k=2):
    """k = nombre de mots avant et après le mot cible."""
    exemples = []
    for phrase in phrases:
        tokens = tokenize(phrase)
        for t in range(k, len(tokens) - k):
            contexte = tokens[t - k:t] + tokens[t + 1:t + k + 1]
            cible = tokens[t]
            exemples.append((contexte, cible))
    return exemples

def one_hot(mot, word2idx):
    """Encodage one-hot d'un mot (vecteur de taille |V|)."""
    v = np.zeros(len(word2idx))
    v[word2idx[mot]] = 1
    return v


def encoder_exemples(exemples, word2idx):
    """X : one-hot de chaque mot du contexte, y : one-hot du mot cible."""
    X = np.array([[one_hot(m, word2idx) for m in ctx] for ctx, _ in exemples])
    y = np.array([one_hot(cible, word2idx) for _, cible in exemples])
    return X, y


if __name__ == "__main__":
    k = 2

    # Vocabulaire
    word2idx, idx2word = construire_vocabulaire(corpus)
    freq = Counter(m for p in corpus for m in tokenize(p))

    print(f"Taille du vocabulaire : {len(word2idx)} mots\n")
    print(f"{'Index':>5} | {'Mot':<15} | Fréquence")
    print("-" * 35)
    for i in range(len(idx2word)):
        print(f"{i:>5} | {idx2word[i]:<15} | {freq[idx2word[i]]}")

    # Exemples CBOW
    exemples = construire_exemples_cbow(corpus, k)
    print(f"\nNombre d'exemples CBOW : {len(exemples)}\n")
    print(f"{'Contexte':45} | Cible")
    print("-" * 60)
    for contexte, cible in exemples:
        print(f"{' – '.join(contexte):45} | {cible}")

    X, y = encoder_exemples(exemples, word2idx)
    print("\nShape X :", X.shape, "| Shape y :", y.shape)