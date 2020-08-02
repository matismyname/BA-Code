import numpy as np
from scipy.stats import norm, expon

#Fall 0
def main():
    n = 30  # Anzahl an zufaellig generierten Zahlen
    NormVertFktListe = []  # Liste, die die berechneten Werte der Verteilungsfunktion der Normalverteilung enthalten wird
    ExpVertFktListe = [] # Liste, die die berechneten Werte der Verteilungsfunktion der Exponentialverteilung enthalten wird
    ExpVertFktListe_eigentlich = [] # Unwichtig für die Rechnungen

    z = np.random.normal(loc=0, scale=1.0, size=n)  # Array normalverteilter Zufallszahlen mit loc = Mittelwert, scale = Standardabweichung
                                                    # und size = Anzahl der normalverteilten Zufallszahlen
    e = np.random.exponential(scale=1, size=n)      # Array exponentialverteilter Zufallszahlen mit scale = 1/lambda und size = Anzahl der exponentialverteilten Zufallszahlen
                                                    # und size Anzahl der normalverteilten Zufallszahlen

    # Array aufsteigend sortiert
    z = sorted(z)
    e = sorted(e)


    # Berechne den Wert der Verteilungsfunktion der Normalverteilung fuer jeden Wert in z und fuege sie in NormVerFktListe ein
    for x in z:
        NormVertFktListe.append(norm.cdf(x))
    
    # Berechne den Wert der Verteilungsfunktion der Exponentialverteilung fuer jeden Wert in e und fuege sie in ExpVertFktListe ein
    for x in e:
        ExpVertFktListe.append(norm.cdf(x))
        ExpVertFktListe_eigentlich.append(norm.cdf(x))

    # Unkommentieren, um die erstellen Zufallszahlen + Verteilungsfunktion zu sehen
    #print("\nNormalverteile Zufallszahlen: \n")
    #for i in range(0, n):
    #    print("Zufallszahlen: {0} \t Wert der Verteilungsfunktion: {1}".format(str(z[i]), str(NormVertFktListe[i])))

    #print("\nExponentialverteilte Zufallszahlen: \n")
    #for i in range(0, n):
    #    print("Zufallszahlen: {0} \t Wert der eigentlichen Verteilungsfunktion: {1} \t Wert der Verteilungsfunktion unter H0: {2}".format(str(z[i]), str(ExpVertFktListe[i]), str(ExpVertFktListe_eigentlich[i])))

    erg, W_stern = berechneWQuad(n, NormVertFktListe)
    U_stern = berechneUQuad(erg, n, NormVertFktListe)
    A_stern = berechneAQuad(n, NormVertFktListe)

    print("\nKritische Werte fuer die Normalverteilung: \n")
    print("W*:", W_stern)
    print("U*:", U_stern)
    print("A*:", A_stern)

    erg, W_stern = berechneWQuad(n, ExpVertFktListe)
    U_stern = berechneUQuad(erg, n, ExpVertFktListe)
    A_stern = berechneAQuad(n, ExpVertFktListe)

    print("\nKritische Werte fuer die Exponentialverteilung: \n")
    print("W*:", W_stern)
    print("U*:", U_stern)
    print("A*:", A_stern)

# Berechne W^2 und W* und gib beide Werte zurueck
def berechneWQuad(n, VertFktListe):
    s0 = 0
    for x in range(1, n + 1):
        s0 = s0 + (VertFktListe[x - 1] - ((2 * x - 1) / (2 * n))) ** 2
    erg = s0 + (1 / (12 * n))

    W_stern = (erg - (0.4 / n) + (0.6 / (n ** 2))) * (1 + (1 / n))
    return erg, W_stern


# Berechne U^2 und gib den Wert zurueck
def berechneUQuad(s0, n, VertFktListe):
    z0 = 0

    # Berechne z Strich
    for x in range(1, n + 1):
        z0 = z0 + VertFktListe[x - 1]

    z_strich = z0 / n

    s1 = s0 - n * ((z_strich - 0.5) ** 2)
    U_stern = (s1 - (0.1 / n) + (0.1 / (n**2)))*(1 + (0.8 / n))
    return U_stern


# Berechne A^2 gib und den Wert zurueck
def berechneAQuad(n, VertFktListe):
    s0 = 0

    for x in range(1, n):
        s0 = s0 + ((2*x - 1)/n)*(np.log(VertFktListe[x - 1]) + \
                                 np.log(1 - VertFktListe[n - x]))
    s1 = - s0 - n
    return s1

#main()