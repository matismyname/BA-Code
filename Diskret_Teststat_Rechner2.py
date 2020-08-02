import numpy as np
import tail_prob
import math
from scipy.stats import ncx2
import math
from scipy.stats import ncx2
from numpy.random import choice
import random
import Simulation_p_Werte_Graph
import Exp_Norm_Rechner

'''
Listen fuer die p_Werte der Teststatistiken. Diese werden auf Gleichverteilung analysiert.
'''
p_Werte_W = []
p_Werte_U = []
p_Werte_A = []

#Liste, mit deren Hilfe die durchschnittliche Abweichung der theoretischen von der eigentlichen Wahrscheinlichkeit berechnet wird
durchschnitt_wkts = []
def main(klassen, N, r, o_i, s):

    n = int(klassen)
    wkts = [(1/n) for i in range(0, n)]

    if r == "zufall":
        gewichtet_wkts = []
        gezogen = []
        o_i = [0 for i in range(klassen)]

        '''
        Der erste for-loop und das erste if-statement setzen zufaellige Gewichte als Wahrscheinlichkeiten fest.
        Hiernach werden N-mal zufaellige Zahlen in {1, ..., n} gezogen mit den festgesetzen Gewichten.
        Im letzten for-loop werden die gezogenen Zahlen nach ihrer Haeufigkeit in Klassen eingeteilt.
        '''

        '''
        k gibt eine Liste von Wahrscheinlichkeiten wieder, die sich zu 1 summieren. Je groeßer der Faktor a
        hierbei ist desto kleiner ist die Abweichung von der theoretischen Wahrscheinlichkeit, d.h.
        zum Beispiel, dass bei einem Wuerfel die Wahrscheinlichkeiten in k bei einem großen a nahe bei 1/6 sein werden
        '''
        k = np.random.dirichlet(np.ones(n)*a, size=1)

        for i in range(0, n):
            gewichtet_wkts.append(k[0][i])

        '''
        Die choice Funktion wählt eine zufaellige Zahl zwischen 1 und n, diese Zahl folgt einer Verteilung mit den Wahrscheinlichkeiten gewichtete_wkts
        '''
        for x in range(0, N):
            zieh = choice([i+1 for i in range(klassen)], 1, p=gewichtet_wkts) #wkts fuer Simulation unter H0, gewichtet_wkts fuer Simulation unter H1
            gezogen.append(zieh)

    
        for i in range(0, klassen):
            for x in gezogen:
                if i+1 == x:
                    o_i[i] += 1
    
        for i in range(0, klassen):
            durchschnitt_wkts.append(abs(wkts[i] - gewichtet_wkts[i]))
        
    else:
        o_i = o_i
    '''
    Hier setzten wir unsere Variablen wie n fest und berechnen alle wichtigen
    Kenngroessen wie z.B. S_j. Wir runden diese Kenngroessen auf 3 Nachkommastellen.
    '''

    N = sum(o_i)
    e_i = [N*wkts[i] for i in range(n)]
    t_j = [round((wkts[i] + wkts[i + 1])/2, 3) for i in range(n-1)]
    t_j.append(round((wkts[n-1] + wkts[0])/2, 3))
    S_j, T_j, H_j, Z_j, Z_strich = fill_calc_nums(o_i, e_i, t_j, N, n)
    H_j_bruch = [round(1/(H_j[i]*(1 - H_j[i])), 3) for i in range(n-1)]
    H_j_bruch.append(0)

    '''
    Die Matrixliste gibt uns die Matrizen wieder, die wir zur Berechnung der Teststatistiken brauchen.
    Hierbei ist die aufgeschriebene Reihenfolge der Matrizen eine Denkstuetze, es gilt also Matrixliste[0] = I.
    '''
    # I, D, E, K, p, o_vektor, e_vektor, d_vektor, A, Z, eins_vektor, Sigma
    Matrixliste = berechne_matrizen(n, wkts, t_j, H_j_bruch, o_i, e_i)

    'Wir berechnen die Teststatistiken.'
    Teststatistiken_liste = berechne_Teststatistik(Matrixliste, N)


    'Hier erstellen wir eine neue Liste von Matrizen, um die p-Werte zu berechnen.'
    Teststatistiken_liste.append(Matrixliste[11])
    Teststatistiken_liste.append(Matrixliste[2])
    Teststatistiken_liste.append(np.zeros(n))
    Teststatistiken_und_Sigma_liste = Teststatistiken_liste

    berechne_QX(Teststatistiken_und_Sigma_liste, r, s)



'''
Mit dieser Funktion berechnen wir Q(X).
Hierbei nutzen wir eine selbstgeschriebene Klasse in tail_prob.py,
die die k-te Kumulante berechnet. Mit dieser berechnen wir dann muh_Q, sigma_Q etc..
'''
def berechne_QX(T_u_S_l, r, s):
    # 0 M_U
    # 1 M_A
    # 2 W_quad
    # 3 U_quad
    # 4 A_quad
    # 5 Sigma
    # 6 E
    # 7 m_x


    kappa_liste_W = []
    kappa_liste_U = []
    kappa_liste_A = []

    for x in range(1, 5):
        tW = tail_prob.Setup(x, T_u_S_l[6], T_u_S_l[5], T_u_S_l[7])
        tU = tail_prob.Setup(x, T_u_S_l[0], T_u_S_l[5], T_u_S_l[7])
        tA = tail_prob.Setup(x, T_u_S_l[1], T_u_S_l[5], T_u_S_l[7])

        kap_W = tW.kte_kum()
        kap_U = tU.kte_kum()
        kap_A = tA.kte_kum()

        kappa_liste_W.append(kap_W)
        kappa_liste_U.append(kap_U)
        kappa_liste_A.append(kap_A)

    #W-Quadrat
    m_Q_W = kappa_liste_W[0]
    sigma_Q_W = math.sqrt(kappa_liste_W[1])
    beta1_W = kappa_liste_W[2] / (math.sqrt(kappa_liste_W[1]))**3
    beta2_W = kappa_liste_W[3] / (kappa_liste_W[1])**2
    s1_W = beta1_W / math.sqrt(8)
    s2_W = beta2_W / 12

    #U-Quadrat
    m_Q_U = kappa_liste_U[0]
    sigma_Q_U = math.sqrt(kappa_liste_U[1])
    beta1_U = kappa_liste_U[2] / (math.sqrt(kappa_liste_U[1]))**3
    beta2_U = kappa_liste_U[3] / (kappa_liste_U[1])**2
    s1_U = beta1_U / math.sqrt(8)
    s2_U = beta2_U / 12

    #A-Quadrat
    m_Q_A = kappa_liste_A[0]
    sigma_Q_A = math.sqrt(kappa_liste_A[1])
    beta1_A = kappa_liste_A[2] / (math.sqrt(kappa_liste_A[1]))**3
    beta2_A = kappa_liste_A[3] / (kappa_liste_A[1])**2
    s1_A = beta1_A / math.sqrt(8)
    s2_A = beta2_A / 12


    if s1_W ** 2 > s2_W:
        a = 1 / (s1_W - math.sqrt(s1_W ** 2 - s2_W))
        delta_W = (s1_W * a**3) - a**2
        l_W = (a ** 2) - (2*delta_W)
    else:
        a = 1 / s1_W
        delta_W = 0
        l_W = a ** 2


    if s1_U ** 2 > s2_U:
        a = 1 / (s1_U - math.sqrt(s1_U ** 2 - s2_U))
        delta_U = (s1_U * a**3) - a**2
        l_U = (a ** 2) - (2*delta_U)
    else:
        a = 1 / s1_U
        delta_U = 0
        l_U = a ** 2


    if s1_A ** 2 > s2_A:
        a = 1 / (s1_A - math.sqrt(s1_A ** 2 - s2_A))
        delta_A = (s1_A * a**3) - a**2
        l_A = (a ** 2) - 2*delta_A
    else:
        a = 1 / s1_A
        delta_A = 0
        l_A = a ** 2


    p_W = berechne_wkts(m_Q_W, sigma_Q_W, delta_W, l_W, T_u_S_l[2])
    p_U = berechne_wkts(m_Q_U, sigma_Q_U, delta_U, l_U, T_u_S_l[3])
    p_A = berechne_wkts(m_Q_A, sigma_Q_A, delta_A, l_A, T_u_S_l[4])

    if s == "":
        p_Werte_W.append(p_W)
        p_Werte_U.append(p_U)
        p_Werte_A.append(p_A)

    if r != "zufall":
        print("p-Werte fuer die p-Werte von {0}".format(s))
        print("p-Wert von W Quadrat:", p_W)
        print("p-Wert von U Quadrat:", p_U)
        print("p-Wert von A Quadrat:", p_A, "\n")


'Mit dieser Funktion berechnen wir die p-Werte, d.h. P(Q(X) > t).'
def berechne_wkts(m_Q, sigma_Q, delta, l, t):
    t_stern = (t - m_Q) / sigma_Q
    m_x = l + delta
    sigma_xi = math.sqrt(2)*math.sqrt(l + 2*delta)

    teststat = t_stern*sigma_xi + m_x

    return 1 - ncx2.cdf(df=l, nc=delta, x=teststat)

    

'Diese Funktion fuellt unsere Kenngroessen wie S_j auf. Wir runden auf 3 Nachkommastellen.'
def fill_calc_nums(o_i, e_i, t_j, N, n):
    S_j = []
    T_j = []
    H_j = []
    Z_j = []
    Z_strich = []

    for x in range(n):
        if x == 0:
            S_j.append(round(o_i[0], 3))
            T_j.append(round(e_i[0], 3))
            H_j.append(round(T_j[0]/N, 3))
            Z_j.append(round((S_j[0] - T_j[0]), 3))
            Z_strich.append(round(Z_j[0]*t_j[0], 3))
        elif x == n-1:
            S_j.append(sum(o_i))
            T_j.append(sum(e_i))
            H_j.append(T_j[x]/N)
            Z_j.append((S_j[x] - T_j[x]))
            Z_strich.append(Z_j[x] * t_j[x])
        else:
            S_j.append(round(S_j[x - 1] + o_i[x], 3))
            T_j.append(round(T_j[x - 1] + e_i[x], 3))
            H_j.append(round(T_j[x]/N, 3))
            Z_j.append(round(S_j[x] - T_j[x], 3))
            Z_strich.append(round(Z_j[x] * t_j[x], 3))

    return S_j, T_j, H_j, Z_j, Z_strich

'Mit dieser Funktion berechnen wir alle wichtigen Matrizen.'
def berechne_matrizen(n, wkts, t_j, H_j_bruch, o_i, e_i):
    I = np.identity(n)
    D = np.diag(wkts)
    E = np.diag(t_j)
    K = np.diag(H_j_bruch)
    p = np.array(wkts)
    o_vektor = np.array(o_i)
    e_vektor = np.array(e_i)
    d_vektor = np.array(o_vektor - e_vektor)
    A = np.tril([1 for i in range(n)])
    Z = np.matmul(A, np.transpose(d_vektor))
    eins_mat = np.ones((n, n))
    Sigma = D - np.outer(p, p)
    Sigma = np.matmul(A, Sigma)
    Sigma = np.matmul(Sigma, np.transpose(A))


    return [I, D, E, K, p, o_vektor, e_vektor, d_vektor, A, Z, eins_mat, Sigma]

'Mit dieser Funktion berechnen wir die Teststatistiken und die symmetrischen Matrizen M bzw. A.'
def berechne_Teststatistik(Matrixliste, N):
    # I 0
    # D 1
    # E 2
    # K 3
    # p 4
    # o_vektor  5
    # e_vektor  6
    # d_vektor  7
    # A 8
    # Z 9
    # eins_mat   10


    W_quad = np.matmul(Matrixliste[9], Matrixliste[2])  	    #Z^T*E
    W_quad = np.matmul(W_quad, np.transpose(Matrixliste[9]))    #W_quad*Z
    W_quad = round(W_quad / N, 3)
    

    M_U = np.matmul(Matrixliste[2], Matrixliste[10])                            #E*11^T
    M_U = Matrixliste[0] - M_U                                                  #I - E*11^T
    M_U = np.matmul(M_U, Matrixliste[2])                                        #(I - E*11^T)*E
    M_U0 = Matrixliste[0] - np.matmul(Matrixliste[10], Matrixliste[2])          #(I - 11^T*E)
    M_U = np.matmul(M_U, M_U0)
    
    U_quad_0 = np.matmul(Matrixliste[9], M_U)
    U_quad = round((np.matmul(U_quad_0, np.transpose(Matrixliste[9])) / N), 3)


    A_quad = np.matmul(Matrixliste[9], Matrixliste[2])
    A_quad = np.matmul(A_quad, Matrixliste[3])
    A_quad = round((np.matmul(A_quad, np.transpose(Matrixliste[9]))) / N, 3)

    M_A = np.matmul(Matrixliste[2], Matrixliste[3])

    return [M_U, M_A, W_quad, U_quad, A_quad]



'''
Hier setzen wir unser N, n und weitere für den Code wichtige Groeßen fest.
Die Variable schritt ist die Groeße unserer Klassen, in die die simulierten p-Werte fallen.
Die Variable a gibt uns eine zufaellige Zahl zwischen der unteren und oberen Grenze wieder.
Die Variable run sagt aus, wie oft simuliert werden soll.
'''
N = 10000
klassen_zufall = 10
schritt = 0.05
klassen_p_Werte = round(1/schritt, 3)
a = 3000
def fahre_test(r, schritt, p_Werte, s):
    run = 0
    if r == "zufall":
        while run < 500:
            main(klassen=klassen_zufall,
            N = N,
            r = r,
            o_i = [],
            s = ""
            )
            run += 1

            print(run)

        print("Annahme: p-Werte diskret gleichverteilt:")
        p_Werte_SW = Simulation_p_Werte_Graph.plotte_nicht(p_Werte_W, schritt=schritt)
        fahre_test(r = "", schritt=schritt, p_Werte=p_Werte_SW[0], s="W Quadrat")

        p_Werte_SU = Simulation_p_Werte_Graph.plotte_nicht(p_Werte_U, schritt=schritt)
        fahre_test(r = "", schritt=schritt, p_Werte=p_Werte_SU[0], s="U Quadrat")

        p_Werte_SA = Simulation_p_Werte_Graph.plotte_nicht(p_Werte_A, schritt=schritt)
        fahre_test(r = "", schritt=schritt, p_Werte=p_Werte_SA[0], s="A Quadrat")

        print("Durchschnittsabweichung von {0}: {1} \n".format(round(1/klassen_zufall, 3), sum(durchschnitt_wkts) / (run*klassen_zufall)))

        Simulation_p_Werte_Graph.plotte(p_Werte_SW[1], p_Werte_SW[0], schritt=schritt, color="blue", name="W Quadrat", N=N, run=run, a=a, klassen=klassen_zufall)
        Simulation_p_Werte_Graph.plotte(p_Werte_SU[1], p_Werte_SU[0], schritt=schritt, color="green", name="U Quadrat", N=N, run=run, a=a, klassen=klassen_zufall)
        Simulation_p_Werte_Graph.plotte(p_Werte_SA[1], p_Werte_SA[0], schritt=schritt, color="red", name="A Quadrat", N=N, run=run, a=a, klassen=klassen_zufall)

        print("Annahme: p-Werte stetig gleichverteilt:")
        erg, W_stern = Exp_Norm_Rechner.berechneWQuad(len(p_Werte_W), sorted(p_Werte_W))
        U_stern = Exp_Norm_Rechner.berechneUQuad(erg, len(p_Werte_W), sorted(p_Werte_W))
        A_stern = Exp_Norm_Rechner.berechneAQuad(len(p_Werte_W), sorted(p_Werte_W))

        print("Teststatistiken fuer die p-Werte von W Quadrat:")
        print("W*:", W_stern)
        print("U*:", U_stern)
        print("A*:", A_stern, "\n")


        erg, W_stern = Exp_Norm_Rechner.berechneWQuad(len(p_Werte_U), sorted(p_Werte_U))
        U_stern = Exp_Norm_Rechner.berechneUQuad(erg, len(p_Werte_U), sorted(p_Werte_U))
        A_stern = Exp_Norm_Rechner.berechneAQuad(len(p_Werte_U), sorted(p_Werte_U))

        print("Teststatistiken fuer die p-Werte von U Quadrat:")
        print("W*:", W_stern)
        print("U*:", U_stern)
        print("A*:", A_stern, "\n")


        erg, W_stern = Exp_Norm_Rechner.berechneWQuad(len(p_Werte_A), sorted(p_Werte_A))
        U_stern = Exp_Norm_Rechner.berechneUQuad(erg, len(p_Werte_A), sorted(p_Werte_A))
        A_stern = Exp_Norm_Rechner.berechneAQuad(len(p_Werte_A), sorted(p_Werte_A))

        print("Teststatistiken fuer die p-Werte von A Quadrat:")
        print("W*:", W_stern)
        print("U*:", U_stern)
        print("A*:", A_stern, "\n")


    else:
        main(klassen=klassen_p_Werte,
            N = N,
            r = r,
            o_i = p_Werte,
            s = s
            )
        
fahre_test("zufall", schritt, [], s="")
