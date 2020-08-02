import matplotlib.pyplot as plt

def plotte(xes, yes, schritt, color, name, N, run, a, klassen):
    xlocs, xlabs = plt.xticks()
    xlocs=[i+1 for i in range(0, len(yes))]
    xlabs=[round((i+1)*schritt, 2) for i in range(0, len(yes))]

    plt.bar(xes, yes, color=color)
    plt.ylabel("Anzahl der p-Werte")
    plt.xlabel("p-Werte für {0}".format(name))
    plt.xticks(xlocs, xlabs)
    for i, v in enumerate(yes):
        if v != 0:
            plt.text(xlocs[i] - 0.25, v + 0.03, str(v))       
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.savefig("Graph-name{0}-durchlauf{1}-N{2}-a{3}-klassenanzahl{4}".format(name, run, N, a, klassen), dpi=None, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format=None,
    transparent=False, bbox_inches=None, pad_inches=0.1,
    frameon=None, metadata=None)
    plt.clf()
    
    
def plotte_nicht(p_werte, schritt):
    r = 0
    xes = [i for i in range(1, int(1/schritt) + 1)]
    yes = [0 for i in range(0, int(1/schritt))]

    for i in range(0, int(1/schritt)):
        for k in p_werte:
            if (round(k, 9) > round((i)*schritt, 9)) & (round(k, 9) <= round((i+1)*schritt, 9)):
                yes[i] += 1
    
    for k in p_werte:
        if k == 0:
            yes[0] += 1

    return [yes, xes]