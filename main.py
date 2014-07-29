from igraph import *
from dateutil.parser import parse
import random
import pandas as pd
import csv
from math import *
import sys
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import timedelta
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import copy
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

microgrid_number = 25; hourly = []; mean_powers = {}

for i in range(microgrid_number):
    nom_fichier = str(5*i)+'.csv'
    path='./traces/%s' % nom_fichier
    hourly.append([])
    hourly[-1] = pd.read_csv(path, delimiter = ';',  index_col='Dates', parse_dates=True)
    hourly[-1].columns = ['Value']
            
Ptot = 0; Pmin = 0

for i in range(len(hourly)):
    hourly[i]['Value'] = np.nan_to_num(hourly[i]['Value'])
    mean_powers[i] = hourly[i]['Value'].mean()
    Ptot += abs( hourly[i]['Value'].mean() )
    
Pmin = 0.1 * Ptot

daily = []; weekly = []

for i in range(len(hourly)):
    daily.append([]); weekly.append([])
    daily[-1] = hourly[i].resample('d', 'mean')
    weekly[-1] = daily[-1].resample('w', 'mean')
    
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for i in range(len(hourly)):
    weekly[i]['month'] = weekly[i]['Value'].index.month
    grouped = weekly[i].groupby('month')['Value'].mean()
    grouped.index = months
    
for i in range(len(weekly)):
    weekly[i]['Value'] = np.nan_to_num( weekly[i]['Value'] )

for j in range(len(hourly)):
    for i in range(12):
        weekly[j][months[i]] = (weekly[j].index.month == i).astype(float)

    X = weekly[j][months]
    y = weekly[j]['Value']
    clf = LinearRegression().fit(X, y)

    weekly[j]['month_trend'] = clf.predict(X)

for j in range(len(hourly)):
    weekly[j]['month_corrected'] = (weekly[j]['Value'] - weekly[j]['month_trend'] + weekly[j]['month_trend'].mean())

#For clustering uncorrelated timeseries :
def metric_1(ts1, ts2):
    return np.corrcoef(ts1, ts2)[0,1]**2
def metric_2(ts1, ts2):
    if 2 * ( 1 - np.corrcoef(ts1, ts2)[0,1] ) < 0:
        return 0
    else:
        return 0.5 * math.sqrt( 2 * ( 1 - np.corrcoef(ts1, ts2)[0,1] ) )

UNCO = []

for i in range(len(hourly)):
    UNCO.append([])
    for j in range(len(hourly)):
        if np.isnan( np.corrcoef(weekly[i]['month_corrected'],weekly[j]['month_corrected'])[0,1] ):
            print 'There is a correlation problem...'
        else:
            UNCO[-1].append( metric_1(weekly[i]['month_corrected'],weekly[j]['month_corrected']) )
UNCO = np.array( UNCO )
print UNCO.shape

#Returns a list of at least nb_seeds cliques 
def generate_seeds(G, nb_seeds):
    security = 0; size_required = False; all_coals = [list(c) for c in G.cliques() if len(c)>=3]; 
    seeds = []
    if len(all_coals) < nb_seeds:
        return seeds
    while len(all_coals)>0 and size_required == False and security < 2:
        for coal in all_coals[::-1]:
            ajout_possible = True
            if len(seeds)>0:
                for seed in seeds:
                    if len(list(set.intersection(set(seed),set(coal)))) >0:
                        ajout_possible = False
                if ajout_possible:
                    seeds.append(coal)
            else:
                seeds.append(coal)
            if len(seeds) >= nb_seeds:
                size_required = True
                break
        security += 1
    return seeds

def plant_biggest_seeds(seeds, nb_seeds):
    if len(seeds)==nb_seeds:
        return seeds
    else:
        return seeds[:nb_seeds]

def plant(seeds, nb_seeds):
    if len(seeds)==nb_seeds:
        return seeds
    else:
        np.random.shuffle(seeds)
        return seeds[:nb_seeds]

def distance_seeds(G, seed1, seed2):
    avg_dist = 0
    for s1 in seed1:
        for s2 in seed2:
            avg_dist += len(G.get_shortest_paths(s1,s2)[0])
    return float(avg_dist) / float( len(seed1) + len(seed2) )

#Returns the frontier of a given clique in a given graph    
def frontier(clique,G):
    frontier = []
    for node in clique:
        frontier+=G.neighbors(node)
    frontier = set(frontier) - set(clique)
    return frontier

#Process which expand the cliques by adding nodes in the frontier
def expand_seeds(seeds, G, size_factor):
    continu = True; t = 0.001
    while continu:
        continu = False
        for i in range(len(seeds)):
            voisins = frontier(seeds[i],G)
            for voisin in voisins:
                if utility(seeds[i]+[voisin])>=size_factor*utility(seeds[i]):#mean_std(seeds[i]+[voisin])<=size_factor*mean_std(seeds[i]):#fitness(seeds[i]+[voisin],G,size_factor)>=fitness(seeds[i],G,size_factor):
                    seeds[i].append(voisin)
                    continu = True
                else:
                    if random.random() <= t:
                        seeds[i].append(voisin)
                        continu = True

#Returns the nodes of the graph that are involved in at least one expansed clique                
def selected_nodes(seeds):
    selected_nodes = set()
    for seed in seeds:
        for node in seed:
            selected_nodes.add(node)
    return list(selected_nodes)

def unselected_nodes(seeds):
    return [node['id'] for node in list(G.vs()) if int(node['id']) not in selected_nodes(seeds)]

def participation_ratio(seeds):
    return float(len(selected_nodes(seeds)))/float(microgrid_number)

def redundant_nodes(seeds):
    redundant = []
    for i in range(len(seeds)):
        for j in range(len(seeds))[i+1:]:
            ajout = list(set.intersection(set(seeds[i]),set(seeds[j])))
            for elt in ajout:
                if elt not in redundant:
                    redundant.append( elt )
    return redundant

#Returns the above mentionned ratio for a given coalition and node
def Tau(coal, node):
    if utility(coal) == 0:
        return 0
    else:
        return  float( utility(coal) - utility([ag for ag in coal if ag != node]))  / float( utility(coal))

#Returns the list of all expansed seeds in wich a given node is involved
def lister_seeds(seeds, node):
    lis = []
    for seed in seeds:
        if node in seed:
            lis.append( seed )
    return lis

#Procedure that map the expansed seeds into coalitions according to the heuristic discussed above
def simplify_seeds(seeds):
    for node in redundant_nodes(seeds):
        Taux = []; L = lister_seeds( seeds, node)
        for S in L:
            Taux.append( Tau(S, node) )
        Taux = np.array( Taux )
        random_curseur = 0#int( abs( random.gauss(0, float(len(Taux)) / float(2) ) ) )
        winner = L[ Taux.argsort()[::-1][min(random_curseur,len(Taux)-1)] ]
        for seed in seeds:
            if seed != winner and node in seed:
                seed.remove(node)

def implant_unselected_nodes(unselected, seeds):
    for node in unselected:
        Taux = []
        for S in seeds:
            Taux.append( Tau(S+[node], node) )
        Taux = np.array( Taux )
        print Taux
        random_curseur = int( abs( random.gauss(0, float(len(Taux)) / float(3) ) ) )
        print Taux.argsort()[::-1][min(random_curseur,len(Taux)-1)]
        winner = seeds[Taux.argsort()[::-1][min(random_curseur,len(Taux)-1)]]
        for seed in seeds:
            if seed == winner and node not in seed:
                seed.append(node)

def uco(coal):
    cor = 0; com = 0;
    for i in range(len(coal)):
        for j in range(len(coal))[i+1:]:
            com += 1
            cor += np.corrcoef( weekly[coal[i]]['month_corrected'], weekly[coal[j]]['month_corrected'] )[0,1] 
    if com == 0:
        return 0
    else:
        return 1 - abs( float(cor)/float(com) )

def prod(coal):
    prod = 0
    for ag in coal:
        prod += abs(mean_powers[ag])
    return prod

def utility(seed):
    return safety_production([seed], 0.05) * float(1)/float(len(seed))
    
def utility_CS(CS):
    return sum(map(utility, [c for c in CS if len(c)>1]))
    
def mean_coal_std(seeds):
    M = []
    for seed in seeds:
        M.append( mean_std(seed) )
    return M
    
def STD(seed):
    M = copy.deepcopy( weekly[seed[0]]['month_corrected'] )
    for i in range(len(seed))[1:]:
        M += copy.deepcopy( weekly[seed[i]]['month_corrected']  )
    return float(np.std(M))
    
def mean_std(seed):
    M = copy.deepcopy( weekly[seed[0]]['month_corrected'])
    for i in range(len(seed))[1:]:
        M += copy.deepcopy( weekly[seed[i]]['month_corrected']   )
    return float(np.std(M))/float(np.mean(M))
    
def summ(seed):
    M = copy.deepcopy( weekly[seed[0]]['month_corrected'] )
    for i in range(len(seed))[1:]:
        M += copy.deepcopy( weekly[seed[i]]['month_corrected']  )
    return M
    
def generate_random_CS(players, nb_coal):
    CS = []
    for i in range(nb_coal):
        CS.append([])
    for p in players:
        CS[random.randint(0,nb_coal-1)].append(p)
    return CS
    
def plot_cdf(data, bins, color, label):
    n_counts,bin_edges = np.histogram(data,bins=bins,normed=True) 
    cdf = np.cumsum(n_counts)  # cdf not normalized, despite above
    scale = 1.0/cdf[-1]
    ncdf = scale * cdf
    plt.plot(np.linspace(0,1,bins),ncdf, color = str(color), label = str(label))
    plt.legend()
    
def danger_proba(data, bins, safety_value):
    n_counts,bin_edges = np.histogram(data,bins=bins,normed=True) 
    cdf = np.cumsum(n_counts)  # cdf not normalized, despite above
    scale = 1.0/cdf[-1]
    ncdf = scale * cdf
    return filter(lambda x: x>safety_value, ncdf)[0]
    
def safety_proba_CS(CS, bins, safety_value):
    prob = 1
    for coal in CS:
        data = summ(coal).values 
        prob = prob * ( 1 - danger_proba(data, bins, safety_value) )
    return prob 
    
def safety_production(CS, seuil):
    safety_production = 0
    for i in range(len(CS)):
        data = summ(CS[i]).values
        XXX = mquantiles(data,[seuil])
        safety_production += 10**-7 * XXX[0]
    return safety_production
    
nb_iter_max = 1; nb_iter = 0; DU_mean = []

while nb_iter <= nb_iter_max:
    
    util = []
    
    for eta in np.linspace(0.3,0.5,3):
        
        util.append([])
        UNCO_eph = UNCO; UNCO_eph = numpy.where(UNCO_eph <= eta, UNCO_eph, 0)
        G = Graph(); G = Graph.Weighted_Adjacency(UNCO_eph.tolist(), mode = ADJ_UNDIRECTED, attr='weight')
        G.vs['id'] = range(len(UNCO_eph))
    
        for nb_coal in range(6)[2:]:
            
            seeds  = generate_seeds( G, nb_coal )
            if len(seeds) >= nb_coal:
                number_of_runs = 0; nb_run = 0; best_score = 0
                
                while nb_run <= number_of_runs:
                    
                    planted_seeds = plant(copy.deepcopy(seeds), nb_coal)
                    expand_seeds( planted_seeds, G, 1)
                    simplify_seeds(planted_seeds)
                    #implant_unselected_nodes( unselected_nodes(planted_seeds), planted_seeds )
                    eph = utility_CS(planted_seeds)
                    if eph > best_score:#np.mean(mean_coal_std(planted_seeds)) < best_score:#G.modularity(get_clusters_labels(G, planted_seeds)) > best_score:
                        best_seeds = planted_seeds
                        best_score = eph   
                    nb_run += 1
            else:
                best_score = 0
            util[-1].append( best_score )
            print 'run completed...'
    util = np.array(util)
    if nb_iter == 0:
        DU_mean = util
    else:
        DU_mean += util
    nb_iter += 1
    print 'MASTER RUN COMPLETED...' + str(nb_iter)
DU_mean = (float(1)/float(nb_iter)) * DU_mean

print DU_mean