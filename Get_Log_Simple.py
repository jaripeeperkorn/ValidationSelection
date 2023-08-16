from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator

import pandas as pd
import json



def get_alphabet(net):
    activities = list({a.label for a in net.transitions if a.label and not '_' in a.label})
    return activities

def get_integer_map_net(net):
    return {x: i+1 for i,x in enumerate(get_alphabet(net))}

def log_to_list(log):
    return [[a['concept:name'] for a in t] for t in log]

def apply_integer_map(log, map):
    return [[map[a] for a in t] for t in log]

def save_log(loglist, filename): #save a list of lists 
    df = pd.DataFrame.from_records(loglist)
    df.to_csv(filename, index=False)
    

def get_variants_list(lst): #get all of the variants in a list, return as list
    st = set(tuple(i) for i in lst) #convert list into set of tuples
    lst2 = list(st) #convert set of tuples into lsit of tuples
    return [list(e) for e in lst2]

def remove_variants(log, variants_to_be_removed):
    return [tr for tr in log if tr not in variants_to_be_removed]


def get_log(model_location):
    net, im, fm = pnml_importer.apply(model_location)
    variants = simulator.apply(net, im, variant=simulator.Variants.EXTENSIVE, 
                               parameters={simulator.Variants.EXTENSIVE.value.Parameters.MAX_TRACE_LENGTH: 100,
                                           simulator.Variants.EXTENSIVE.value.Parameters.MAX_MARKING_OCC: 3})
    

    number_of_variants = len(variants)

    number_of_traces = number_of_variants * 100
    simulated_log = simulator.apply(net, im, variant=simulator.Variants.BASIC_PLAYOUT, 
                                    parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: number_of_traces})

    mapping = get_integer_map_net(net)
    #save number mapping
    mappingfilename = 'Mapping.txt'  
    with open(mappingfilename, 'w') as f:
        f.write(json.dumps(mapping))
        
    var_list = apply_integer_map(log_to_list(variants), mapping)
    save_log(var_list, "Variants.csv")
    super_log = apply_integer_map(log_to_list(simulated_log), mapping)
    save_log(super_log, "Log.csv")

    
    

