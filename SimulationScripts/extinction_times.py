import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)
sys.path.append('..')
import household_contact_tracing.BranchingProcessSimulation as hct
import pickle
import pandas as pd
import numpy.random as npr
from multiprocessing import Pool

# Simulation Script Purpose:
# This simulation script simulates an epidemic until either it goes extinct, times out, or the limit is hit.

repeats = 1000
days_to_simulate = 730
starting_infections = 100

# Importing the calibration dictionary
#with open('hazard_rate_detection_prob_pairs.pickle', 'rb') as handle:
#    pairs_dict = pickle.load(handle)
     
pairs_dict = {0.1: 0.21523276281929152,
              0.2: 0.22242098020187553,
              0.3: 0.22960919758445958,
              0.4: 0.23679741496704362,
              0.5: 0.24398563234962767}

def run_simulation(repeat):
    
    npr.seed(None)

    infection_reporting_prob = npr.choice([0.1, 0.2, 0.3, 0.4, 0.5])

    haz_rate_scale = pairs_dict[infection_reporting_prob]

    contact_tracing_success_prob = npr.uniform(0.7, 0.95)

    contact_trace_delay_par = npr.uniform(1.5, 2.5)

    # Select scenario as desired

    # Full prior
    # reduce_contacts_by = npr.uniform(0, 0.9)
    # Scenario A
    #reduce_contacts_by=(0.68, 0.83, 0.83, 0.821, 0.846, 0.836)
    # Scenario B
    #reduce_contacts_by=(0.638, 0.786, 0.76, 0.733, 0.765, 0.755)
    # Scenario C
    reduce_contacts_by=(0.628, 0.76, 0.685, 0.632, 0.668, 0.668)
    #Scenario D
    #reduce_contacts_by=(0.561, 0.698, 0.61, 0.543, 0.589, 0.577)
    # Scenario E
    #reduce_contacts_by = (0.413, 0.544, 0.393, 0.278, 0.348, 0.315)

    do_2_step = npr.choice([True, False])

    backwards_trace = npr.choice([True, False])

    prob_has_trace_app = npr.uniform(0, 0.5)

    simulation = hct.household_sim_contact_tracing(haz_rate_scale=haz_rate_scale,
                                                     household_haz_rate_scale=0.77729,
                                                     contact_tracing_success_prob=contact_tracing_success_prob,
                                                     contact_trace_delay_par=contact_trace_delay_par,
                                                     overdispersion=0.36,
                                                     infection_reporting_prob=infection_reporting_prob,
                                                     contact_trace=True,
                                                     reduce_contacts_by=reduce_contacts_by,
                                                     do_2_step=do_2_step,
                                                     backwards_trace=backwards_trace,
                                                     test_before_propagate_tracing=False,
                                                     prob_has_trace_app=prob_has_trace_app,
                                                     starting_infections=starting_infections)

    simulation.run_simulation(days_to_simulate, stop_when_X_infections = True)

    parameters = [
        haz_rate_scale,
        infection_reporting_prob,
        contact_tracing_success_prob,
        contact_trace_delay_par,
        reduce_contacts_by,
        do_2_step,
        prob_has_trace_app,
        backwards_trace
    ]
    
    return(parameters + [simulation.end_reason, simulation.day_extinct] + simulation.inf_counts)


param_names = [
    "hazard_rate_scale",
    "infection_reporting_prob",
    "contact_tracing_success_prob",
    "contact_trace_delay_par",
    "global_contact_reduction",
    "two_step_tracing",
    "prob_has_trace_app",
    "backwards_trace"
]

simulation_names = [
    "end_reason",
    "extinction_time"
]

col_names = param_names + simulation_names + [str(i) for i in range(days_to_simulate)]
col_names_dict = {}
for i in range(len(col_names)):
    col_names_dict.update({i: col_names[i]})

if __name__ == '__main__':
    with Pool(8) as p:
        results = p.map(run_simulation, range(repeats))
        results = pd.DataFrame(results)
        results = results.rename(columns=col_names_dict)
        results.to_excel("100_infection_scenC.xlsx")