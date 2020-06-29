from multiprocessing import Pool
import sys
sys.path.append("..")
import os
sys.path.append(os.getcwd())
import household_contact_tracing.BranchingProcessSimulation as hct
import pandas as pd
import pickle
import numpy.random as npr

# npr.seed(1)

repeats = 100
days_to_simulate = 25
starting_infections = 5000

# Importing the calibration dictionary
with open('Data/Calibration/hazard_rate_detection_prob_pairs.pickle', 'rb') as handle:
    pairs_dict = pickle.load(handle)

def run_simulation(repeat):

    infection_reporting_prob = npr.choice([0.1, 0.2, 0.3, 0.4, 0.5])

    haz_rate_scale = pairs_dict[infection_reporting_prob]

    contact_tracing_success_prob = npr.uniform(0.7, 0.95)

    contact_trace_delay_par = npr.uniform(1.5, 2.5)

    reduce_contacts_by = npr.uniform(0.0, 0.9)

    # Scenario A
    # reduce_contacts_by=(0.68, 0.83, 0.83, 0.821, 0.846, 0.836)
    # Scenario B
    # reduce_contacts_by=(0.638, 0.786, 0.76, 0.733, 0.765, 0.755)
    # Scenario C
    # reduce_contacts_by=(0.628, 0.76, 0.685, 0.632, 0.668, 0.668)
    #Scenario D
    # reduce_contacts_by=(0.561, 0.698, 0.61, 0.543, 0.589, 0.577)
    # Scenario E
    # reduce_contacts_by = (0.413, 0.544, 0.393, 0.278, 0.348, 0.315)

    #do_2_step = npr.choice([True, False])

    prob_has_trace_app = npr.uniform(0, 0.5)

    test_delay_mean = npr.uniform(1, 2)

    probable_infections_need_test = npr.choice([True, False])

    test_before_propagate_tracing = True

    backwards_tracing_time_limit = npr.choice(list(range(7, 22)))

    number_of_days_to_trace_backwards = npr.choice(list(range(2, 14)))

    recall_probability_fall_off = npr.uniform(0.9, 0.95)

    simulation = hct.uk_model(haz_rate_scale=haz_rate_scale,
                            household_haz_rate_scale=0.77729,
                            contact_tracing_success_prob=contact_tracing_success_prob,
                            contact_trace_delay_par=contact_trace_delay_par,
                            overdispersion=0.36,
                            infection_reporting_prob=infection_reporting_prob,
                            contact_trace=True,
                            reduce_contacts_by=reduce_contacts_by,
                            test_before_propagate_tracing=test_before_propagate_tracing,
                            probable_infections_need_test=probable_infections_need_test,
                            test_delay_mean=test_delay_mean,
                            backwards_tracing_time_limit=backwards_tracing_time_limit,
                            number_of_days_to_trace_backwards=number_of_days_to_trace_backwards,
                            prob_has_trace_app=prob_has_trace_app,
                            starting_infections=starting_infections,
                            recall_probability_fall_off=recall_probability_fall_off)

    simulation.run_simulation(days_to_simulate)

    parameters = [
        haz_rate_scale,
        infection_reporting_prob,
        contact_tracing_success_prob,
        contact_trace_delay_par,
        reduce_contacts_by,
        prob_has_trace_app,
        test_before_propagate_tracing,
        test_delay_mean,
        probable_infections_need_test,
        number_of_days_to_trace_backwards,
        backwards_tracing_time_limit,
        recall_probability_fall_off
    ]
    return(parameters + simulation.inf_counts)

param_names = [
    "hazard_rate_scale",
    "infection_reporting_prob",
    "contact_tracing_success_prob",
    "contact_trace_delay_par",
    "global_contact_reduction",
    "prob_has_trace_app",
    "test_before_propagate_tracing",
    "test_delay_mean",
    "probable_infections_need_test",
    "number_of_days_to_trace_backwards",
    "backwards_tracing_time_limit",
    "recall_probability_fall_off"
]

col_names = param_names + [str(i) for i in range(days_to_simulate)]
col_names_dict = {}
for i in range(len(col_names)):
    col_names_dict.update({i: col_names[i]})

if __name__ == '__main__':
    with Pool(10) as p:
        results = p.map(run_simulation, range(repeats))
        results = pd.DataFrame(results)
        results = results.rename(columns=col_names_dict)
        results.to_excel("Data/Simulation Results/UK Model/backwards tracing test recall.xlsx")