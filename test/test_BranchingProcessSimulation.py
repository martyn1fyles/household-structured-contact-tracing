import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)
import household_contact_tracing.BranchingProcessSimulation as hct    # The code to test
import numpy as np

# generate coverage report using:
# pytest --cov=. --cov-report xml:cov.xml
# in the terminal

test_model = hct.household_sim_contact_tracing(
    haz_rate_scale=0.805,
    contact_tracing_success_prob=0.66,
    contact_trace_delay_par=2,
    overdispersion=0.36,
    infection_reporting_prob=0.8,
    contact_trace=True,
    do_2_step=True,
    test_before_propagate_tracing=True
)


def test_two_step():
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        do_2_step=True
    )

    assert model.max_tracing_index == 2

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
    )

    assert model.max_tracing_index == 1


def test_incubation_period():

    incubation_periods = np.array([
        test_model.incubation_period()
        for i in range(10000)])

    # True mean = 4.83
    assert incubation_periods.mean() < 5
    assert incubation_periods.mean() > 4.5
    # True var = 7.7
    assert incubation_periods.var() > 7.5
    assert incubation_periods.var() < 8.5


def test_testing_delays():

    test_delays = np.array([
        test_model.testing_delay()
        for i in range(10000)])

    # True mean = 1.52 (default)
    assert test_delays.mean() < 1.75
    assert test_delays.mean() > 1.25
    # True var = 1.11
    assert test_delays.var() > 0.8
    assert test_delays.var() < 1.5


def test_reporting_delay():

    reporting_delays = np.array([
        test_model.reporting_delay()
        for i in range(10000)])

    # True mean = 2.68
    assert reporting_delays.mean() < 3
    assert reporting_delays.mean() > 2
    # True var = 2.38^2 = 5.66 ish
    assert reporting_delays.var() > 4
    assert reporting_delays.var() < 7


def test_contacts_made_today():

    house_size = 1
    contacts = np.array(
        [
            test_model.contacts_made_today(house_size)
            for i in range(10000)
        ]
    )

    # true value = 8.87
    assert contacts.mean() < 10.5
    assert contacts.mean() > 8

    house_size = 6
    contacts = np.array(
        [
            test_model.contacts_made_today(house_size)
            for i in range(1000)
        ]
    )

    # true value = 17.69
    assert contacts.mean() < 20
    assert contacts.mean() > 17


def test_contact_trace_delay():

    assert test_model.contact_trace_delay(True) == 0

    trace_delays = np.array(
        [
            test_model.contact_trace_delay(False)
            for i in range(1000)
        ]
    )

    assert trace_delays.mean() < 2.5
    assert trace_delays.mean() > 1.5


def test_new_household():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
    )

    model.time = 100

    model.new_household(new_household_number=10,
                        generation=5,
                        infected_by=6,
                        infected_by_node=3)

    house = model.houses.household(10)

    assert house.size in [1, 2, 3, 4, 5, 6]
    assert house.time == 100
    assert house.size - 1 == house.susceptibles
    assert house.generation == 5
    assert house.infected_by_id == 6
    assert house.infected_by_node == 3


def test_get_edge_between_household():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True)

    # household 1
    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    # infection 1
    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    # household 2
    model.new_household(
        new_household_number=2,
        generation=2,
        infected_by=1,
        infected_by_node=1)

    # infection 2
    model.new_infection(
        node_count=2,
        generation=2,
        household_id=2)

    # add an edge between the infections
    model.nodes.G.add_edge(1, 2)

    house1 = model.houses.household(1)
    house2 = model.houses.household(2)
    assert model.get_edge_between_household(house1, house2) == (1, 2)


def test_is_app_traced():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

    # household 1
    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    # infection 1
    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    # household 2
    model.new_household(
        new_household_number=2,
        generation=2,
        infected_by=1,
        infected_by_node=1)

    # infection 2
    model.new_infection(
        node_count=2,
        generation=2,
        household_id=2)

    # add an edge between the infections
    model.nodes.G.add_edge(1, 2)
    assert model.is_edge_app_traced((1, 2))


def test_new_outside_household_infection():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

    # household 1
    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    node1 = model.nodes.node(1)

    # infection 1
    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    model.new_outside_household_infection(
        infecting_node=node1,
        serial_interval=1
    )

    assert model.house_count == 2
    assert node1.spread_to == [2]
    assert model.nodes.G.has_edge(1, 2)


def test_within_household_infection():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    node1 = model.nodes.node(1)
    house = model.houses.household(1)
    house.house_size = 2
    house.susceptibles = 1

    model.new_within_household_infection(
        infecting_node=node1,
        serial_interval=10)

    node2 = model.nodes.node(2)

    assert house.susceptibles == 0
    assert node1.spread_to == [2]
    assert node2.household_id == 1
    assert node2.serial_interval == 10
    assert node2.generation == 2
    assert model.nodes.G.edges[1, 2]["colour"] == "black"
    assert house.within_house_edges == [(1, 2)]


def test_perform_recoveries():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    node1 = model.nodes.node(1)
    node1.recovery_time = 0
    model.perform_recoveries()
    assert node1.recovered is True


def test_colour_edges_between_houses():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    node1 = model.nodes.node(1)

    model.new_infection(
        node_count=1,
        generation=1,
        household_id=1)

    model.node_count = 2

    model.new_outside_household_infection(
        infecting_node=node1,
        serial_interval=10
    )

    house1 = model.houses.household(1)
    house2 = model.houses.household(2)
    model.colour_node_edges_between_houses(house1, house2, "yellow")
    assert model.nodes.G.edges[1, 2]["colour"] == "yellow"


def test_overide_testing_delay():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False)

    assert model.testing_delay() == 0


def test_hh_prob_leave_iso_default():
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False)
    assert model.hh_propensity_to_leave_isolation() == 0


def test_hh_prob_leave_iso():
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False,
        hh_prob_propensity_to_leave_isolation=1)
    assert model.hh_propensity_to_leave_isolation() == 1


def test_hh_has_propensity_attr():
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False,
        hh_prob_propensity_to_leave_isolation=0.5)

    assert model.houses.household(1).propensity_to_leave_isolation in (True, False)


def test_leave_isolation():

    # All households have the propensity
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False,
        hh_prob_propensity_to_leave_isolation=1,
        leave_isolation_prob=1)

    # set node 1 to the isolation status
    model.nodes.node(1).isolated = True

    # see if the node leaves isolation over the next 50 days
    for _ in range(50):
        model.decide_if_leave_isolation(node=model.nodes.node(1))
        model.time += 1

    assert model.nodes.node(1).isolated is False


def test_update_adherence_to_isolation():

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.803782,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1,
        leave_isolation_prob=0.1
    )

    # let a simulation run for a bit
    model.run_simulation(20)

    # record how many nodes are currently isolated
    initially_isolated_ids = [
        node.node_id for node in model.nodes.all_nodes()
        if node.isolated
        and not node.recovered
        and node.household().propensity_to_leave_isolation
    ]

    # update the isolation 10 
    for _ in range(10):
        model.update_adherence_to_isolation()

    secondary_isolated_ids = [
        node.node_id for node in model.nodes.all_nodes()
        if node.isolated
        and not node.recovered
        and node.household().propensity_to_leave_isolation
    ]

    assert initially_isolated_ids != secondary_isolated_ids

def test_update_adherence_to_isolation_manually():

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.803782,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1,
        leave_isolation_prob=0
    )

    # isolate a household
    # current leave isolation prob is 0, so it shouldn't have left
    model.isolate_household(model.houses.household(1))

    assert model.nodes.node(1).isolated is True

    # set the leave isolation probability to 1
    model.leave_isolation_prob = 1

    model.update_adherence_to_isolation()

    assert model.nodes.node(1).isolated is False


def test_default_household_haz_rate_scale():

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1
    )

    assert model.household_haz_rate_scale == 0.77729


def test_node_colour():

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1
    )
    model.nodes.node(1).isolated = True
    model.nodes.node(2).had_contacts_traced = True
    assert model.node_colour(model.nodes.node(1)) is "yellow"
    assert model.node_colour(model.nodes.node(2)) is "orange"
    assert model.node_colour(model.nodes.node(3)) is "white"


def test_onset_to_isolation_times():

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1
    )

    model.isolate_household(model.houses.household(1))
    node1 = model.nodes.node(1)

    assert model.onset_to_isolation_times() == [-node1.symptom_onset_time]

    assert model.onset_to_isolation_times(include_self_reports=False) == []


def test_infection_to_isolation_times():

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1
    )

    model.isolate_household(model.houses.household(1))

    assert model.infected_to_isolation_times() == [0]

    assert model.infected_to_isolation_times(include_self_reports=False) == []


def test_household_haz_rate_scale():

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        # household_haz_rate_scale=0
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1
    )

    # check the default value which is 0.77729
    assert model.household_haz_rate_scale == 0.77729

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        household_haz_rate_scale=1,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_propensity_to_leave_isolation=1
    )

    assert model.household_haz_rate_scale == 1


def test_household_not_uptake_isolation():

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        household_haz_rate_scale=1,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=10,
        hh_prob_will_take_up_isolation=0
    )

    model.isolate_household(model.houses.household(1))

    assert model.houses.household(1).isolated is False
    assert model.houses.household(1).contact_traced is True

def test_nodes_inherit_contact_traced_status():
    # If a household has been contact traced nodes should inherit this status
    # new household, contact trace it, new infection, assert new infection has contact traced status

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        household_haz_rate_scale=1,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        prob_has_trace_app=0.0,
        starting_infections=1,
    )

    # Give the household a size of atleast 2
    model.houses.household(1).house_size = 6

    model.isolate_household(model.houses.household(1))

    model.new_within_household_infection(model.nodes.node(1), serial_interval = 2)

    assert model.nodes.node(2).contact_traced == True

def test_default_uptake_isolation():
    # tests that the default value for uptake isolation is 1, otherwise households never uptake isolation

    # set up a model
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.8,
        household_haz_rate_scale=1,
        contact_tracing_success_prob=0.7,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.5,
        contact_trace=True,
        reduce_contacts_by=0.0,
        do_2_step=False,
        test_before_propagate_tracing=False,
        hh_prob_propensity_to_leave_isolation=0,
        prob_has_trace_app=0.0,
        starting_infections=1,
    )

    model.isolate_household(model.houses.household(1))

    assert model.houses.household(1).isolated is True
    assert model.houses.household(1).isolated is True

    nodes_in_hh = [node for node in model.houses.household(1).nodes()]
    node0 = nodes_in_hh[0]

    assert node0.isolated is True
    assert node0.contact_traced is True

def test_household_propensity_app():

    simulation = hct.household_sim_contact_tracing(
    haz_rate_scale=0.2,
    household_haz_rate_scale=0.8,
    overdispersion=0.32,
    contact_tracing_success_prob=0.9,
    contact_trace_delay_par=1.5,
    infection_reporting_prob=0.2,
    contact_trace=True,
    starting_infections=1,
    prob_has_trace_app=1,
    hh_propensity_to_use_trace_app=0)

    # No households have the propensity to use the trace app
    assert simulation.nodes.node(1).has_contact_tracing_app is False

    simulation = hct.household_sim_contact_tracing(
    haz_rate_scale=0.2,
    household_haz_rate_scale=0.8,
    overdispersion=0.32,
    contact_tracing_success_prob=0.9,
    contact_trace_delay_par=1.5,
    infection_reporting_prob=0.2,
    contact_trace=True,
    starting_infections=1,
    prob_has_trace_app=1,
    hh_propensity_to_use_trace_app=1)

    # No households have the propensity to use the trace app
    assert simulation.nodes.node(1).has_contact_tracing_app is True
