from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import networkx as nx
import scipy as s
import scipy.integrate as si
import math
from matplotlib.lines import Line2D
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")

# Code for demonstrating contact tracing on networks

# parameters for the generation time distribution
# mean 5, sd = 1.9
gen_shape = 2.826
gen_scale = 5.665

def weibull_pdf(t):
    out = (gen_shape / gen_scale) * (t / gen_scale)**(gen_shape - 1) * math.exp(-(t / gen_scale)**gen_shape)
    return out


def weibull_hazard(t):
    return (gen_shape / gen_scale) * (t / gen_scale)**(gen_shape - 1)


def weibull_survival(t):
    return math.exp(-(t / gen_scale)**gen_shape)


# Probability of a contact causing infection
def unconditional_hazard_rate(t, survive_forever):
    """
    Borrowed from survival analysis.

    To get the correct generation time distribution, set the probability
    of a contact on day t equal to the generation time distribution's hazard rate on day t

    Since it is not guaranteed that an individual will be infected, we use improper variables and rescale appropriately.
    The R0 scaling parameter controls this, as R0 is closely related to the probability of not being infected
    The relationship does not hold exactly in the household model, hence model tuning is required.

    Notes on the conditional variable stuff https://data.princeton.edu/wws509/notes/c7s1

    Returns
    The probability that a contact made on day t causes an infection.

    Notes:
    Currently this is using a weibull distribution, as an example.
    """
    unconditional_pdf = (1 - survive_forever) * weibull_pdf(t)
    unconditional_survival = (1 - survive_forever) * weibull_survival(t) + survive_forever
    return unconditional_pdf / unconditional_survival 


def current_hazard_rate(t, survive_forever):
    """Integrates over the unconditional hazard rate to get the probability of a contact causing infection on day t.

    survive_forever controls the probability that an infection never occurs, and is important to set R0.

    Arguments:
        t {int} -- current day
        survive_forever {float} -- rescales the hazard rate so that it is possible to not be infected
    """
    if t == 0:
        return si.quad(lambda t: unconditional_hazard_rate(t, survive_forever), 0, 0.5)[0]
    else:
        return si.quad(lambda t: unconditional_hazard_rate(t, survive_forever), t - 0.5, t + 0.5)[0]


def current_rate_infection(t):
    """Integrates over the unconditional hazard rate to get the probability of a contact causing infection on day t.

    survive_forever controls the probability that an infection never occurs, and is important to set R0.

    Arguments:
        t {int} -- current day
        survive_forever {float} -- rescales the hazard rate so that it is possible to not be infected
    """
    if t == 0:
        return si.quad(lambda t: weibull_pdf(t), 0, 0.5)[0]
    else:
        return si.quad(lambda t: weibull_pdf(t), t - 0.5, t + 0.5)[0]

def current_prob_leave_isolation(t, survive_forever):
    """Integrates over the unconditional hazard rate to get the probability of a contact causing infection on day t.

    survive_forever controls the probability that an infection never occurs, and is important to set R0.

    Arguments:
        t {int} -- current day
        survive_forever {float} -- rescales the hazard rate so that it is possible to not be infected
    """
    return si.quad(lambda t: unconditional_hazard_rate(t, survive_forever), t, t+1)[0]

def negbin_pdf(x, m, a):
    """
    We need to draw values from an overdispersed negative binomial distribution, with non-integer inputs. Had to
    generate the numbers myself in order to do this.
    This is the generalized negbin used in glm models I think.

    m = mean
    a = overdispertion 
    """
    A = math.gamma(x + 1 / a) / (math.gamma(x + 1) * math.gamma(1 / a))
    B = (1 / (1 + a * m))**(1 / a)
    C = (a * m / (1 + a * m))**x
    return A * B * C


def compute_negbin_cdf(mean, overdispersion, length_out):
    """
    Computes the overdispersed negative binomial cdf, which we use to generate random numbers by generating uniform(0,1)
    rv's.
    """
    pdf = [negbin_pdf(i, mean, overdispersion) for i in range(length_out)]
    cdf = [sum(pdf[:i]) for i in range(length_out)]
    return cdf


class Node:

    def __init__(
        self,
        nodes: 'NodeCollection',
        houses: 'HouseholdCollection',
        node_id: int,
        time_infected: int,
        generation: int,
        household: int,
        isolated: bool,
        symptom_onset_time: int,
        serial_interval: int,
        recovery_time: int,
        will_report_infection: bool,
        time_of_reporting: int,
        has_contact_tracing_app: bool,
        testing_delay: int,
        contact_traced: bool,
        had_contacts_traced=False,
        outside_house_contacts_made=0,
        spread_to: Optional[List[int]] = None,
        recovered=False
    ):
        self.nodes = nodes
        self.houses = houses
        self.node_id = node_id
        self.time_infected = time_infected
        self.generation = generation
        self.household_id = household
        self.isolated = isolated
        self.symptom_onset_time = symptom_onset_time
        self.serial_interval = serial_interval
        self.recovery_time = recovery_time
        self.will_report_infection = will_report_infection
        self.time_of_reporting = time_of_reporting
        self.has_contact_tracing_app = has_contact_tracing_app
        self.testing_delay = testing_delay
        self.contact_traced = contact_traced
        self.had_contacts_traced = had_contacts_traced
        self.outside_house_contacts_made = outside_house_contacts_made
        self.spread_to = spread_to if spread_to else []
        self.spread_to_global_node_time_tuples = []
        self.recovered = recovered

    def household(self) -> 'Household':
        return self.houses.household(self.household_id)


class NodeCollection:

    def __init__(self, houses: 'HouseholdCollection'):
        self.G = nx.Graph()
        self.houses = houses
        # TODO: put node_count in this class

    def add_node(
        self, node_id, time, generation, household, isolated,
        symptom_onset_time, serial_interval, recovery_time, will_report_infection,
        time_of_reporting, has_contact_tracing_app, testing_delay, contact_traced
    ) -> Node:
        self.G.add_node(node_id)
        node = Node(
            self, self.houses, node_id,
            time, generation, household, isolated,
            symptom_onset_time, serial_interval, recovery_time, will_report_infection,
            time_of_reporting, has_contact_tracing_app, testing_delay, contact_traced
        )
        self.G.nodes[node_id]['node_obj'] = node
        return node

    def node(self, node_id) -> Node:
        return self.G.nodes[node_id]['node_obj']

    def all_nodes(self) -> Iterator[Node]:
        return (self.node(n) for n in self.G)


class Household:

    def __init__(
        self,
        houses: 'HouseholdCollection',
        nodecollection: NodeCollection,
        house_id: int,
        house_size: int,
        time_infected: int,
        propensity,
        generation: int,
        infected_by: int,
        infected_by_node: int,
        propensity_trace_app: bool
    ):
        self.houses = houses
        self.nodecollection = nodecollection
        self.house_id = house_id
        self.size = house_size                  # Size of the household
        self.time = time_infected               # The time at which the infection entered the household
        self.susceptibles = house_size - 1      # How many susceptibles remain in the household
        self.isolated = False                   # Has the household been isolated, so there can be no more infections from this household
        self.isolated_time = float('inf')       # When the house was isolated
        self.propensity_to_leave_isolation = propensity
        self.propensity_trace_app = propensity_trace_app
        self.contact_traced = False             # If the house has been contact traced, it is isolated as soon as anyone in the house shows symptoms
        self.time_until_contact_traced = float('inf')  # The time until quarantine, calculated from contact tracing processes on connected households
        self.contact_traced_household_ids: List[int] = []  # The list of households contact traced from this one
        self.being_contact_traced_from: Optional[int] = None   # If the house if being contact traced, this is the house_id of the first house that will get there
        self.propagated_contact_tracing = False  # The house has not yet propagated contact tracing
        self.time_propagated_tracing: Optional[int] = None     # Time household propagated contact tracing
        self.contact_tracing_index = 0          # The house is which step of the contact tracing process
        self.generation = generation            # Which generation of households it belongs to
        self.infected_by_id = infected_by       # Which house infected the household
        self.spread_to_ids: List[int] = []          # Which households were infected by this household
        self.node_ids: List[int] = []           # The ID of currently infected nodes in the household
        self.infected_by_node = infected_by_node  # Which node infected the household
        self.within_house_edges: List[Tuple[int, int]] = []  # Which edges are contained within the household
        self.had_contacts_traced = False         # Have the nodes inside the household had their contacts traced?

    def nodes(self) -> Iterator[Node]:
        return (self.nodecollection.node(n) for n in self.node_ids)

    def add_node_id(self, node_id: int):
        self.node_ids.append(node_id)

    def contact_traced_households(self) -> Iterator['Household']:
        return (self.houses.household(hid) for hid in self.contact_traced_household_ids)

    def spread_to(self) -> Iterator['Household']:
        return (self.houses.household(hid) for hid in self.spread_to_ids)

    def infected_by(self) -> 'Household':
        if self.infected_by_id is None:
            return None
        return self.houses.household(self.infected_by_id)


class HouseholdCollection:

    def __init__(self, nodes: NodeCollection):
        self.house_dict: Dict[int, Household] = {}
        self.nodes = nodes
        # TODO: put house_count in this class

    def add_household(
        self,
        house_id: int,
        house_size: int,
        time_infected: int,
        propensity: bool,
        generation: int,
        infected_by: int,
        infected_by_node: int,
        propensity_trace_app: bool
    ) -> Household:
        new_household = Household(
            self, self.nodes, house_id,
            house_size, time_infected, propensity, generation, infected_by,
            infected_by_node, propensity_trace_app
        )
        self.house_dict[house_id] = new_household
        return new_household

    def household(self, house_id) -> Household:
        return self.house_dict[house_id]

    @property
    def count(self) -> int:
        return len(self.house_dict)

    def all_households(self) -> Iterator[Household]:
        return (self.household(hid) for hid in self.house_dict)


# Precomputing the cdf's for generating the overdispersed contact data, saves a lot of time later
class household_sim_contact_tracing:
    # We assign each node a recovery period of 14 days, after 14 days the probability of causing a new infections is 0,
    # due to the generation time distribution
    effective_infectious_period = 21

    # Working out the parameters of the incubation period
    ip_mean = 4.83
    ip_var = 2.78**2
    ip_scale = ip_var / ip_mean
    ip_shape = ip_mean ** 2 / ip_var

    # Visual Parameters:
    contact_traced_edge_colour_within_house = "blue"
    contact_traced_edge_between_house = "magenta"
    default_edge_colour = "black"
    failed_contact_tracing = "red"
    app_traced_edge = "green"

    # Local contact probability:
    local_contact_probs = [0, 0.826, 0.795, 0.803, 0.787, 0.819]

    # The mean number of contacts made by each household
    total_contact_means = [7.238, 10.133, 11.419, 12.844, 14.535, 15.844]

    def __init__(self,
                 haz_rate_scale,
                 contact_tracing_success_prob,
                 contact_trace_delay_par,
                 overdispersion,
                 infection_reporting_prob,
                 contact_trace,
                 household_haz_rate_scale=0.77729,
                 do_2_step=False,
                 backwards_trace=True,
                 reduce_contacts_by=0,
                 prob_has_trace_app=0,
                 hh_propensity_to_use_trace_app=1,
                 test_delay_mean=1.52,
                 test_before_propagate_tracing=True,
                 starting_infections=1,
                 hh_prob_will_take_up_isolation=1,
                 hh_prob_propensity_to_leave_isolation=0,
                 leave_isolation_prob=0):
        """Initializes parameters and distributions for performing a simulation of contact tracing.
        The epidemic is modelled as a branching process, with nodes assigned to households.

        Arguments:
            proportion_of_within_house_contacts {[type]} -- [description]
            haz_rate_scale {[type]} -- controls the R_0 by rescaling the hazard rate function
            contact_tracing_success_prob {[type]} -- [description]
            prob_of_successful_contact_trace_today {[type]} -- [description]
        """
        # Probability of each household size
        house_size_probs = [0.294591195, 0.345336927, 0.154070081, 0.139478886, 0.045067385, 0.021455526]

        # Precomputing the cdf's for generating the overdispersed contact data
        self.cdf_dict = {
            1: compute_negbin_cdf(self.total_contact_means[0], overdispersion, 100),
            2: compute_negbin_cdf(self.total_contact_means[1], overdispersion, 100),
            3: compute_negbin_cdf(self.total_contact_means[2], overdispersion, 100),
            4: compute_negbin_cdf(self.total_contact_means[3], overdispersion, 100),
            5: compute_negbin_cdf(self.total_contact_means[4], overdispersion, 100),
            6: compute_negbin_cdf(self.total_contact_means[5], overdispersion, 100)
        }

        # Calculate the expected local contacts
        expected_local_contacts = [self.local_contact_probs[i] * i for i in range(6)]

        # Calculate the expected global contacts
        expected_global_contacts = np.array(self.total_contact_means) - np.array(expected_local_contacts)

        # Size biased distribution of households (choose a node, what is the prob they are in a house size 6, this is
        # biased by the size of the house)
        size_mean_contacts_biased_distribution = [(i + 1) * house_size_probs[i] * expected_global_contacts[i] for i in range(6)]
        total = sum(size_mean_contacts_biased_distribution)
        self.size_mean_contacts_biased_distribution = [prob / total for prob in size_mean_contacts_biased_distribution]

        # Parameter Inputs:
        self.haz_rate_scale = haz_rate_scale
        # If a household hazard rate scale is not specified, we assume it is the same as the outside-household
        # hazard rate scaling
        if household_haz_rate_scale is None:
            self.household_haz_rate_scale = self.haz_rate_scale
        else:
            self.household_haz_rate_scale = household_haz_rate_scale
        self.contact_tracing_success_prob = contact_tracing_success_prob
        self.contact_trace_delay_par = contact_trace_delay_par
        self.overdispersion = overdispersion
        self.infection_reporting_prob = infection_reporting_prob
        self.contact_trace = contact_trace
        self.prob_has_trace_app = prob_has_trace_app
        self.hh_propensity_to_use_trace_app = hh_propensity_to_use_trace_app
        self.reduce_contacts_by = reduce_contacts_by
        self.do_2_step = do_2_step
        self.backwards_trace = backwards_trace
        self.test_before_propagate_tracing = test_before_propagate_tracing
        self.test_delay_mean = test_delay_mean
        self.starting_infections = starting_infections
        self.hh_prob_will_take_up_isolation = hh_prob_will_take_up_isolation
        self.hh_prob_propensity_to_leave_isolation = hh_prob_propensity_to_leave_isolation
        self.leave_isolation_prob = leave_isolation_prob
        if do_2_step:
            self.max_tracing_index = 2
        else:
            self.max_tracing_index = 1
        if type(self.reduce_contacts_by) is tuple:
            self.contact_rate_reduction_by_household = True
        else:
            self.contact_rate_reduction_by_household = False

        # Precomputing the infection probabilities for the within household epidemics.
        contact_prob = 0.8
        day_0_infection_prob = current_hazard_rate(0, self.household_haz_rate_scale)/contact_prob
        infection_probs = np.array(day_0_infection_prob)
        for day in range(1, 15):
            survival_function = (1 - infection_probs*contact_prob).prod()
            hazard = current_hazard_rate(day, self.household_haz_rate_scale)
            current_prob_infection = hazard * survival_function / contact_prob
            infection_probs = np.append(infection_probs, current_prob_infection)
        self.hh_infection_probs = infection_probs

        
        # Precomputing the global infection probabilities
        self.global_infection_probs = []
        for day in range(15):
            self.global_infection_probs.append(self.haz_rate_scale * current_rate_infection(day))

        # Calls the simulation reset function, which creates all the required dictionaries
        self.reset_simulation()

    def contact_trace_delay(self, app_traced_edge):
        if app_traced_edge:
            return 0
        else:
            return npr.poisson(self.contact_trace_delay_par)

    def incubation_period(self):
        return round(npr.gamma(
            shape=self.ip_shape,
            scale=self.ip_scale))

    def testing_delay(self):
        if self.test_before_propagate_tracing is False:
            return 0
        else:
            return round(npr.gamma(
                shape=self.test_delay_mean**2 / 1.11**2,
                scale=1.11**2 / self.test_delay_mean))

    def reporting_delay(self):
        return round(npr.gamma(
            shape=2.62**2/2.38**2,
            scale=2.38**2/2.62))

    def hh_propensity_to_leave_isolation(self):
        if npr.binomial(1, self.hh_prob_propensity_to_leave_isolation) == 1:
            return True
        else:
            return False
        
    def hh_will_take_up_isolation(self):
        if npr.binomial(1, self.hh_prob_will_take_up_isolation) == 1:
            return True
        else:
            return False

    def hh_propensity_use_trace_app(self):
        if npr.binomial(1, self.hh_propensity_to_use_trace_app) == 1:
            return True
        else:
            return False

    def contacts_made_today(self, household_size):
        """Generates the number of contacts made today by a node, given the house size of the node. Uses an
        overdispersed negative binomial distribution.

        Arguments:
            house_size {int} -- size of the nodes household
        """
        random = npr.uniform()
        cdf = self.cdf_dict[household_size]
        obs = sum([int(cdf[i] < random) for i in range(100)])
        return obs

    def size_of_household(self):
        """Generates a random household size

        Returns:
        household_size {int}
        """
        return npr.choice([1, 2, 3, 4, 5, 6], p=self.size_mean_contacts_biased_distribution)

    def has_contact_tracing_app(self):
        return npr.binomial(1, self.prob_has_trace_app) == 1

    def count_non_recovered_nodes(self) -> int:
        """Returns the number of nodes not in the recovered state.
        Returns:
            [int] -- Number of non-recovered nodes.
        """
        # TODO: BUG - if **not** recovered ??
        return len([node for node in self.nodes.all_nodes() if node.recovered])

    def new_infection(self, node_count: int, generation: int, household_id: int, serial_interval=None, infecting_node=None):
        """
        Adds a new infection to the graph along with the following attributes:
        t - when they were infected
        offspring - how many offspring they produce

        Inputs::
        G - the network object
        time - the time when the new infection happens
        node_count - how many nodes are currently in the network
        """
        # Symptom onset time
        symptom_onset_time = self.time + self.incubation_period()
        # When a node reports it's infection
        if npr.binomial(1, self.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self.reporting_delay()
        else:
            will_report_infection = False
            time_of_reporting = float('Inf')

        # We assign each node a recovery period of 21 days, after 21 days the probability of causing a new infections is
        # 0, due to the generation time distribution
        recovery_time = self.time + 14

        household = self.houses.household(household_id)

        # If the household has the propensity to use the contact tracing app, decide if the node uses the app.
        if household.propensity_trace_app:
            has_trace_app = self.has_contact_tracing_app()
        else:
            has_trace_app = False

        node = self.nodes.add_node(
            node_id=node_count,
            time=self.time,
            generation=generation,
            household=household_id,
            isolated=household.isolated,
            contact_traced=household.contact_traced,
            symptom_onset_time=symptom_onset_time,
            serial_interval=serial_interval,
            recovery_time=recovery_time,
            will_report_infection=will_report_infection,
            time_of_reporting=time_of_reporting,
            has_contact_tracing_app=has_trace_app,
            testing_delay=self.testing_delay(),
        )

        # Updates to the household dictionary
        # Each house now stores a the ID's of which nodes are stored inside the house, so that quarantining can be done at the household level
        household.node_ids.append(node_count)

        # A number of days may have passed since the house was isolated
        # We need to decide if the node has left isolation already, since it did not previously exist
        if household.isolated:
            days_isolated = int(self.time - household.isolated_time)
            for _ in range(days_isolated):
                self.decide_if_leave_isolation(node)

        node.infected_by_node = infecting_node

        if infecting_node:
            if infecting_node.household().house_id == household_id:
                node.locally_infected = True
        else:
            node.locally_infected = False

    def new_household(self, new_household_number: int, generation: int, infected_by: int, infected_by_node: int):
        """Adds a new household to the household dictionary

        Arguments:
            new_household_number {int} -- The house id
            generation {int} -- The household generation of this household
            infected_by {int} -- Which household spread the infection to this household
            infected_by_node {int} -- Which node spread the infection to this household
        """
        house_size = self.size_of_household()

        propensity = self.hh_propensity_to_leave_isolation()

        propensity_trace_app = self.hh_propensity_use_trace_app()

        self.houses.add_household(
            house_id=new_household_number,
            house_size=house_size,
            time_infected=self.time,
            propensity=propensity,
            generation=generation,
            infected_by=infected_by,
            infected_by_node=infected_by_node,
            propensity_trace_app=propensity_trace_app
        )

    def get_edge_between_household(self, house1: Household, house2: Household):
        for node1 in house1.nodes():
            for node2 in house2.nodes():
                if self.G.has_edge(node1.node_id, node2.node_id):
                    return (node1.node_id, node2.node_id)

    def is_edge_app_traced(self, edge):
        """Returns whether both ends of an edge have the app, and the app does the tracing.
        """
        return self.nodes.node(edge[0]).has_contact_tracing_app and self.nodes.node(edge[1]).has_contact_tracing_app

    @property
    def active_infections(self):
        return [
            node for node in self.nodes.all_nodes()
            if node.time_of_reporting >= self.time
            and not node.recovered
        ]

    def get_contact_rate_reduction(self, house_size):
        """For a house size input, returns a contact rate reduction

        Arguments:
            house_size {int} -- The household size
        """
        if self.contact_rate_reduction_by_household is True:
            return self.reduce_contacts_by[house_size - 1]
        else:
            return self.reduce_contacts_by

    def increment_infection(self):
        """
        Creates a new days worth of infections
        """

        for node in self.active_infections:
            household = node.household()

            # Extracting useful parameters from the node
            days_since_infected = self.time - node.time_infected

            outside_household_contacts = -1

            while outside_household_contacts < 0:

                # The number of contacts made that day
                contacts_made = self.contacts_made_today(household.size)

                # How many of the contacts are within the household
                local_contacts = npr.binomial(household.size - 1, self.local_contact_probs[household.size - 1])

                # How many of the contacts are outside household contacts
                outside_household_contacts = contacts_made - local_contacts

            # If there is social distancing perform bernoulli thinning of the global contacts
            if node.isolated:
                outside_household_contacts = 0
            else:
                outside_household_contacts = npr.binomial(
                    outside_household_contacts,
                    1 - self.get_contact_rate_reduction(house_size=household.size)
                )

            # Within household, how many of the infections would cause new infections
            # These contacts may be made with someone who is already infected, and so they will again be thinned
            local_infective_contacts = npr.binomial(
                local_contacts,
                self.hh_infection_probs[days_since_infected]
            )

            for _ in range(local_infective_contacts):
                # A further thinning has to happen since each attempt may choose an already infected person
                # That is to say, if everyone in your house is infected, you have 0 chance to infect a new person in your house

                # A one represents a susceptibles node in the household
                # A 0 represents an infected member of the household
                # We choose a random subset of this vector of length local_infective_contacts to determine infections
                # i.e we are choosing without replacement
                household_composition = [1]*household.susceptibles + [0]*(household.size - 1 - household.susceptibles)
                within_household_new_infections = sum(npr.choice(household_composition, local_infective_contacts, replace=False))

                # If the within household infection is successful:
                for _ in range(within_household_new_infections):
                    self.new_within_household_infection(
                        infecting_node=node,
                        serial_interval=days_since_infected
                    )

            # Update how many contacts the node made
            node.outside_house_contacts_made += outside_household_contacts

            # How many outside household contacts cause new infections
            outside_household_new_infections = npr.binomial(outside_household_contacts, self.global_infection_probs[days_since_infected])

            for _ in range(outside_household_new_infections):
                self.new_outside_household_infection(
                    infecting_node=node,
                    serial_interval=days_since_infected)

                node_time_tuple = (nx.number_of_nodes(self.G), self.time)

                node.spread_to_global_node_time_tuples.append(node_time_tuple)

    def new_within_household_infection(self, infecting_node: Node, serial_interval: Optional[int]):
        # Add a new node to the network, it will be a member of the same household that the node that infected it was
        node_count = nx.number_of_nodes(self.G) + 1

        # We record which node caused this infection
        infecting_node.spread_to.append(node_count)

        infecting_node_household = infecting_node.household()

        # Adds the new infection to the network
        self.new_infection(node_count=node_count,
                           generation=infecting_node.generation + 1,
                           household_id=infecting_node_household.house_id,
                           serial_interval=serial_interval,
                           infecting_node=infecting_node)

        # Add the edge to the graph and give it the default colour
        self.G.add_edge(infecting_node.node_id, node_count)
        self.G.edges[infecting_node.node_id, node_count].update({"colour": self.default_edge_colour})

        # Decrease the number of susceptibles in that house by 1
        infecting_node_household.susceptibles -= 1

        # We record which edges are within this household for visualisation later on
        infecting_node_household.within_house_edges.append((infecting_node.node_id, node_count))

    def new_outside_household_infection(self, infecting_node: Node, serial_interval: Optional[int]):
        # We assume all new outside household infections are in a new household
        # i.e: You do not infect 2 people in a new household
        # you do not spread the infection to a household that already has an infection
        self.house_count += 1
        node_count = nx.number_of_nodes(self.G) + 1
        infecting_household = infecting_node.household()

        # We record which node caused this infection
        infecting_node.spread_to.append(node_count)

        # We record which house spread to which other house
        infecting_household.spread_to_ids.append(self.house_count)

        # Create a new household, since the infection was outside the household
        self.new_household(new_household_number=self.house_count,
                           generation=infecting_household.generation + 1,
                           infected_by=infecting_node.household_id,
                           infected_by_node=infecting_node.node_id)

        # add a new infection in the house just created
        self.new_infection(node_count=node_count,
                           generation=infecting_node.generation + 1,
                           household_id=self.house_count,
                           serial_interval=serial_interval,
                           infecting_node=infecting_node)

        # Add the edge to the graph and give it the default colour
        self.G.add_edge(infecting_node.node_id, node_count)
        self.G.edges[infecting_node.node_id, node_count].update({"colour": "black"})

    def update_isolation(self):
        # Update the contact traced status for all households that have had the contact tracing process get there
        [
            self.contact_trace_household(household)
            for household in self.houses.all_households()
            if household.time_until_contact_traced <= self.time
            and not household.contact_traced
        ]

        # Isolate all non isolated households where the infection has been reported (excludes those who will not take up isolation if prob <1)
        [
            self.isolate_household(node.household())
            for node in self.nodes.all_nodes()
            if node.time_of_reporting <= self.time
            and not node.isolated
            and not node.household().contact_traced
        ]


    def increment_contact_tracing(self):
        """
        Performs a days worth of contact tracing by:
        * Looking for nodes that have been admitted to hospital. Once a node is admitted to hospital, it's house is isolated
        * Looks for houses where the contact tracing delay is over and moves them to the contact traced state
        * Looks for houses in the contact traced state, and checks them for symptoms. If any of them have symptoms, the house is isolated

        The isolation function also assigns contact tracing times to any houses that had contact with that household

        For each node that is contact traced
        """

        # Isolate all households under observation that now display symptoms (excludes those who will not take up isolation if prob <1)
        # TODO can this be removed?
        [
            self.isolate_household(node.household())
            for node in self.nodes.all_nodes()
            if node.symptom_onset_time <= self.time
            and node.contact_traced
            and not node.isolated
        ]

        # Look for houses that need to propagate the contact tracing because their test result has come back
        # Necessary conditions: household isolated, symptom onset + testing delay = time

        # Propagate the contact tracing for all households that self-reported and have had their test results come back
        [
            self.propagate_contact_tracing(node.household())
            for node in self.nodes.all_nodes()
            if node.time_of_reporting + node.testing_delay == self.time
            and not node.household().propagated_contact_tracing
        ]

        # Propagate the contact tracing for all households that are isolated due to exposure, have developed symptoms and had a test come back
        [
            self.propagate_contact_tracing(node.household())
            for node in self.nodes.all_nodes()
            if node.symptom_onset_time <= self.time
            and not node.household().propagated_contact_tracing
            and node.household().isolated_time + node.testing_delay <= self.time
        ]

        # Update the contact tracing index of households
        # That is, re-evaluate how far away they are from a known infected case (traced + symptom_onset_time + testing_delay)
        self.update_contact_tracing_index()

        if self.do_2_step:
            # Propagate the contact tracing from any households with a contact tracing index of 1
            [
                self.propagate_contact_tracing(household)
                for household in self.houses.all_households()
                if household.contact_tracing_index == 1
                and not household.propagated_contact_tracing
                and household.isolated
            ]

    def contact_trace_household(self, household: Household):
        """
        When a house is contact traced, we need to place all the nodes under surveillance.

        If any of the nodes are symptomatic, we need to isolate the household.
        """
        # Update the house to the contact traced status
        household.contact_traced = True

        # Update the nodes to the contact traced status
        for node in household.nodes():
            node.contact_traced = True

        # Colour the edges within household
        [
            self.G.edges[edge[0], edge[1]].update({"colour": self.contact_traced_edge_colour_within_house})
            for edge in household.within_house_edges
        ]

        # If there are any nodes in the house that are symptomatic, isolate the house:
        symptomatic_nodes = [node for node in household.nodes() if node.symptom_onset_time <= self.time]
        if symptomatic_nodes != []:
            self.isolate_household(household)
        else:
            self.isolate_household(household)


    def perform_recoveries(self):
        """
        Loops over all nodes in the branching process and determins recoveries.

        time - The current time of the process, if a nodes recovery time equals the current time, then it is set to the recovered state
        """
        for node in self.nodes.all_nodes():
            if node.recovery_time == self.time:
                node.recovered = True


    def colour_node_edges_between_houses(self, house_to: Household, house_from: Household, new_colour):
        # Annoying bit of logic to find the edge and colour it
        for node_1 in house_to.nodes():
            for node_2 in house_from.nodes():
                if self.nodes.G.has_edge(node_1.node_id, node_2.node_id):
                    self.nodes.G.edges[node_1.node_id, node_2.node_id].update({"colour": new_colour})


    def attempt_contact_trace_of_household(self, house_to: Household, house_from: Household, contact_trace_delay: int = 0):
        # Decide if the edge was traced by the app
        app_traced = self.is_edge_app_traced(self.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing_success_prob

        # is the trace successful
        if (npr.binomial(1, success_prob) == 1):
            # Update the list of traced households from this one
            house_from.contact_traced_household_ids.append(house_to.house_id)

            # Assign the household a contact tracing index, 1 more than it's parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            contact_trace_delay = contact_trace_delay + self.contact_trace_delay(app_traced)
            proposed_time_until_contact_trace = self.time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time until contact trace
            # Note this starts as infinity
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from.house_id

            # Edge colouring
            if app_traced:
                self.colour_node_edges_between_houses(house_to, house_from, self.app_traced_edge)
            else:
                self.colour_node_edges_between_houses(house_to, house_from, self.contact_traced_edge_between_house)
        else:
            self.colour_node_edges_between_houses(house_to, house_from, self.failed_contact_tracing)


    def isolate_household(self, household: Household):
        """
        Isolates a house so that all infectives in that household may no longer infect others.

        If the house is being surveillance due to a successful contact trace, and not due to reporting symptoms,
        update the edge colour to display this.

        For households that were connected to this household, they are assigned a time until contact traced

        When a house has been contact traced, all nodes in the house are under surveillance for symptoms. When a node becomes symptomatic, the house moves to isolation status.
        """
        household.contact_traced = True

        for node in household.nodes():
            node.contact_traced = True

        # Households have a probability to take up isolation if traced
        if self.hh_will_take_up_isolation():
            
        # The house moves to isolated status if it has been assigned to take up isolation if trace, given a probability
            household.isolated = True
        # household.contact_traced = True
            household.isolated_time = self.time

            # Update every node in the house to the isolated status
            for node in household.nodes():
                node.isolated = True


            # Which house started the contact trace that led to this house being isolated, if there is one
            # A household may be being isolated because someone in the household self reported symptoms
            # Hence sometimes there is a None value for House which contact traced
            if household.being_contact_traced_from is not None:
                house_which_contact_traced = self.houses.household(household.being_contact_traced_from)
                
                # Initially the edge is assigned the contact tracing colour, may be updated if the contact tracing does not succeed
                if self.is_edge_app_traced(self.get_edge_between_household(household, house_which_contact_traced)):
                    self.colour_node_edges_between_houses(household, house_which_contact_traced, self.app_traced_edge)
                else:
                    self.colour_node_edges_between_houses(household, house_which_contact_traced, self.contact_traced_edge_between_house)
                        
                    # We update the colour of every edge so that we can tell which household have been contact traced when we visualise
                    [
                        self.G.edges[edge[0], edge[1]].update({"colour": self.contact_traced_edge_colour_within_house})
                        for edge in household.within_house_edges
                        ]


    def decide_if_leave_isolation(self, node: Node):
        """
        If a node lives in a household with the propensity to not adhere to isolation, then this
        function decides if the node will leave isolation, conditional upon how many days it's been
        since the node was isolated.

        Only makes sense to apply this function to isolated nodes, in a household with propensity to
        leave isolation
        """
        if npr.binomial(1, self.leave_isolation_prob) == 1:
            node.isolated = False


    def propagate_contact_tracing(self, household: Household):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household that is under surveillance develops symptoms + gets tested.
        """
        # update the propagation data
        household.propagated_contact_tracing = True
        household.time_propagated_tracing = self.time

        # Contact tracing attempted for the household that infected the household currently propagating the infection
        
        infected_by = household.infected_by()

        # If infected by = None, then it is the origin node, a special case
        if self.backwards_trace is True:
            if infected_by and not infected_by.isolated:
                self.attempt_contact_trace_of_household(infected_by, household)

        # Contact tracing for the households infected by the household currently traced
        child_households_not_traced = [h for h in household.spread_to() if not h.isolated]
        for child in child_households_not_traced:
            self.attempt_contact_trace_of_household(child, household)


    def update_contact_tracing_index(self):
        for household in self.houses.all_households():
            # loop over households with non-zero indexes, those that have been contact traced but with
            if household.contact_tracing_index != 0:
                for node in household.nodes():

                    # Necessary conditions for an index 1 household to propagate tracing:
                    # The node must have onset of symptoms
                    # The node households must be isolated
                    # The testing delay must be passed
                    # The testing delay starts when the house have been isolated and symptoms have onset
                    critical_time = max(node.symptom_onset_time, household.isolated_time)

                    if critical_time + node.testing_delay <= self.time:
                        household.contact_tracing_index = 0

                        for index_1_hh in household.contact_traced_households():
                            if index_1_hh.contact_tracing_index == 2:
                                index_1_hh.contact_tracing_index = 1


    def update_adherence_to_isolation(self):
        """Loops over nodes currently in quarantine, and updates whether they are currently adhering to
        quarantine, if their household has the propensity to not adhere.
        """
        [
            self.decide_if_leave_isolation(node)
            for node in self.nodes.all_nodes()
            if node.isolated
            and node.household().propensity_to_leave_isolation
        ]


    def release_nodes_from_quarantine(self):
        """If a node has completed the quarantine according to the following rules, they are released from
        quarantine.

        You are released from isolation if:
            * it has been 7 days since your symptoms onset
            and
            * it has been a minimum of 14 days since your household was isolated
        """
        for node in self.nodes.all_nodes():
            if self.time >= node.symptom_onset_time + 7 and self.time >= node.household().isolated_time + 14:
                node.isolated = False


    def simulate_one_day(self):
        """Simulates one day of the epidemic and contact tracing.

        Useful for bug testing and visualisation.
        """
        self.increment_infection()
        self.update_isolation()
        if self.contact_trace is True:
            for _ in range(1):
                self.increment_contact_tracing()
        self.perform_recoveries()
        self.release_nodes_from_quarantine()
        self.update_adherence_to_isolation()
        self.time += 1


    def reset_simulation(self):
        """
        Returns the simulation to it's initially specified values
        """

        self.time = 0

        # Stores information about the contact tracing that has occurred.
        self.contact_tracing_dict = {
            "contacts_to_be_traced": 0,         # connections made by nodes that are contact traced and symptomatic
            "possible_to_trace_contacts": 0,    # contacts that are possible to trace assuming a failure rate, not all connections will be traceable
            "total_traced_each_day": [0],       # A list recording the the number of contacts added to the system each day
            "daily_active_surveillances": [],   # A list recording how many surveillances were happening each day
            "currently_being_surveilled": 0,    # Ongoing surveillances
            "day_800_cases_traced": None        # On which day was 800 cases reached
        }

        # Create the empty graph - we add the houses properly below
        self.nodes = NodeCollection(None)

        # Stores information about the households.
        self.houses = HouseholdCollection(self.nodes)
        self.nodes.houses = self.houses

        # make things available as before
        self.G = self.nodes.G

        # Create first household
        self.house_count = 0

        # Initial values
        node_count = 1
        generation = 0

        # Create the starting infectives
        for _ in range(self.starting_infections):
            self.house_count += 1
            node_count = nx.number_of_nodes(self.G) + 1
            self.new_household(self.house_count, 1, None, None)
            self.new_infection(node_count, generation, self.house_count)


    def run_simulation_hitting_times(self, time_out):

        # Return the simulation to it's initial starting state
        self.reset_simulation()

        # Get the number of current nodes in the network
        node_count = nx.number_of_nodes(self.G)

        # For recording the number of cases over time
        total_cases = []

        # Setting up parameters for this run of the experiment
        self.time_800 = None    # Time when we hit 800 under surveillance
        self.time_8000 = None   # Time when we hit 8000 under surveillance
        self.hit_800 = False    # flag used for the code that records the first time we hit 800 under surveillance
        self.hit_8000 = False   # same but for 8000
        self.died_out = False   # flag for whether the epidemic has died out
        self.timed_out = False  # flag for whether the simulation reached it's time limit without another stop condition being met

        # While loop ends when there are no non-isolated infections
        currently_infecting = len([node for node in self.nodes.all_nodes() if not node.recovered])

        while (currently_infecting != 0 and self.hit_8000 is False and self.timed_out is False):

            # This chunk of code executes a days worth on infections and contact tracings
            node_count = nx.number_of_nodes(self.G)
            self.simulate_one_day()

            self.house_count = self.houses.count
            total_cases.append(node_count)

            # While loop ends when there are no non-isolated infections
            currently_infecting = len([node for node in self.nodes.all_nodes() if not node.isolated])

            # Records the first time we hit 800 under surveillance
            if (self.contact_tracing_dict["currently_being_surveilled"] > 800 and self.hit_800 is False):
                self.time_800 = self.time
                self.hit_800 = True

            # Records the first time we hit 8000 surveilled
            if (self.contact_tracing_dict["currently_being_surveilled"] > 8000 and self.hit_8000 is False):
                self.time_8000 = self.time
                self.hit_8000 = True

            if currently_infecting == 0:
                self.died_out = True

            if self.time == time_out:
                self.timed_out = True

        # Infection Count output
        self.inf_counts = total_cases
        

    def run_simulation_detection_times(self):
        
         # Create all the required dictionaries and reset parameters
        self.reset_simulation()

        # For recording the number of cases over time
        self.total_cases = []

        # Initial values
        self.end_reason = ''
        self.timed_out = False
        self.extinct = False
        self.day_extinct = -1

        # While loop ends when there are no non-isolated infections
        currently_infecting = len([node for node in self.G.nodes() if self.G.nodes[node]["recovered"] is False])

        nodes_reporting_infection = [
            node
            for node in self.nodes.all_nodes() 
            if (node.time_of_reporting + node.testing_delay == self.time)
        ]


        while self.end_reason == '':

            nodes_reporting_infection = [
                node
                for node in self.nodes.all_nodes() 
                if (node.time_of_reporting + node.testing_delay == self.time)
            ]

            # This chunk of code executes a days worth on infections and contact tracings
            node_count = nx.number_of_nodes(self.G)
            self.simulate_one_day()
            
            self.house_count = len(self.houses)
            self.total_cases.append(node_count)

            # While loop ends when there are no non-isolated infections
            currently_infecting = len([node for node in self.G.nodes() if self.G.nodes[node]["recovered"] is False])

            if currently_infecting == 0:
                self.end_reason = 'extinct'
                self.died_out = True
                self.day_extinct = self.time

            if len(nodes_reporting_infection) != 0:
                self.end_reason = 'infection_detected'

        # Infection Count output
        self.inf_counts = self.total_cases


    def run_simulation(self, time_out, stop_when_X_infections=False):

        # Create all the required dictionaries and reset parameters
        self.reset_simulation()

        # For recording the number of cases over time
        self.total_cases = []

        # Initial values
        self.end_reason = ''
        self.timed_out = False
        self.extinct = False
        self.day_extinct = -1

        # While loop ends when there are no non-isolated infections
        currently_infecting = len([node for node in self.nodes.all_nodes() if not node.recovered])

        while self.end_reason == '':

            # This chunk of code executes a days worth on infections and contact tracings
            node_count = nx.number_of_nodes(self.G)
            self.simulate_one_day()

            self.house_count = self.houses.count
            self.total_cases.append(node_count)

            # While loop ends when there are no non-isolated infections
            currently_infecting = len([node for node in self.nodes.all_nodes() if not node.recovered])

            if currently_infecting == 0:
                self.end_reason = 'extinct'
                self.died_out = True
                self.day_extinct = self.time

            if self.time == time_out:
                self.end_reason = 'timed_out'
                self.timed_out = True

            if stop_when_X_infections is True and currently_infecting > 5000:
                self.end_reason = 'more_than_X'
                self.timed_out = True

        # Infection Count output
        self.inf_counts = self.total_cases

    def onset_to_isolation_times(self, include_self_reports=True):
        if include_self_reports:
            return [
                node.household().isolated_time - node.symptom_onset_time
                for node in self.nodes.all_nodes()
                if node.isolated
            ]
        else:
            return [
                node.household().isolated_time - node.symptom_onset_time
                for node in self.nodes.all_nodes()
                if node.isolated
                and node.household().being_contact_traced_from is not None
            ]

    def infected_to_isolation_times(self, include_self_reports=True):
        if include_self_reports:
            return [
                node.household().isolated_time - node.time_infected
                for node in self.nodes.all_nodes()
                if node.isolated
            ]
        else:
            return [
                node.household().isolated_time - node.time_infected
                for node in self.nodes.all_nodes()
                if node.isolated
                and node.household().being_contact_traced_from is not None
            ]


    def make_proxy(self, clr, **kwargs):
        """Used to draw the lines we use in the draw network legend.

        Arguments:
            clr {str} -- the colour of the line to be drawn.

        Returns:
            Line2D -- A Line2D object to be passed to the 
        """
        return Line2D([0, 1], [0, 1], color=clr, **kwargs)


    def node_colour(self, node: Node):
        """Returns a node colour, given the current status of the node.

        Arguments:
            node {int} -- The node id

        Returns:
            str -- The colour assigned
        """
        if node.isolated is True:
            return "yellow"
        elif node.had_contacts_traced:
            return "orange"
        else:
            return "white"


    def draw_network(self):
        """Draws the network generated by the model."""

        node_colour_map = [self.node_colour(node) for node in self.nodes.all_nodes()]

        # The following chunk of code draws the pretty branching processes
        edge_colour_map = [self.nodes.G.edges[edge]["colour"] for edge in self.nodes.G.edges()]

        # Legend for explaining edge colouring
        proxies = [
            self.make_proxy(clr, lw=1) for clr in (
                self.default_edge_colour,
                self.contact_traced_edge_colour_within_house,
                self.contact_traced_edge_between_house,
                self.app_traced_edge,
                self.failed_contact_tracing
            )
        ]
        labels = (
            "Transmission, yet to be traced",
            "Within household contact tracing",
            "Between household contact tracing",
            "App traced edge",
            "Failed contact trace"
        )

        node_households = {}
        for node in self.nodes.all_nodes():
            node_households.update({node.node_id: node.household_id})

        #pos = graphviz_layout(self.G, prog='twopi')
        plt.figure(figsize=(8, 8))

        nx.draw(
            self.nodes.G,
            # pos,
            node_size=150, alpha=0.9, node_color=node_colour_map, edge_color=edge_colour_map,
            labels=node_households
        )
        plt.axis('equal')
        plt.title("Household Branching Process with Contact Tracing")
        plt.legend(proxies, labels)


class uk_model(household_sim_contact_tracing):


    def __init__(self,
        haz_rate_scale,
        contact_tracing_success_prob,
        contact_trace_delay_par,
        overdispersion,
        infection_reporting_prob,
        contact_trace,
        household_haz_rate_scale=0.77729,
        number_of_days_to_trace_backwards=2,
        number_of_days_to_trace_forwards=7,
        reduce_contacts_by=0,
        prob_has_trace_app=0,
        hh_propensity_to_use_trace_app=1,
        test_delay_mean=1.52,
        test_before_propagate_tracing=True,
        probable_infections_need_test=False,
        backwards_tracing_time_limit=None,
        starting_infections=1,
        hh_prob_will_take_up_isolation=1,
        hh_prob_propensity_to_leave_isolation=0,
        leave_isolation_prob=0,
        do_2_step=False,
        recall_probability_fall_off=1):

        super().__init__(
            haz_rate_scale=haz_rate_scale,
            contact_tracing_success_prob=contact_tracing_success_prob,
            contact_trace_delay_par=contact_trace_delay_par,
            overdispersion=overdispersion,
            infection_reporting_prob=infection_reporting_prob,
            contact_trace=contact_trace,
            household_haz_rate_scale=household_haz_rate_scale,
            do_2_step=False,
            backwards_trace=True,
            reduce_contacts_by=reduce_contacts_by,
            prob_has_trace_app=prob_has_trace_app,
            hh_propensity_to_use_trace_app=hh_propensity_to_use_trace_app,
            test_delay_mean=test_delay_mean,
            test_before_propagate_tracing=test_before_propagate_tracing,
            starting_infections=starting_infections,
            hh_prob_will_take_up_isolation=hh_prob_will_take_up_isolation,
            hh_prob_propensity_to_leave_isolation=hh_prob_propensity_to_leave_isolation,
            leave_isolation_prob=leave_isolation_prob
        )
        
        self.probable_infections_need_test = probable_infections_need_test
        self.backwards_tracing_time_limit = backwards_tracing_time_limit
        if self.backwards_tracing_time_limit is None:
            self.backwards_tracing_time_limit = float('inf')
        self.number_of_days_to_trace_backwards = number_of_days_to_trace_backwards
        self.number_of_days_to_trace_forwards = number_of_days_to_trace_forwards
        self.recall_probability_fall_off = recall_probability_fall_off

    def testing_delay(self):
        if self.test_before_propagate_tracing is False:
            return 0
        else:
            return round(npr.gamma(
                shape=self.test_delay_mean**2 / 1.11**2,
                scale=1.11**2 / self.test_delay_mean))

    def increment_contact_tracing(self):
        """
        Performs a days worth of contact tracing by:
        * Looking for nodes that have been admitted to hospital. Once a node is admitted to hospital, it's house is isolated
        * Looks for houses where the contact tracing delay is over and moves them to the contact traced state
        * Looks for houses in the contact traced state, and checks them for symptoms. If any of them have symptoms, the house is isolated

        The isolation function also assigns contact tracing times to any houses that had contact with that household

        For each node that is contact traced
        """

        # Isolate all households under observation that now display symptoms (excludes those who will not take up isolation if prob <1)
        # TODO can this be removed?
        [
            self.isolate_household(node.household())
            for node in self.nodes.all_nodes()
            if node.symptom_onset_time <= self.time
            and node.contact_traced
            and not node.isolated
        ]


        # Look for nodes in households that have been isolated, and a node in that household has had symptom onset
        # and the test result has come back. Then this is a list of households where the infection has been confirmed
        # and the node has not already propagated contact tracing
        # Households with symptoms are isoalt
        households_with_confirmed_infection = [
            node.household()
            for node in self.nodes.all_nodes()
            if node.household().isolated
            and node.household().isolated_time + node.testing_delay <= self.time
        ]

        # Remove duplicates
        households_with_confirmed_infection = list(set(households_with_confirmed_infection))

        # Propagate the contact tracing for nodes that have had symptom onset in a household that has a confirmed infection
        
        for household in households_with_confirmed_infection:
            for node in household.nodes():

                # A node is only tested when their household has been isolated and they have had symptom onset
                node_positive_test_time = max(node.symptom_onset_time, node.household().isolated_time) + node.testing_delay

                if node.propagated_contact_tracing is False and node.symptom_onset_time <= self.time and not self.probable_infections_need_test:
                    self.propagate_contact_tracing(node)
                elif node.propagated_contact_tracing is False and node_positive_test_time <= self.time and self.probable_infections_need_test:
                    self.propagate_contact_tracing(node)

        # Propagate the contact tracing for all nodes that self-reported and have had their test results come back
        # [
        #     self.propagate_contact_tracing(node)
        #     for node in self.nodes.all_nodes()
        #     if node.time_of_reporting + node.testing_delay == self.time
        #     and not node.household().propagated_contact_tracing
        # ]

        # # Propagate the contact tracing for all households that are isolated due to exposure, have developed symptoms and had a test come back
        # [
        #     self.propagate_contact_tracing(node.household())
        #     for node in self.nodes.all_nodes()
        #     if node.symptom_onset_time <= self.time
        #     and not node.household().propagated_contact_tracing
        #     and node.household().isolated_time + node.testing_delay <= self.time
        # ]

        # Update the contact tracing index of households
        # That is, re-evaluate how far away they are from a known infected case (traced + symptom_onset_time + testing_delay)
        # self.update_contact_tracing_index()

        # if self.do_2_step:
        #     # Propagate the contact tracing from any households with a contact tracing index of 1
        #     [
        #         self.propagate_contact_tracing(household)
        #         for household in self.houses.all_households()
        #         if household.contact_tracing_index == 1
        #         and not household.propagated_contact_tracing
        #         and household.isolated
        #     ]
    
    def propagate_contact_tracing(self, node: Node):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household that is under surveillance develops symptoms + gets tested.
        """
        # update the propagation data
        node.propagated_contact_tracing = True
        node.time_propagated_tracing = self.time

        # If the node's symptom onset time was more that the backwards tracing time limit days in the past, do nothing
        if node.symptom_onset_time < self.time - self.backwards_tracing_time_limit:
            pass

        # Contact tracing attempted for the household that infected the household currently propagating the infection

        infected_by_node = node.infected_by_node

        days_since_symptom_onset = self.time - node.symptom_onset_time

        # Determine if the infection was so long ago, that it is not worth contact tracing
        if days_since_symptom_onset > self.backwards_tracing_time_limit:
            time_limit_hit = True
        else:
            time_limit_hit = False

        # If the node was globally infected, we are backwards tracing and the infecting node is not None
        if not node.locally_infected and self.backwards_trace and infected_by_node and not time_limit_hit:

            # if the infector is not already isolated and the time the node was infected captured by going backwards
            # the node.time_infected is when they had a contact with their infector.
            if  not infected_by_node.isolated and node.time_infected >= self.time - self.number_of_days_to_trace_backwards:

                # Then attempt to contact trace the household of the node that infected you
                self.attempt_contact_trace_of_household(
                    house_to=infected_by_node.household(),
                    house_from=node.household(),
                    days_since_contact_occurred=node.time_infected
                    )

        # spread_to_global_node_time_tuples stores a list of tuples, where the first element is the node_id
        # of a node who was globally infected by the node, and the second element is the time of transmission
        for global_infection in node.spread_to_global_node_time_tuples:
            
            # Get the child node_id and the time of transmission/time of contact
            child_node_id, time = global_infection

            child_node = self.nodes.node(child_node_id)

            # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
            if time > node.symptom_onset_time - self.number_of_days_to_trace_backwards and time < node.symptom_onset_time + self.number_of_days_to_trace_forwards and not child_node.isolated:

                self.attempt_contact_trace_of_household(
                    house_to=child_node.household(),
                    house_from=node.household(),
                    days_since_contact_occurred=time
                    )


    def attempt_contact_trace_of_household(self, house_to: Household, house_from: Household, days_since_contact_occurred: int, contact_trace_delay: int = 0):
        # Decide if the edge was traced by the app
        app_traced = self.is_edge_app_traced(self.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing_success_prob * self.recall_probability_fall_off ** days_since_contact_occurred

        # is the trace successful
        if (npr.binomial(1, success_prob) == 1):
            # Update the list of traced households from this one
            house_from.contact_traced_household_ids.append(house_to.house_id)

            # Assign the household a contact tracing index, 1 more than it's parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            contact_trace_delay = contact_trace_delay + self.contact_trace_delay(app_traced)
            proposed_time_until_contact_trace = self.time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time until contact trace
            # Note this starts as infinity
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from.house_id

            # Edge colouring
            if app_traced:
                self.colour_node_edges_between_houses(house_to, house_from, self.app_traced_edge)
            else:
                self.colour_node_edges_between_houses(house_to, house_from, self.contact_traced_edge_between_house)
        else:
            self.colour_node_edges_between_houses(house_to, house_from, self.failed_contact_tracing)


    def new_infection(self, node_count: int, generation: int, household_id: int, serial_interval=None, infecting_node=None):
        """
        Adds a new infection to the graph along with the following attributes:
        t - when they were infected
        offspring - how many offspring they produce

        Inputs::
        G - the network object
        time - the time when the new infection happens
        node_count - how many nodes are currently in the network
        """
        # Symptom onset time
        symptom_onset_time = self.time + self.incubation_period()
        # When a node reports it's infection
        if npr.binomial(1, self.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self.reporting_delay()
        else:
            will_report_infection = False
            time_of_reporting = float('Inf')

        # We assign each node a recovery period of 21 days, after 21 days the probability of causing a new infections is
        # 0, due to the generation time distribution
        recovery_time = self.time + 14

        household = self.houses.household(household_id)

        # If the household has the propensity to use the contact tracing app, decide if the node uses the app.
        if household.propensity_trace_app:
            has_trace_app = self.has_contact_tracing_app()
        else:
            has_trace_app = False

        node = self.nodes.add_node(
            node_id=node_count,
            time=self.time,
            generation=generation,
            household=household_id,
            isolated=household.isolated,
            contact_traced=household.contact_traced,
            symptom_onset_time=symptom_onset_time,
            serial_interval=serial_interval,
            recovery_time=recovery_time,
            will_report_infection=will_report_infection,
            time_of_reporting=time_of_reporting,
            has_contact_tracing_app=has_trace_app,
            testing_delay=self.testing_delay(),
        )

        # Updates to the household dictionary
        # Each house now stores a the ID's of which nodes are stored inside the house, so that quarantining can be done at the household level
        household.node_ids.append(node_count)

        # A number of days may have passed since the house was isolated
        # We need to decide if the node has left isolation already, since it did not previously exist
        if household.isolated:
            days_isolated = int(self.time - household.isolated_time)
            for _ in range(days_isolated):
                self.decide_if_leave_isolation(node)

        node.infected_by_node = infecting_node

        if infecting_node:
            if infecting_node.household().house_id == household_id:
                node.locally_infected = True
            else:
                node.locally_infected = False
        else:
            node.locally_infected = False

        node.propagated_contact_tracing = False


class model_calibration(household_sim_contact_tracing):

    def gen_mu_local_house(self, house_size):
        """
        Generates an observation of the number of members of each generation in a local (household) epidemic.

        The definition is specific, see the paper by Pellis et al.

        Brief description:
        1) Pretend every node is infected
        2) Draw edges from nodes to other nodes in household if they make an infective contact
        3) The generation of a node is defined as the shotest path length from node 0
        4) Get the vector V where v_i is the number of members of generation i
        """

        # Set up the graph
        G = nx.DiGraph()
        G.add_nodes_from(range(house_size))

        # If the house size is 1 there is no local epidemic
        if house_size == 1:
            return [1]

        # Loop over every node in the household
        for node in G.nodes():

            # Other nodes in house
            other_nodes = [member for member in range(house_size) if member != node]

            # Get the infectious period:
            effective_infectious_period = 21

            for day in range(1, effective_infectious_period+1):

                # How many infectious contact does the node make
                contacts_within_house = npr.binomial(house_size - 1, self.local_contact_probs[house_size - 1])
                infectious_contacts = npr.binomial(contacts_within_house, self.hh_infection_probs[day])

                # Add edges to the graph based on who was contacted with an edge
                infected_nodes = npr.choice(other_nodes, infectious_contacts, replace=False)

                for infected in infected_nodes:
                    G.add_edge(node, infected)

        # Compute the generation of each node
        generations = []
        for node in G.nodes:
            if nx.has_path(G, 0, node) is True:
                generations.append(nx.shortest_path_length(G, 0, node))

        # Work of the size of each generation
        mu_local = []
        for gen in range(house_size):
            mu_local.append(sum([int(generations[i] == gen) for i in range(len(generations))]))

        return mu_local

    def estimate_mu_local(self):
        """
        Computes the expected size of each generation for a within household epidemic by simulation
        """

        repeats = 1000
        mu_local = np.array([0.]*6)

        # Loop over the possible household sizes
        for house_size in range(1, 7):
            mu_local_house = np.array([0]*6)

            # Generate some observations of the size of each generation and keep adding to an empty array
            for _ in range(repeats):
                sample = self.gen_mu_local_house(house_size)
                sample = np.array(sample + [0]*(6 - len(sample)))
                mu_local_house += sample

            # Get the mean
            mu_local_house = mu_local_house/repeats

            # Normalize by the size-biased distribution (prob of household size h * house size h) and the normalized to unit probability
            update = mu_local_house*self.size_mean_contacts_biased_distribution[house_size - 1]
            mu_local += update

        return mu_local

    def estimate_mu_global(self):
        "Performs a Monte-Carlo simulation of the number of global infections for a given house and generation"
        repeats = 1000

        total_infections = 0
        for _ in range(repeats):

            # Need to use size-biased distribution here maybe?
            house_size = self.size_of_household()

            effective_infectious_period = 21

            for day in range(0, effective_infectious_period + 1):

                # How many infectious contact does the node make
                prob = self.haz_rate_scale*current_rate_infection(day)
                contacts = self.contacts_made_today(house_size)
                contacts_within_house = npr.binomial(house_size - 1, self.local_contact_probs[house_size - 1])
                contacts_within_house = min(house_size-1, contacts_within_house)
                contacts_outside_house = contacts - contacts_within_house
                infectious_contacts = npr.binomial(contacts_outside_house, prob)
                total_infections += infectious_contacts

        return total_infections/repeats

    def calculate_R0(self):
        """
        The following function calculates R_0 for a given (alpha, p_inf) pair, using the method described in the paper by Pellis et. al.
        """

        mu_global = self.estimate_mu_global()

        mu_local = self.estimate_mu_local()

        g = lambda x: 1-sum([mu_global*mu_local[i]/(x**(i+1)) for i in range(6)])
        output = s.optimize.root_scalar(g, x0=1, x1=4)
        return output.root

    def estimate_secondary_attack_rate(self):
        """Simulates a household epidemic, with a single starting case. Outside household infections are performed but will not propagate.
        """

        # Reset the simulation to it's initial state
        self.reset_simulation()

        # Initial households are allowed to run the household epidemics
        starting_households = list(range(1, self.starting_infections))

        while len(self.active_infections) is not 0:

            # Increment the infection process
            self.increment_infection()

            # recover nodes that need it
            self.perform_recoveries()

            # set any node that was an outside-household infection to the recovered state, so that they are not simulated.
            for node in self.nodes.all_nodes():
                if node.household_id not in starting_households and not node.recovered:
                    node.recovered = True

            self.time += 1

        total_infected = sum([
            len(self.houses.household(house_id).node_ids) - 1
            for house_id in starting_households
        ])

        total_exposed = sum([
            self.houses.household(house_id).size - 1
            for house_id in starting_households
        ])

        return total_infected/total_exposed

    def generate_secondary_infection_distribution(self):

        # Reset the simulation to it's initial state
        self.reset_simulation()

        # Initial households are allowed to run the household epidemics
        starting_households = list(range(1, self.starting_infections))

        while len(self.active_infections) is not 0:

            # Increment the infection process
            self.increment_infection()

            # recover nodes that need it
            self.perform_recoveries()

            # set any node that was an outside-household infection to the recovered state, so that they are
            # not simulated.
            for node in self.nodes.all_nodes():
                if node.household_id not in starting_households and not node.recovered:
                    node.recovered = True

            self.time += 1

        return [
            len(node.spread_to)
            for node in self.nodes.all_nodes()
            if node.household_id in starting_households
        ]
