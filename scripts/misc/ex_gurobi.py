import gurobipy as gp
from gurobipy import GRB

# example 1: mixed integer program
#  maximize
#        x + y + 2z
#  subject to
#        x + 2y + 3z <= 4
#        x +  y      >= 1
#        x, y, z binary
try:
    # optimization environment (suppress outputs)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    # defining model
    m = gp.Model("ex1", env=env)

    # defining variables
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")

    # defining objective
    m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

    # defining constraints
    m.addConstr(x + 2 * y + 3 * z <= 4, "constr1")
    m.addConstr(x + y >= 1, "constr2")

    # optimizing and printing optimal values
    m.optimize()
    print("Example 1")
    for v in m.getVars():  # getVars() returns a list
        print(f"{v.VarName}: {v.X}")
    print(f"Optimal Value: {m.ObjVal}\n")

except gp.GurobiError as e:
    print("Error code " + str(e.errno) + ": " + str(e))

except AttributeError:
    print("Encountered an attribute error")


# example 2
# Solve a multi-commodity flow problem.  Two products ('Pencils' and 'Pens')
# are produced in 2 cities ('Detroit' and 'Denver') and must be sent to
# warehouses in 3 cities ('Boston', 'New York', and 'Seattle') to
# satisfy demand ('inflow[h,i]').
#
# Flows on the transportation network must respect arc capacity constraints
# ('capacity[i,j]'). The objective is to minimize the sum of the arc
# transportation costs ('cost[i,j]').
# Base data
commodities = ["Pencils", "Pens"]
nodes = ["Detroit", "Denver", "Boston", "New York", "Seattle"]

arcs, capacity = gp.multidict(
    {
        ("Detroit", "Boston"): 100,
        ("Detroit", "New York"): 80,
        ("Detroit", "Seattle"): 120,
        ("Denver", "Boston"): 120,
        ("Denver", "New York"): 120,
        ("Denver", "Seattle"): 120,
    }
)

# Cost for triplets commodity-source-destination
cost = {
    ("Pencils", "Detroit", "Boston"): 10,
    ("Pencils", "Detroit", "New York"): 20,
    ("Pencils", "Detroit", "Seattle"): 60,
    ("Pencils", "Denver", "Boston"): 40,
    ("Pencils", "Denver", "New York"): 40,
    ("Pencils", "Denver", "Seattle"): 30,
    ("Pens", "Detroit", "Boston"): 20,
    ("Pens", "Detroit", "New York"): 20,
    ("Pens", "Detroit", "Seattle"): 80,
    ("Pens", "Denver", "Boston"): 60,
    ("Pens", "Denver", "New York"): 70,
    ("Pens", "Denver", "Seattle"): 30,
}

# Demand for pairs of commodity-city
inflow = {
    ("Pencils", "Detroit"): 50,
    ("Pencils", "Denver"): 60,
    ("Pencils", "Boston"): -50,
    ("Pencils", "New York"): -50,
    ("Pencils", "Seattle"): -10,
    ("Pens", "Detroit"): 60,
    ("Pens", "Denver"): 40,
    ("Pens", "Boston"): -40,
    ("Pens", "New York"): -30,
    ("Pens", "Seattle"): -30,
}

# Create optimization model
env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()
m = gp.Model("netflow", env=env)

# Create variables, returns a tupledict object
# (commodities & arcs) is (jointly) the index set.
# for commodity c and arc from i to j, flow[c, i, j] is a continuous variable
flow = m.addVars(commodities, arcs, obj=cost, name="flow")

# Arc-capacity constraints
# the sum of commodity flows on arc(i, j) must be less than the arc capacity
m.addConstrs((flow.sum("*", i, j) <= capacity[i, j] for i, j in arcs), "cap")

# Equivalent version using Python looping
# for i, j in arcs:
#   m.addConstr(sum(flow[h, i, j] for h in commodities) <= capacity[i, j],
#               "cap[%s, %s]" % (i, j))


# Flow-conservation constraints
# consider node j and commodity h.
# the net inflow of h must equal the flow out minus the flow in
m.addConstrs(
    (
        flow.sum(h, "*", j) + inflow[h, j] == flow.sum(h, j, "*")
        for h in commodities
        for j in nodes
    ),
    "node",
)

# Alternate version:
# m.addConstrs(
#   (gp.quicksum(flow[h, i, j] for i, j in arcs.select('*', j)) + inflow[h, j] ==
#     gp.quicksum(flow[h, j, k] for j, k in arcs.select(j, '*'))
#     for h in commodities for j in nodes), "node")

# Compute optimal solution
m.optimize()

# Print solution
if m.Status == GRB.OPTIMAL:
    solution = m.getAttr("X", flow)
    print("Example 2")
    for h in commodities:
        print("Optimal flows for %s:" % h)
        for i, j in arcs:
            if solution[h, i, j] > 0:
                print("%s -> %s: %g" % (i, j, solution[h, i, j]))
        print()
