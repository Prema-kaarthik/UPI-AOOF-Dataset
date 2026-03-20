import pulp
import numpy as np
import pandas as pd

def two_stage_stochastic_opt(forecast_mean, num_scenarios=200):
    # Stage 1: Strategic server allocation
    servers = pulp.LpVariable("servers", lowBound=5000, upBound=15000, cat='Integer')
    prob = pulp.LpProblem("AOOF_Stochastic", pulp.LpMinimize)
    
    # Scenario generation (Gaussian around forecast)
    scenarios = np.random.normal(forecast_mean, 0.87e6, num_scenarios)  # 0.87M from paper
    
    recourse_cost = 0
    for xi in scenarios:
        # Simple recourse cost: extra servers needed if demand > capacity
        extra = pulp.LpVariable(f"extra_{xi}", lowBound=0, cat='Continuous')
        recourse_cost += 1200 * extra  # Rs 1200/server-hour approx
    
    prob += 1200 * servers + (1/num_scenarios) * recourse_cost   # expected cost
    
    # Capacity constraint (simplified)
    prob += servers * 1500 * 86400 >= np.mean(scenarios)  # tx per day capacity
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    print(f"Optimal proactive servers: {int(servers.value())}")
    print(f"Expected total cost: Rs {pulp.value(prob.objective):,.0f}")
    
    return int(servers.value())

if __name__ == "__main__":
    forecast = 15_000_000  # example daily volume
    two_stage_stochastic_opt(forecast)
