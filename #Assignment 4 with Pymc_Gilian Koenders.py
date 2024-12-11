import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt

data = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv')

# I have converted 'version' column to binary: gate_30 = 0, gate_40 = 1
data['version_numeric'] = data['version'].map({'gate_30': 0, 'gate_40': 1})

# You need two groups with one the gate 30 and the other gate 40
group_control = data[data['version_numeric'] == 0] # I assumed that people that get to gate 30 will retain less people because they have to spend less time on the game.
group_treatment = data[data['version_numeric'] == 1]

print(data.head(50)) # With outcome I assumed a higher retention rate for people that get to back one day later, as there went little time over the game.
# I am only not sure if getting to gate 40 with retain more people because you have to spend more time than.

# Extract retention_1 and retention_7 for each group
retention1_control = group_control['retention_1']
retention1_treatment = group_treatment['retention_1']
retention7_control = group_control['retention_7']
retention7_treatment = group_treatment['retention_7']

alpha_prior_1, beta_prior_1 = 1, 1  # Prior for retention 1 (50%) because I believe that people are more likely to retain if they come back to the game the next day.
alpha_prior_7, beta_prior_7 = 1, 9  # Prior for retention 7 (10%)


# Bayesian model for retention_1
with pm.Model() as model_retention1:
    
    p_control_1 = pm.Beta('p_control_1', alpha_prior_1, beta_prior_1)
    p_treatment_1 = pm.Beta('p_treatment_1', alpha_prior_1, beta_prior_1)
    
    retention1_control_obs = pm.Bernoulli('retention1_control_obs', p_control_1, observed=retention1_control)
    
    retention1_treatment_obs = pm.Bernoulli('retention1_treatment_obs', p_treatment_1, observed=retention1_treatment)
    
    trace_retention_1 = pm.sample(20000, return_inferencedata=True) # I picked a higher number of samples as I was not completly sure about the outcome.

# Bayesian model for retention_7
with pm.Model() as model_retention7:
    
    p_control_7 = pm.Beta('p_control_7', alpha_prior_7, beta_prior_7)
    p_treatment_7 = pm.Beta('p_treatment_7', alpha_prior_7, beta_prior_7)
    
    retention7_control_obs = pm.Bernoulli('retention7_control_obs', p_control_7, observed=retention7_control)
    
    retention7_treatment_obs = pm.Bernoulli('retention7_treatment_obs', p_treatment_7, observed=retention7_treatment)
    
    trace_retention_7 = pm.sample(20000, return_inferencedata=True)

# I started with calculating probability instead of a graph because I thought that it would be faster and less programming.
prob_treatment_better_1 = (trace_retention_1.posterior['p_treatment_1'].mean(dim=["chain", "draw"]) >
                           trace_retention_1.posterior['p_control_1'].mean(dim=["chain", "draw"])).mean().item()
prob_treatment_better_7 = (trace_retention_7.posterior['p_treatment_7'].mean(dim=["chain", "draw"]) >
                           trace_retention_7.posterior['p_control_7'].mean(dim=["chain", "draw"])).mean().item()

# Results printing for retention_1
print(f"Control Mean Retention_1 Rate: {trace_retention_1.posterior['p_control_1'].mean(dim=['chain', 'draw']).item():.4f}")
print(f"Treatment Mean Retention_1 Rate: {trace_retention_1.posterior['p_treatment_1'].mean(dim=['chain', 'draw']).item():.4f}")

# Results printing for retention_7
print(f"Control Mean Retention_7 Rate: {trace_retention_7.posterior['p_control_7'].mean(dim=['chain', 'draw']).item():.4f}")
print(f"Treatment Mean Retention_7 Rate: {trace_retention_7.posterior['p_treatment_7'].mean(dim=['chain', 'draw']).item():.4f}")

#The outcome was very close together and with that I actually could not make conclusions. So I looked at the baseball case, but was not able to make the same plots quickly
# So I rewrote it in a way that worked for me and now I would get a visual representation of the data.

# Extracting the posterior samples so I got plot them
posterior_control_1 = trace_retention_1.posterior['p_control_1'].values.flatten()
posterior_treatment_1 = trace_retention_1.posterior['p_treatment_1'].values.flatten()
posterior_control_7 = trace_retention_7.posterior['p_control_7'].values.flatten()
posterior_treatment_7 = trace_retention_7.posterior['p_treatment_7'].values.flatten()

# Plot posterior distributions for retention_1
plt.figure(figsize=(10, 6))
plt.hist(posterior_control_1, bins=50, alpha=0.5, label='Control (Gate 30)', density=True)
plt.hist(posterior_treatment_1, bins=50, alpha=0.5, label='Treatment (Gate 40)', density=True)
plt.axvline(posterior_control_1.mean(), color='blue', linestyle='--', label='Control Mean')
plt.axvline(posterior_treatment_1.mean(), color='orange', linestyle='--', label='Treatment Mean')
plt.title("Posterior Distributions of Retention1 Rates")
plt.xlabel("Retention Rate")
plt.ylabel("Density")
plt.legend()
plt.show()

# Plot posterior distributions for retention_7
plt.figure(figsize=(10, 6))
plt.hist(posterior_control_7, bins=50, alpha=0.5, label='Control (Gate 30)', density=True)
plt.hist(posterior_treatment_7, bins=50, alpha=0.5, label='Treatment (Gate 40)', density=True)
plt.axvline(posterior_control_7.mean(), color='blue', linestyle='--', label='Control Mean')
plt.axvline(posterior_treatment_7.mean(), color='orange', linestyle='--', label='Treatment Mean')
plt.title("Posterior Distributions of Retention7 Rates")
plt.xlabel("Retention Rate")
plt.ylabel("Density")
plt.legend()
plt.show()


'''
''' # this is just some code I made to verify if I did it correct. I still had the feeling that the pymc code is kind of a black box where I don't know what happens exactly inside.
# I mean I know what you need to do to connect the dots, but does the computer know what is going on in my brain.


def analyze_player_engagement(data):
    total_plays = data['sum_gamerounds'].sum() # The variable sum_gamerounds is the total number of plays in 14 days
    avg_plays = data['sum_gamerounds'].mean()
    active_players = data[(data['retention_1'] == 1) | (data['retention_7'] == 1)].shape[0]
    total_players = data.shape[0]
    active_percentage = (active_players / total_players) * 100

    print(f"Total plays in 14 days: {total_plays}")
    print(f"Average plays per player in 14 days: {avg_plays:.2f}")
    print(f"Active players (1-day or 7-day retention): {active_players}")
    print(f"Percentage of active players: {active_percentage:.2f}%\n")

    num_unique_users = data['userid'].nunique()
    print(f"Number of unique user IDs: {num_unique_users}") # I have add this to do some retention analysis on my calculator

# Below is the code for the rest of the analysis
def analyze_conversion(data, gate_level):
    if gate_level == 30:
        group = data[data['version'] == 'gate_30']
    elif gate_level == 40:
        group = data[data['version'] == 'gate_40']
    else:
        raise ValueError("Invalid gate level. Use 30 or 40.")

    passed_players = group[group['sum_gamerounds'] > gate_level].shape[0]
    kept_playing = group[(group['sum_gamerounds'] > gate_level) & 
                         ((group['retention_1'] == 1) | (group['retention_7'] == 1))].shape[0]

    conversion_rate = (kept_playing / passed_players) * 100 if passed_players > 0 else 0

    print(f"Gate Level: {gate_level}")
    print(f"Players that passed level {gate_level}: {passed_players}")
    print(f"Players that passed level {gate_level} and kept playing: {kept_playing}")
    print(f"Conversion rate: {conversion_rate:.2f}%\n")

# This is the code for calculating retention rates
def calculate_retention_rates(data):
    retention_1_rate = data['retention_1'].mean() * 100
    retention_7_rate = data['retention_7'].mean() * 100

    print(f"Overall Retention Rate after 1 day: {retention_1_rate:.2f}%")
    print(f"Overall Retention Rate after 7 days: {retention_7_rate:.2f}%\n")

# Printing all the anlysis results for gate 30 and 40
print("Overall Engagement Analysis:")
analyze_player_engagement(data)

print("Conversion Analysis for Gate Level 30:")
analyze_conversion(data, gate_level=30)

print("Conversion Analysis for Gate Level 40:")
analyze_conversion(data, gate_level=40)

print("Retention Rates Analysis:")
calculate_retention_rates(data)