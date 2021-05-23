import streamlit as st
import pandas as pd
import plotly.express as px
from functions import generate_data, single_step_sim, plot_arm_dist, get_df_distribution

st.set_page_config(layout="wide")
header = st.beta_container()
sim_1 = st.beta_container()

with header:
	st.title("Learning Multi-Armed Bandit")

with sim_1:
	st.header("Multi-Armed Bandit Simulation")
	input_col, display_col = st.beta_columns(2)

	input_col.markdown("### Generate Dummy Data")
	input_form = input_col.form("sim_param")
	num_arm = input_form.slider("Number of Arms", min_value=2, max_value=7, value=4)
	num_obs = input_form.slider("Number of Observation", min_value=100, max_value=10000, value=1000, step=100)
	arm_prob_dist = input_form.radio("Probability Across Arms",("Random (similar across arms)", "Bias"))
	simulate_button = input_form.form_submit_button("Simulate")
	

	# display_col.markdown("### Arms Movement")
	
	if simulate_button:
		df, probs = generate_data(num_arm, num_obs, arm_prob_dist)
		slot_selected, rewards, penalties, total_reward, beta_params = single_step_sim(df)
		input_col.markdown("### Probability of Each Arm")
		input_col.write(probs)
		input_col.markdown("### Generated Data (Head)")
		input_col.write(df.head(20))
		
		# fig, axs = plot_arm_dist(beta_params)
		# display_col.write(fig)

		snapshot_param, unpivot_dist = get_df_distribution(beta_params)
		display_col.markdown("### 10 Snapshot of Beta Distribution Parameters")
		display_col.write(snapshot_param)

		last_snapshot = unpivot_dist[unpivot_dist['iteration'] == unpivot_dist['iteration'].max()]
		# fig = px.line(last_snapshot, x="x", y="y_values", color="y_group", title='Arm Movement' % (unpivot_dist['iteration'].max()))
		# display_col.write(fig)
		display_col.markdown("### Arm Movement")
		fig = px.line(unpivot_dist, x="x", y="y_values", color="y_group", range_y=[0,unpivot_dist['y_values'].max()*1.02], animation_frame='iteration')
		display_col.write(fig)