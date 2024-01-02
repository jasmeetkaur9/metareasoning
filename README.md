# Metareasoning
# Setup 
```
1. git clone https://github.com/jasmeetkaur9/metareasoning.git
2. pip install -r requirements.txt
3. install [https://github.com/caelan/pybullet-planning](Pybullet-Planning)
4. update model files provided in data
```

# Run experiments 
```
  python scripts/exp_performance.py
  python scripts/exp_ppo.py
```
# Plots
<table>
  <tr>
    <td align="center"><img src="plots/deadline_score_1.png" alt="Alt text" style="display: block; margin: 0 auto;" title="Performance with increasing deadline" width="900" height="400">
          MCTS Agent </td>
    <td align="center"><img src="plots/navigation.png" alt="Alt text" style="display: block; margin: 0 auto;" title="PPO Agent" width="600" height="300">
          PPO Agent</td>
  </tr>
  <tr>
    <td align="center"><img src="plots/exp_11.png" alt="Alt text" style="display: block; margin: 0 auto;" title="Deadline 30, 3 symbolic plans" width="500" height="300">
    Deadline 30, 3 symbolic plans</td>
    <td align="center"><img src="plots/exp_15.png" alt="Alt text" style="display: block; margin: 0 auto;" title="Deadline 40, 2 symbolic plans" width="500" height="300">
    Deadline 30, 2 symbolic plans</td>
  </tr>
</table>


