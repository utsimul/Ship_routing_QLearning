# Observations

### I modified theta = <output from network> to theta = goal_direction + <output from network>

(consider plots in Obs1)
In some episodes the agent did reach goal state. What we need to consider now is the weather cost - if "goal_direction +" benefits distance but significantly reduces weather.
- change weather conditions to test what the agent does when there is a tradeoff between distance and weather. (static weather again - from the start)