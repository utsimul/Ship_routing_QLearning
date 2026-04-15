# Representing a large global weather "image" into a snapshot 


### Visualization of the dynamic weather
![[reports/imgs/Map_plot.png]]

As can be seen I've plotted wind directions which are mostly going in a circular manner. 

Now I wanted to create a compact representation of the global weather, because you can't feed this entire global weather to the RL agent. That would be computationally heavy AND redundant.

Redundant because the weather at place x would be similar to the weather at place x + delta. This is also true for images. So i thought that it would be better to get a "snapshot" of the global weather.

Several techniques are there to do this.

### 1. Pooling
The first technique I'm trying is pooling. There are 2 types of pooling I am using over a given SCALE:
- mean
- max

Now this pooled result is for one SCALE. Like a filter size + stride combo (like in computer vision).

I will use different SCALES and create a stacked pyramid of the pooled snapshots at these different scales. That is what the function `build_vector_pyramid` does.

#### Analysing 

###### 20 cross 20 heatmap
![[reports/imgs/20_cross_20_heatmap.png]]

###### 10 cross 10 heatmap
[[!reports/imgs/10_cross_10_heatmap.png]]

###### 5 cross 5 heatmap
[[!reports/imgs/5_cross_5_heatmap.png]]

The 20 cross 20 one records weather data at comparatively the highest resolution, followed by 10 cross 10, and lastly by 5 cross 5. 

We can see that the weather at the bottom midde clearly stands out in all three - where the mean v and mean u are both more. comparing with the actual weather map, we can see that the wind speed there is clearly higher than the areas above. 

Okay now while writing this, I am thinking - we do need higher precision for nearby points. But we just need a "vague idea" for faraway points - , if i were some ship captain I'd have thought that. 

The less the distance from the agent's current point, the more the precision. 