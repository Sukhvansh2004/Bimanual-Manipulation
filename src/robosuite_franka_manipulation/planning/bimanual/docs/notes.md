create the trajectory of the data in 6D space

Generate set of grasps for both the gripper types. 

Filter the grasps based on some collision criteria

Create a scene and define the collisions. take into account the 





get the trajectory (For now load it from somewhere)

Get the trajectory of the end effector poses through multiple grasps. This would result in multiple trajectories.

Check which trajectory poses are feasible and which are not. Consider collision with all items present in the scene. 




Step 1: 


Get all possible grasps for both the arms. DONE 

Create all possible pairs of the grasps. DONE 

Reject out the pairs  which are not possible at the same time. using collision checks. DONE 

How do the grasps look like numerically? Good. We can apply the transforms later on. 

Visualize and fix the SE3 frame associated with it. --DONE They should be good enoough.

Create a transformed trajectory from the grasps generated for each pair. 
I have the poses of the object. I have the transforms with respect to the end effector. Just multiply those. 


Visualize the resulting trajectories. 



Can we rank them somehow? We want to reduce/ prune possible pair of trajectories. Rank the pairs based on some metric. This should include the interaction with the object as well as the 



From the resulting set of the grasps. Create a set of trajectories over all. 




ISSUE: 
1) When the object is very small, there are chances that the parallel jaw gripper and the suctionb gripper may collide with each other. 


Grasp feasibility: Done

Segment wise feasibilty analysis : At which positions what robot might coliide with the objects. Which segment is feasible and which is not. This will give me a 2D matrix where the columns are number of segments and rows are if kth gripper can execute it or not. 


I manually define a function which accepts the trajectories of each of the gripper models them as cuboids and see if that cuboid is colliding with the environment or not. In experiments this should be a function that puts the robot at that position and see if it is feasible or not through a motion planner. 









Based on this matrix define the switch points. manually for now. Calculate switch feasibility. At which point switch is feasible. 

How to define the time? That is in our hand. Defining the cost is a major challenge here. What if multiple segments are feasible. 










Smoother joint space actions 
acceleration velocity profile 



Sort or Filter the grasps based on some rank. Assuming that the individual grasps are feasible without collision. ie grasps are correct. 

Step 1: 
Grasp Pair feasibitly check: Check if those grasps are feasible at the same time. How? There should not be any collision between the grippers at that time. This returns the possible pairs of grasps (A,B) where A is suction grasp poses and B is parallel jaw grasp poses

Step 2: 

