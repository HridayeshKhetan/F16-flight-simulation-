# F16-flight-simulation-

This is a real time, 17-state nonlinear flight dynamics simulator of the F 16 aircraft using the publicly available f16 model published in 1979 by NASA and Caltech . 

https://www.cds.caltech.edu/~murray/projects/afosr95-vehicles/models/f16/ 
https://ntrs.nasa.gov/citations/19770009539

It utilizes Python, specifically Taichi in the backend to simulate complex physics, including a layered atmospheric model that uses data from the U.S. Standard Atmosphere model 1976 for accurate information on temperature,pressure,local air density and lapse rate.
The simulator runs a 200 Hz physics engine in parallel with a 60 FPS Pygame visualization suite, enabling the execution of complex aerodynamic maneuvers without the need of dedicated gpu.

The simulation uses Newton's laws of motion in the rotating frame of the aircraft and thus takes account of the coriolis acceleration as well as rotational acceleration from the aircraft's various manoevers. 
Furthermore, fluid mechanics is also incorporated by simulating the drag and lift the F16 receives in its various maneuvers such as the Kvochur bell maneuver.
Lift,Drag and Sideforce are all computed using non dimensional coefficients that are evaluated as functions of Angle of Attack alpha’’, Sideslip “beta’”, and control deflections.
A 4 dimensional Quaternion unit is used for tracking the spatial orientation of the aircraft without running into gimbal lock errors at +-90 degree pitch.

Lastly, all the non linear differential equations are resolved mathematically using a 4th-order Runge Kutta (RK4) numerical integration algorithm which evaluates the derivatives at four distinct intermediate stages over a time step delta.
