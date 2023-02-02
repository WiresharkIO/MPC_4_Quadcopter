# Flight-Control
This repository contains information and contents for flight control.

![image](https://user-images.githubusercontent.com/14985440/209779929-f99364ab-e37d-41b7-8ba9-7d6061df09ba.png)

The theory notes in this repository are self explanatory but requires prior knowledge of notations and exposure to control eqautions.

Discussion/Problems:
1. https://github.com/orgs/bitcraze/discussions/528#discussion-4776334
2. https://github.com/orgs/bitcraze/discussions/535#discussion-4795575

DONE:
- Simple visualization of controls on flight path but without considering exact equations
- Visualize with motion equations
- Implement controls with MPC
- Add static obstacles to the objective function and frame the constraints related to it
- Operate crazy-flie drone in pitch-roll invariant frame.(Literature)
- Tested velocity commands with Flowdeck and Multiranger
- Tested and callibrated positions with Lighthouse positioning system

TODO:
- Integrate MPC formulation with Crazy-flie client(50% done)
- Add dynamic obstacles to the objective function and frame the constraints related to it
