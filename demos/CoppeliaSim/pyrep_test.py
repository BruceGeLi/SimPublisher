from pyrep import PyRep
from simpub.parser.coppelia_sim_pyrep import CoppeliasSimPyRepParser
from simpub.sim.coppelia_sim_publisher import CoppeliaSimPublisher

pr = PyRep()
# Launch the application with a scene file in headless mode
# pr.launch('/home/lige/CoppeliaSim/table_tennis.ttt', headless=False)
pr.launch('/home/lige/Codes/xinkai/PyRep/examples/scene_panda_reach_target.ttt', headless=False)
# visual_layer_list = list(range(7))
visual_layer_list = None
# visual_keyword_list = ["visualization", "table", "ball", "target"]
visual_keyword_list = ["visual", "visible"]

max_steps = 300000
# CoppeliasSimPyRepParser(pr, visual_layer_list)
# CoppeliaSimPublisher(pr, host='192.168.0.143', visual_layer_list=visual_layer_list,
CoppeliaSimPublisher(pr, host='127.0.0.1',
                     visual_layer_list=visual_layer_list,
                     visual_keyword_list=visual_keyword_list,
                     use_pyrep=True)

pr.start()  # Start the simulation

for time_step in range(max_steps):
    pr.step()  # Step the simulation
    time_step += 1

pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application