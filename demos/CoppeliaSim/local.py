from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import simpub.parser.coppelia_sim as cs_parser
from simpub.sim.coppelia_sim_publisher import CoppeliaSimPublisher

client = RemoteAPIClient()
sim = client.require('sim')

# sim.loadScene('scenes/gears.ttt')
sim.loadScene('table_tennis.ttt')
visual_layer_list = list(range(7))



sim.setStepping(True)

# cs_parser.CoppeliasSimParser(sim)

CoppeliaSimPublisher(sim, host='192.168.0.143', visual_layer_list=visual_layer_list)

sim.startSimulation()
while (t := sim.getSimulationTime()) < 30:
    # print(f'Simulation time: {t:.2f} [s]')
    sim.step()
sim.stopSimulation()


