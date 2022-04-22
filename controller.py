from env import Environment
from agent import *
import argparse



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-agents', '--agents', nargs='+', default=[])
    args = parser.parse_args()
    agents = []
    for idx,agent in enumerate(args.agents):
        agents.append(globals()[args.a1](1,idx))
        agents.append(globals()[args.a2](1,idx))
    env = Environment(agents)
    Agent.env = env
    env.run_sim()