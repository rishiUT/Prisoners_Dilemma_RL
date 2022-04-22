from env import Environment
from agent import *
import argparse



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1",type=str,default="Cynic")
    parser.add_argument("--a2",type=str,default="Easy_Mark")
    args = parser.parse_args()
    agents = []
    agents.append(globals()[args.a1](1,0))
    agents.append(globals()[args.a2](1,1))
    env = Environment(agents)
    Agent.env = env
    env.run_sim()