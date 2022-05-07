from csv import DictReader
from csv import writer
from ctypes import sizeof


result = [[],[],[],[],[]]
totals = dict()

print("TEST")

for d in DictReader(open('team_dqn_ppo.csv')):
    result[0].append(d['EPISODES'])
    result[1].append(d['DQN'])
    result[2].append(d['PPO'])
    result[3].append(d['TFT1'])
    result[4].append(d['TFT2'])

    if int(d['EPISODES']) not in totals:
        totals[int(d['EPISODES'])] = [[0, 0], [0, 0]]

    if d['DQN'] is not '-':
        totals[int(d['EPISODES'])][0][0] += 1
        if d['PPO'] is not '-' or d['TFT2'] is not '-':
            totals[int(d['EPISODES'])][0][1] += int(d['DQN'])
        else:
            totals[int(d['EPISODES'])][0][1] += 1 - int(d['DQN'])

    
    if d['PPO'] is not '-':
        totals[int(d['EPISODES'])][1][0] += 1
        if d['DQN'] is not '-' or d['TFT1'] is not '-':
            totals[int(d['EPISODES'])][1][1] += int(d['PPO'])
        else:
            totals[int(d['EPISODES'])][1][1] += 1 - int(d['PPO'])
    

def print_avgs():
    to_print = ["EPISODES", "DQN", "PPO"]
    with open('alliance_averages.csv', 'w', newline='') as f_object:  
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(to_print)
        for (key, vals) in totals.items():
            print(key)
            to_print[0] = key
            to_print[1] = vals[0][1] / vals[0][0]
            to_print[2] = vals[1][1] / vals[1][0]
            writer_object.writerow(to_print)
        # Close the file object
        f_object.close()

def print_all():
    to_print = ["EPISODES", "DQN_runs", "DQN_correct", "PPO_runs", "PPO_correct"]
    with open('results.csv', 'w', newline='') as f_object:  
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(to_print)
        for (key, vals) in totals.items():
            print(key)
            to_print[0] = key
            to_print[1] = vals[0][0]
            to_print[2] = vals[0][1]
            to_print[3] = vals[1][0]
            to_print[4] = vals[1][1]
            writer_object.writerow(to_print)
        # Close the file object
        f_object.close()

print_avgs()