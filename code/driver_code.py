import pandas as pd
import matplotlib.pyplot as plt
import data
import testing_data
import statistics
import numpy as np

pd.set_option('display.max_columns', None)

def findWaitingTime(arrival_time, processes, total_processes, burst_time, waiting_time, quantum):
    rem_bt = [0] * total_processes
    for i in range(total_processes):
        rem_bt[i] = burst_time[i]
    t = 0
    while (1):

        done = True
        for i in range(total_processes):
            var = True
            if (rem_bt[i] > 0):
                done = False
                if (arrival_time[i] <= t):
                    if (rem_bt[i] > quantum):
                        t += quantum
                        rem_bt[i] -= quantum
                    else:
                        t = t + rem_bt[i]
                        waiting_time[i] = t - burst_time[i] - arrival_time[i]
                        rem_bt[i] = 0
                else:
                    t += 1
                    var = False
            if var == False:
                break
        if (done == True):
            break

def fairnessFunc(waiting_time):
    largest_diff = 0
    for i in range(waiting_time):
        diff = abs(waiting_time[i] - waiting_time[i+1])
        if(diff > largest_diff):
            largest_diff = diff


def findTurnAroundTime(arrival_time, processes, total_processes, burst_time, waiting_time, turnaroundtime):
    for i in range(total_processes):
        turnaroundtime[i] = burst_time[i] + waiting_time[i]


def findavgTime(arrival_time, processes, total_processes, burst_time, quantum):
    waiting_time = [0] * total_processes
    turnaroundtime = [0] * total_processes
    findWaitingTime(arrival_time, processes, total_processes, burst_time, waiting_time, quantum)
    findTurnAroundTime(arrival_time, processes, total_processes, burst_time, waiting_time, turnaroundtime)
    total_waitingtime = []
    total_turnaroundtime = []
    total_wt = 0
    total_tat = 0
    for i in range(total_processes):
        total_wt = total_wt + waiting_time[i]
        total_tat = total_tat + turnaroundtime[i]
        total_waitingtime.append(total_wt)
        total_turnaroundtime.append(total_tat)
    avg_wt = total_wt / total_processes
    avg_tat = total_tat / total_processes

    process_df = pd.DataFrame()
    # process_df['process_id'] = processes
    process_df['burst_time'] = burst_time
    process_df['arrival_time'] = arrival_time
    process_df['total_waitingtime'] = total_waitingtime
    process_df['total_turnarounftime'] = total_turnaroundtime
    #####
    diff_list = []
    # largest_diff_list = []
    for i in range(len(total_waitingtime)-1):
        diff = abs(total_waitingtime[i] - total_waitingtime[i + 1])
        diff_list.append(diff)
    # process_df['diff_waiting_time'] = diff_list
    largest_diff = max(diff_list)
    # largest_diff_list.append(largest_diff)
    # print(largest_diff_list)

    return process_df, avg_tat, avg_wt, largest_diff


def plotGraphs(quantum_df, i):

    quantum_df = quantum_df.sort_values('quantum')
    plt.plot('quantum', 'fairness_score', data=quantum_df, color='magenta', label = 'fair_wt')
    plt.plot('quantum', 'average_waitingtime', data=quantum_df, color='blue', label ="avg_wt")
    plt.legend()
    plt.title('train_set_' + str(i))
    plt.grid()
    plt.xlabel('quantum value')
    plt.ylabel('time')
    plt.tight_layout()
    plt.savefig('train_set_'+ str(i) +'.png')
    plt.show()


    # quantum_df.plot.scatter(x = 'quantum', y = 'fair_waitingtime')
    # plt.show()


quantum_assignment_df = pd.DataFrame()
quantum_df = pd.DataFrame()
quantum_df_1 = pd.DataFrame()
list_dataframes = []
# Driver code
i = 1
if __name__ == "__main__":

    for train_set, at_set in zip(data.training, data.arrival_time):

        len_set = len(train_set)
        process_id = [i for i in range(0, len_set)]
        total_processes = len(process_id)

        burst_time = train_set
        burst_time = list(map(int, burst_time))

        arrival_time = at_set
        arrival_time = list(map(int, at_set))

        #quantum_list = [i for i in range(2, 100)]
        quantum_list_1 = [i for i in range(2, 9)]
        quantum_list_2 = [i for i in range(2, 90)]

        quantum_list = []
        quantum_list.append(list(quantum_list_1))
        quantum_list.append(list(quantum_list_2))

        avg_wt_list = []
        avg_tat_list = []
        largest_diff_list = []
        if train_set == data.training[0]:
            for quantum in quantum_list[0]:

                print('-------------------------------Quantum Value: '+str(quantum))
                process_df, avg_tat, avg_wt, largest_diff = findavgTime(arrival_time, process_id, total_processes, burst_time, quantum)
                largest_diff_list.append(largest_diff)


                avg_wt_list.append(avg_wt)
                avg_tat_list.append(avg_tat)
                print(process_df)

            quantum_df['quantum'] = quantum_list[0]
            quantum_df['average_waitingtime'] = avg_wt_list

            fair_waitingtime_list = [abs(avg_wt - largest_diff) for avg_wt, largest_diff in zip(avg_wt_list, largest_diff_list) ]
            quantum_df['fairness_score'] = fair_waitingtime_list
            quantum_df['average_turnaroundtime'] = avg_tat_list
            quantum_df = quantum_df.sort_values('fairness_score')
            print(quantum_df)
            list_dataframes.append(quantum_df)
            train_set_names = ['train_set_1', 'train_set_2', 'train_set_3', 'train_set_4', 'train_set_5', 'train_set_6', 'train_set_7', 'train_set_8', 'train_set_9', 'train_set_10', 'train_set_11', 'train_set_12', 'train_set_13', 'train_set_14', 'train_set_15', 'train_set_16']

            quantum_assignment_df = quantum_assignment_df.append(quantum_df.iloc[0], ignore_index=True)

            plotGraphs(quantum_df, i)
            i += 1

        else:
            for quantum in quantum_list[1]:
                print('-------------------------------Quantum Value: ' + str(quantum))
                process_df, avg_tat, avg_wt, largest_diff = findavgTime(arrival_time, process_id, total_processes, burst_time, quantum)
                largest_diff_list.append(largest_diff)

                avg_wt_list.append(avg_wt)
                avg_tat_list.append(avg_tat)
                print(process_df)


            quantum_df_1['quantum'] = quantum_list[1]
            quantum_df_1['average_waitingtime'] = avg_wt_list
            fair_waitingtime_list = [abs(avg_wt - largest_diff) for avg_wt, largest_diff in
                                     zip(avg_wt_list, largest_diff_list)]
            quantum_df_1['fairness_score'] = fair_waitingtime_list
            quantum_df_1['average_turnaroundtime'] = avg_tat_list
            quantum_df_1 = quantum_df_1.sort_values('fairness_score')
            print(quantum_df_1)
            list_dataframes.append(quantum_df_1)
            train_set_names = ['train_set_1', 'train_set_2', 'train_set_3', 'train_set_4', 'train_set_5', 'train_set_6', 'train_set_7', 'train_set_8', 'train_set_9', 'train_set_10', 'train_set_11', 'train_set_12', 'train_set_13', 'train_set_14', 'train_set_15', 'train_set_16']

            quantum_assignment_df = quantum_assignment_df.append(quantum_df_1.iloc[0], ignore_index = True)

            plotGraphs(quantum_df_1, i)
            i+=1
    quantum_assignment_df.index = ['train_set_1', 'train_set_2', 'train_set_3', 'train_set_4', 'train_set_5', 'train_set_6', 'train_set_7', 'train_set_8', 'train_set_9', 'train_set_10', 'train_set_11', 'train_set_12', 'train_set_13', 'train_set_14', 'train_set_15', 'train_set_16' ]
    print(quantum_assignment_df)

test_process = testing_data.testing_set_b
test_process_atd = testing_data.atd_set_testa
train_processes = data.training

test_processes = []
test_processes_atd = []
for _ in range(16):
    test_processes.append(list(test_process))

for _ in range(16):
    test_processes_atd.append(list(test_process_atd))



quantum_assignment_df.reset_index(inplace=True)


stats_df = pd.DataFrame()
stats_df['train_bt'] = train_processes
stats_df['test_bt'] = test_processes
stats_df['train_bt'] = stats_df['train_bt'].apply(lambda x: list(map(int, x)))
stats_df['test_bt'] = stats_df['test_bt'].apply(lambda x: list(map(int, x)))
#######
stats_df['train_atd'] = data.diff_at
stats_df['test_atd'] = test_processes_atd
stats_df['train_atd'] = stats_df['train_atd'].apply(lambda x: list(map(int, x)))
stats_df['test_atd'] = stats_df['test_atd'].apply(lambda x: list(map(int, x)))
########
stats_df['train_mean_bt'] = stats_df['train_bt'].apply(lambda x: statistics.mean(x))
stats_df['test_mean_bt'] = stats_df['test_bt'].apply(lambda x: statistics.mean(x))
########
stats_df['train_mean_atd'] = stats_df['train_atd'].apply(lambda x: statistics.mean(x))
stats_df['test_mean_atd'] = stats_df['test_atd'].apply(lambda x: statistics.mean(x))
########
stats_df['allocated_quantum'] = quantum_assignment_df['quantum']
stats_df['mean_difference_bt'] = stats_df.apply(lambda x: np.linalg.norm(x['train_mean_bt'] - x['test_mean_bt']), axis=1)
stats_df['mean_difference_at'] = stats_df.apply(lambda x: np.linalg.norm(x['train_mean_atd'] - x['test_mean_atd']), axis=1)
stats_df['min_pair'] = stats_df['mean_difference_at'] + stats_df['mean_difference_bt']
stats_df = stats_df.sort_values(by='min_pair')
stats_df = stats_df.reset_index(drop = True)
print(stats_df)

###FINAL RESULT###
print('RESULTING DATAFRAME')
result_df = pd.DataFrame(columns = ['test_set', 'allocated_quantum'])
result_df.loc[0,['test_set','allocated_quantum']] = stats_df.loc[0,['test_bt','allocated_quantum']].values
print(result_df)

##########################AT##############################

#------------------Plot-------------------#
data_boxplot_bt = data.box_plot_bt
test_set_bt = testing_data.testing_set_b
data_boxplot_bt.append(test_set_bt)

plot_df_bt = pd.DataFrame(data_boxplot_bt).T
plot_df_bt = plot_df_bt.applymap(int)
plot_df_bt.plot.box(figsize = (16, 6))
plt.title('Boxplot for burst time')
plt.savefig('bt_boxplot.png')
plt.show()

####plot for AT########
data_boxplot_at = data.box_plot_atd
test_set_atd = testing_data.atd_set_testb
data_boxplot_at.append(test_set_atd)

plot_df_at = pd.DataFrame(data_boxplot_at).T
plot_df_at = plot_df_at.applymap(int)
plot_df_at.plot.box(figsize = (16, 6))
plt.title('Boxplot for arrival time')
plt.savefig('at_boxplot.png')
plt.show()