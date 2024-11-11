import numpy as np
import pandas as pd
import random
import os

# Parameters
zeta = 10
randomDecisionListProb = 0
initDLProb = 0.006
deltaZ = 0.01
deltaPi = 1
maxCountZ0 = 500
maxCountPI0 = 100
maxCountZ2 = 500
maxCountPI2 = 100
noActionPenaltyItemQ = 100
noActionPenaltyItemJ = 100

# Persistent variables
VfuncList = None
stateValueListJ = None
fullyDecisionListMeanStd = None
stateActionsCell = None
ZinPE = []


def DDP_DL_path(muGraph, sigmaGraph, traversalGraph, currentState, destinationNode, synchronous):
    path = [currentState]
    while currentState != destinationNode:
        nextAction = DDP_DL(muGraph, sigmaGraph, traversalGraph, currentState, destinationNode, synchronous)
        if nextAction == currentState:
            # If no progress is made, break to avoid infinite loop
            break
        path.append(int(nextAction))
        currentState = int(nextAction)
    return path


def DDP_DL(muGraph, sigmaGraph, traversalGraph, currentState, destinationNode, synchronous):
    global VfuncList, stateValueListJ, fullyDecisionListMeanStd, stateActionsCell, ZinPE

    nodeNum = muGraph.shape[0]
    decisionListBuf = np.zeros((nodeNum, nodeNum))

    if synchronous == 0:
        VfuncList = np.zeros(nodeNum)
        stateValueListJ = np.zeros(nodeNum)
        fullyDecisionListMeanStd = np.zeros((nodeNum, nodeNum))
        stateActionsCell = [None] * nodeNum

        fullyDecisionListMeanStd, stateActionsCell = init_state_actions_cell(muGraph, traversalGraph, destinationNode)
        VfuncList = update_vfunc_list(VfuncList, stateActionsCell, noActionPenaltyItemQ, destinationNode, nodeNum)

        countPolicy = 1
        while countPolicy < maxCountPI0:
            countVfunc = 0
            while countVfunc < maxCountZ0:
                stateActionsCell = update_q_in_state_actions_cell(muGraph, VfuncList, stateActionsCell, destinationNode,
                                                                  nodeNum)
                VfuncList = update_vfunc_list(VfuncList, stateActionsCell, noActionPenaltyItemQ, destinationNode,
                                              nodeNum)
                stateActionsCell, delta = update_jsa_and_qvar_in_state_actions_cell(muGraph, sigmaGraph,
                                                                                    stateValueListJ, VfuncList,
                                                                                    stateActionsCell, zeta,
                                                                                    destinationNode)
                stateValueListJ = update_state_value_list_j(stateValueListJ, stateActionsCell, VfuncList,
                                                            noActionPenaltyItemJ, destinationNode, nodeNum)

                if delta < deltaZ:
                    break
                if random.random() < randomDecisionListProb:
                    fullyDecisionListMeanStd, stateActionsCell = update_decision_lists(stateActionsCell,
                                                                                       randomDecisionListProb,
                                                                                       destinationNode, nodeNum)
                countVfunc += 1

            fullyDecisionListMeanStd, stateActionsCell = update_decision_lists(stateActionsCell, randomDecisionListProb,
                                                                               destinationNode, nodeNum)
            errorPI = np.sum(np.abs(decisionListBuf - fullyDecisionListMeanStd))
            if errorPI < deltaPi:
                break
            decisionListBuf = fullyDecisionListMeanStd.copy()
            countPolicy += 1

        nextAction = fullyDecisionListMeanStd[0, currentState]
        return nextAction

    elif synchronous == 1:
        for i in range(nodeNum):
            if fullyDecisionListMeanStd[i, currentState] != 0:
                if traversalGraph[currentState, int(fullyDecisionListMeanStd[i, currentState])] != 0:
                    return int(fullyDecisionListMeanStd[i, currentState])
        return currentState

    elif synchronous == 2:
        stateActionsCell = update_traversal(traversalGraph, stateActionsCell, destinationNode, nodeNum)
        countPolicy = 1
        while countPolicy < maxCountPI2:
            countVfunc = 0
            while countVfunc < maxCountZ0:
                stateActionsCell = update_q_in_state_actions_cell(muGraph, VfuncList, stateActionsCell, destinationNode,
                                                                  nodeNum)
                VfuncList = update_vfunc_list(VfuncList, stateActionsCell, noActionPenaltyItemQ, destinationNode,
                                              nodeNum)
                stateActionsCell, delta = update_jsa_and_qvar_in_state_actions_cell(muGraph, sigmaGraph,
                                                                                    stateValueListJ, VfuncList,
                                                                                    stateActionsCell, zeta,
                                                                                    destinationNode)
                stateValueListJ = update_state_value_list_j(stateValueListJ, stateActionsCell, VfuncList,
                                                            noActionPenaltyItemJ, destinationNode, nodeNum)
                if delta < deltaZ:
                    break
                countVfunc += 1

            fullyDecisionListMeanStd, stateActionsCell = update_decision_lists(stateActionsCell, randomDecisionListProb,
                                                                               destinationNode, nodeNum)
            if np.sum(np.abs(decisionListBuf - fullyDecisionListMeanStd)) < deltaPi:
                break
            decisionListBuf = fullyDecisionListMeanStd.copy()
            countPolicy += 1

        nextAction = fullyDecisionListMeanStd[0, currentState]
        return nextAction


# Helper functions

def init_state_actions_cell(muGraph, traversalGraph, destinationNode):
    nodeNum = muGraph.shape[0]
    outputCell = [None] * nodeNum
    fullyDecisionListMeanStd = np.zeros((nodeNum, nodeNum))
    shortestPathList = np.zeros(nodeNum)

    for i in range(nodeNum):
        try:
            dist = np.linalg.norm(muGraph[i] - muGraph[destinationNode])
            shortestPathList[i] = dist
        except:
            shortestPathList[i] = np.inf

    for i in range(nodeNum):
        if i != destinationNode and shortestPathList[i] != np.inf:
            bufQPJlist = []
            for j in range(nodeNum):
                if muGraph[i, j] != 0 and shortestPathList[j] != np.inf:
                    bufQPJ = {
                        'Q': shortestPathList[j] + muGraph[i, j],
                        'P': traversalGraph[i, j],
                        'Jsa': 0,
                        'nextState': j,
                        'Qvar': 0,
                        'Z': 0,
                        'preZ': 0
                    }
                    bufQPJlist.append(bufQPJ)
            bufQPJlist.sort(key=lambda x: x['Q'])
            for idx, item in enumerate(bufQPJlist):
                fullyDecisionListMeanStd[idx, i] = item['nextState']
            outputCell[i] = bufQPJlist
        else:
            bufQPJ = {
                'Q': 0,
                'P': 0,
                'Jsa': 0,
                'nextState': i,
                'Qvar': 0,
                'Z': 0
            }
            outputCell[i] = [bufQPJ]

    return fullyDecisionListMeanStd, outputCell


def update_vfunc_list(VfuncList, stateActionsCell, noActionPenaltyItemV, destinationNode, nodeNum):
    for i in range(nodeNum):
        if i != destinationNode:
            bufQPJlist = stateActionsCell[i]
            value = 0
            for bi in range(len(bufQPJlist)):
                multiplicative = 1
                for ci in range(bi):
                    multiplicative *= (1 - bufQPJlist[ci]['P'])
                value += multiplicative * bufQPJlist[bi]['P'] * bufQPJlist[bi]['Q']
            penaltyV = noActionPenaltyItemV
            for di in range(len(bufQPJlist)):
                penaltyV *= (1 - bufQPJlist[di]['P'])
            VfuncList[i] = value + penaltyV
        else:
            VfuncList[i] = 0
    return VfuncList


def update_q_in_state_actions_cell(muGraph, VfuncList, stateActionsCell, destinationNode, nodeNum):
    for i in range(nodeNum):
        if i != destinationNode:
            for action in stateActionsCell[i]:
                nextState = action['nextState']
                action['Q'] = muGraph[i, nextState] + VfuncList[nextState]
    return stateActionsCell


def update_jsa_and_qvar_in_state_actions_cell(muGraph, sigmaGraph, stateValueListJ, VfuncList, stateActionsCell, zeta,
                                              destinationNode):
    delta = 0
    for i in range(len(stateActionsCell)):
        if i != destinationNode:
            for action in stateActionsCell[i]:
                nextState = action['nextState']
                action['preZ'] = action['Z']
                action['Jsa'] = sigmaGraph[i, nextState] ** 2 + muGraph[i, nextState] ** 2 + stateValueListJ[
                    nextState] + 2 * muGraph[i, nextState] * VfuncList[nextState]
                action['Qvar'] = abs(action['Jsa'] - action['Q'] ** 2)
                action['Z'] = action['Q'] + zeta * np.sqrt(action['Qvar'])
                error = abs(action['Z'] - action['preZ'])
                if error > delta:
                    delta = error
    return stateActionsCell, delta


def update_state_value_list_j(stateValueListJ, stateActionsCell, VfuncList, noActionPenaltyItemJ, destinationNode,
                              nodeNum):
    for i in range(nodeNum):
        if i != destinationNode:
            bufQPJlist = stateActionsCell[i]
            value = 0
            for bi in range(len(bufQPJlist)):
                multiplicative = 1
                for ci in range(bi):
                    multiplicative *= (1 - bufQPJlist[ci]['P'])
                value += multiplicative * bufQPJlist[bi]['P'] * bufQPJlist[bi]['Jsa']
            noActionProb = 1
            for di in range(len(bufQPJlist)):
                noActionProb *= (1 - bufQPJlist[di]['P'])
            penaltyJ = noActionProb * (
                        noActionPenaltyItemJ ** 2 + stateValueListJ[i] + 2 * noActionPenaltyItemJ * VfuncList[i])
            stateValueListJ[i] = value + penaltyJ
    return stateValueListJ


def update_decision_lists(stateActionsCell, randomDecisionListProb, destinationNode, nodeNum):
    fullyDecisionListMeanStd = np.zeros((nodeNum, nodeNum))
    for i in range(nodeNum):
        if i != destinationNode:
            bufQPJlist = stateActionsCell[i]
            if random.random() > randomDecisionListProb:
                bufQPJlist.sort(key=lambda x: x['Z'])
            else:
                random.shuffle(bufQPJlist)
            for idx, action in enumerate(bufQPJlist):
                fullyDecisionListMeanStd[idx, i] = action['nextState']
            stateActionsCell[i] = bufQPJlist
    return fullyDecisionListMeanStd, stateActionsCell


def update_traversal(traversalGraph, stateActionsCell, destinationNode, nodeNum):
    for i in range(nodeNum):
        if i != destinationNode:
            stateActionsCell[i] = [action for action in stateActionsCell[i] if
                                   traversalGraph[i, action['nextState']] != 0]
            for ti in range(nodeNum):
                if traversalGraph[i, ti] != 0 and all(action['nextState'] != ti for action in stateActionsCell[i]):
                    new_action = {
                        'Q': 0,
                        'P': traversalGraph[i, ti],
                        'Jsa': 0,
                        'nextState': ti,
                        'Qvar': 0,
                        'Z': 0,
                        'preZ': 0
                    }
                    stateActionsCell[i].append(new_action)
    return stateActionsCell


if __name__ == "__main__":
    # Load graph data from a CSV file
    map_name = 'Anaheim'  # Options: SiouxFalls, Anaheim, Friedrichshain, Winnipeg
    network_file = f'../Networks/{map_name}/{map_name}_network.csv'
    df = pd.read_csv(network_file)

    # Initialize graph matrices
    nodeNum = max(df['From'].max(), df['To'].max()) + 1
    muGraph = np.zeros((nodeNum, nodeNum))
    sigmaGraph = np.zeros((nodeNum, nodeNum))
    traversalGraph = np.zeros((nodeNum, nodeNum))

    # Populate graph data
    for _, row in df.iterrows():
        start = int(row['From'])
        end = int(row['To'])
        muGraph[start, end] = row['Cost']
        sigmaGraph[start, end] = row['sigma']
        traversalGraph[start, end] = row['prob']

    # Load OD pairs from a CSV file
    od_pairs_file = f'../Networks/{map_name}/{map_name}_OD.csv'  # Modify with actual file path
    od_pairs_df = pd.read_csv(od_pairs_file)

    # Create a folder to store the paths
    output_folder = f'../Networks/{map_name}/Benchmark_Record'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'DDP_DL.txt')

    # Write paths to a text file
    with open(output_file, 'w') as f:
        # Iterate over each OD pair and calculate the path
        for _, row in od_pairs_df.iterrows():
            origin = int(row['O'])
            destination = int(row['D'])
            path = DDP_DL_path(muGraph, sigmaGraph, traversalGraph, origin, destination, synchronous=0)
            path_str = f"Path from {origin} to {destination}: {path}\n"
            print(path_str)
            f.write(path_str)
