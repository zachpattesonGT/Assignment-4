import time
import mdptoolbox.example
import numpy as np


def MDPValue(P, R, mainTitle):
    # This code was originally taken and modified from https://pymdptoolbox.readthedocs.io/en/latest/api/mdp.html
    start = time.time()

    discount = 0.9

    VI = mdptoolbox.mdp.ValueIteration(P, R, discount)
    VI.run()

    end = time.time()

    totalTime = end-start

    print("MDP Value: " + mainTitle)
    # print("Time: " + str(end-start))
    print("Policy: " + str(VI.policy))
    # print("Iterations: " + str(VI.iter))
    return totalTime, VI.iter

def MDPPolicy(P, R, mainTitle):
    # This code was originally taken and modified from https://pymdptoolbox.readthedocs.io/en/latest/api/mdp.html
    start = time.time()

    discount = 0.9

    PI = mdptoolbox.mdp.PolicyIteration(P, R, discount)
    PI.run()

    end = time.time()

    totalTime = end - start

    print("MDP Policy: " + mainTitle)
    # print("Time: " + str(end - start))
    print("Policy: " + str(PI.policy))
    # print("Iterations: " + str(PI.iter))
    # print("V: " + str(PI.V))
    return totalTime, PI.iter

def QLearning(P, R, mainTitle):
    # This code was originally taken and modified from https://pymdptoolbox.readthedocs.io/en/latest/api/mdp.html
    start = time.time()

    discount = 0.9

    QI = mdptoolbox.mdp.QLearning(P, R, discount)
    QI.run()

    end = time.time()

    totalTime = end - start

    print("Q Learning: " + mainTitle)
    # print("Time: " + str(end - start))
    # print("Q Matrix: " + str(QI.Q))
    print("Policy: " + str(QI.policy))
    # print("V: " + str(QI.V))
    return totalTime


# Calling main function
if __name__ == "__main__":
    P1, R1 = mdptoolbox.example.forest(3, 4, 2, .75)
    P2, R2 = mdptoolbox.example.rand(25, 5)
    PArray = [P1, P2]
    RArray = [R1, R2]
    titleArray = ["Forest", "Random"]
    timeV = np.zeros(10)
    iterV = np.zeros(10)
    timeP = np.zeros(10)
    iterP = np.zeros(10)
    timeQ = np.zeros(10)
    for x in range(len(titleArray)):
        for y in range(0, 10):
            timeV[y], iterV[y] = MDPValue(PArray[x], RArray[x], titleArray[x])
            timeP[y], iterP[y] = MDPPolicy(PArray[x], RArray[x], titleArray[x])
            timeQ[y] = QLearning(PArray[x], RArray[x], titleArray[x])

        print("Average MDP Value Time for " + titleArray[x] + ": " + str(np.mean(timeV)))
        print("Average MDP Policy Time for " + titleArray[x] + ": " + str(np.mean(timeP)))
        print("Average MDP QLearning Time for " + titleArray[x] + ": " + str(np.mean(timeQ)))
        print("Average MDP Value Iteration for " + titleArray[x] + ": " + str(np.mean(iterV)))
        print("Average MDP Policy Iteration for " + titleArray[x] + ": " + str(np.mean(iterP)))