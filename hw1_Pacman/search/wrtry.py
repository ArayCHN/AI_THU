# DFS
# problem的类型是可以以searchAgent.py中的PositionSearchProblem class为例
# problem的构成：对于PositionSearchProblem, problem.walls[x][y]==True/False表示这里是否有墙
# problem.getStartState==(x,y)初始位置
# problem.getSuccessors(state)：返回state后面可以走到的所有地点，是一个数组
# problem.isGoalState(state):是否达到目标
# 这里的state指的是一个tuple，（x, y）

# 调用这个函数的地方：searchAgent.py/ searchAgent...:(line 115) self.actions  = self.searchFunction(problem)

#state = problem.getStartState() #不太确定是不是tuple类型。source: searchAgents.py/positionSearchProblem.getStartState
                                #->self.startstate --> pacman.py/gameState.getPacmanPosition()
                                #->self.data.agentStates[0].getPosition()
                                #-->game.py/gameStateData

state = problem.getStartState()
path = []
frontier = util.Stack() # the frontier of nodes waiting to be expanded
prev = {}
prev[state] = None # prev[state][0,1]: prev[state][0]:previous state; prev[state][1]:the action from prev to curr
frontier.push(state)
while True:
    if frontier.isEmpty():
        return [] # can't find a path
    state = frontier.pop()
    if problem.isGoalState(state): # found the goal, dfs reaches end
        while prev[state] != None:
            path.append(prev[state][1])
            state = prev[state][0]
        else:
            path.reverse() # the path is reversed
        return path

    successors = problem.getSuccessors(state)
    for successor in successors:
        if not prev.has_key(successor[0]): # successor == (state, action, cost)
            prev[successor[0]] = state, successor[1]
            frontier.push(successor[0])

############# BFS #################
    state = problem.getStartState()
    path = []
    frontier = util.Queue() # the frontier of nodes waiting to be expanded
    prev = {}
    prev[state] = None # prev[state][0,1]: prev[state][0]:previous state; prev[state][1]:the action from prev to curr
    frontier.push(state)
    while True:
        if frontier.isEmpty():
            return [] # can't find a path
        state = frontier.pop()
        if problem.isGoalState(state): # found the goal, dfs reaches end
            while prev[state] != None:
                path.append(prev[state][1])
                state = prev[state][0]
            else:
                path.reverse() # the path is reversed
            return path

        successors = problem.getSuccessors(state)
        for successor in successors:
            if not prev.has_key(successor[0]): # successor == (state, action, cost)
                prev[successor[0]] = state, successor[1]
                frontier.push(successor[0])

############  UCS  #################
    state = problem.getStartState()
    path = []
    frontier = util.PriorityQueue() # the frontier of nodes waiting to be expanded
    prev = {}
    cost = {}
    prev[state] = None # prev[state][0,1,2]: prev[state][0]:previous state; prev[state][1]:the action from prev to curr
    cost[state] = 0 # the current minima cost to reach this state
    frontier.push(state, 0)
    while True:
        if frontier.isEmpty():
            return [] # can't find a path
        state = frontier.pop()
        if problem.isGoalState(state): # found the goal, dfs reaches end
            while prev[state] != None:
                path.append(prev[state][1])
                state = prev[state][0]
            else:
                path.reverse() # the path is reversed
            return path

        successors = problem.getSuccessors(state)
        for successor in successors:
            newcost = cost[state] + successor[2]
            if not prev.has_key(successor[0]): # successor == (state, action, cost)
                frontier.push(successor[0], newcost)
                prev[successor[0]] = state, successor[1]
                cost[successor[0]] = newcost
            else:
                if newcost < cost[successor[0]]: # else, no need to add
                    frontier.update(successor[0], newcost)
                    cost[successor[0]] = newcost
                    prev[successor[0]] = state, successor[1] # update path

############# A* ##################
    state = problem.getStartState()
    path = []
    frontier = util.PriorityQueue() # the frontier of nodes waiting to be expanded
    prev = {}
    cost = {}
    h = lambda x: heuristic(state, problem) # could be optimized, to avoid repetitive calculation
    prev[state] = None # prev[state][0,1,2]: prev[state][0]:previous state; prev[state][1]:the action from prev to curr
    frontier.push(state, h(state))
    cost[state] = 0
    while True:
        if frontier.isEmpty():
            return [] # can't find a path
        state = frontier.pop()
        if problem.isGoalState(state): # found the goal, dfs reaches end
            while prev[state] != None:
                path.append(prev[state][1])
                state = prev[state][0]
            else:
                path.reverse() # the path is reversed
            return path

        successors = problem.getSuccessors(state)
        for successor in successors:
            newcost = cost[state] + successor[2] # newcost is the g of the successor
            fvalue = newcost + h(successor[0]) # fvalue is 'f' in A*, where f = g + h, acting as 'priority' in PQ
            if not prev.has_key(successor[0]): # successor == (state, action, cost)
                frontier.push(successor[0], fvalue)
                prev[successor[0]] = state, successor[1]
                cost[successor[0]] = newcost
            else:
                if newcost < cost[successor[0]]: # else, no need to add
                    frontier.update(successor[0], fvalue)
                    cost[successor[0]] = newcost
                    prev[successor[0]] = state, successor[1] # update path

############### PRIM #############
        foodList = foodGrid.asList()
        if len(foodList) == 0: return 0
        m = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) # m-dist
        alr = {} # nodes already in the tree
        alr[foodList[0]] = True # add the first node to the tree
        foodList = {}
        food0 = foodList[0]
        for food in foodList:
            dist[food] = m(food0, food) # the dist to the tree

        ans = 0
        for i in range(0, len(foodList) - 1): # repeat n-1 times
            minima = 99999
            for food in foodList:
                if not alr.has_key(food):
                    if dist[food] < minima:
                        minima = dist[food]
                        minfood = food
            alr[minfood] = True
            ans += minima
            for food in foodList:
                dist[food] = min(dist[food], m(food, minfood))
        return ans

