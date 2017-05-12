# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    ################ BELOW: MY INI VERSION AND I BELIEVE IT IS THE BEST
    # state = problem.getStartState()
    # if problem.isGoalState(state):
    #     return [Directions.STOP]
    # path = []
    # frontier = util.Stack() # the frontier of nodes waiting to be expanded
    # prev = {}
    # prev[state] = None # prev[state][0,1]: prev[state][0]:previous state; prev[state][1]:the action from prev to curr
    # frontier.push(state)
    # while True:
    #     if frontier.isEmpty():
    #         return [Directions.STOP] # can't find a path
    #     state = frontier.pop()

    #     successors = problem.getSuccessors(state)
    #     #print successors
    #     for successor in successors:
    #         if not prev.has_key(successor[0]): # successor == (state, action, cost)
    #             prev[successor[0]] = state, successor[1]
    #             #print successor
    #             if problem.isGoalState(successor[0]): # found the goal, dfs reaches end
    #                 state = successor[0]
    #                 while prev[state] != None:
    #                     path.append(prev[state][1])
    #                     state = prev[state][0]
    #                 else:
    #                     path.reverse() # the path is reversed
    #                 return path
    #             frontier.push(successor[0])
    ################### BELOW: ONE POSSIBLE VERSION FROM INTERNET #
    # fringe = util.Stack()
    # fringe.push( (problem.getStartState(), [], []) )
    # while not fringe.isEmpty():
    #     node, actions, visited = fringe.pop()
    #     if problem.isGoalState(node):
    #         return actions

    #     for coord, direction, steps in problem.getSuccessors(node):
    #         if not coord in visited:
    #             fringe.push((coord, actions + [direction], visited + [node] ))
    ################## BELOW: ONE VERSION ADAPTED FROM WEB, BUT WASTES TIME AND SPACE
    # frontier = util.Stack()
    # frontier.push((problem.getStartState(), [], []))
    # while not frontier.isEmpty():
    #     state, actions, expanded = frontier.pop()
    #     if state in expanded: continue
    #     if problem.isGoalState(state):
    #         return actions
    #     successors = problem.getSuccessors(state)
    #     for successor in successors: # successor == (state, action, cost)
    #         if not successor[0] in expanded:
    #             frontier.push((successor[0], actions + [successor[1]], expanded + [state] ))
    # else:
    #     return(Directions.STOP)
    ############### The final (optimized) version ############
    expanded = {}
    prev = {}
    state = problem.getStartState()
    prev[state] = None
    frontier = util.Stack()
    frontier.push(problem.getStartState())
    while not frontier.isEmpty():
        state = frontier.pop()
        if expanded.has_key(state): continue
        expanded[state] = True
        if problem.isGoalState(state):
            path = []
            while prev[state] != None:
                path.append(prev[state][1])
                state = prev[state][0]
            else:
                path.reverse() # the path is reversed
            return path
        successors = problem.getSuccessors(state)
        for successor in successors: # successor == (state, action, cost)
            if not expanded.has_key(successor[0]):
                frontier.push(successor[0])
                prev[successor[0]] = state, successor[1] # prev[state][0]:previous state; prev[state][1]:the action
    else:
        return(Directions.STOP)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState() # state here must be an (x, y) tuple, but in cornerprob not...
    path = []
    frontier = util.Queue() # the frontier of nodes waiting to be expanded
    prev = {}
    prev[state] = None # prev[state][0,1]: prev[state][0]:previous state; prev[state][1]:the action from prev to curr
    frontier.push(state)
    while True:
        if frontier.isEmpty():
            return [Directions.STOP] # can't find a path
        state = frontier.pop()
        if problem.isGoalState(state): # found the goal, bfs reaches end
            while prev[state] != None:
                path.append(prev[state][1])
                state = prev[state][0]
            else:
                path.reverse() # the path is reversed
            return path

        successors = problem.getSuccessors(state)
        for successor in successors:
            if not prev.has_key(successor[0]): # successor == (state, action, cost)
            # TO BE IMPROVED HERE: here a same pos will still be visited if bool status of four corners are diff
                prev[successor[0]] = state, successor[1]
                frontier.push(successor[0])

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
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
            return [Directions.STOP] # can't find a path
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

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    path = []
    frontier = util.PriorityQueue() # the frontier of nodes waiting to be expanded
    prev = {}
    cost = {}
    h = lambda x: heuristic(x, problem) # could be optimized, to avoid repetitive calculation
    prev[state] = None # prev[state][0,1,2]: prev[state][0]:previous state; prev[state][1]:the action from prev to curr
    frontier.push(state, h(state))
    cost[state] = 0
    while True:
        if frontier.isEmpty():
            return [Directions.STOP] # can't find a path
        state = frontier.pop()
        if problem.isGoalState(state): # found the goal, dfs reaches end
            while prev[state] != None:
                path.append(prev[state][1])
                state = prev[state][0]
            else:
                path.reverse() # the path is reversed
            return path

        successors = problem.getSuccessors(state)
        #print successors
        for successor in successors:
            newcost = cost[state] + successor[2] # newcost is the g of the successor
            fvalue = newcost + h(successor[0]) # fvalue is 'f' in A*, where f = g + h, acting as 'priority' in PQ
            #print successor, cost[state], newcost, h(successor[0])
            if not prev.has_key(successor[0]): # successor == (state, action, cost)
                frontier.push(successor[0], fvalue)
                prev[successor[0]] = state, successor[1]
                cost[successor[0]] = newcost
            else:
                if newcost < cost[successor[0]]: # else, no need to add
                    frontier.update(successor[0], fvalue)
                    cost[successor[0]] = newcost
                    prev[successor[0]] = state, successor[1] # update path

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
