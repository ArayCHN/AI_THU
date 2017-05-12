# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        # print legalMoves
        # print scores
        # print bestIndices
        # print("OK!")
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) # next gameState after the step is taken
        newFoodPos = currentGameState.getFood().asList() # (x, y), actually old pos here
        #newFoodPos = currentGameState.getFood().asList()
        #newFoodPos = successorGameState.getFood().asList() # food in new state will be eaten if eaten!
        newFood = successorGameState.getFood() # a gird, with True / False representing food or not
        newGhostStates = successorGameState.getGhostStates() # a list of ghostState pointers
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        pacmanPosition = successorGameState.getPacmanPosition()
        numGhosts = len(newGhostStates)
        newGhostPositions = successorGameState.getGhostPositions()
        dists = [manhattanDistance(pacmanPosition, newGhostPositions[i])\
            for i in range(0, numGhosts) if newScaredTimes[i] == 0] # distance of pacman to each unscared ghost
        penalty = 0
        for dist in dists:
            penalty = penalty - 3 / (dist - 0.5)**2
        dist = 99999
        for foodPos in newFoodPos:
          if manhattanDistance(foodPos, pacmanPosition) < dist:
            dist = manhattanDistance(foodPos, pacmanPosition)
        bonus = 2 / (dist + 0.2)

        return penalty + bonus
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        ###### USE a recursive function #####
        def dfs(state, ig, d): # ig: index of the ghost taking step now; d: the current depth(which round)
            """
              Returns the minimal score of the current state (with pacman's step already taken).
            """
            if ig == numAgents: # no more ghosts in this depth, go to next depth
                if d == self.depth: # has reached last depth
                    # print 'score in dfs:',self.evaluationFunction(state)
                    return self.evaluationFunction(state)
                # else: have not yet reached the last depth, go deeper
                actions = state.getLegalActions(0) # first step is taken by pacman
                # print 'pacman step in depth',d + 1,':', actions
                maxScore = -99999
                if actions == []:
                    maxScore = self.evaluationFunction(state) # when there's no choice for agent, stop this exploration
                for pacAction in actions:
                    newState = state.generateSuccessor(0, pacAction)
                    score = dfs(newState, 1, d + 1)
                    # print 'pacman score in depth', d,':', score
                    if score > maxScore:
                          maxScore = score # get the max score possible so as to decide which step pacman will be taking
                return maxScore

            actions = state.getLegalActions(ig)
            # print 'actions of depth', d, ':', actions ###
            minScore = 99999
            if actions == []:
                return self.evaluationFunction(state) # when there's no choice for agent,stop this exploration
            for agAction in actions: # agent action
                newState = state.generateSuccessor(ig, agAction)
                score = dfs(newState, ig + 1, d)
                if score < minScore:
                    minScore = score
            return minScore

        numAgents = gameState.getNumAgents()
        # print 'numAgents:', numAgents ###
        pacActions = gameState.getLegalActions(0)
        maxScore = -99999
        maxAction = None
        for pacAction in pacActions:
          state = gameState.generateSuccessor(0, pacAction) # assume first step is taken by pacman
          # print 'pacAction:',pacAction ###
          currScore = dfs(state, 1, 1)
          # print 'pac Score:',currScore ###
          if currScore > maxScore:
              # print 'OK!'
              maxScore = currScore
              maxAction = pacAction
        # print 'Fished!' ###
        return maxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def dfs(state, ig, d, a, b): # ig: index of the ghost taking step now; d: the current depth(which round);
                                     # a, b: alpha, beta, the current upper and lower limit of a node
            """
              Returns the minimal score of the current state (with pacman's step already taken).
            """
            if ig == numAgents: # no more ghosts in this depth, go to next depth
                if d == self.depth: # has reached last depth
                    # print 'score in dfs:',self.evaluationFunction(state)
                    ans = self.evaluationFunction(state)
                    return ans
                # else: have not yet reached the last depth, go deeper
                actions = state.getLegalActions(0) # first step is taken by pacman
                # print 'pacman step in depth',d + 1,':', actions
                maxScore = -99999
                newa = a
                if actions == []:
                    maxScore = self.evaluationFunction(state) # when there's no choice for agent, stop this exploration
                for pacAction in actions:
                    newState = state.generateSuccessor(0, pacAction)
                    score = dfs(newState, 1, d + 1, newa, b) # ab is the current smallest for last layer
                    # print 'pacman score in depth', d,':', score, 'ab:', ab
                    if score == None: continue
                    if score > b: return None # beta-prunning
                    if score > newa: newa = score
                    if score > maxScore:
                        maxScore = score # get the max score possible so as to decide which step pacman will be taking
                return maxScore

            actions = state.getLegalActions(ig)
            newb = b # the new domain for 
            minScore = 99999
            if actions == []:
                return self.evaluationFunction(state) # when there's no choice for agent,stop this exploration
            for agAction in actions: # agent action
                newState = state.generateSuccessor(ig, agAction)
                score = dfs(newState, ig + 1, d, a, newb)
                if score == None:
                    if ig < numAgents - 1: 
                        return None # son is still min, so min will always be too small
                    else:
                        continue
                if score < a: return None
                if score < newb: newb = score
                if score < minScore: minScore = score
            # if minScore == 99999 return None
            return minScore

        numAgents = gameState.getNumAgents()
        # print 'numAgents:', numAgents ###
        pacActions = gameState.getLegalActions(0)
        b = 99999
        a = -99999
        ansAction = None
        for pacAction in pacActions:
          state = gameState.generateSuccessor(0, pacAction) # assume first step is taken by pacman
          # print 'pacAction:',pacAction ###
          currScore = dfs(state, 1, 1, a, b)
          if currScore == None: continue
          # print 'pac Score:',currScore ###
          if currScore > a:
              a = currScore
              ansAction = pacAction
        # print 'Fished!' ###
        return ansAction

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def dfs(state, ig, d): # ig: index of the ghost taking step now; d: the current depth(which round)
            """
              Returns the minimal score of the current state (with pacman's step already taken).
            """
            if ig == numAgents: # no more ghosts in this depth, go to next depth
                if d == self.depth: # has reached last depth
                    # print 'score in dfs:',self.evaluationFunction(state)
                    return self.evaluationFunction(state)
                # else: have not yet reached the last depth, go deeper
                actions = state.getLegalActions(0) # first step is taken by pacman
                # print 'pacman step in depth',d + 1,':', actions
                maxScore = -99999
                if actions == []:
                    maxScore = self.evaluationFunction(state) # when there's no choice for agent, stop this exploration
                for pacAction in actions:
                    newState = state.generateSuccessor(0, pacAction)
                    score = dfs(newState, 1, d + 1)
                    # print 'pacman score in depth', d,':', score
                    if score > maxScore:
                          maxScore = score # get the max score possible so as to decide which step pacman will be taking
                return maxScore

            actions = state.getLegalActions(ig)
            # print 'actions of depth', d, ':', actions ###
            if actions == []:
                return self.evaluationFunction(state) # when there's no choice for agent,stop this exploration
            num = len(actions) # number of total possible moves
            tot = 0.0
            for agAction in actions: # agent action
                newState = state.generateSuccessor(ig, agAction)
                score = float(dfs(newState, ig + 1, d))
                tot += score
            return tot / num

        numAgents = gameState.getNumAgents()
        pacActions = gameState.getLegalActions(0)
        maxScore = -99999
        maxAction = None
        for pacAction in pacActions:
          state = gameState.generateSuccessor(0, pacAction) # assume first step is taken by pacman
          # print 'pacAction:',pacAction ###
          currScore = dfs(state, 1, 1)
          # print 'pac Score:',currScore ###
          if currScore > maxScore:
              # print 'OK!'
              maxScore = currScore
              maxAction = pacAction
        # print 'Fished!' ###
        return maxAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():  return 9999
    if currentGameState.isLose(): return -999
    foodPos = currentGameState.getFood().asList() # (x, y)
    numFood = len(foodPos)
    ghostStates = currentGameState.getGhostStates() # a list of ghostState pointers
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    numUnScared = 0 # the number of unscared ghosts
    for scareTime in scaredTimes:
        if scareTime == 0:
            numUnScared += 1
    pacmanPos = currentGameState.getPacmanPosition()
    numGhosts = len(ghostStates)
    ghostPositions = currentGameState.getGhostPositions()
    capsulePos = currentGameState.getCapsules()

    dists = [manhattanDistance(pacmanPos, ghostPositions[i])\
         for i in range(0, numGhosts) if scaredTimes[i] == 0] # distance of pacman to each unscared ghost
    penalty = 0
    for dist in dists:
        # print 'dist to ghost:', dist
        penalty = penalty - 5 / (dist - 0.5)**2
    # print 'penalty:', penalty
    dist = 99999
    for foodPos in foodPos:
        if manhattanDistance(foodPos, pacmanPos) <= dist:
            dist = manhattanDistance(foodPos, pacmanPos)
    #bonus = 10 / (dist - 0.5) + 20 - numFood
    bonus = 1 / (dist - 0.5)**2 + 10 - numFood * 5 - 100 * numUnScared
    return penalty + bonus + currentGameState.getScore()
    
# Abbreviation
better = betterEvaluationFunction

