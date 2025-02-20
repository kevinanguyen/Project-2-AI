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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Distance to closest food
        foodList = newFood.asList()
        if not foodList:
            return float('inf')  # If no food is left, that's the best state
        closestFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])

        # Ghost distances
        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        closestGhostDistance = min(ghostDistances) if ghostDistances else float('inf')

        score = successorGameState.getScore()

        # Reward eating food
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            score += 500

        # Reward for getting closer to food
        score += 100.0 / (closestFoodDistance + 1)

        # Discourage stopping
        if action == Directions.STOP:
            score -= 50

        # Ghost handling: less aggressive penalty and use scared timer
        if closestGhostDistance <= 1 and min(newScaredTimes) == 0:
            score -= 1000
        else:
            score += 2 * closestGhostDistance

        # Encourage chasing scared ghosts
        score += sum(newScaredTimes) * 100

        # Handle capsules if available
        capsuleList = currentGameState.getCapsules()
        if capsuleList:
            capsuleDistances = [manhattanDistance(newPos, cap) for cap in capsuleList]
            score += 50.0 / (min(capsuleDistances) + 1)

        return score
     

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Start the minimax recursion from Pacman (agent index 0) and the initial depth
        def minimax(agentIndex, depth, state):
            # If the state is terminal (win/lose) or depth limit reached, return evaluation
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # If agentIndex is 0, it's Pacman's turn (maximize score)
            if agentIndex == 0:
                return max_value(agentIndex, depth, state)
            else:
                # Otherwise, it's a ghost's turn (minimize score)
                return min_value(agentIndex, depth, state)

        def max_value(agentIndex, depth, state):
            # Pacman tries to maximize the evaluation score
            best_score = float('-inf')
            best_action = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                score = minimax(1, depth, successor)  # Go to the first ghost
                if score > best_score:
                    best_score = score
                    best_action = action
            # If we are at the root call, return the action. Otherwise, return the score.
            return best_action if depth == 0 else best_score

        def min_value(agentIndex, depth, state):
            # Ghost tries to minimize the evaluation score
            best_score = float('inf')
            nextAgent = agentIndex + 1
            # If the next agent exceeds total agents, reset to Pacman and increase depth
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                depth += 1

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                score = minimax(nextAgent, depth, successor)
                best_score = min(best_score, score)
            return best_score

        # Call minimax for Pacman at the root level
        return minimax(0, 0, gameState)
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Alpha starts at -∞ (best value for maximizer so far)
        # Beta starts at +∞ (best value for minimizer so far)
        alpha = float('-inf')
        beta = float('inf')
        best_action = None

        # Iterate over all legal moves for Pacman (agent index 0)
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Compute value using the minimizer for the next agent (ghost)
            value = self.min_value(successor, 1, 0, alpha, beta)
            # Update best action if we find a higher value
            if value > alpha:
                alpha = value
                best_action = action

        return best_action

    def max_value(self, gameState, depth, alpha, beta):
        # Return evaluation if depth reached or game ended
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        v = float('-inf')
        for action in gameState.getLegalActions(0):  # Pacman moves
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.min_value(successor, 1, depth, alpha, beta))

            # Alpha-beta pruning
            if v > beta:
                return v
            alpha = max(alpha, v)

        return v

    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        # Return evaluation if depth reached or game ended
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        v = float('inf')
        num_agents = gameState.getNumAgents()

        for action in gameState.getLegalActions(agentIndex):  # Ghost moves
            successor = gameState.generateSuccessor(agentIndex, action)

            # If last ghost, next state handled by max_value (next ply)
            if agentIndex == num_agents - 1:
                v = min(v, self.max_value(successor, depth + 1, alpha, beta))
            else:
                v = min(v, self.min_value(successor, agentIndex + 1, depth, alpha, beta))

            # Alpha-beta pruning
            if v < alpha:
                return v
            beta = min(beta, v)

        return v
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Starts the expectimax search and returns the best action
        def expectimax(agentIndex, depth, state):
            # If at terminal state or depth limit, return the evaluated score
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman (Max node)
            if agentIndex == 0:
                return max_value(agentIndex, depth, state)
            else:
                return exp_value(agentIndex, depth, state)

        # Handles Pacman's turn (maximizing player)
        def max_value(agentIndex, depth, state):
            bestScore = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                score = expectimax((agentIndex + 1) % state.getNumAgents(),
                                   depth + 1 if (agentIndex + 1) % state.getNumAgents() == 0 else depth,
                                   successor)
                # Choose action with the highest score
                if score > bestScore:
                    bestScore, bestAction = score, action
            if depth == 0:
                return bestAction  # Return best action at root
            return bestScore

        # Handles ghosts' turn (chance node - expectation over actions)
        def exp_value(agentIndex, depth, state):
            actions = state.getLegalActions(agentIndex)
            probability = 1 / len(actions)  # Uniform probability
            expectedValue = 0
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                score = expectimax((agentIndex + 1) % state.getNumAgents(),
                                   depth + 1 if (agentIndex + 1) % state.getNumAgents() == 0 else depth,
                                   successor)
                expectedValue += probability * score
            return expectedValue

        # Start the expectimax search from the root
        return expectimax(0, 0, gameState)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Get Pacman's position, food, ghost states, and capsules
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # Start with the current score
    score = currentGameState.getScore()

    # Handle ghost distances (reward distance from active ghosts, penalize proximity)
    ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates]
    for ghostState, dist in zip(ghostStates, ghostDistances):
        if ghostState.scaredTimer > 0:
            score += 200 / (dist + 1)  # Incentivize chasing scared ghosts
        else:
            if dist <= 1:
                score -= 1000  # Heavy penalty for dangerous proximity
            else:
                score += dist * 5  # Reward staying away from active ghosts

    # Handle food distance (incentivize eating closest food)
    if foodList:
        closestFoodDist = min([manhattanDistance(pacmanPos, food) for food in foodList])
        score += 10 / (closestFoodDist + 1)  # Closer food = higher score
        score -= len(foodList) * 20  # Penalize remaining food

    # Handle capsules (incentivize eating capsules)
    if capsules:
        closestCapsuleDist = min([manhattanDistance(pacmanPos, cap) for cap in capsules])
        score += 25 / (closestCapsuleDist + 1)  # Closer capsule = higher score
        score -= len(capsules) * 100  # Penalize remaining capsules

    # Additional incentive if winning state
    if currentGameState.isWin():
        score += 10000

    # Heavy penalty if losing state
    if currentGameState.isLose():
        score -= 10000

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
