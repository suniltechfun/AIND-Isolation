"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """   
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    aggression_weight_factor = 1.5
    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    return float(my_moves - aggression_weight_factor * opponent_moves)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    center_y_pos, center_x_pos = int(game.height / 2), int(game.width / 2)
    
    my_position = game.get_player_location(player)
    opponent_position = game.get_player_location(game.get_opponent(player))

    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    my_dist_x = abs(center_x_pos - my_position[0])
    my_dist_y = abs(center_y_pos - my_position[1])

    opponent_dist_x = abs(center_x_pos - opponent_position[0])
    opponent_dist_y = abs(center_y_pos - opponent_position[1])

    my_abs_distance = (my_dist_x * my_dist_x + my_dist_y * my_dist_y)
    opponent_abs_distance = (opponent_dist_x * opponent_dist_x + opponent_dist_y * opponent_dist_y)

    return float(10 * (my_moves - opponent_moves) + (my_abs_distance - opponent_abs_distance))



def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    """Maximize the distance between the player and the opponent, i.e., run
    away from the opponent. Returns the absolute difference between the sum of
    the location vectors, where larger differences equal higher scores."""
      
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    own_v_wall = [move for move in own_moves if move[0] == 0
                                             or move[0] == (game.height - 1)
                                             or move[1] == 0
                                             or move[1] == (game.width - 1)]

    opp_moves = game.get_legal_moves(game.get_opponent(player))
    opp_v_wall = [move for move in opp_moves if move[0] == 0
                                             or move[0] == (game.height - 1)
                                             or move[1] == 0
                                             or move[1] == (game.width - 1)]
    
    # Penalize/reward move count if some moves are against the wall
    return float(len(own_moves) - len(own_v_wall)
                 - len(opp_moves) + len(opp_v_wall))

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score_2, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return best_move
        else:
            best_move = legal_moves[0]
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def terminal_test(self, game):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        return not bool(game.get_legal_moves())

    def min_value(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game) or depth == 0 or game.is_winner(self):
            return self.score(game, self) 

        v = float('inf')
        for move in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(move), depth - 1))
          
        return v

    def max_value(self, game, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
               
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(game) or depth == 0 or game.is_winner(self):
            return self.score(game, self)
            
        v = float('-inf')
        for move in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(move), depth - 1))           
        return v


    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
               
        best_move = (-1,-1)
        # Get the legal moves for the active player
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return best_move
        else:
            best_move = legal_moves[0]

        if depth == 0:
             return game.get_player_location(self)  
        """ When we start take the center position for the first move"""
        if game.move_count == 0:
            return(int(game.height/2), int(game.width/2))
        
        best_score = float("-inf")        
           
        for move in legal_moves:
            cur_score = self.min_value(game.forecast_move(move), depth - 1)
            if cur_score >= best_score:
                best_score = cur_score
                best_move = move
        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        
        try:
            depth = 1
            while True:
                best_move = self.alphabeta(game, depth)
                depth = depth + 1
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
   
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        """*********************alpha-beta helper functions*************************
        *********************START***********************************************"""
        
        def terminal_test(self, game):
            """ Return True if the game is over or False otherwise """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            return not bool(game.get_legal_moves())
    
        def max_play(self, game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if terminal_test(self, game) or depth == 0 or game.is_winner(self):
                return self.score(game, self) 
            
            best_score = float('-inf')

            for move in game.get_legal_moves():
                best_score = max(best_score, min_play(self, game.forecast_move(move), depth - 1, alpha, beta))
                if best_score >= beta:
                    return best_score
                alpha = max(alpha, best_score)            
            return best_score

        def min_play(self, game, depth, alpha, beta):    
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
        
            if terminal_test(self, game) or depth == 0 or game.is_winner(self):
                return self.score(game, self) 
        
            best_score = float('inf')

            for move in game.get_legal_moves():
                best_score = min(best_score, max_play(self, game.forecast_move(move), depth - 1, alpha, beta))
                if best_score <= alpha:
                    return best_score
                beta = min(beta, best_score)                            
            return best_score

        """*********************alpha-beta helper functions*************************
        ******************************END****************************************"""

        """*********************alpha-beta  search**********************************
        *************************************************************************"""
                       
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)
        legal_moves = game.get_legal_moves()
        if depth == 0 or not legal_moves:
            return game.get_player_location(self)
        else:
            best_move = legal_moves[0]
			
        best_score = alpha
        move_score = float("-inf")
        
        for move in legal_moves:
            move_score = min_play(self, game.forecast_move(move), depth - 1, alpha, beta)
            """ Update alpha with in alphabeta method"""   
            alpha = max(alpha, move_score)
            if move_score > best_score:               
                best_score = move_score
                best_move = move
            
        return best_move
