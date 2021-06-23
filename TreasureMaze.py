# This class represents the environment, which includes a maze object defined as a matrix. 

import numpy as np

visited_mark = 0.8  # The visited cells are marked by an 80% gray shade.
pirate_mark = 0.5   # The current cell where the pirate is located is marked by a 50% gray shade.

# The agent can move in one of four directions.
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

class TreasureMaze(object):

    # The maze is a two-dimensional Numpy array of floats between 0.0 and 1.0.
    # 1.0 corresponds to a free cell and 0.0 to an occupied cell.
    # pirate = (row, col) initial pirate position (defaults to (0,0))

    def __init__(self, maze, pirate=(0,0)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows-1, ncols-1)   # target cell where the "treasure" is
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not pirate in self.free_cells:
            raise Exception("Invalid Pirate Location: must sit on a free cell")
        self.reset(pirate)

    # This method resets the pirate's position.
    
    def reset(self, pirate):
        self.pirate = pirate
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = pirate
        self.maze[row, col] = pirate_mark
        self.state = (row, col, 'start')
        # To prevent the game from running excessively long, a minimum reward is defined.
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    # This method updates the state based on agent movement (valid, invalid, or blocked).
    
    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = pirate_row, pirate_col, mode = self.state

        if self.maze[pirate_row, pirate_col] > 0.0:
            self.visited.add((pirate_row, pirate_col))  # marks a visited cell

        valid_actions = self.valid_actions()
                
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:                  
            mode = 'invalid' # invalid action, no change in pirate position

        # New state
        self.state = (nrow, ncol, nmode)

    # This method returns a reward based on the agent movement guidelines.
    # The agent will be rewarded with positive or negative points, ranging from -1 to 1, for every movement. 
    # The highest reward is granted when the agent reaches the treasure cell. 
    # If the agent hits an occupied cell or attempts to go outside the maze boundary, it will incur the highest penalty. 
    # A penalty is also applied when the agent tries to revisit a cell, to prevent wandering within free cells. 
    
    def get_reward(self):
        pirate_row, pirate_col, mode = self.state
        nrows, ncols = self.maze.shape
        if pirate_row == nrows-1 and pirate_col == ncols-1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (pirate_row, pirate_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    # This method keeps track of the state and total reward based on agent action.

    def act(self,  action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    # This method returns the current environment state.
    
    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    # To help with visualization, this class includes a draw method to visualize the cells. 
    # Free cells are marked with white and occupied cells with black. 

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the pirate
        row, col, valid = self.state
        canvas[row, col] = pirate_mark
        return canvas

    # This method returns the game status.
    
    def game_status(self):
        # If the agent’s total reward goes below the minimum reward, the game is over.
        if self.total_reward < self.min_reward:
            return 'lose'
        pirate_row, pirate_col, mode = self.state
        nrows, ncols = self.maze.shape
        # If the agent reaches the treasure cell, the game is won.
        if pirate_row == nrows-1 and pirate_col == ncols-1:
            return 'win'

        # Game is not complete yet
        return 'not_over'

    # This method returns the set of valid actions starting from the current cell.
    
    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)

        if row>0 and self.maze[row-1,col] == 0.0:
            actions.remove(1)
        if row<nrows-1 and self.maze[row+1,col] == 0.0:
            actions.remove(3)

        if col>0 and self.maze[row,col-1] == 0.0:
            actions.remove(0)
        if col<ncols-1 and self.maze[row,col+1] == 0.0:
            actions.remove(2)

        return actions

