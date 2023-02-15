import numpy as np 

class State:
    """
    State to store player's turn and state of table
    State is represented as an array. 
    In each cell:
        - No orb: value = 0
        - Red orbs: value = +1 * number of orbs
        - Green orbs: value = -1 * number of orbs
    
    """
    def __init__(self, M=8, N=8, givenstate=None, verbose=False) -> None:
        self.turn  = 'red'
        
        self.height = M
        self.width = N
        self.verbose = verbose
        # Color of table
        self.color = np.zeros((self.height, self.width), dtype=np.int16)
        # number of orbs 
        self.orbs = np.zeros((self.height, self.width), dtype=np.int16)

        # Generate degree array to check which 
        self.gererate_degree_array()
        if givenstate is not None:
            self.set_state(givenstate["array_view"], givenstate["player_turn"])

    def gererate_degree_array(self, ):
        self.degree = np.ones((self.height, self.width), dtype=np.int16) + 1
        self.degree[1:-1, 1:-1] += 1
        # minus at four corner
        self.degree[0, 0] -= 1
        self.degree[0, self.width-1] -= 1
        self.degree[self.height-1, 0] -= 1
        self.degree[self.height-1, self.width-1] -= 1
        if self.verbose:
            for row in self.height:
                for col in self.width:
                    print(self.degree[row, col], end=" ")
                print()

    def set_state(self, state_view, player_turn):
        array = np.array(state_view)
        self.orbs = np.abs(array)
        self.color = np.sign(array)
        self.turn = player_turn

    def get_array_view(self, ):
        return {
            'array_view': self.orbs * self.color,
            'player_turn': self.turn,
        }
