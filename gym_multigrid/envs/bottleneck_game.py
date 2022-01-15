from math import ceil, floor
from gym_multigrid.multigrid import *


class BottleneckGame(MultiGridEnv):
    """
    Environment in which the agents have pass through a bottleneck.
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        agents_index=[],
        zero_sum=False,
        view_size=7

    ):
        self.zero_sum = zero_sum
        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size,
            actions_set=SmallActions
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Add bottleneck wall
        self.grid.horz_wall(self.world, 0, floor(height/2), length=floor(width/2))
        self.grid.horz_wall(self.world, ceil(height/2), floor(height/2), length=floor(width/2))

        for i, a in enumerate(self.agents):
            if i == 0:
                a.pos = (1, 1)
                a.dir = 0
                self.put_obj(a, 1, 1)
                self.put_obj(Goal(self.world, a.index), width - 2, height - 2)
            elif i == 1:
                a.pos = (1, height - 2)
                a.dir = 0
                self.put_obj(a, 1, height - 2)
                self.put_obj(Goal(self.world, a.index), width-2, 1)

    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        for j, a in enumerate(self.agents):
            if a.index == i or a.index == 0:
                rewards[j] += reward
            if self.zero_sum:
                if a.index != i or a.index == 0:
                    rewards[j] -= reward

    def _handle_goal(self, i, rewards, fwd_pos, fwd_cell):
        # print(i == fwd_cell.index)
        # done only if agent has reached correct goal
        if i == fwd_cell.index:
            return True

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info


class BottleneckGame1A5x5(BottleneckGame):
    def __init__(self):
        super().__init__(size=4,
                         agents_index=[0],
                         zero_sum=False)


class BottleneckGame2A5x5(BottleneckGame):
    def __init__(self):
        super().__init__(size=5,
                         agents_index=[0],
                         zero_sum=False)
