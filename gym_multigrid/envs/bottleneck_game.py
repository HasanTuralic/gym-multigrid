from math import ceil, floor
import random
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
        view_size=5,
        see_through_walls=False,
        fixed_pos=True,
        actions_set=SmallActions
    ):
        self.zero_sum = zero_sum
        self.world = World
        self.fixed_pos = fixed_pos

        self.dones = []
        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))
            self.dones.append(False)

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=1024,
            # Set this to True for maximum speed
            see_through_walls=see_through_walls,
            agents=agents,
            agent_view_size=view_size,
            actions_set=actions_set
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
        self.grid.horz_wall(self.world, ceil(width/2), floor(height/2), length=floor(width/2))

        top_corners = [[1, 1], [width-2, 1]]
        bot_corners = [[1, height-2], [width-2, height-2]]
        corners = [top_corners, bot_corners]

        if self.fixed_pos:
            rand_corner, rand_a, rand_g = [0, 0, 0]
        else:
            rand_corner, rand_a, rand_g = [
                random.randint(0, 1),
                random.randint(0, 1),
                random.randint(0, 1)
            ]

        for i, a in enumerate(self.agents):

            a_pos = corners[rand_corner][rand_a]
            g_pos = corners[1-rand_corner][rand_g]

            a.pos = (a_pos[0], a_pos[1])
            a.dir = 0
            self.put_obj(a, *a_pos)
            self.put_obj(Goal(self.world, a.index), *g_pos)

            rand_corner = 1 - rand_corner
            rand_a = 1 - rand_a
            rand_g = 1 - rand_g

    def _handle_goal(self, i, rewards, fwd_pos, fwd_cell):
        # done only if agent has reached correct goal
        if i == fwd_cell.index:
            self.dones[i] = True
        return all(self.dones)

    def step(self, actions):
        self.dones = [False] * len(self.agents)
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info


class BottleneckGame1A5x5(BottleneckGame):
    def __init__(self):
        super().__init__(size=5,
                         agents_index=[0],
                         zero_sum=False,
                         fixed_pos=False)


class BottleneckGame1A5x5F(BottleneckGame):
    def __init__(self):
        super().__init__(size=5,
                         agents_index=[0],
                         zero_sum=False,
                         fixed_pos=True)


class BottleneckGame1A5x5Move(BottleneckGame):
    def __init__(self):
        super().__init__(size=5,
                         agents_index=[0],
                         zero_sum=False,
                         fixed_pos=False,
                         actions_set=MoveActions)


class BottleneckGame1A5x5FMove(BottleneckGame):
    def __init__(self):
        super().__init__(size=5,
                         agents_index=[0],
                         zero_sum=False,
                         fixed_pos=True,
                         actions_set=MoveActions)


class BottleneckGame2A7x5F(BottleneckGame):
    # Easiest possible environment for 2 agents
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=5,
                         see_through_walls=False,
                         fixed_pos=True,
                         actions_set=MoveActions)


class BottleneckGame2A7x5(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=5,
                         see_through_walls=False,
                         fixed_pos=False,
                         actions_set=MoveActions)
