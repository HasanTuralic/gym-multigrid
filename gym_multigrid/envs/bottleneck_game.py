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
        actions_set=SmallActions,
        goal_zone=2
    ):
        self.zero_sum = zero_sum
        self.world = World
        self.fixed_pos = fixed_pos
        self.goal_zone = goal_zone

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=512,
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

        if self.height <= 6:
            # Add bottleneck wall
            self.grid.horz_wall(self.world, 0, floor(height/2), length=floor(width/2))
            self.grid.horz_wall(self.world, ceil(width/2), floor(height/2), length=floor(width/2))
        elif self.height > 6:
            first, second = random.sample(range(2, 4), 2)
            # First wall
            self.grid.horz_wall(self.world, 0, 2, length=first)
            self.grid.horz_wall(self.world, first+1, 2, length=6-first)
            # Second wall
            self.grid.horz_wall(self.world, 0, 4, length=second)
            self.grid.horz_wall(self.world, second+1, 4, length=6-second)

        top_corners = [[1, 1], [width-2, 1]]
        bot_corners = [[1, height-2], [width-2, height-2]]
        corners = [top_corners, bot_corners]

        if self.fixed_pos:
            rand_corner, rand_a, rand_g = [0, 0, 0]
            # TODO remove this
            # self.put_obj(Lava(self.world), 2, 1)
        else:
            rand_corner, rand_a, rand_g = [
                random.randint(0, 1),
                random.randint(0, 1),
                random.randint(0, 1)
            ]

        for _, a in enumerate(self.agents):

            a_pos = corners[rand_corner][rand_a]
            g_pos = corners[1-rand_corner][rand_g]

            a.pos = (a_pos[0], a_pos[1])
            a.dir = 0
            self.put_obj(a, *a_pos)

            i, j = g_pos
            for k in range(self.goal_zone):
                if i+k < self.width and self.grid.get(i+k, j) is None:
                    self.put_obj(Goal(self.world, a.index), i+k, j)
                if i-k >= 0 and self.grid.get(i-k, j) is None:
                    self.put_obj(Goal(self.world, a.index), i-k, j)
                # if self.grid.get(i, j+k) is None:
                #    self.put_obj(Goal(self.world, a.index), i, j+k)
                # if self.grid.get(i, j-k) is None:
                #    self.put_obj(Goal(self.world, a.index), i, j-k)

            rand_corner = 1 - rand_corner
            temp = rand_a
            rand_a = 1 - rand_g
            rand_g = 1 - temp

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        success = all([_reached_goal(a) for a in self.agents])
        done = done or success
        if success:
            rewards = [self._reward(i, rewards, 1) for i in range(len(self.agents))]
        return obs, rewards, done, info


def _reached_goal(agent: Agent) -> bool:
    """Returns true if the agent is standing on the correct goal."""
    if agent.standing_on:
        return agent.standing_on.index == agent.index
    else:
        return False


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


class BottleneckGame2A7x7F(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=7,
                         see_through_walls=False,
                         fixed_pos=True,
                         actions_set=MoveActions)


class BottleneckGame2A7x7(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=7,
                         see_through_walls=False,
                         fixed_pos=False,
                         actions_set=MoveActions)


class BottleneckGame2A7x7FZ(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=7,
                         see_through_walls=False,
                         fixed_pos=True,
                         actions_set=MoveActions,
                         goal_zone=10)


class BottleneckGame2A7x7Z(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=7,
                         see_through_walls=False,
                         fixed_pos=False,
                         actions_set=MoveActions,
                         goal_zone=10)
